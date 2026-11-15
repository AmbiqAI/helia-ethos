"""Join catalog hardware facts with explicit, consumer-owned simulation policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from helia_platforms import builtin
from helia_platforms.model import Provenance as CatalogProvenance
from pydantic import Field, ValidationError

from .catalog import CatalogNpuFacts, read_catalog_npu
from .soc import (
    MemoryDef,
    ModeClocks,
    NpuFamily,
    Provenance,
    ProvenanceStatus,
    SocDefinition,
    SocError,
    SocOrigin,
)
from .soc_schema import MemoryModel, SocBoundaryModel, describe_soc_validation_error, memory_from_model
from .yaml_loading import load_yaml_mapping


class VelaPortClass(IntEnum):
    """Compiler/timing-adapter classes, not physical AXI manager numbers."""

    SRAM = 0
    EXTERNAL = 1


@dataclass(frozen=True)
class PortDefault:
    """The explicit default occupant of a compiler port class."""

    port: VelaPortClass
    memory: str


@dataclass(frozen=True)
class SocModelPolicy:
    """Timing inputs and placement policy for a specific NPU configuration.

    Memory definitions contain tool timings and provenance; catalog path ceilings
    are deliberately not substituted for them. An explicit caller-selected policy
    also works for a derived SoC without guessing its unexposed overlay ancestry.
    """

    id: str
    name: str
    npu: NpuFamily
    macs: int
    memories: tuple[MemoryDef, ...]
    defaults: tuple[PortDefault, ...]
    assumed_memories: tuple[str, ...] = ()


@dataclass(frozen=True)
class CatalogSocProjection:
    """An executable SoC projection alongside its full catalog facts and policy id."""

    soc: SocDefinition
    facts: CatalogNpuFacts
    policy_id: str


class _PolicyDocument(SocBoundaryModel):
    name: str
    reference_target: str
    revision: str
    defaults: dict[VelaPortClass, str]
    memories: dict[str, MemoryModel]
    comparison_memories: dict[str, MemoryModel] = Field(default_factory=dict)


def at110_model_policy(
    *, comparison_memories: tuple[str, ...] = (), on_read: Callable[[Path, bytes], None] | None = None
) -> SocModelPolicy:
    """Load consumer-owned AT110 timings, with explicit optional comparison memories.

    Compatibility geometry comes from the unmodified catalog target. Comparison
    memory parameters describe an experiment, not populated hardware or NPU reach.
    """
    path = Path(__file__).parent / "model_policies/atomiq110.yaml"
    try:
        document = _PolicyDocument.model_validate(load_yaml_mapping(path, what="NPU model policy", on_read=on_read))
    except ValidationError as exc:
        raise SocError(f"{path}: {describe_soc_validation_error(exc)}") from exc
    except (OSError, ValueError) as exc:
        raise SocError(f"{path}: {exc}") from exc
    unknown = set(comparison_memories) - document.comparison_memories.keys()
    if unknown or len(comparison_memories) != len(set(comparison_memories)):
        raise SocError(f"{path}: unknown or repeated comparison memories: {comparison_memories}")
    overlap = document.memories.keys() & document.comparison_memories.keys()
    if overlap:
        raise SocError(f"{path}: comparison memories overlap catalog memories: {sorted(overlap)}")
    reference = read_catalog_npu(document.reference_target, catalog=builtin(include_plugins=False))
    npu = reference.npu.npu
    selected = {**document.memories, **{name: document.comparison_memories[name] for name in comparison_memories}}
    return SocModelPolicy(
        id=f"{document.reference_target}-model-{document.revision}",
        name=document.name,
        npu=NpuFamily(f"{npu.family.rsplit('-', 1)[0]}-{npu.model}"),
        macs=npu.macs_per_cycle,
        memories=tuple(memory_from_model(name, value) for name, value in selected.items()),
        defaults=tuple(PortDefault(port, name) for port, name in document.defaults.items()),
        assumed_memories=comparison_memories,
    )


def _provenance(record: CatalogProvenance) -> Provenance:
    # Preserve the catalog's original status in the source description. Only
    # measurements are promoted to HELX's narrower "confirmed" meaning.
    status = {
        "measured": ProvenanceStatus.CONFIRMED,
        "fitted": ProvenanceStatus.FITTED,
    }.get(record.status, ProvenanceStatus.UNVERIFIED)
    return Provenance(
        status=status,
        source=f"helia-platforms ({record.status}): " + ("; ".join(record.source) or "source not recorded"),
        reviewed_by=record.verified_by,
        reviewed_on=record.verified_on,
        notes=record.notes,
    )


def project_catalog_soc(facts: CatalogNpuFacts, *, policy: SocModelPolicy) -> CatalogSocProjection:
    """Project catalog facts with explicitly selected policy, preserving both sources.

    Missing/denied memory reach, incompatible NPU configuration, and ambiguous
    defaults are user-actionable errors. No catalog ordering, Access.port or
    path characterization selects compiler behavior. Physical capacity remains in
    ``facts``; it is not a Corstone linker capacity or a memory-placement decision.
    """
    prefix = f"{facts.target_id}: policy '{policy.id}'"
    if facts.implementation_id is not None:
        raise SocError(f"{prefix} requires a SoC target; implementation timing needs a separate policy")
    npu = facts.npu.npu
    try:
        family = NpuFamily(f"{npu.family.rsplit('-', 1)[0]}-{npu.model}")
    except ValueError as exc:
        raise SocError(f"{prefix} cannot model NPU '{npu.accelerator_config}'") from exc
    if family != policy.npu or npu.macs_per_cycle != policy.macs:
        raise SocError(f"{prefix} does not support '{npu.accelerator_config}'")
    if facts.npu.axi_width_bits is None:
        raise SocError(f"{prefix} requires a catalog AXI bus width")
    if family is NpuFamily.ETHOS_U85 and facts.npu.sram_port_hash is None:
        raise SocError(f"{prefix} requires a catalog SRAM port hash")
    paths = {path.region.name: path for path in facts.memory_paths}
    memories = {memory.name: memory for memory in policy.memories}
    if not memories or len(memories) != len(policy.memories):
        raise SocError(f"{prefix} needs uniquely named modeled memories")
    if not set(policy.assumed_memories).issubset(memories):
        raise SocError(f"{prefix} assumes a memory not present in its model")
    for memory in memories.values():
        try:
            VelaPortClass(memory.axi_port)
        except ValueError as exc:
            raise SocError(f"{prefix} has an unknown compiler port class for '{memory.name}'") from exc
        if memory.name not in policy.assumed_memories and (
            memory.name not in paths or not paths[memory.name].reachable
        ):
            raise SocError(f"{prefix} requires catalog NPU reach to '{memory.name}'")
    defaults = {item.port: item.memory for item in policy.defaults}
    ports = {memory.axi_port for memory in memories.values()}
    if len(defaults) != len(policy.defaults) or set(defaults) != ports:
        raise SocError(f"{prefix} needs exactly one explicit default per modeled port class")
    for port, name in defaults.items():
        if name not in memories or memories[name].axi_port != port:
            raise SocError(f"{prefix} default '{name}' does not occupy compiler port {port}")
    # Existing SocDefinition consumers select the first memory in each class.
    # Construct that order from explicit defaults, never catalog/policy input order.
    ordered = sorted(
        memories.values(), key=lambda m: (m.axi_port, m.name != defaults[VelaPortClass(m.axi_port)], m.name)
    )
    modes = {point.name.upper(): ModeClocks(point.name.upper(), point.mhz * 1e6) for point in facts.clock_points}
    if not modes or len(modes) != len(facts.clock_points):
        raise SocError(f"{prefix} needs distinct NPU clock points")
    provenance = {f"memories.{m.name}": m.provenance for m in ordered if m.provenance is not None}
    for name in policy.assumed_memories:
        provenance[f"catalog.reach.{name}"] = Provenance(
            status=ProvenanceStatus.UNVERIFIED,
            source=f"Explicit comparison memory in policy {policy.id}; no populated device or confirmed reach",
        )
    for path in facts.memory_paths:
        if path.region.name in memories and path.region.name not in policy.assumed_memories and path.access is not None:
            record = path.access.field_provenance.get("access", path.access.provenance)
            provenance[f"catalog.reach.{path.region.name}"] = _provenance(record)
    for field in ("npu.family", "npu.model", "npu.macs_per_cycle", "axi_width_bits", "sram_port_hash"):
        key = f"npus.{facts.npu.id}.{field}"
        record = facts.soc.field_provenance.get(key, facts.soc.provenance)
        provenance["sram_ports" if field == "sram_port_hash" else f"catalog.{field}"] = _provenance(record)
    for point in facts.clock_points:
        provenance[f"catalog.clock.{point.name}"] = _provenance(point.field_provenance.get("mhz", point.provenance))
    soc = SocDefinition(
        id=facts.soc.id,
        name=policy.name,
        npu=family.value,
        macs=npu.macs_per_cycle,
        axi_bus_width_bits=facts.npu.axi_width_bits,
        modes=modes,
        memories={m.name: m for m in ordered},
        origin=SocOrigin.DERIVED,
        provenance=provenance,
        sram_port_hash=facts.npu.sram_port_hash,
    )
    return CatalogSocProjection(soc=soc, facts=facts, policy_id=policy.id)

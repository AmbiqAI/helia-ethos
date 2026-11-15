"""The SoC YAML boundary: Pydantic models for a SoC definition, and the loader.

Tools that carry more than the SoC (helx adds an energy model and a revision
history) extend :class:`SocModel` and pass the validated SoC part to
:func:`soc_from_model`.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .soc import (
    MemoryDef,
    MemoryParams,
    ModeClocks,
    Provenance,
    ProvenanceStatus,
    SocDefinition,
    SocError,
    SocOrigin,
    SocRevision,
    SocStage,
    default_axi_beat_bits,
)
from .yaml_loading import load_yaml_mapping


class SocSchemaError(ValueError):
    pass


class SocBoundaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)


class ProvenanceModel(SocBoundaryModel):
    """Where a value came from and how far it can be trusted.

    ``status`` is the only required field: ``confirmed`` (measured or reviewed
    against measurements by the owning team), ``fitted`` (tuned so the model
    reproduces RTL or FPGA measurements), or ``unverified`` (specification,
    estimate or placeholder). ``fields`` carries a record per value for the numbers in
    the block whose status differs from the block's; the field names are
    checked by the block that owns them.
    """

    status: Literal["confirmed", "fitted", "unverified"]
    """confirmed (measured or reviewed against measurements), fitted (tuned to reproduce measurements) or unverified
    (spec, estimate or placeholder).
    """
    source: str | None = None
    """Where the value came from: a document, a measurement, a review."""
    reviewed_by: str | None = None
    """The team that reviewed the value."""
    reviewed_on: datetime.date | None = None
    """Date of that review."""
    notes: str | None = None
    """Caveats a reader of the number should know."""
    fields: dict[str, ProvenanceModel] | None = None
    """Per-value records for the numbers in the block whose status differs from the block's."""

    @field_validator("status", mode="before")
    @classmethod
    def reject_retired_statuses(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() == "derived":
            raise ValueError(
                "status 'derived' is retired: use 'fitted' for a value tuned to reproduce measurements, "
                "'unverified' for an estimate or assumption"
            )
        if isinstance(value, str) and value.strip().lower() == "reviewed":
            raise ValueError("status 'reviewed' is retired: write 'confirmed'")
        return value

    @model_validator(mode="after")
    def field_records_are_flat(self) -> ProvenanceModel:
        for name, record in (self.fields or {}).items():
            if record.fields:
                raise ValueError(f"provenance.fields.{name}: per-field records cannot carry fields of their own")
        return self

    def check_field_names(self, allowed: set[str], where: str) -> None:
        unknown = sorted(set(self.fields or {}) - allowed)
        if unknown:
            raise ValueError(f"{where}.provenance.fields names unknown values: {', '.join(unknown)}")

    def to_provenance(self) -> Provenance:
        return Provenance(
            status=ProvenanceStatus(self.status),
            source=self.source,
            reviewed_by=self.reviewed_by,
            reviewed_on=self.reviewed_on,
            notes=self.notes,
            fields={name: record.to_provenance() for name, record in (self.fields or {}).items()},
        )


class ModeModel(SocBoundaryModel):
    npu_clock_hz: float
    """NPU clock in this mode, Hz."""

    @field_validator("npu_clock_hz", mode="before")
    @classmethod
    def reject_boolean_clock(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("must be a number, not a boolean")
        return value


class MemoryModel(SocBoundaryModel):
    axi_port: int
    """Port class the memory sits on: 0 (Arm AXI_SRAM, Vela Sram) or 1 (Arm AXI_EXT, Vela Dram)."""
    read_latency_ns: float
    """Read latency the NPU sees, ns."""
    write_latency_ns: float
    """Write latency the NPU sees, ns."""
    max_outstanding: int
    """Outstanding read transactions the path sustains; also the write limit unless max_outstanding_writes is set."""
    max_outstanding_writes: int | None = None
    """Outstanding write transactions the path sustains, when it differs from max_outstanding."""
    burst_length: int
    """Burst length, bytes. Ethos-U85 FVP runs need 64, 128 or 256: the runner's driver splits AXI bursts at it."""
    ports_used: int = 1
    """How many ports of this class Vela is told exist, 1 or 2."""
    axi_clock_hz: float | None = None
    """AXI clock of the memory path, Hz; the bandwidth per NPU cycle is derived from it and the NPU clock."""
    bandwidth_fraction: float | None = None
    """Alternative to axi_clock_hz: fraction of one data beat per NPU cycle the port sustains."""
    provenance: ProvenanceModel | None = None
    """Where the memory's values came from."""

    @field_validator(
        "axi_port",
        "max_outstanding",
        "max_outstanding_writes",
        "burst_length",
        "ports_used",
        "read_latency_ns",
        "write_latency_ns",
        "axi_clock_hz",
        "bandwidth_fraction",
        mode="before",
    )
    @classmethod
    def reject_boolean_numbers(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("must be a number, not a boolean")
        return value

    PROVENANCE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "axi_port",
            "read_latency_ns",
            "write_latency_ns",
            "max_outstanding",
            "max_outstanding_writes",
            "burst_length",
            "ports_used",
            "axi_clock_hz",
            "bandwidth_fraction",
        }
    )

    @model_validator(mode="after")
    def validate_memory(self) -> MemoryModel:
        if self.axi_port not in (0, 1):
            raise ValueError(f"axi_port must be 0 or 1, got {self.axi_port}")
        if self.axi_clock_hz is None and self.bandwidth_fraction is None:
            raise ValueError("must provide 'axi_clock_hz' or 'bandwidth_fraction'")
        if self.max_outstanding_writes is not None and self.max_outstanding_writes < 1:
            raise ValueError(f"max_outstanding_writes must be >= 1, got {self.max_outstanding_writes}")
        if self.provenance is not None:
            self.provenance.check_field_names(set(self.PROVENANCE_FIELDS), "memories")
        return self


class SramPortsModel(SocBoundaryModel):
    """How the NPU spreads AXI SRAM traffic over its SRAM ports."""

    hash: int = Field(ge=0)
    """Address mask whose parity picks the port for each transaction (the Ethos-U85's HASH0, the Corstone-320 FVP's
    sram_hash0): 0x80 sends alternate 128 B stripes to alternate ports. Without this block the FVP default applies,
    0x100 (address bit 8).
    """
    provenance: ProvenanceModel | None = None
    """Where the value came from."""

    @field_validator("hash", mode="before")
    @classmethod
    def reject_boolean_hash(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("must be an integer, not a boolean")
        return value

    @model_validator(mode="after")
    def validate_provenance_fields(self) -> SramPortsModel:
        if self.provenance is not None:
            self.provenance.check_field_names({"hash"}, "sram_ports")
        return self


class SocRevisionModel(SocBoundaryModel):
    label: str = Field(min_length=1)
    """Revision label, e.g. v8 or A0."""
    stage: Literal["spec", "rtl", "silicon", "released"] = "spec"
    """Where the revision sits in the part's life: spec, rtl, silicon or released."""
    date: datetime.date | None = None
    """Date of the revision."""
    notes: str | None = None
    """What the revision is based on."""

    def to_revision(self) -> SocRevision:
        return SocRevision(label=self.label, stage=SocStage(self.stage), date=self.date, notes=self.notes)


_LABEL_WITH_DATE = re.compile(r"^(?P<label>.+?)\s*\((?P<date>\d{4}-\d{2}-\d{2})\)$")


def parse_revision(value: str | SocRevisionModel | None) -> SocRevision | None:
    """``"v7 (2026-09-18)"`` or ``"A0"`` as a string, or the structured form."""
    if value is None:
        return None
    if isinstance(value, SocRevisionModel):
        return value.to_revision()
    text = value.strip()
    match = _LABEL_WITH_DATE.match(text)
    if match:
        return SocRevision(label=match.group("label"), date=datetime.date.fromisoformat(match.group("date")))
    return SocRevision(label=text)


class SocModel(SocBoundaryModel):
    """The SoC half of a definition file. ``id`` defaults to the lowercased name."""

    name: str = Field(min_length=1)
    """The product name, e.g. AT110."""
    id: str | None = None
    """Canonical lowercase id used in generated files and results; defaults to the lowercased name."""
    origin: Literal["golden", "derived", "custom"] = "custom"
    """golden (packaged, values reviewed by the owners), derived (a variant of a golden SoC) or custom."""
    revision: str | SocRevisionModel | None = None
    """Which drop or stepping the values describe: a label such as v8 (2026-09-22), or {label, stage, date, notes}."""
    npu: str = Field(min_length=1)
    """NPU family, e.g. ethos-u85."""
    macs: int
    """MAC units per cycle, e.g. 256."""
    axi_bus_width_bits: int | None = Field(default=None, gt=0)
    """Width of one AXI data beat, bits; the NPU's (64 on the Ethos-U55, 128 on the U65 and U85) when unset."""
    modes: dict[str, ModeModel] = Field(min_length=1)
    """Operating modes by name, each with its NPU clock."""
    memories: dict[str, MemoryModel] = Field(min_length=1)
    """Every memory wired to the NPU, by name, each on one port class. A port may list several (HBLRAM, PSRAM and
    RRAM on AT110's port 1); the first declared is the port's default and a run's memory_placement picks another.
    """
    sram_ports: SramPortsModel | None = None
    """How the NPU spreads SRAM traffic over its SRAM ports; the FVP default when unset."""

    @field_validator("name", "npu", mode="before")
    @classmethod
    def normalize_required_strings(cls, value: object) -> object:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()

    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip().lower()

    @field_validator("macs", "axi_bus_width_bits", mode="before")
    @classmethod
    def reject_boolean_integers(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("must be an integer, not a boolean")
        return value

    @model_validator(mode="after")
    def validate_axi_bindings(self) -> SocModel:
        seen_names: set[str] = set()
        for memory_name in self.memories:
            canonical = memory_name.lower()
            if canonical in seen_names:
                raise ValueError(f"duplicate memory name ignoring case: {memory_name}")
            seen_names.add(canonical)
        if not any(memory.axi_port == 0 for memory in self.memories.values()):
            raise ValueError("at least one memory must sit on axi_port 0 (Vela's Sram class)")
        if self.origin == "golden":
            unrecorded = sorted(name for name, memory in self.memories.items() if memory.provenance is None)
            if unrecorded:
                raise ValueError(
                    f"origin: golden requires a provenance record on every memory; none on: {', '.join(unrecorded)}"
                )
            if self.sram_ports is not None and self.sram_ports.provenance is None:
                raise ValueError("origin: golden requires a provenance record on sram_ports")
        return self


def memory_from_model(name: str, memory: MemoryModel) -> MemoryDef:
    """Build one modeled memory and its provenance from validated parameters."""
    return MemoryDef(
        name=name.lower(),
        axi_port=memory.axi_port,
        provenance=memory.provenance.to_provenance() if memory.provenance is not None else None,
        params=MemoryParams(
            read_latency_ns=memory.read_latency_ns,
            write_latency_ns=memory.write_latency_ns,
            bandwidth_fraction=memory.bandwidth_fraction if memory.bandwidth_fraction is not None else 1.0,
            max_outstanding=memory.max_outstanding,
            max_outstanding_writes=memory.max_outstanding_writes,
            burst_length=memory.burst_length,
            ports_used=memory.ports_used,
            axi_clock_hz=memory.axi_clock_hz,
        ),
    )


def soc_from_model(document: SocModel, *, revision: str | SocRevision | None = None) -> SocDefinition:
    """Build the immutable definition from a validated document."""
    modes = {name: ModeClocks(name=name, npu_clock_hz=mode.npu_clock_hz) for name, mode in document.modes.items()}
    provenance: dict[str, Provenance] = {}
    memories: dict[str, MemoryDef] = {}
    for mem_name, memory in document.memories.items():
        key = mem_name.lower()
        mem_provenance = memory.provenance.to_provenance() if memory.provenance is not None else None
        if mem_provenance is not None:
            provenance[f"memories.{key}"] = mem_provenance
        memories[key] = memory_from_model(key, memory)
    if document.sram_ports is not None and document.sram_ports.provenance is not None:
        provenance["sram_ports"] = document.sram_ports.provenance.to_provenance()
    return SocDefinition(
        id=document.id or document.name.lower(),
        name=document.name,
        npu=document.npu.strip().lower(),
        macs=document.macs,
        axi_bus_width_bits=(
            document.axi_bus_width_bits
            if document.axi_bus_width_bits is not None
            else default_axi_beat_bits(document.npu)
        ),
        modes=modes,
        memories=memories,
        origin=SocOrigin(document.origin),
        revision=(
            revision
            if isinstance(revision, SocRevision)
            else parse_revision(revision if revision is not None else document.revision)
        ),
        provenance=provenance,
        sram_port_hash=document.sram_ports.hash if document.sram_ports is not None else None,
    )


class SocOverridesModel(SocBoundaryModel):
    """A diff on top of a base SoC: the what-if and variant mechanism.

    Scalars replace the base's. ``modes`` replaces the whole mode table when
    given. ``memories`` is merged per memory: a mapping replaces or adds that
    memory, ``null`` removes it. The result's ``origin`` is ``derived`` unless
    nothing but identity changed.
    """

    name: str | None = None
    """Replaces the base SoC's name."""
    id: str | None = None
    """Replaces the base SoC's id; a derived variant should have its own."""
    origin: Literal["golden", "derived", "custom"] | None = None
    """Replaces the origin; when omitted, an override that changes hardware makes the SoC derived, one that only
    renames keeps the base origin.
    """
    revision: str | SocRevisionModel | None = None
    """Replaces the revision."""
    npu: str | None = None
    """Replaces the NPU family."""
    macs: int | None = None
    """Replaces the MAC count."""
    axi_bus_width_bits: int | None = Field(default=None, gt=0)
    """Replaces the beat width; a new npu without one takes that NPU's."""
    modes: dict[str, ModeModel] | None = None
    """Replaces the modes."""
    memories: dict[str, MemoryModel | None] | None = None
    """Memories to add or replace by name; null removes one. A replaced memory keeps its place in the order; when a
    port's first memory is removed, the next one declared on that port becomes its default.
    """
    sram_ports: SramPortsModel | None = None
    """Replaces the SRAM port hash."""

    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip().lower()

    @property
    def changes_hardware(self) -> bool:
        return any(
            v is not None
            for v in (self.npu, self.macs, self.axi_bus_width_bits, self.modes, self.memories, self.sram_ports)
        )


def derive_soc(
    base: SocDefinition, overrides: SocOverridesModel, *, revision: SocRevision | None = None
) -> SocDefinition:
    """A new definition from *base* with *overrides* applied; see :class:`SocOverridesModel`."""
    modes = base.modes
    if overrides.modes is not None:
        modes = {name: ModeClocks(name=name, npu_clock_hz=mode.npu_clock_hz) for name, mode in overrides.modes.items()}
    memories = dict(base.memories)
    provenance = {k: v for k, v in base.provenance.items()}
    if overrides.memories is not None:
        for mem_name, memory in overrides.memories.items():
            key = mem_name.lower()
            provenance.pop(f"memories.{key}", None)
            if memory is None:
                if key not in memories:
                    raise SocError(f"cannot remove memory '{mem_name}': base '{base.id}' has no such memory")
                del memories[key]
                continue
            mem_provenance = memory.provenance.to_provenance() if memory.provenance is not None else None
            if mem_provenance is not None:
                provenance[f"memories.{key}"] = mem_provenance
            memories[key] = memory_from_model(key, memory)
        if not any(m.axi_port == 0 for m in memories.values()):
            raise SocError(f"derived SoC '{overrides.id or base.id}' has no memory on AXI port 0")
    sram_port_hash = base.sram_port_hash
    if overrides.sram_ports is not None:
        sram_port_hash = overrides.sram_ports.hash
        provenance.pop("sram_ports", None)
        if overrides.sram_ports.provenance is not None:
            provenance["sram_ports"] = overrides.sram_ports.provenance.to_provenance()
    if overrides.origin is not None:
        origin = SocOrigin(overrides.origin)
    elif overrides.changes_hardware:
        origin = SocOrigin.DERIVED
    else:
        origin = base.origin
    if revision is None:
        revision = parse_revision(overrides.revision) if overrides.revision is not None else base.revision
    if base.catalog_identity is not None:
        # The catalog identity describes the base; explicit what-if settings must
        # not inherit the original fact's measurement status.
        assumed = Provenance(status=ProvenanceStatus.UNVERIFIED, source="Explicit soc_overrides modeling assumption")
        for value, keys in (
            (overrides.npu, ("catalog.npu.family", "catalog.npu.model", "catalog.axi_width_bits")),
            (overrides.macs, ("catalog.npu.macs_per_cycle",)),
            (overrides.axi_bus_width_bits, ("catalog.axi_width_bits",)),
        ):
            if value is not None:
                provenance.update(dict.fromkeys(keys, assumed))
        if overrides.modes is not None:
            provenance = {key: record for key, record in provenance.items() if not key.startswith("catalog.clock.")}
            provenance.update({f"catalog.clock.{name.casefold()}": assumed for name in modes})
        for name in overrides.memories or {}:
            if name.lower() in memories and name.lower() not in base.memories:
                provenance[f"catalog.reach.{name.lower()}"] = assumed
    return SocDefinition(
        id=overrides.id or base.id,
        name=overrides.name or base.name,
        npu=(overrides.npu or base.npu).strip().lower(),
        macs=overrides.macs if overrides.macs is not None else base.macs,
        axi_bus_width_bits=(
            overrides.axi_bus_width_bits
            if overrides.axi_bus_width_bits is not None
            else default_axi_beat_bits(overrides.npu)
            if overrides.npu
            else base.axi_bus_width_bits
        ),
        modes=modes,
        memories=memories,
        origin=origin,
        revision=revision,
        provenance=provenance,
        sram_port_hash=sram_port_hash,
        catalog_identity=base.catalog_identity,
    )


def describe_soc_validation_error(exc: ValidationError) -> str:
    """Return the first invalid field and its explanation as one actionable line."""
    error = exc.errors(include_url=False)[0]
    location = ".".join(str(part) for part in error["loc"])
    message = " ".join(str(error["msg"]).removeprefix("Value error, ").splitlines())
    return f"{location}: {message}" if location else message


def validate_soc_document(raw: object) -> SocModel:
    try:
        return SocModel.model_validate(raw)
    except ValidationError as exc:
        raise SocSchemaError(describe_soc_validation_error(exc)) from exc


class CatalogSocReference(SocBoundaryModel):
    """A packaged SoC reference to catalog facts and explicit consumer policy."""

    catalog: str = Field(min_length=1)
    policy: Literal["atomiq110"]
    comparison_memories: list[str] = Field(default_factory=list)


SOCS_DIR = Path(__file__).resolve().parent / "socs"


def list_socs() -> list[str]:
    """The ids of the reference SoC definitions shipped with the package."""
    return sorted(p.stem for p in SOCS_DIR.glob("*.yaml"))


def load_soc(path_or_id: str | Path, *, on_read: Callable[[Path, bytes], None] | None = None) -> SocDefinition:
    """Load a SoC definition by packaged id (``"atomiq110"``) or from a SoC-only YAML file.

    Extra keys are rejected; tools with richer files (helx's platform YAML) use
    their own loader and build the SoC through :func:`soc_from_model`.
    """
    path = Path(path_or_id)
    if not path.suffix:
        wanted = str(path_or_id).strip().lower()
        matched = next((p for p in SOCS_DIR.glob("*.yaml") if p.stem.lower() == wanted), None)
        if matched is None:
            raise SocError(f"No packaged SoC named '{path_or_id}'. Available: {', '.join(list_socs())}")
        path = matched
    if not path.exists():
        raise SocError(f"SoC definition not found: {path}")
    try:
        raw = load_yaml_mapping(path, what="SoC file", on_read=on_read)
    except ValueError as exc:
        raise SocError(str(exc)) from exc
    try:
        if "catalog" in raw:
            from .catalog_loading import CatalogPolicy, load_catalog_model

            reference = CatalogSocReference.model_validate(raw)
            return load_catalog_model(
                reference.catalog,
                policy=CatalogPolicy(reference.policy),
                comparison_memories=tuple(reference.comparison_memories),
                on_read=on_read,
            ).soc
        return soc_from_model(validate_soc_document(raw))
    except ValidationError as exc:
        raise SocError(f"{path}: {describe_soc_validation_error(exc)}") from exc
    except SocSchemaError as exc:
        raise SocError(f"{path}: {exc}") from exc

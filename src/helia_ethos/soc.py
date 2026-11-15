"""What a SoC looks like to the NPU: the NPU instance, its memories, its clocks.

This is the silicon description Vela and the FVP timing adapters need, and
nothing more: no energy model, no peripherals, no board. Every value can carry
provenance, and every definition says where it came from (``origin``) so a
number computed against a what-if cannot be mistaken for one computed against
silicon.
"""

from __future__ import annotations

import dataclasses
import datetime
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from .catalog_identity import CatalogModelIdentity


class SocError(ValueError):
    """A SoC definition is inconsistent, or was asked for something it does not have."""


class SocOrigin(StrEnum):
    """Where a definition's authority comes from.

    ``golden``: the maintained definition of a real part. ``derived``: a golden
    definition with something deliberately changed. ``custom``: anything else,
    which is the default so an unrecognised file cannot borrow a characterised
    part's authority.
    """

    GOLDEN = "golden"
    DERIVED = "derived"
    CUSTOM = "custom"


class SocStage(StrEnum):
    """Where a SoC revision sits in the part's life: what its numbers can be."""

    SPEC = "spec"  # pre-RTL: functional spec, estimates
    RTL = "rtl"  # an RTL drop: verification and power-analysis results exist
    SILICON = "silicon"  # a stepping: measured on the part
    RELEASED = "released"  # a production part


@dataclass(frozen=True)
class SocRevision:
    """A dated point in a SoC's history: which drop or stepping the values describe."""

    label: str  # "v7", "A0", "rtl-2026-09"
    stage: SocStage = SocStage.SPEC
    date: datetime.date | None = None
    notes: str | None = None

    def __str__(self) -> str:
        when = f" ({self.date.isoformat()})" if self.date else ""
        return f"{self.label}{when}"


class ProvenanceStatus(StrEnum):
    """How much a hardware number can be trusted, from most to least.

    ``confirmed``: measured, or reviewed against measurements, by the team that
    owns the value. ``fitted``: tuned so the model reproduces measurements it
    was compared with (RTL, FPGA); right for that comparison, not a property
    of the part. ``unverified``: a specification figure, an estimate or a
    placeholder nobody has checked yet.
    """

    CONFIRMED = "confirmed"
    FITTED = "fitted"
    UNVERIFIED = "unverified"

    @property
    def rank(self) -> int:
        """Lower is more trusted; a rollup reports the highest rank among the values it used."""
        return {ProvenanceStatus.CONFIRMED: 0, ProvenanceStatus.FITTED: 1, ProvenanceStatus.UNVERIFIED: 2}[self]


def least_trusted(statuses: Iterable[ProvenanceStatus]) -> ProvenanceStatus | None:
    """The status a result must carry when it depends on all of *statuses*."""
    found = list(statuses)
    return max(found, key=lambda s: s.rank) if found else None


@dataclass(frozen=True)
class Provenance:
    """Who vouched for a value, and on what basis.

    A number computed from a confirmed definition is a different claim from one
    computed against an assumption, so every reported value should be able to
    say which it is. ``label()`` is the human-facing form used by reports.
    A block-level record can carry ``fields``: per-value records for the
    numbers in the block whose status differs from the block's own.
    """

    status: ProvenanceStatus
    source: str | None = None
    reviewed_by: str | None = None
    reviewed_on: datetime.date | None = None
    notes: str | None = None
    fields: Mapping[str, Provenance] = dataclasses.field(default_factory=dict)

    def for_field(self, name: str) -> Provenance:
        """The record that applies to one value of the block: its own if it has one, else the block's."""
        return self.fields.get(name, self)

    def label(self) -> str:
        if self.status is ProvenanceStatus.CONFIRMED:
            who = self.reviewed_by or "reviewer not recorded"
            when = self.reviewed_on.isoformat() if self.reviewed_on else "date not recorded"
            return f"Confirmed ({who}, {when})"
        if self.status is ProvenanceStatus.FITTED:
            return f"Fitted — {self.source}" if self.source else "Fitted"
        return f"Unverified — {self.source}" if self.source else "Unverified"


NO_PROVENANCE = "No provenance recorded"


class NpuFamily(StrEnum):
    """Ethos-U variants this project knows how to compile and simulate for."""

    ETHOS_U55 = "ethos-u55"
    ETHOS_U65 = "ethos-u65"
    ETHOS_U85 = "ethos-u85"

    @classmethod
    def _missing_(cls, value: object) -> NpuFamily | None:
        canonical = str(value).strip().lower().replace("_", "-")
        return next((member for member in cls if member.value == canonical), None)

    @property
    def axi_beat_bits(self) -> int:
        """Width of one AXI data beat on this NPU's ports: Vela 5.2 moves 8 bytes a cycle on the U55, 16 on the rest."""
        return 64 if self is NpuFamily.ETHOS_U55 else 128


def default_axi_beat_bits(npu: str) -> int:
    """The beat width a SoC with *npu* has; 128 for an NPU this project does not know (it is refused at compile)."""
    try:
        return NpuFamily(npu).axi_beat_bits
    except ValueError:
        return 128


class VelaMemoryMode(StrEnum):
    """The three standard Vela memory modes; also the modes the Corstone runtime can execute.

    Members spell Vela's own ``[Memory_Mode.*]`` section names. A placement
    whose AXI pattern matches none of them gets a custom section label instead
    (:meth:`SocDefinition.vela_memory_mode_label`). Lookup is case-insensitive.
    """

    SRAM_ONLY = "Sram_Only"
    SHARED_SRAM = "Shared_Sram"
    DEDICATED_SRAM = "Dedicated_Sram"

    @classmethod
    def _missing_(cls, value: object) -> VelaMemoryMode | None:
        canonical = str(value).strip().lower()
        return next((member for member in cls if member.value.lower() == canonical), None)


#: NPU cycles over which a Corstone timing adapter enforces a port's rate cap, unless a run overrides it.
TA_RATE_WINDOW_CYCLES = 4000


@dataclass(frozen=True)
class TimingAdapterPort:
    """What the Corstone timing adapter on one AXI port is programmed with, in NPU cycles.

    ``bandwidth`` is the fraction of the NPU clock rate the port sustains, the
    same number Vela reads as ``clock_scale``. The adapter enforces it over
    ``rate_window_cycles``: at most ``bandwidth x window`` beats per window, so a
    long window lets a port burst above its rate for a while and a short one
    holds it to the rate.
    """

    read_latency: int
    write_latency: int
    bandwidth: float
    max_transactions: int
    rate_window_cycles: int = TA_RATE_WINDOW_CYCLES
    max_write_transactions: int | None = None
    """Outstanding writes when they differ from reads; the adapter's MAXW."""

    @property
    def write_transactions(self) -> int:
        """Outstanding writes the adapter is programmed with: ``max_write_transactions``, or ``max_transactions``
        when no separate write limit is set."""
        return self.max_transactions if self.max_write_transactions is None else self.max_write_transactions


#: Timing-adapter settings keyed by AXI port index (0 is Vela's Sram/Axi0, 1 its Dram/Axi1).
TimingAdapterProfile = Mapping[int, TimingAdapterPort]

# Vela port type names by AXI index (Ethos-U embedded convention).
VELA_PORT_TYPE: dict[int, str] = {0: "Sram", 1: "Dram"}
VELA_AXI_NAME: dict[int, str] = {0: "Axi0", 1: "Axi1"}
#: Arm's name for each port class; what the NPU's PMU counters and the Corstone timing adapters are keyed by.
NPU_PORT_CLASS: dict[int, str] = {0: "AXI_SRAM", 1: "AXI_EXT"}
#: The prefix of the host-side PMU fields for each port (``npu_sram_rd_data_beats``, ``npu_ext_wr_data_beats``).
PMU_FIELD_PREFIX: dict[int, str] = {0: "npu_sram_", 1: "npu_ext_"}


def port_join_label(axi_port: int) -> str:
    """Every name one port carries, for people reading a platform: ``port 0: Vela Sram/Axi0, PMU npu_sram_*, TA0``."""
    return (
        f"port {axi_port}: Arm {NPU_PORT_CLASS[axi_port]}, Vela {VELA_PORT_TYPE[axi_port]}/{VELA_AXI_NAME[axi_port]}, "
        f"PMU {PMU_FIELD_PREFIX[axi_port]}*, TA{axi_port}"
    )


# Standard AXI-port patterns (const, arena, cache) -> Vela memory mode.
AXI_PATTERN_TO_VELA_MODE: dict[tuple[int, int, int], VelaMemoryMode] = {
    (0, 0, 0): VelaMemoryMode.SRAM_ONLY,
    (1, 0, 0): VelaMemoryMode.SHARED_SRAM,
    (1, 1, 0): VelaMemoryMode.DEDICATED_SRAM,
}
VELA_MODE_TO_AXI: dict[VelaMemoryMode, tuple[int, int, int]] = {v: k for k, v in AXI_PATTERN_TO_VELA_MODE.items()}


def standard_vela_mode(name: str) -> VelaMemoryMode | None:
    """The standard Vela memory mode *name* spells (case-insensitively), or ``None``."""
    try:
        return VelaMemoryMode(name)
    except ValueError:
        return None


@dataclass(frozen=True)
class MemoryParams:
    """Physical memory interface parameters."""

    read_latency_ns: float
    write_latency_ns: float
    bandwidth_fraction: float  # static fallback (used when axi_clock_hz absent)
    max_outstanding: int
    burst_length: int
    ports_used: int
    axi_clock_hz: float | None = None  # if set, bandwidth derived per mode
    max_outstanding_writes: int | None = None  # outstanding writes when they differ from reads


@dataclass(frozen=True)
class MemoryDef:
    """A named physical memory with its AXI port binding."""

    name: str  # canonical lowercase name (e.g. "ssram", "hblram")
    axi_port: int  # 0 or 1
    params: MemoryParams
    provenance: Provenance | None = None


@dataclass(frozen=True)
class MemoryPlacement:
    """Where to place const (weights), arena (activations), and cache.

    Values are physical memory names from the platform definition
    (e.g. ``"ssram"``, ``"hblram"``).  Always stored lowercase.
    """

    const: str
    arena: str
    cache: str


@dataclass(frozen=True)
class ModeClocks:
    """Clocks for a single operating mode."""

    name: str
    npu_clock_hz: float


@dataclass(frozen=True)
class MemoryTiming:
    """Per-memory timing in NPU clock cycles."""

    name: str
    axi_port: int
    read_latency: int
    write_latency: int
    bandwidth: float
    max_outstanding: int
    burst_length: int
    ports_used: int
    rate_window_cycles: int = TA_RATE_WINDOW_CYCLES
    max_outstanding_writes: int | None = None

    @property
    def write_outstanding(self) -> int:
        """Outstanding writes the memory path sustains: ``max_outstanding_writes``, or ``max_outstanding`` when the
        SoC sets no separate write limit."""
        return self.max_outstanding if self.max_outstanding_writes is None else self.max_outstanding_writes


@dataclass(frozen=True)
class DerivedTimingProfile:
    """Timing values in NPU clock cycles, ready for Vela .ini and FVP TA."""

    npu_clock_hz: float
    memories: dict[str, MemoryTiming]  # keyed by lowercase memory name

    def port_timing(self, occupancy: Mapping[int, str] | None = None) -> dict[int, MemoryTiming]:
        """The timing of the memory each port carries, keyed by port; the first memory on a port by default."""
        out: dict[int, MemoryTiming] = {}
        for mt in self.memories.values():
            wanted = occupancy.get(mt.axi_port) if occupancy is not None else None
            if wanted is None:
                out.setdefault(mt.axi_port, mt)
            elif wanted == mt.name:
                out[mt.axi_port] = mt
        return out

    def as_ta_profile(self, occupancy: Mapping[int, str] | None = None) -> dict[int, TimingAdapterPort]:
        """The timing-adapter settings of the memory each port carries, keyed by **AXI port index**."""
        return {
            port: TimingAdapterPort(
                read_latency=mt.read_latency,
                write_latency=mt.write_latency,
                bandwidth=mt.bandwidth,
                max_transactions=mt.max_outstanding,
                rate_window_cycles=mt.rate_window_cycles,
                max_write_transactions=mt.max_outstanding_writes,
            )
            for port, mt in self.port_timing(occupancy).items()
        }


@dataclass(frozen=True)
class MemoryTimingOverride:
    """Optional per-memory overrides applied after timing derivation."""

    read_latency_cycles: int | None = None
    write_latency_cycles: int | None = None
    bandwidth: float | None = None
    max_outstanding: int | None = None
    max_outstanding_writes: int | None = None
    burst_length: int | None = None
    ports_used: int | None = None
    rate_window_cycles: int | None = None
    """FVP only: the timing adapter's rate window. Vela never sees it."""
    deliberate: bool = True
    """False for an override a tool derives rather than one a person sets (e.g. helia-estimator's emulated
    application traffic): it reaches Vela but is no sweep of the port, so Vela's port-1 latency clamp still applies."""

    @property
    def reaches_vela(self) -> bool:
        """True when the override changes a value Vela compiles with (everything but the rate window)."""
        return any(
            value is not None
            for value in (
                self.read_latency_cycles,
                self.write_latency_cycles,
                self.bandwidth,
                self.max_outstanding,
                self.max_outstanding_writes,
                self.burst_length,
                self.ports_used,
            )
        )


@dataclass(frozen=True)
class TimingOverrides:
    """Run-scoped timing overrides: per memory, keyed by physical memory name, and the SRAM port hash."""

    memories: dict[str, MemoryTimingOverride] = field(default_factory=dict)
    sram_port_hash: int | None = None
    """Replaces the SoC's :attr:`SocDefinition.sram_port_hash` for the run."""

    @property
    def enabled(self) -> bool:
        return bool(self.memories) or self.sram_port_hash is not None


@dataclass(frozen=True)
class SocDefinition:
    """One SoC as the NPU sees it."""

    id: str  # canonical, stable identifier: generates names and keys
    name: str  # human-facing name; also the Vela system-config prefix today
    npu: str  # NPU family, e.g. "ethos-u85"
    macs: int
    axi_bus_width_bits: int
    modes: dict[str, ModeClocks]  # keyed by original-case mode name
    memories: dict[str, MemoryDef]  # keyed by lowercase memory name
    origin: SocOrigin = SocOrigin.CUSTOM
    revision: SocRevision | None = None
    # Provenance keyed by dotted path ("memories.ssram", "sram_ports"); absent when none recorded.
    provenance: dict[str, Provenance] = field(default_factory=dict)
    sram_port_hash: int | None = None
    """Address mask whose parity picks which of the NPU's AXI SRAM ports a transaction uses (the Ethos-U85's
    HASH0; the Corstone-320 FVP's ``sram_hash0``). ``None`` leaves the FVP default, 0x100 (address bit 8)."""
    catalog_identity: CatalogModelIdentity | None = None
    """Catalog/policy basis before any explicit caller overrides; absent for inline definitions."""

    def __post_init__(self) -> None:
        # The beat width is the NPU's, not a design choice: a wrong one scales every memory energy term.
        try:
            family = NpuFamily(self.npu)
        except ValueError:
            return
        if self.axi_bus_width_bits != family.axi_beat_bits:
            raise SocError(
                f"SoC '{self.id}': an {family} moves {family.axi_beat_bits}-bit AXI data beats, "
                f"but axi_bus_width_bits is {self.axi_bus_width_bits}"
            )

    def provenance_label(self, key: str) -> str:
        record = self.provenance.get(key)
        return record.label() if record is not None else NO_PROVENANCE

    def revision_label(self) -> str | None:
        return str(self.revision) if self.revision is not None else None

    def memories_on_axi(self, port: int) -> list[MemoryDef]:
        """Every memory wired to *port*, in declaration order; the first is the port's default."""
        return [mem for mem in self.memories.values() if mem.axi_port == port]

    def memory_for_axi(self, port: int) -> MemoryDef | None:
        """The port's default memory (the first declared on it), or ``None`` when nothing is wired there."""
        wired = self.memories_on_axi(port)
        return wired[0] if wired else None

    def occupants(self, placement: MemoryPlacement | None = None) -> dict[int, str]:
        """Which memory each port carries for a run: the one *placement* uses there, else the port's default.

        The NPU sees one memory per port class, so a run picks one occupant per
        port out of everything wired; Vela's timing, the timing adapters and the
        energy terms are the occupant's.
        """
        chosen: dict[int, str] = {}
        if placement is not None:
            self.validate_placement(placement)
            for name in (placement.const, placement.arena, placement.cache):
                chosen[self.resolve_memory(name).axi_port] = name.lower()
        out: dict[int, str] = {}
        for port in (0, 1):
            default = self.memory_for_axi(port)
            if port in chosen:
                out[port] = chosen[port]
            elif default is not None:
                out[port] = default.name
        return out

    def resolve_memory(self, name: str) -> MemoryDef:
        """Case-insensitive memory lookup."""
        key = name.lower()
        if key not in self.memories:
            raise SocError(f"Memory '{name}' not found in '{self.name}'. Available memories: {sorted(self.memories)}")
        return self.memories[key]

    def validate_placement(self, placement: MemoryPlacement) -> None:
        """Every name is a memory of this SoC, and no port carries two different memories in one run."""
        by_port: dict[int, str] = {}
        for field_name in ("const", "arena", "cache"):
            memory = self.resolve_memory(getattr(placement, field_name))
            other = by_port.setdefault(memory.axi_port, memory.name)
            if other != memory.name:
                raise SocError(
                    f"memory_placement uses {other} and {memory.name} on port {memory.axi_port} at once; the NPU sees "
                    "one memory per port, so pick one occupant per port"
                )

    def placement_axi_ports(self, placement: MemoryPlacement) -> tuple[int, int, int]:
        """``(const_axi, arena_axi, cache_axi)``."""
        return (
            self.resolve_memory(placement.const).axi_port,
            self.resolve_memory(placement.arena).axi_port,
            self.resolve_memory(placement.cache).axi_port,
        )

    def fvp_memory_mode(self, placement: MemoryPlacement) -> VelaMemoryMode:
        """The closest of the three memory modes the Corstone runtime can execute."""
        const_port, arena_port, _cache_port = self.placement_axi_ports(placement)
        if arena_port != 0:
            return VelaMemoryMode.DEDICATED_SRAM
        if const_port != 0:
            return VelaMemoryMode.SHARED_SRAM
        return VelaMemoryMode.SRAM_ONLY

    def standard_placements(self, occupancy: Mapping[int, str] | None = None) -> dict[VelaMemoryMode, MemoryPlacement]:
        """The placements Vela's three mode names mean on this SoC, for the modes its ports can express.

        *occupancy* names the memory each port carries (a run's choice); the
        default is each port's first memory. A SoC with nothing on port 1 has
        only ``Sram_Only``.
        """
        by_port = dict(occupancy) if occupancy is not None else self.occupants()
        out: dict[VelaMemoryMode, MemoryPlacement] = {}
        for mode, (const_port, arena_port, cache_port) in VELA_MODE_TO_AXI.items():
            if all(port in by_port for port in (const_port, arena_port, cache_port)):
                out[mode] = MemoryPlacement(
                    const=by_port[const_port], arena=by_port[arena_port], cache=by_port[cache_port]
                )
        return out

    def placement_label(self, placement: MemoryPlacement) -> str:
        """The placement spelled by physical memory: ``const_hblram__arena_ssram__cache_ssram``."""
        return f"const_{placement.const}__arena_{placement.arena}__cache_{placement.cache}"

    def vela_memory_mode_label(self, placement: MemoryPlacement) -> str:
        """Vela ``[Memory_Mode.*]`` section name: a standard mode's when the AXI pattern matches."""
        standard = AXI_PATTERN_TO_VELA_MODE.get(self.placement_axi_ports(placement))
        if standard is not None:
            return standard.value
        return self.placement_label(placement)

    def resolve_mode(self, mode: str) -> ModeClocks:
        """Case-insensitive operating-mode lookup."""
        if mode in self.modes:
            return self.modes[mode]
        wanted = mode.lower()
        for name, clocks in self.modes.items():
            if name.lower() == wanted:
                return clocks
        raise SocError(f"Mode '{mode}' not defined in '{self.name}'. Available modes: {sorted(self.modes)}")


def parse_memory_placement(raw: str | dict, soc: SocDefinition | None = None) -> MemoryPlacement:
    """A placement from a YAML value: a standard Vela mode name, or ``{const, arena, cache}``.

    A mode name needs a SoC to resolve to physical memory names.
    """
    if isinstance(raw, str):
        canonical = standard_vela_mode(raw)
        if canonical is None:
            raise SocError(
                f"Unknown memory mode '{raw}'. Use one of {sorted(mode.value for mode in VelaMemoryMode)} "
                "or specify {const, arena, cache}"
            )
        if soc is None:
            raise SocError(
                f"memory_placement with mode name '{raw}' requires a platform (SoC definition) "
                "to resolve physical memory names"
            )
        names: list[str] = []
        for port in VELA_MODE_TO_AXI[canonical]:
            mem = soc.memory_for_axi(port)
            if mem is None:
                raise SocError(f"Cannot resolve '{raw}': '{soc.name}' has no memory on AXI port {port}")
            names.append(mem.name)
        return MemoryPlacement(const=names[0], arena=names[1], cache=names[2])

    if not isinstance(raw, dict):
        raise SocError("memory_placement must be a string or mapping with {const, arena, cache}")

    default_const = default_arena = default_cache = ""
    if soc is not None:
        axi0 = soc.memory_for_axi(0)
        axi1 = soc.memory_for_axi(1)
        default_const = axi1.name if axi1 else ""
        default_arena = default_cache = axi0.name if axi0 else ""
    const = str(raw.get("const", default_const)).strip().lower()
    arena = str(raw.get("arena", default_arena)).strip().lower()
    cache = str(raw.get("cache", default_cache)).strip().lower()
    if not const or not arena or not cache:
        raise SocError("memory_placement must specify const, arena, and cache memory names")
    placement = MemoryPlacement(const=const, arena=arena, cache=cache)
    if soc is not None:
        soc.validate_placement(placement)
    return placement


#: The Ethos-U driver's AXI burst-split codes, the same on the U55, U65 and U85 (U85 ``AXI_LIMIT_SRAM/EXT_MAX_BEATS``,
#: U55/U65 ``AXI_LIMIT0..3_MAX_BEATS_BYTES``).
AXI_MAX_BEATS_CODE: dict[int, int] = {64: 0, 128: 1, 256: 2}


def axi_max_beats_code(burst_bytes: int) -> int:
    """The Ethos-U driver code that splits AXI bursts at *burst_bytes*; the NPUs support 64, 128 and 256 B only."""
    try:
        return AXI_MAX_BEATS_CODE[burst_bytes]
    except KeyError:
        raise ValueError(
            f"burst_length {burst_bytes} B is not a split the Ethos-U NPUs support (64, 128 or 256 B)"
        ) from None


def ns_to_cycles(latency_ns: float, clock_hz: float) -> int:
    """A latency in nanoseconds as NPU clock cycles, rounded up and at least one."""
    return max(1, math.ceil(latency_ns * clock_hz / 1e9))


def derive_timing(
    soc: SocDefinition,
    mode: str,
    overrides: TimingOverrides | None = None,
) -> DerivedTimingProfile:
    """Timing in NPU cycles for one operating mode, ready for Vela and the FVP timing adapters.

    When a memory specifies ``axi_clock_hz``, the effective bandwidth
    (Vela ``clock_scale``) is derived as ``min(1.0, axi_clock / npu_clock)``.
    Otherwise the static ``bandwidth_fraction`` is used.
    """
    mc = soc.resolve_mode(mode)
    clock = mc.npu_clock_hz
    mem_timings: dict[str, MemoryTiming] = {}
    for mem_name, mem_def in soc.memories.items():
        p = mem_def.params
        bandwidth = min(1.0, p.axi_clock_hz / clock) if p.axi_clock_hz is not None else p.bandwidth_fraction
        mem_timing = MemoryTiming(
            name=mem_name,
            axi_port=mem_def.axi_port,
            read_latency=ns_to_cycles(p.read_latency_ns, clock),
            write_latency=ns_to_cycles(p.write_latency_ns, clock),
            bandwidth=bandwidth,
            max_outstanding=p.max_outstanding,
            burst_length=p.burst_length,
            ports_used=p.ports_used,
            max_outstanding_writes=p.max_outstanding_writes,
        )
        if overrides is not None and mem_name in overrides.memories:
            override = overrides.memories[mem_name]
            mem_timing = MemoryTiming(
                name=mem_timing.name,
                axi_port=mem_timing.axi_port,
                read_latency=(
                    override.read_latency_cycles
                    if override.read_latency_cycles is not None
                    else mem_timing.read_latency
                ),
                write_latency=(
                    override.write_latency_cycles
                    if override.write_latency_cycles is not None
                    else mem_timing.write_latency
                ),
                bandwidth=override.bandwidth if override.bandwidth is not None else mem_timing.bandwidth,
                max_outstanding=(
                    override.max_outstanding if override.max_outstanding is not None else mem_timing.max_outstanding
                ),
                burst_length=override.burst_length if override.burst_length is not None else mem_timing.burst_length,
                ports_used=override.ports_used if override.ports_used is not None else mem_timing.ports_used,
                rate_window_cycles=(
                    override.rate_window_cycles
                    if override.rate_window_cycles is not None
                    else mem_timing.rate_window_cycles
                ),
                max_outstanding_writes=(
                    override.max_outstanding_writes
                    if override.max_outstanding_writes is not None
                    else mem_timing.max_outstanding_writes
                ),
            )
        mem_timings[mem_name] = mem_timing
    return DerivedTimingProfile(npu_clock_hz=clock, memories=mem_timings)

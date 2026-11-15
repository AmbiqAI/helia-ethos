"""Ethos-U PMU programs: which events go in which counter slots, and what comes out.

A profile name ("default", "dv_align") used to be the whole contract between
the firmware and the host, but two profiles can report the same field names
while programming different events, and two tools can program the same names
into different slots with different cycle-counter semantics. The contract is
the *program*: the ordered driver events per slot, their units, how the cycle
counter is used, and how derived fields are computed. Everything else (the C
table the firmware compiles, the fields the host validates) is rendered from
it, so helx and heliaPROFILER can select the same program by id. Whether two
captures are comparable also depends on how each tool reads the counters;
that is recorded per consumer as :class:`Acquisition`.

Event names are the Ethos-U core driver's ``enum ethosu_pmu_event_type``
symbols. U55/U65 name the ports AXI0/AXI1; U85 names them SRAM/EXT and adds
the MAC and per-port counters. Slot counts: 4 on U55/U65, 8 on U85.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .fvp import NpuFamily

__all__ = [
    "AccumulationSite",
    "Acquisition",
    "CycleCounter",
    "CycleSource",
    "DerivedField",
    "OverflowScope",
    "PmuProfile",
    "PmuProgram",
    "PmuSlot",
    "PmuUnit",
    "list_programs",
    "program_for",
    "render_c_header",
    "slot_count",
]


class PmuProfile(StrEnum):
    """Which PMU counter set the runtime firmware reports."""

    DEFAULT = "default"
    EXT_STALL = "ext_stall"
    DEFAULT_EXT_RD_STALL = "default_ext_rd_stall"
    #: MAC_ACTIVE and MAC stall breakdown plus the four energy beat counters (Ethos-U85 only).
    DV_ALIGN = "dv_align"
    #: Per-port read stall-limit and request-stalled counters plus MAC activity and read beats (U85 only).
    PORT_STALL = "port_stall"
    #: SRAM read/write stalls and beats plus MAC activity (U85 only; no EXT counters).
    SRAM_STALL = "sram_stall"
    #: Aggregate SRAM accepted transactions and matching beats, retaining all energy beat counters (U85 only).
    SRAM_TRANSACTIONS = "sram_transactions"


class PmuUnit(StrEnum):
    CYCLES = "cycles"
    BEATS = "beats"
    TRANSACTIONS = "transactions"


@dataclass(frozen=True)
class PmuSlot:
    """One event counter: the driver event it counts, the host field it feeds, its unit."""

    event: str  # driver symbol, e.g. ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED
    field: str  # host report field, e.g. npu_sram_rd_data_beats
    unit: PmuUnit
    #: The label the firmware prints; the host parser matches on it. Kept
    #: verbatim so existing bundles stay readable.
    report_name: str
    #: Whether the host insists the field is present (some slots are informational).
    required: bool = True


class CycleSource(StrEnum):
    CCNT = "ccnt"  # the dedicated cycle counter
    EVENT_SLOT = "event_slot"  # ETHOSU_PMU_CYCLE programmed into an event counter (costs a slot; 32-bit)
    NONE = "none"


@dataclass(frozen=True)
class CycleCounter:
    """How cycles are counted: the dedicated CCNT, an event slot, or not at all."""

    source: CycleSource = CycleSource.CCNT
    field: str = "npu_total_cycles"
    #: Ungated: the counter runs for the whole invocation, including clocked-idle time.
    gated_by_event: str | None = None
    report_name: str = "NPU TOTAL"
    bits: int = 64  # CCNT is 64-bit; an event slot is 32-bit


@dataclass(frozen=True)
class DerivedField:
    field: str
    formula: str  # in terms of other fields; documentation and validation, not executed
    unit: PmuUnit
    report_name: str


class AccumulationSite(StrEnum):
    FIRMWARE = "firmware"  # the runner sums counter reads on the target and prints totals
    HOST = "host"  # the runner prints each read; the host sums


class OverflowScope(StrEnum):
    ANY_COUNTER = "any_counter"  # any PMOVSSET bit (event slots and CCNT) marks the capture overflowed
    SELECTED_EVENT_SLOTS = "selected_event_slots"  # only the programmed event slots are checked


@dataclass(frozen=True)
class Acquisition:
    """How one consumer reads and sums a program's counters.

    Consumer-owned, and recorded per consumer: two tools that program the same
    events (``PmuProgram.events``) but accumulate at different widths, in
    different places or per different units of work, or that check overflow
    over different counters, do not produce interchangeable numbers. Equal
    events say the counters were set up the same way; nothing more.
    """

    consumer: str  # who reads the counters this way, e.g. "helx-fvp-runner"
    accumulation_bits: int  # width of the running sum (the hardware event counters are 32-bit)
    accumulation_site: AccumulationSite
    accumulation_unit: str  # what one addend is: "dispatch" (one command stream) or "layer"
    overflow_scope: OverflowScope
    #: Counters reset before each addend; None when the consumer has not stated it. Two records
    #: that both say "not stated" compare equal, which is right for the same consumer and says
    #: nothing about a different one.
    reset_per_unit: bool | None = True
    #: Provenance: where this description of the consumer comes from. Not part of equality,
    #: so two consumers described from different sources but reading the same way compare equal.
    source: str = field(default="", compare=False)


#: helx's Corstone FVP runner (``ethosu_profiler.c``): per-dispatch reads summed in 64-bit in
#: firmware, any overflow bit marks the whole capture.
HELX_RUNNER_ACQUISITION = Acquisition(
    consumer="helx-fvp-runner",
    accumulation_bits=64,
    accumulation_site=AccumulationSite.FIRMWARE,
    accumulation_unit="dispatch",
    overflow_scope=OverflowScope.ANY_COUNTER,
    source="helia_estimator/local_fvp_runtime/src/ethosu_profiler.c",
)

#: heliaPROFILER's firmware (its PMU template, 2026-09-20): 32-bit per-layer accumulation
#: on the target, overflow checked over the selected event mask. heliaPROFILER owns this; it is
#: recorded so the difference from helx is data.
HPX_FIRMWARE_ACQUISITION = Acquisition(
    consumer="hpx-firmware",
    accumulation_bits=32,
    accumulation_site=AccumulationSite.FIRMWARE,
    accumulation_unit="layer",
    overflow_scope=OverflowScope.SELECTED_EVENT_SLOTS,
    reset_per_unit=None,  # not documented by heliaPROFILER yet
    source="heliaPROFILER firmware PMU template (2026-09-20)",
)


@dataclass(frozen=True)
class PmuProgram:
    """One simultaneous program: what the counters are set to for one dispatch.

    A capture that needs more events than slots runs several programs in
    separate passes; that is a list of programs, never one program, and the
    passes are not equivalent to a single simultaneous capture.
    """

    name: str  # profile name as selected in configs
    npu_family: NpuFamily
    slots: tuple[PmuSlot, ...]
    cycle_counter: CycleCounter = CycleCounter()
    derived: tuple[DerivedField, ...] = ()
    description: str = ""
    version: int = 1
    event_counter_bits: int = 32  # hardware: every event slot is 32-bit
    #: How the consumer that runs this program reads and sums it. Not part of the program's
    #: identity (``id``, ``events``); part of whether two captures are comparable.
    acquisition: Acquisition | None = None
    # Retained result identity: a package rename must not change the PMU program.
    owner: str = "helia-npu"

    def __post_init__(self) -> None:
        expected = slot_count(self.npu_family)
        if not 1 <= len(self.slots) <= expected:
            raise ValueError(
                f"{self.npu_family.value} has {expected} event counters; "
                f"program '{self.name}' defines {len(self.slots)}"
            )
        fields = [slot.field for slot in self.slots]
        if len(set(fields)) != len(fields):
            raise ValueError(f"program '{self.name}' maps two slots to one field")
        cycle_slots = [slot for slot in self.slots if slot.event == "ETHOSU_PMU_CYCLE"]
        if cycle_slots and self.cycle_counter.source is not CycleSource.EVENT_SLOT:
            raise ValueError(
                f"program '{self.name}' puts ETHOSU_PMU_CYCLE in a slot but cycle_counter.source is not event_slot"
            )
        if self.cycle_counter.source is CycleSource.EVENT_SLOT and not cycle_slots:
            raise ValueError(f"program '{self.name}' declares an event-slot cycle counter but no ETHOSU_PMU_CYCLE slot")

    @property
    def id(self) -> str:
        """Stable identity a result can record: ``<owner>/<family>/<name>/v<version>``."""
        return f"{self.owner}/{self.npu_family.value}/{self.name}/v{self.version}"

    @property
    def measured_fields(self) -> tuple[str, ...]:
        """Fields read from hardware counters (slots and the cycle counter), as opposed to derived."""
        fields = [slot.field for slot in self.slots]
        if self.cycle_counter.source is CycleSource.CCNT:
            fields.append(self.cycle_counter.field)
        return tuple(fields)

    @property
    def events(self) -> tuple[str, ...]:
        """The ordered driver events, slot 0 first: what the firmware actually programs."""
        return tuple(slot.event for slot in self.slots)

    @property
    def required_fields(self) -> tuple[str, ...]:
        """Fields the host requires in a report: slot 0, the derived fields, the cycle counter, and required slots."""
        head = [self.slots[0].field, *(d.field for d in self.derived)]
        if self.cycle_counter.source is CycleSource.CCNT:
            head.append(self.cycle_counter.field)
        return (*head, *(slot.field for slot in self.slots[1:] if slot.required))

    @property
    def fields(self) -> tuple[str, ...]:
        return (*self.measured_fields, *(d.field for d in self.derived))

    def same_program_as(self, other: PmuProgram) -> bool:
        """The counters are set up identically: same family, ordered events and cycle-counter use.

        Host labels (field names, printed report names) are not part of it.
        """
        mine, theirs = self.cycle_counter, other.cycle_counter
        return (
            self.npu_family is other.npu_family
            and self.events == other.events
            and (mine.source, mine.gated_by_event, mine.bits) == (theirs.source, theirs.gated_by_event, theirs.bits)
        )

    def comparable_with(self, other: PmuProgram) -> bool:
        """Numbers from the two can be compared: same program *and* the same, known acquisition."""
        return (
            self.same_program_as(other)
            and self.acquisition is not None
            and other.acquisition is not None
            and self.acquisition == other.acquisition
        )


def slot_count(family: NpuFamily) -> int:
    return 8 if family is NpuFamily.ETHOS_U85 else 4


# ── The programs helx's firmware has always run, now as data ────────────────

_ACTIVE = PmuSlot("ETHOSU_PMU_NPU_ACTIVE", "npu_active_cycles", PmuUnit.CYCLES, "NPU ACTIVE")
_IDLE = DerivedField("npu_idle_cycles", "npu_total_cycles - npu_active_cycles", PmuUnit.CYCLES, "NPU IDLE")


def _slot(event: str, field: str, unit: PmuUnit, *, required: bool = True) -> PmuSlot:
    return PmuSlot(event, field, unit, f"NPU {event}", required=required)


def _beats(event: str, field: str, *, required: bool = True) -> PmuSlot:
    return _slot(event, field, PmuUnit.BEATS, required=required)


def _cycles(event: str, field: str, *, required: bool = True) -> PmuSlot:
    return _slot(event, field, PmuUnit.CYCLES, required=required)


_U55_PROGRAMS: dict[str, tuple[str, tuple[PmuSlot, ...]]] = {
    "default": (
        "Energy counters: port beats on AXI0 (SRAM) and AXI1 reads.",
        (
            _ACTIVE,
            _beats("ETHOSU_PMU_AXI0_RD_DATA_BEAT_RECEIVED", "npu_sram_rd_data_beats"),
            _beats("ETHOSU_PMU_AXI0_WR_DATA_BEAT_WRITTEN", "npu_sram_wr_data_beats"),
            _beats("ETHOSU_PMU_AXI1_RD_DATA_BEAT_RECEIVED", "npu_ext_rd_data_beats"),
        ),
    ),
    "default_ext_rd_stall": (
        "Same fields as default; on U55/U65 there is no spare slot for a stall counter.",
        (
            _ACTIVE,
            _beats("ETHOSU_PMU_AXI0_RD_DATA_BEAT_RECEIVED", "npu_sram_rd_data_beats"),
            _beats("ETHOSU_PMU_AXI0_WR_DATA_BEAT_WRITTEN", "npu_sram_wr_data_beats"),
            _beats("ETHOSU_PMU_AXI1_RD_DATA_BEAT_RECEIVED", "npu_ext_rd_data_beats"),
        ),
    ),
    "ext_stall": (
        "AXI1 (external) traffic and read-request stalls.",
        (
            _ACTIVE,
            _beats("ETHOSU_PMU_AXI1_RD_DATA_BEAT_RECEIVED", "npu_ext_rd_data_beats"),
            _beats("ETHOSU_PMU_AXI1_WR_DATA_BEAT_WRITTEN", "npu_ext_wr_data_beats"),
            _cycles("ETHOSU_PMU_AXI1_RD_TRAN_REQ_STALLED", "npu_ext_rd_tran_req_stalled"),
        ),
    ),
}

_U85_PROGRAMS: dict[str, tuple[str, tuple[PmuSlot, ...]]] = {
    "default": (
        "Energy counters: SRAM and EXT beats, plus the per-port splits.",
        (
            _ACTIVE,
            _beats("ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED", "npu_sram_rd_data_beats"),
            _beats("ETHOSU_PMU_SRAM_WR_DATA_BEAT_WRITTEN", "npu_sram_wr_data_beats"),
            _beats("ETHOSU_PMU_EXT_RD_DATA_BEAT_RECEIVED", "npu_ext_rd_data_beats"),
            _beats("ETHOSU_PMU_EXT_WR_DATA_BEAT_WRITTEN", "npu_ext_wr_data_beats", required=False),
            _beats("ETHOSU_PMU_EXT0_RD_DATA_BEAT_RECEIVED", "npu_ext0_rd_data_beats", required=False),
            _beats("ETHOSU_PMU_EXT1_RD_DATA_BEAT_RECEIVED", "npu_ext1_rd_data_beats", required=False),
            _beats("ETHOSU_PMU_SRAM0_RD_DATA_BEAT_RECEIVED", "npu_sram0_rd_data_beats", required=False),
        ),
    ),
    "default_ext_rd_stall": (
        "The default energy counters with EXT read-request stalls in place of the EXT1 split.",
        (
            _ACTIVE,
            _beats("ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED", "npu_sram_rd_data_beats"),
            _beats("ETHOSU_PMU_SRAM_WR_DATA_BEAT_WRITTEN", "npu_sram_wr_data_beats"),
            _beats("ETHOSU_PMU_EXT_RD_DATA_BEAT_RECEIVED", "npu_ext_rd_data_beats"),
            _beats("ETHOSU_PMU_EXT_WR_DATA_BEAT_WRITTEN", "npu_ext_wr_data_beats", required=False),
            _beats("ETHOSU_PMU_EXT0_RD_DATA_BEAT_RECEIVED", "npu_ext0_rd_data_beats", required=False),
            _cycles("ETHOSU_PMU_EXT_RD_TRAN_REQ_STALLED", "npu_ext_rd_tran_req_stalled", required=False),
            _beats("ETHOSU_PMU_SRAM0_RD_DATA_BEAT_RECEIVED", "npu_sram0_rd_data_beats", required=False),
        ),
    ),
    "ext_stall": (
        "EXT (port 1) traffic and every EXT stall counter the U85 has.",
        (
            _ACTIVE,
            _beats("ETHOSU_PMU_EXT_RD_DATA_BEAT_RECEIVED", "npu_ext_rd_data_beats"),
            _beats("ETHOSU_PMU_EXT_WR_DATA_BEAT_WRITTEN", "npu_ext_wr_data_beats"),
            _cycles("ETHOSU_PMU_EXT_RD_TRAN_REQ_STALLED", "npu_ext_rd_tran_req_stalled"),
            _cycles("ETHOSU_PMU_EXT_WR_TRAN_REQ_STALLED", "npu_ext_wr_tran_req_stalled"),
            _cycles("ETHOSU_PMU_EXT_WR_DATA_BEAT_STALLED", "npu_ext_wr_data_beat_stalled", required=False),
            _cycles("ETHOSU_PMU_EXT_RD_STALL_LIMIT", "npu_ext_rd_stall_limit", required=False),
            _cycles("ETHOSU_PMU_EXT_WR_STALL_LIMIT", "npu_ext_wr_stall_limit", required=False),
        ),
    ),
    "port_stall": (
        "Where the NPU waits for memory: read stall-limit and request-stalled cycles per port, plus MAC activity and "
        "read beats. The capture the RTL testbench and the FPGA use to locate a memory stall.",
        (
            _ACTIVE,
            _cycles("ETHOSU_PMU_MAC_ACTIVE", "npu_mac_active_cycles"),
            _cycles("ETHOSU_PMU_SRAM_RD_STALL_LIMIT", "npu_sram_rd_stall_limit"),
            _cycles("ETHOSU_PMU_SRAM_RD_TRAN_REQ_STALLED", "npu_sram_rd_tran_req_stalled"),
            _cycles("ETHOSU_PMU_EXT_RD_STALL_LIMIT", "npu_ext_rd_stall_limit"),
            _cycles("ETHOSU_PMU_EXT_RD_TRAN_REQ_STALLED", "npu_ext_rd_tran_req_stalled"),
            _beats("ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED", "npu_sram_rd_data_beats"),
            _beats("ETHOSU_PMU_EXT_RD_DATA_BEAT_RECEIVED", "npu_ext_rd_data_beats"),
        ),
    ),
    "sram_stall": (
        "SRAM read/write queue pressure and traffic, plus MAC activity; no EXT traffic or full memory energy.",
        (
            _ACTIVE,
            _cycles("ETHOSU_PMU_MAC_ACTIVE", "npu_mac_active_cycles"),
            _cycles("ETHOSU_PMU_SRAM_RD_STALL_LIMIT", "npu_sram_rd_stall_limit"),
            _cycles("ETHOSU_PMU_SRAM_RD_TRAN_REQ_STALLED", "npu_sram_rd_tran_req_stalled"),
            _cycles("ETHOSU_PMU_SRAM_WR_STALL_LIMIT", "npu_sram_wr_stall_limit"),
            _cycles("ETHOSU_PMU_SRAM_WR_TRAN_REQ_STALLED", "npu_sram_wr_tran_req_stalled"),
            _beats("ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED", "npu_sram_rd_data_beats"),
            _beats("ETHOSU_PMU_SRAM_WR_DATA_BEAT_WRITTEN", "npu_sram_wr_data_beats"),
        ),
    ),
    "sram_transactions": (
        "Aggregate SRAM accepted transactions and matching beats, MAC activity and all four energy beat counters; "
        "replaces default's SRAM0/EXT0/EXT1 read splits, not a per-port capture or stall measure.",
        (
            _ACTIVE,
            _cycles("ETHOSU_PMU_MAC_ACTIVE", "npu_mac_active_cycles"),
            _beats("ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED", "npu_sram_rd_data_beats"),
            _beats("ETHOSU_PMU_SRAM_WR_DATA_BEAT_WRITTEN", "npu_sram_wr_data_beats"),
            _beats("ETHOSU_PMU_EXT_RD_DATA_BEAT_RECEIVED", "npu_ext_rd_data_beats"),
            _beats("ETHOSU_PMU_EXT_WR_DATA_BEAT_WRITTEN", "npu_ext_wr_data_beats"),
            _slot("ETHOSU_PMU_SRAM_RD_TRANS_ACCEPTED", "npu_sram_rd_transactions", PmuUnit.TRANSACTIONS),
            _slot("ETHOSU_PMU_SRAM_WR_TRANS_ACCEPTED", "npu_sram_wr_transactions", PmuUnit.TRANSACTIONS),
        ),
    ),
    "dv_align": (
        "What the RTL testbench and FPGA (heliaPROFILER) datasets report: MAC activity and stalls plus the four "
        "energy beats.",
        (
            _ACTIVE,
            _cycles("ETHOSU_PMU_MAC_ACTIVE", "npu_mac_active_cycles"),
            _cycles("ETHOSU_PMU_MAC_STALLED_BY_IB", "npu_mac_stalled_by_ib_cycles", required=False),
            _cycles("ETHOSU_PMU_MAC_STALLED_BY_W_OR_ACC", "npu_mac_stalled_by_w_or_acc_cycles", required=False),
            _beats("ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED", "npu_sram_rd_data_beats"),
            _beats("ETHOSU_PMU_SRAM_WR_DATA_BEAT_WRITTEN", "npu_sram_wr_data_beats"),
            _beats("ETHOSU_PMU_EXT_RD_DATA_BEAT_RECEIVED", "npu_ext_rd_data_beats"),
            _beats("ETHOSU_PMU_EXT_WR_DATA_BEAT_WRITTEN", "npu_ext_wr_data_beats"),
        ),
    ),
}


#: heliaPROFILER's curated single-pass program as of 2026-09-20. Four slots on the
#: U85, ETHOSU_PMU_CYCLE in slot 0, CCNT enabled but not read.
#: Named here so both tools can select the same program by id; heliaPROFILER owns its contents.
_HPX_DEFAULT_V1 = PmuProgram(
    "hpx_default_v1",
    NpuFamily.ETHOS_U85,
    (
        PmuSlot("ETHOSU_PMU_CYCLE", "npu_total_cycles", PmuUnit.CYCLES, "ETHOSU_PMU_CYCLE"),
        PmuSlot("ETHOSU_PMU_NPU_ACTIVE", "npu_active_cycles", PmuUnit.CYCLES, "ETHOSU_PMU_NPU_ACTIVE"),
        PmuSlot("ETHOSU_PMU_MAC_ACTIVE", "npu_mac_active_cycles", PmuUnit.CYCLES, "ETHOSU_PMU_MAC_ACTIVE"),
        PmuSlot(
            "ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED",
            "npu_sram_rd_data_beats",
            PmuUnit.BEATS,
            "ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED",
        ),
    ),
    cycle_counter=CycleCounter(source=CycleSource.EVENT_SLOT, field="npu_total_cycles", bits=32),
    derived=(),
    description="heliaPROFILER default: cycle count in an event slot, no derived idle.",
    acquisition=HPX_FIRMWARE_ACQUISITION,
    owner="hpx",
)


def _build() -> dict[tuple[NpuFamily, str], PmuProgram]:
    programs: dict[tuple[NpuFamily, str], PmuProgram] = {}
    for family in (NpuFamily.ETHOS_U55, NpuFamily.ETHOS_U65):
        for name, (description, slots) in _U55_PROGRAMS.items():
            programs[(family, name)] = PmuProgram(
                name, family, slots, derived=(_IDLE,), description=description, acquisition=HELX_RUNNER_ACQUISITION
            )
    for name, (description, slots) in _U85_PROGRAMS.items():
        programs[(NpuFamily.ETHOS_U85, name)] = PmuProgram(
            name,
            NpuFamily.ETHOS_U85,
            slots,
            derived=(_IDLE,),
            description=description,
            acquisition=HELX_RUNNER_ACQUISITION,
        )
    programs[(NpuFamily.ETHOS_U85, _HPX_DEFAULT_V1.name)] = _HPX_DEFAULT_V1
    return programs


PROGRAMS: dict[tuple[NpuFamily, str], PmuProgram] = _build()
DEFAULT_PROGRAM = "default"


def list_programs(family: NpuFamily | str) -> tuple[str, ...]:
    family = NpuFamily(family)
    return tuple(name for (fam, name) in PROGRAMS if fam is family)


def program_for(family: NpuFamily | str, name: str | None = None) -> PmuProgram:
    """The program a family runs under a profile name; ``None`` means the default."""
    family = NpuFamily(family)
    key = (family, name or DEFAULT_PROGRAM)
    try:
        return PROGRAMS[key]
    except KeyError:
        available = ", ".join(list_programs(family))
        raise ValueError(f"No PMU program '{name}' for {family.value}. Available: {available}") from None


def render_c_header(program: PmuProgram) -> str:
    """The C header the firmware compiles the program from: one table, no #if ladder."""
    lines = [
        "// Generated by helia_ethos.pmu from the PMU program descriptor. Do not edit.",
        f"// Program: {program.npu_family.value}/{program.name}: {program.description}",
        "#pragma once",
        "",
        f'#define HELX_PMU_PROGRAM_NAME "{program.npu_family.value}/{program.name}"',
        f"#define HELX_PMU_NSLOTS {len(program.slots)}U",
        "",
        "// {driver event, report label, unit} per counter slot, slot 0 first.",
        "#define HELX_PMU_SLOTS \\",
    ]
    for index, slot in enumerate(program.slots):
        sep = "," if index < len(program.slots) - 1 else ""
        lines.append(f'    {{{slot.event}, "{slot.report_name}", "{slot.unit.value}"}}{sep} \\')
    lines[-1] = lines[-1].rstrip(" \\")
    lines.extend(
        [
            "",
            f'#define HELX_PMU_PROGRAM_ID "{program.id}"',
            f'#define HELX_PMU_CCNT_REPORT_NAME "{program.cycle_counter.report_name}"',
            f"#define HELX_PMU_CCNT_GATED {1 if program.cycle_counter.gated_by_event else 0}",
            f"#define HELX_PMU_CCNT_SOURCE_{program.cycle_counter.source.value.upper()} 1",
            f"#define HELX_PMU_CCNT_BITS {program.cycle_counter.bits}",
            f"#define HELX_PMU_SLOT0_IS_NPU_ACTIVE {1 if program.slots[0].event == 'ETHOSU_PMU_NPU_ACTIVE' else 0}",
            f"#define HELX_PMU_NDERIVED {len(program.derived)}U",
        ]
    )
    for derived in program.derived:
        lines.append(f'#define HELX_PMU_DERIVED_{derived.field.upper()}_REPORT_NAME "{derived.report_name}"')
    return "\n".join(lines) + "\n"

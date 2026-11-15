"""A target: one realization of a SoC revision, and what it actually implements.

A SoC definition says what the part *is*. A target says what a particular
instance *does*: silicon of a given stepping, an FPGA drop that carries a
subset of the design at a fraction of the clock, or a simulator with known
fidelity limits. Every one of them has been called "atomiq110" in results,
which is how an FPGA running the NPU and fabric at 25 MHz 1:1 with no HBLRAM
gets compared to a simulator modelling 500/250 MHz with HBLRAM. A target
record makes that difference data.

Targets are consumed by name (``load_target("atomiq110-fpga-turbo-npu")``) or
from a YAML file, and are meant to be recorded, with the SoC id and revision,
in every result a tool produces.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .fvp import FvpSubsystem
from .yaml_loading import load_yaml_mapping


class TargetError(ValueError):
    """A target definition is inconsistent or was asked for something it does not implement."""


class TargetKind(StrEnum):
    SILICON = "silicon"  # the part itself, on a board
    FPGA = "fpga"  # an RTL drop synthesised onto an FPGA: a subset, at a fraction of the clock
    SIMULATOR = "simulator"  # a Fast Model or equivalent: timing is a model, not a measurement
    RTL = "rtl"  # the RTL testbench itself (verification and power-analysis runs)


class Feature(StrEnum):
    """Blocks a target may or may not implement. Coarse on purpose; refine when a consumer needs to."""

    CPU = "cpu"
    NPU = "npu"
    GPU = "gpu"
    DISPLAY = "display"
    SSRAM = "ssram"
    TCM = "tcm"
    HBLRAM = "hblram"
    MRAM = "mram"
    RRAM = "rram"
    PSRAM = "psram"
    XSPI = "xspi"
    PMU = "pmu"


@dataclass(frozen=True)
class ClockConfig:
    """Hz. ``None`` means the target does not have that domain (or it is not known)."""

    cpu_hz: float | None = None
    npu_hz: float | None = None
    fabric_hz: float | None = None

    @property
    def npu_to_fabric_ratio(self) -> float | None:
        if self.npu_hz is None or self.fabric_hz is None or self.fabric_hz == 0:
            return None
        return self.npu_hz / self.fabric_hz


@dataclass(frozen=True)
class Emulation:
    """How a memory or block on this target stands in for the real one."""

    feature: Feature
    stand_in: str  # e.g. "bram", "ddr", "absent", "fast-model"
    notes: str | None = None


@dataclass(frozen=True)
class TargetDefinition:
    id: str  # canonical, stable: "atomiq110-fpga-turbo-npu"
    soc: str  # SoC id this realizes: "atomiq110"
    kind: TargetKind
    name: str
    soc_revision: str | None = None  # RTL drop / stepping the target was built from, when known
    board: str | None = None  # "atomiq110_fpga_turbo", "apollo510_evb"; the BSP's name
    implements: frozenset[Feature] = frozenset()
    clocks: ClockConfig = field(default_factory=ClockConfig)
    emulation: tuple[Emulation, ...] = ()
    #: For FPGA: the bitstream / drop identifier. For silicon: the stepping. For simulators: the model version.
    build_id: str | None = None
    #: For a Corstone FVP simulator: which one. A run is attributed to it only when the SoC's NPU runs there.
    fvp_subsystem: FvpSubsystem | None = None
    since: date | None = None
    fidelity_notes: tuple[str, ...] = ()
    #: Whether cycle counts taken on this target represent the SoC's timing. Every target file states it:
    #: false for FPGA drops running at other clock ratios and simulators with partial memory models;
    #: the shipped RTL and silicon targets say true. The default is false so a new target must opt in.
    timing_is_representative: bool = False

    def implements_all(self, *features: Feature) -> bool:
        return all(feature in self.implements for feature in features)

    def require(self, *features: Feature) -> None:
        missing = [feature.value for feature in features if feature not in self.implements]
        if missing:
            raise TargetError(f"Target '{self.id}' does not implement: {', '.join(missing)}")

    def emulation_for(self, feature: Feature) -> Emulation | None:
        return next((e for e in self.emulation if e.feature is feature), None)

    def label(self) -> str:
        """``"atomiq110-fpga-turbo-npu (fpga, rtl-2026-09)"``-style, for reports."""
        rev = f", {self.soc_revision}" if self.soc_revision else ""
        return f"{self.id} ({self.kind.value}{rev})"


# ── YAML boundary ───────────────────────────────────────────────────────────


class _TargetBoundaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClockConfigModel(_TargetBoundaryModel):
    cpu_hz: float | None = None
    npu_hz: float | None = None
    fabric_hz: float | None = None

    @field_validator("cpu_hz", "npu_hz", "fabric_hz", mode="before")
    @classmethod
    def reject_booleans(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("must be a number, not a boolean")
        return value


class EmulationModel(_TargetBoundaryModel):
    feature: Feature
    stand_in: str = Field(min_length=1)
    notes: str | None = None


class TargetModel(_TargetBoundaryModel):
    id: str = Field(min_length=1)
    soc: str = Field(min_length=1)
    kind: Literal["silicon", "fpga", "simulator", "rtl"]
    name: str = Field(min_length=1)
    soc_revision: str | None = None
    board: str | None = None
    implements: list[Feature] = Field(default_factory=list)
    clocks: ClockConfigModel = Field(default_factory=ClockConfigModel)
    emulation: list[EmulationModel] = Field(default_factory=list)
    build_id: str | None = None
    fvp_subsystem: FvpSubsystem | None = None
    since: date | None = None
    fidelity_notes: list[str] = Field(default_factory=list)
    timing_is_representative: bool = False

    @field_validator("id", "soc", mode="before")
    @classmethod
    def normalize_ids(cls, value: object) -> object:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip().lower()

    @model_validator(mode="after")
    def emulated_blocks_are_not_implemented_as_real(self) -> TargetModel:
        for entry in self.emulation:
            if entry.stand_in == "absent" and entry.feature in self.implements:
                raise ValueError(f"{entry.feature.value} is listed both as implemented and as absent")
        if self.fvp_subsystem is not None and self.kind != "simulator":
            raise ValueError(
                f"fvp_subsystem names a Corstone FVP, which only a simulator target has; this one is {self.kind}"
            )
        return self


def target_from_model(document: TargetModel) -> TargetDefinition:
    return TargetDefinition(
        id=document.id,
        soc=document.soc,
        kind=TargetKind(document.kind),
        name=document.name,
        soc_revision=document.soc_revision,
        board=document.board,
        implements=frozenset(document.implements),
        clocks=ClockConfig(
            cpu_hz=document.clocks.cpu_hz, npu_hz=document.clocks.npu_hz, fabric_hz=document.clocks.fabric_hz
        ),
        emulation=tuple(Emulation(e.feature, e.stand_in, e.notes) for e in document.emulation),
        build_id=document.build_id,
        fvp_subsystem=document.fvp_subsystem,
        since=document.since,
        fidelity_notes=tuple(document.fidelity_notes),
        timing_is_representative=document.timing_is_representative,
    )


def validate_target_document(raw: object) -> TargetModel:
    try:
        return TargetModel.model_validate(raw)
    except ValidationError as exc:
        error = exc.errors(include_url=False)[0]
        location = ".".join(str(part) for part in error["loc"])
        message = str(error["msg"]).removeprefix("Value error, ")
        raise TargetError(f"{location + ': ' if location else ''}{message}") from exc


TARGETS_DIR = Path(__file__).resolve().parent / "targets"


def list_targets(soc: str | None = None) -> list[str]:
    """Ids of the packaged targets, optionally only those realizing *soc*."""
    ids = []
    for path in sorted(TARGETS_DIR.glob("*.yaml")):
        if soc is None:
            ids.append(path.stem)
        else:
            try:
                raw = load_yaml_mapping(path, what="Target file")
            except ValueError as exc:
                raise TargetError(str(exc)) from exc
            if str(raw.get("soc", "")).lower() == soc.lower():
                ids.append(path.stem)
    return ids


def load_target(path_or_id: str | Path, *, on_read: Callable[[Path, bytes], None] | None = None) -> TargetDefinition:
    """Load a target by packaged id or from a YAML file."""
    path = Path(path_or_id)
    if not path.suffix:
        wanted = str(path_or_id).strip().lower()
        matched = next((p for p in TARGETS_DIR.glob("*.yaml") if p.stem.lower() == wanted), None)
        if matched is None:
            raise TargetError(f"No packaged target named '{path_or_id}'. Available: {', '.join(list_targets())}")
        path = matched
    if not path.exists():
        raise TargetError(f"Target definition not found: {path}")
    try:
        raw = load_yaml_mapping(path, what="Target file", on_read=on_read)
    except ValueError as exc:
        raise TargetError(str(exc)) from exc
    try:
        return target_from_model(validate_target_document(raw))
    except TargetError as exc:
        raise TargetError(f"{path}: {exc}") from exc

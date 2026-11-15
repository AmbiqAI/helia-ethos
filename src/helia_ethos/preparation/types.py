"""Typed requests, observations and retained preparation evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer, model_validator

import helia_ethos

from ..catalog import CatalogNpuFacts
from ..catalog_identity import CatalogContentIdentity
from ..model import ModelTensor
from ..soc import MemoryPlacement, NpuFamily, SocDefinition, TimingOverrides


class PreparationError(ValueError):
    """Preparation could not produce or inspect a usable model."""


class InvalidPreparation(PreparationError):
    """The source or explicit preparation settings are invalid."""


class CompilerUnavailable(PreparationError):
    """The requested compiler cannot be located or started."""


class CompilationFailed(PreparationError):
    """The compiler failed; the destination retains diagnostics."""


class InvalidPreparedArtifact(PreparationError):
    """Retained preparation evidence is missing, inconsistent or incompatible."""


class AcceleratorCompatibility(StrEnum):
    """Relationship between decoded regions and an explicit accelerator request."""

    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Accelerator:
    """Ethos-U family and MAC count, independent of compiler estimates."""

    family: NpuFamily
    macs: int


@dataclass(frozen=True)
class CustomNpuTarget:
    """Caller-owned what-if target identity; makes no catalog hardware claim."""

    identity: str
    accelerator: Accelerator


@dataclass(frozen=True)
class NpuCompileSettings:
    """Explicit compiler model and placement, with its caller-selected policy identity."""

    soc: SocDefinition
    mode: str
    placement: MemoryPlacement
    policy_id: str
    arena_cache_size: int = 0
    timing_overrides: TimingOverrides | None = None
    vela_bin: str | None = None


@dataclass(frozen=True)
class ExplicitFences:
    """Cut after these zero-based original operator indices."""

    boundaries: tuple[int, ...]


@dataclass(frozen=True)
class BudgetFences:
    """Plan cuts from Vela estimated cycles; makes no execution deadline promise."""

    cycle_budget: int
    min_coverage: float = 1.0


FenceSelection = ExplicitFences | BudgetFences


class NpuRequirement(StrEnum):
    """Whether a successful compile must contain an Ethos-U region."""

    REQUIRED = "required"
    OPTIONAL = "optional"


@dataclass(frozen=True)
class PrepareRequest:
    """Prepare one original model in a fresh, caller-owned directory."""

    source: Path
    target: CatalogNpuFacts | CustomNpuTarget
    settings: NpuCompileSettings
    output_dir: Path
    fences: FenceSelection | None = None
    retain_source: bool = False
    npu_requirement: NpuRequirement = NpuRequirement.REQUIRED


class ArtifactRole(StrEnum):
    """The role of a file in the retained preparation record."""

    MODEL = "model"
    CONFIG = "config"
    BOUNDARIES = "boundaries"
    SOURCE = "source"
    REWRITTEN = "rewritten"
    DIAGNOSTIC = "diagnostic"


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class RecordValue(BaseModel):
    """Closed, immutable JSON-boundary value."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Artifact(RecordValue):
    """A relative retained file identity; optional absent files do not prevent use."""

    role: ArtifactRole
    path: str
    sha256: Sha256
    required: bool


class TargetEvidence(RecordValue):
    """Selected target identity, kept separate from compiler modeling assumptions."""

    kind: Literal["catalog", "custom"]
    target_id: str
    npu_id: str | None = None
    implementation_id: str | None = None
    catalog: CatalogContentIdentity | None = None


class CompileEvidence(RecordValue):
    """Effective compiler inputs and executable evidence for the final compilation."""

    accelerator: Accelerator
    mode: str
    placement: MemoryPlacement
    arena_cache_size: int = Field(ge=0)
    policy_id: str
    effective_model_json: str
    effective_model_sha256: Sha256
    arguments: tuple[str, ...]
    version_output: str
    executable_sha256: Sha256
    reported_accelerator: str | None
    reported_system_config: str | None
    reported_memory_mode: str | None


class CustomOperator(RecordValue):
    """A custom operator version the consumer's runtime must implement."""

    name: str
    version: int


class Region(RecordValue):
    """One observed Ethos-U invocation; undecodable accelerator identity is unknown."""

    subgraph: int
    operator: int
    accelerator: Accelerator | None
    unknown_reason: str | None = None


class Segment(RecordValue):
    """Original fence segment interval covered by one final NPU invocation."""

    first: int
    last: int


class FenceEvidence(RecordValue):
    """Selected boundaries and estimate coverage, plus verified invocation mapping."""

    kind: Literal["explicit", "budget"]
    boundaries: tuple[int, ...]
    cycle_budget: int | None = None
    min_coverage: float | None = None
    matched_cycles: int | None = None
    unmatched_cycles: int | None = None
    spans: tuple[Segment, ...]


class PackageVersion(RecordValue):
    """A package version in the preparing Python environment."""

    name: str
    version: str


class PreparationRecord(RecordValue):
    """Versioned compilation evidence; not a measurement or execution bundle."""

    schema_version: Literal[1, 2] = 2
    npu_requirement: NpuRequirement = NpuRequirement.REQUIRED
    source_sha256: Sha256
    target: TargetEvidence
    compiler: CompileEvidence
    artifacts: tuple[Artifact, ...]
    fences: FenceEvidence | None = None
    packages: tuple[PackageVersion, ...]
    regions: tuple[Region, ...]
    custom_operators: tuple[CustomOperator, ...]
    inputs: tuple[ModelTensor, ...]
    outputs: tuple[ModelTensor, ...]

    @model_serializer(mode="wrap")
    def _serialize_version(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        payload = handler(self)
        if self.schema_version == 1:
            payload.pop("npu_requirement", None)
        return payload

    @model_validator(mode="after")
    def _version_policy(self) -> PreparationRecord:
        if self.schema_version == 1 and self.npu_requirement is not NpuRequirement.REQUIRED:
            raise ValueError("Preparation record version 1 requires Ethos-U regions")
        return self


@dataclass(frozen=True)
class PreparedModelInfo:
    """Observed model properties and optional verified retained compilation evidence."""

    tflite: Path
    sha256: str
    regions: tuple[Region, ...]
    custom_operators: tuple[CustomOperator, ...]
    inputs: tuple[ModelTensor, ...]
    outputs: tuple[ModelTensor, ...]
    record: PreparationRecord | None = None
    unavailable_optional: tuple[str, ...] = ()

    def compatibility(self, expected: Accelerator) -> AcceleratorCompatibility:
        """Compare every decoded region with an explicit expected accelerator."""
        if any(region.accelerator is not None and region.accelerator != expected for region in self.regions):
            return AcceleratorCompatibility.INCOMPATIBLE
        if not self.regions or any(region.accelerator is None for region in self.regions):
            return AcceleratorCompatibility.UNKNOWN
        return AcceleratorCompatibility.COMPATIBLE


@dataclass(frozen=True)
class PreparedModel:
    """A compiled model, its relocatable preparation record and observed requirements."""

    tflite: Path
    record_path: Path
    info: PreparedModelInfo
    compilation: helia_ethos.CompiledModel
    compiler_log: Path

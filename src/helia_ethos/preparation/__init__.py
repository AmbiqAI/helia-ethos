"""Shared Ethos-U preparation and inspection; consumers own all execution."""

from .inspection import inspect_prepared_model
from .prepare import prepare_model
from .types import (
    Accelerator,
    AcceleratorCompatibility,
    Artifact,
    ArtifactRole,
    BudgetFences,
    CompilationFailed,
    CompilerUnavailable,
    CustomNpuTarget,
    ExplicitFences,
    FenceSelection,
    InvalidPreparation,
    InvalidPreparedArtifact,
    NpuCompileSettings,
    PreparationError,
    PreparationRecord,
    PreparedModel,
    PreparedModelInfo,
    PrepareRequest,
)

__all__ = [
    "Accelerator",
    "AcceleratorCompatibility",
    "Artifact",
    "ArtifactRole",
    "BudgetFences",
    "CompilationFailed",
    "CompilerUnavailable",
    "CustomNpuTarget",
    "ExplicitFences",
    "FenceSelection",
    "InvalidPreparation",
    "InvalidPreparedArtifact",
    "NpuCompileSettings",
    "PreparationError",
    "PreparationRecord",
    "PrepareRequest",
    "PreparedModel",
    "PreparedModelInfo",
    "inspect_prepared_model",
    "prepare_model",
]

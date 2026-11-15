"""Shared Ethos-U preparation and inspection; consumers own all execution."""

from pathlib import Path

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


def prepare_model(request: PrepareRequest) -> PreparedModel:
    """Compile the source and retain its preparation evidence in a fresh directory."""
    from .prepare import prepare_model as prepare

    return prepare(request)


def inspect_prepared_model(path: Path) -> PreparedModelInfo:
    """Inspect a model or retained preparation record without compiling or executing it."""
    from .inspection import inspect_prepared_model as inspect

    return inspect(path)


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

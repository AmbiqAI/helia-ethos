"""Ethos-U NPU support for the helia toolchain.

The core, using pydantic, PyYAML and helia-platforms: SoC and target definitions,
timing derivation, the Vela system config and compile driver, the FVP target
matrix and the PMU programs. Reading or rewriting a ``.tflite`` file
(:mod:`helia_ethos.model`, :mod:`helia_ethos.fencing`,
:func:`helia_ethos.vela.read_tflite_graph`) needs the ``models`` extra; those
names are imported on first use so the core imports without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .fvp import (
    DEFAULT_FVP_SUBSYSTEM,
    FVP_TARGET_PROFILES,
    FvpSubsystem,
    FvpTargetProfile,
    NpuFamily,
    required_pmu_fields,
    resolve_fvp_target_profile,
    validate_fvp_memory_mode,
    validate_fvp_runtime_engine,
)
from .npu_reference import ARM_REFERENCE_ENERGY, NpuReferenceEnergy, reference_energy
from .pmu import CycleSource, PmuProgram, list_programs, program_for, render_c_header
from .preparation import (
    Accelerator,
    AcceleratorCompatibility,
    BudgetFences,
    CompilationFailed,
    CompilerUnavailable,
    CustomNpuTarget,
    ExplicitFences,
    FenceSelection,
    InvalidPreparation,
    InvalidPreparedArtifact,
    NpuCompileSettings,
    NpuRequirement,
    PreparationError,
    PreparationRecord,
    PreparedModel,
    PreparedModelInfo,
    PrepareRequest,
    inspect_prepared_model,
    prepare_model,
)
from .soc import (
    AXI_PATTERN_TO_VELA_MODE,
    NO_PROVENANCE,
    NPU_PORT_CLASS,
    PMU_FIELD_PREFIX,
    VELA_AXI_NAME,
    VELA_PORT_TYPE,
    DerivedTimingProfile,
    MemoryDef,
    MemoryParams,
    MemoryPlacement,
    MemoryTiming,
    MemoryTimingOverride,
    ModeClocks,
    Provenance,
    ProvenanceStatus,
    SocDefinition,
    SocError,
    SocOrigin,
    SocRevision,
    SocStage,
    TimingAdapterPort,
    TimingAdapterProfile,
    TimingOverrides,
    VelaMemoryMode,
    derive_timing,
    least_trusted,
    parse_memory_placement,
    port_join_label,
    standard_vela_mode,
)
from .soc_schema import derive_soc, list_socs, load_soc, soc_from_model
from .target import ClockConfig, Feature, TargetDefinition, TargetError, TargetKind, list_targets, load_target
from .yaml_loading import DuplicateKeyError, load_yaml_mapping, load_yaml_text

if TYPE_CHECKING:  # the names resolved by __getattr__, visible to type checkers
    from .fencing import (  # noqa: F401
        Fence,
        FenceBindingError,
        FenceBoundaries,
        FenceConfig,
        FenceCoverage,
        FenceMode,
        FencePlan,
        FencePlanError,
        FenceSegment,
        FenceSidecarError,
        FenceSpan,
        RewriteResult,
        check_plan_binding,
        fence_model,
        load_fence_boundaries,
        npu_invocation_spans,
        plan_fences,
        rewrite_model_with_fences,
    )
    from .model import ModelCharacterization, characterize_model, command_stream_bytes  # noqa: F401
    from .vela import CompiledModel as CompiledModel

__version__ = "0.3.0"  # x-release-please-version

#: Names behind the ``models`` extra, resolved on first use; kept out of ``__all__`` so
#: ``from helia_ethos import *`` works on a core-only install.
_FENCING = {
    "Fence",
    "FenceBindingError",
    "FenceBoundaries",
    "FenceConfig",
    "FenceCoverage",
    "FenceMode",
    "FencePlan",
    "FencePlanError",
    "FenceSegment",
    "FenceSidecarError",
    "FenceSpan",
    "RewriteResult",
    "check_plan_binding",
    "fence_model",
    "load_fence_boundaries",
    "npu_invocation_spans",
    "plan_fences",
    "rewrite_model_with_fences",
}
_MODEL = {"ModelCharacterization", "characterize_model", "command_stream_bytes"}


def __getattr__(name: str) -> Any:
    """Resolve compiler, fencing and model names on demand, keeping core imports lightweight."""
    if name == "CompiledModel":
        from .vela import CompiledModel

        return CompiledModel
    if name in _FENCING:
        from . import fencing

        return getattr(fencing, name)
    if name in _MODEL:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """The module's names plus the ones resolved on first use, so ``dir()`` and completion see them."""
    return sorted(set(globals()) | _FENCING | _MODEL | {"CompiledModel"})


__all__ = [
    "ARM_REFERENCE_ENERGY",
    "AXI_PATTERN_TO_VELA_MODE",
    "DEFAULT_FVP_SUBSYSTEM",
    "FVP_TARGET_PROFILES",
    "NO_PROVENANCE",
    "NPU_PORT_CLASS",
    "PMU_FIELD_PREFIX",
    "VELA_AXI_NAME",
    "VELA_PORT_TYPE",
    "Accelerator",
    "AcceleratorCompatibility",
    "BudgetFences",
    "ClockConfig",
    "CompilationFailed",
    "CompilerUnavailable",
    "CustomNpuTarget",
    "CycleSource",
    "DerivedTimingProfile",
    "DuplicateKeyError",
    "ExplicitFences",
    "Feature",
    "FenceSelection",
    "FvpSubsystem",
    "FvpTargetProfile",
    "InvalidPreparation",
    "InvalidPreparedArtifact",
    "MemoryDef",
    "MemoryParams",
    "MemoryPlacement",
    "MemoryTiming",
    "MemoryTimingOverride",
    "ModeClocks",
    "NpuCompileSettings",
    "NpuFamily",
    "NpuReferenceEnergy",
    "NpuRequirement",
    "PmuProgram",
    "PreparationError",
    "PreparationRecord",
    "PrepareRequest",
    "PreparedModel",
    "PreparedModelInfo",
    "Provenance",
    "ProvenanceStatus",
    "SocDefinition",
    "SocError",
    "SocOrigin",
    "SocRevision",
    "SocStage",
    "TargetDefinition",
    "TargetError",
    "TargetKind",
    "TimingAdapterPort",
    "TimingAdapterProfile",
    "TimingOverrides",
    "VelaMemoryMode",
    "__version__",
    "derive_soc",
    "derive_timing",
    "inspect_prepared_model",
    "least_trusted",
    "list_programs",
    "list_socs",
    "list_targets",
    "load_soc",
    "load_target",
    "load_yaml_mapping",
    "load_yaml_text",
    "parse_memory_placement",
    "port_join_label",
    "prepare_model",
    "program_for",
    "reference_energy",
    "render_c_header",
    "required_pmu_fields",
    "resolve_fvp_target_profile",
    "soc_from_model",
    "standard_vela_mode",
    "validate_fvp_memory_mode",
    "validate_fvp_runtime_engine",
]

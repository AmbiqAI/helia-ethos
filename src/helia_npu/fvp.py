"""Which Corstone FVP subsystem can execute what.

Each Ethos-U variant is simulated by a specific Corstone Fixed Virtual
Platform with its own CPU, base addresses, timing-adapter layout and
launch parameters. This is Arm-side fact, shared by anything that builds
or runs against an FVP.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from .soc import NpuFamily as NpuFamily  # defined with the SoC (it sets the AXI beat); imported from here too
from .soc import VelaMemoryMode


class FvpSubsystem(StrEnum):
    """Corstone subsystem the local runtime targets."""

    CORSTONE_300 = "corstone_300"
    CORSTONE_320 = "corstone_320"

    @classmethod
    def _missing_(cls, value: object) -> FvpSubsystem | None:
        canonical = str(value).strip().lower().replace("-", "_")
        return next((member for member in cls if member.value == canonical), None)


DEFAULT_FVP_SUBSYSTEM = FvpSubsystem.CORSTONE_320


@dataclass(frozen=True)
class FvpTargetProfile:
    """Build and launch contract for one Corstone FVP model: a subsystem with one Ethos-U variant."""

    name: FvpSubsystem
    npu_family: NpuFamily
    ethos_arch: str
    target_cpu: str
    cmsis_device: str
    arm_cpu: str
    cpu_compile_define: str
    ethos_compile_define: str
    platform_target: str
    fvp_dependency: str
    fvp_dirname: str
    fvp_model_relpath: str
    fvp_model_namespace: str
    fvp_npu_parameter_namespace: str
    fvp_extra_visualisation_parameter: str | None
    ethos_u_base_address: str
    ethos_u_irqn: int
    uart0_base: str
    timing_adapter_bases: tuple[str, str]
    fvp_extra_args: tuple[str, ...]
    use_simlimit: bool
    supported_memory_modes: frozenset[VelaMemoryMode]
    supported_runtime_engines: frozenset[str]
    host_transport: str
    fvp_sram_hash_parameter: str | None = None
    """The FVP parameter carrying the NPU's SRAM port hash (HASH0); ``None`` when the model has no such setting."""
    fvp_bus_loggers: tuple[str, str] | None = None
    """The PVBusLogger components on the NPU's two AXI masters, as the FVP's trace sources name them: the SRAM-class
    master first, then EXT. ``None`` when the model has none, so the NPU's bus cannot be traced."""
    fvp_trace_plugin_relpath: str = "plugins/Linux64_GCC-9.3/GenericTrace.so"
    """Arm's GenericTrace plugin, relative to the FVP home."""


_CORSTONE_300_U55 = FvpTargetProfile(
    name=FvpSubsystem.CORSTONE_300,
    npu_family=NpuFamily.ETHOS_U55,
    ethos_arch="u55",
    target_cpu="cortex-m55",
    cmsis_device="ARMCM55.h",
    arm_cpu="ARMCM55",
    cpu_compile_define="CPU_CORTEX_M55",
    ethos_compile_define="ETHOSU55",
    platform_target="corstone-300",
    fvp_dependency="fvp_corstone_300",
    fvp_dirname="FVP_Corstone_SSE-300",
    fvp_model_relpath="models/Linux64_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U55",
    fvp_model_namespace="mps3_board",
    fvp_npu_parameter_namespace="ethosu",
    fvp_extra_visualisation_parameter=None,
    ethos_u_base_address="0x48102000",
    ethos_u_irqn=56,
    uart0_base="0x49303000",
    timing_adapter_bases=("0x48103000", "0x48103200"),
    fvp_extra_args=("-C", "cpu0.CFGDTCMSZ=12", "-C", "cpu0.CFGITCMSZ=12"),
    use_simlimit=False,
    supported_memory_modes=frozenset({VelaMemoryMode.SHARED_SRAM}),
    supported_runtime_engines=frozenset({"tflite_micro", "helia_rt", "helia_aot"}),
    host_transport="uart_file",
)

# The Corstone-300 archive ships a second model with an Ethos-U65 in place of the U55: the same
# subsystem, memory map, timing adapters and parameter names; another binary and NPU.
_CORSTONE_300_U65 = replace(
    _CORSTONE_300_U55,
    npu_family=NpuFamily.ETHOS_U65,
    ethos_arch="u65",
    ethos_compile_define="ETHOSU65",
    fvp_model_relpath="models/Linux64_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U65",
)

_CORSTONE_320_U85 = FvpTargetProfile(
    name=FvpSubsystem.CORSTONE_320,
    npu_family=NpuFamily.ETHOS_U85,
    ethos_arch="u85",
    target_cpu="cortex-m85",
    cmsis_device="ARMCM85.h",
    arm_cpu="ARMCM85",
    cpu_compile_define="CPU_CORTEX_M85",
    ethos_compile_define="ETHOSU85",
    platform_target="corstone-320",
    fvp_dependency="fvp_corstone_320",
    fvp_dirname="FVP_Corstone_SSE-320",
    fvp_model_relpath="models/Linux64_GCC-9.3/FVP_Corstone_SSE-320",
    fvp_model_namespace="mps4_board",
    fvp_npu_parameter_namespace="mps4_board.subsystem.ethosu",
    fvp_extra_visualisation_parameter="vis_hdlcd.disable_visualisation=1",
    ethos_u_base_address="0x50004000",
    ethos_u_irqn=16,
    uart0_base="0x48203000",
    timing_adapter_bases=("0x50006000", "0x50006800"),
    fvp_extra_args=(),
    use_simlimit=True,
    supported_memory_modes=frozenset(VelaMemoryMode),
    supported_runtime_engines=frozenset({"tflite_micro", "helia_rt", "helia_aot"}),
    host_transport="uart_file",
    fvp_sram_hash_parameter="mps4_board.subsystem.sram_hash0",
    # Listed by the ListTraceSources plugin, FVP 11.27 (2026-10-09); the Corstone-300 models have no NPU loggers.
    fvp_bus_loggers=("ethosu_m0_logger", "ethosu_m1_logger"),
)

FVP_TARGET_PROFILES: tuple[FvpTargetProfile, ...] = (_CORSTONE_300_U55, _CORSTONE_300_U65, _CORSTONE_320_U85)
"""Every FVP model helx runs. A subsystem's first entry is the one its name alone selects."""


def resolve_fvp_target_profile(subsystem: str | None = None, npu: str | None = None) -> FvpTargetProfile:
    """The FVP model that simulates *npu*, on *subsystem* when one is named.

    With a subsystem and no NPU, that subsystem's first model (the U55 on Corstone-300); with
    neither, Corstone-320. Raises ``ValueError`` for an unknown name, or for a subsystem that
    does not simulate *npu*.
    """
    wanted = _subsystem(subsystem) if subsystem is not None else None
    if npu is None:
        wanted = wanted or DEFAULT_FVP_SUBSYSTEM
        return next(profile for profile in FVP_TARGET_PROFILES if profile.name == wanted)
    try:
        family = NpuFamily(npu)
    except ValueError as exc:
        available = ", ".join(member.value for member in NpuFamily)
        raise ValueError(f"No FVP simulates NPU '{npu}'. Known NPUs: {available}") from exc
    for profile in FVP_TARGET_PROFILES:
        if profile.npu_family == family and wanted in (None, profile.name):
            return profile
    simulated = " or ".join(profile.npu_family.value for profile in FVP_TARGET_PROFILES if profile.name == wanted)
    raise ValueError(f"FVP subsystem '{wanted}' simulates {simulated}, but the platform uses '{family}'")


def _subsystem(name: str) -> FvpSubsystem:
    try:
        return FvpSubsystem(name)
    except ValueError as exc:
        available = ", ".join(member.value for member in FvpSubsystem)
        raise ValueError(f"Unknown FVP subsystem '{name}'. Available: {available}") from exc


def validate_fvp_memory_mode(profile: FvpTargetProfile, memory_mode: str) -> None:
    """Reject memory placements that are not physically validated on a target."""
    if memory_mode not in profile.supported_memory_modes:
        available = ", ".join(sorted(mode.value for mode in profile.supported_memory_modes))
        raise ValueError(
            f"FVP subsystem '{profile.name}' does not support memory mode '{memory_mode}'. Validated modes: {available}"
        )


def validate_fvp_runtime_engine(profile: FvpTargetProfile, runtime_engine: str) -> None:
    """Reject runtime engines without a validated target/linker contract."""
    if runtime_engine not in profile.supported_runtime_engines:
        available = ", ".join(sorted(profile.supported_runtime_engines))
        raise ValueError(
            f"FVP subsystem '{profile.name}' does not support runtime engine '{runtime_engine}'. "
            f"Validated engines: {available}"
        )


def required_pmu_fields(profile: FvpTargetProfile, pmu_profile: str | None) -> tuple[str, ...]:
    """Host-report fields the selected PMU program must emit; see :mod:`helia_npu.pmu`."""
    from .pmu import program_for

    return program_for(profile.npu_family, pmu_profile).required_fields

from __future__ import annotations

import pytest

from helia_npu.fvp import (
    required_pmu_fields,
    resolve_fvp_target_profile,
    validate_fvp_memory_mode,
    validate_fvp_runtime_engine,
)


def test_default_profile_is_corstone_320() -> None:
    profile = resolve_fvp_target_profile(None)

    assert profile.name == "corstone_320"
    assert profile.target_cpu == "cortex-m85"
    assert profile.npu_family == "ethos-u85"


def test_corstone_300_profile_targets_m55_and_u55() -> None:
    profile = resolve_fvp_target_profile("corstone-300")

    assert profile.name == "corstone_300"
    assert profile.target_cpu == "cortex-m55"
    assert profile.npu_family == "ethos-u55"
    assert profile.ethos_arch == "u55"


def test_profile_rejects_mismatched_npu_family() -> None:
    with pytest.raises(
        ValueError, match=r"'corstone_300' simulates ethos-u55 or ethos-u65, but the platform uses 'ethos-u85'"
    ):
        resolve_fvp_target_profile("corstone_300", "ethos-u85")
    with pytest.raises(ValueError, match="'corstone_320' simulates ethos-u85, but the platform uses 'ethos-u55'"):
        resolve_fvp_target_profile("corstone_320", "ethos-u55")


def test_the_npu_picks_the_fvp_model() -> None:
    """No subsystem named: each NPU runs on the one FVP model that simulates it."""
    u55 = resolve_fvp_target_profile(npu="ethos-u55")
    u65 = resolve_fvp_target_profile(npu="Ethos_U65")
    u85 = resolve_fvp_target_profile(npu="ethos-u85")

    assert (u55.name, u55.fvp_model_relpath.rsplit("/", 1)[1]) == ("corstone_300", "FVP_Corstone_SSE-300_Ethos-U55")
    assert (u65.name, u65.fvp_model_relpath.rsplit("/", 1)[1]) == ("corstone_300", "FVP_Corstone_SSE-300_Ethos-U65")
    assert (u85.name, u85.fvp_model_relpath.rsplit("/", 1)[1]) == ("corstone_320", "FVP_Corstone_SSE-320")
    assert resolve_fvp_target_profile("corstone-300", "ethos-u65") is u65


def test_u65_differs_from_u55_only_in_npu_and_model() -> None:
    """The Corstone-300 U65 model shares the U55 model's subsystem: addresses, CPU, dependency, limits."""
    u55 = resolve_fvp_target_profile(npu="ethos-u55")
    u65 = resolve_fvp_target_profile(npu="ethos-u65")
    differing = {name for name in u55.__dataclass_fields__ if getattr(u55, name) != getattr(u65, name)}

    assert differing == {"npu_family", "ethos_arch", "ethos_compile_define", "fvp_model_relpath"}
    assert (u65.ethos_arch, u65.ethos_compile_define) == ("u65", "ETHOSU65")


def test_profile_rejects_unknown_npu() -> None:
    with pytest.raises(ValueError, match="No FVP simulates NPU 'ethos-u99'"):
        resolve_fvp_target_profile(npu="ethos-u99")


def test_profile_rejects_unknown_subsystem() -> None:
    with pytest.raises(ValueError, match="Unknown FVP subsystem"):
        resolve_fvp_target_profile("corstone_999")


def test_corstone_300_accepts_only_validated_shared_sram() -> None:
    profile = resolve_fvp_target_profile("corstone_300")

    validate_fvp_memory_mode(profile, "Shared_Sram")
    with pytest.raises(ValueError, match="does not support memory mode 'Sram_Only'"):
        validate_fvp_memory_mode(profile, "Sram_Only")
    with pytest.raises(ValueError, match="does not support memory mode 'Dedicated_Sram'"):
        validate_fvp_memory_mode(profile, "Dedicated_Sram")


def test_corstone_320_accepts_standard_memory_modes() -> None:
    profile = resolve_fvp_target_profile("corstone_320")

    for memory_mode in ("Sram_Only", "Shared_Sram", "Dedicated_Sram"):
        validate_fvp_memory_mode(profile, memory_mode)


def test_corstone_300_accepts_all_runtime_engines() -> None:
    profile = resolve_fvp_target_profile("corstone_300")

    assert profile.host_transport == "uart_file"
    for runtime_engine in ("tflite_micro", "helia_rt", "helia_aot"):
        validate_fvp_runtime_engine(profile, runtime_engine)


def test_required_pmu_fields_follow_u55_profile() -> None:
    profile = resolve_fvp_target_profile("corstone_300")

    assert required_pmu_fields(profile, "default") == (
        "npu_active_cycles",
        "npu_idle_cycles",
        "npu_total_cycles",
        "npu_sram_rd_data_beats",
        "npu_sram_wr_data_beats",
        "npu_ext_rd_data_beats",
    )
    assert "npu_ext_rd_tran_req_stalled" in required_pmu_fields(profile, "ext_stall")
    u85 = resolve_fvp_target_profile("corstone_320")
    assert "npu_mac_active_cycles" in required_pmu_fields(u85, "dv_align")
    assert "npu_ext_wr_data_beats" in required_pmu_fields(u85, "dv_align")

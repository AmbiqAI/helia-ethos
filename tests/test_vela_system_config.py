"""The Vela .ini written for a SoC and mode."""

from __future__ import annotations

from pathlib import Path

from helia_npu.soc import MemoryPlacement
from helia_npu.soc_schema import load_soc
from helia_npu.vela import system_config_name_for, write_system_config


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_section_name_is_soc_id_and_mode(at110_soc_file: Path) -> None:
    assert system_config_name_for(load_soc(at110_soc_file), "ULP") == "atomiq110_ULP"


def test_ulp_system_config(at110_soc_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "vela.ini"
    name = write_system_config(load_soc(at110_soc_file), "ULP", out)
    lines = _lines(out)
    assert name == "atomiq110_ULP"
    assert "[System_Config.atomiq110_ULP]" in lines
    assert "core_clock=1e+08" in lines
    assert "axi0_port=Sram" in lines and "axi1_port=Dram" in lines
    for expected in (
        "Sram_clock_scale=1.0",
        "Sram_ports_used=2",
        "Sram_burst_length=256",
        "Sram_read_latency=7",
        "Sram_write_latency=7",
        "Sram_max_reads=8",
        "Sram_max_writes=8",
        "Dram_clock_scale=1.0",
        "Dram_ports_used=1",
        "Dram_read_latency=9",
        "Dram_write_latency=9",
        "Dram_max_reads=12",
    ):
        assert expected in lines, expected
    for section in ("[Memory_Mode.Sram_Only]", "[Memory_Mode.Shared_Sram]", "[Memory_Mode.Dedicated_Sram]"):
        assert section in lines


def _single_port(soc):
    """The test SoC with one SRAM port declared; memory overrides must be complete definitions."""
    from helia_npu import derive_soc
    from helia_npu.soc_schema import SocOverridesModel

    ssram = soc.memories["ssram"].params
    single = {
        "axi_port": 0,
        "axi_clock_hz": ssram.axi_clock_hz,
        "read_latency_ns": ssram.read_latency_ns,
        "write_latency_ns": ssram.write_latency_ns,
        "max_outstanding": ssram.max_outstanding,
        "burst_length": ssram.burst_length,
        "ports_used": 1,
    }
    return derive_soc(soc, SocOverridesModel.model_validate({"id": f"{soc.id}-1p", "memories": {"ssram": single}}))


def test_hp_system_config_presents_port1_no_slower_than_port0(at110_soc_file: Path, tmp_path: Path) -> None:
    """At HP the NPU (500 MHz) outruns a single SRAM port (250 MHz); Vela must stream weights from port 1, not cache them."""
    out = tmp_path / "vela.ini"
    write_system_config(_single_port(load_soc(at110_soc_file)), "HP", out)
    lines = _lines(out)
    assert "core_clock=5e+08" in lines
    assert "Sram_read_latency=32" in lines and "Sram_write_latency=32" in lines
    assert "Dram_read_latency=32" in lines and "Dram_write_latency=32" in lines  # physical 44, clamped
    assert "Dram_max_reads=12" in lines  # only latency is clamped
    assert "Sram_clock_scale=0.5" in lines and "Dram_clock_scale=0.5" in lines
    assert any(
        line.startswith("; Port 1 (hblram) latency is presented as no worse than port 0 (ssram)") for line in lines
    )


def test_a_rate_limited_port1_is_not_clamped(tmp_path: Path) -> None:
    """A port-1 memory no deeper than port 0 cannot hide latency; streaming from it is not known to help."""
    from helia_npu import derive_soc
    from helia_npu.soc_schema import SocOverridesModel

    rram = derive_soc(
        load_soc("atomiq110"),
        SocOverridesModel.model_validate(
            {
                "id": "x-rram",
                "memories": {
                    "hblram": None,
                    "psram": None,
                    "rram": {
                        "axi_port": 1,
                        "axi_clock_hz": 250e6,
                        "read_latency_ns": 240,
                        "write_latency_ns": 240,
                        "max_outstanding": 2,
                        "burst_length": 256,
                        "ports_used": 1,
                    },
                },
            }
        ),
    )
    out = tmp_path / "vela.ini"
    write_system_config(rram, "HP", out)
    lines = _lines(out)
    assert "Dram_read_latency=120" in lines and "Sram_read_latency=32" in lines
    assert not any(line.startswith("; Port 1") for line in lines)


def test_two_port_sram_is_not_clamped(at110_soc_file: Path, tmp_path: Path) -> None:
    """With two port-0 ports declared Vela caches regardless of latency, so the clamp would only mislabel the ini."""
    out = tmp_path / "vela.ini"
    write_system_config(load_soc(at110_soc_file), "HP", out)  # conftest SoC: ports_used 2
    lines = _lines(out)
    assert "Sram_ports_used=2" in lines and "Dram_read_latency=44" in lines
    assert not any(line.startswith("; Port 1") for line in lines)


def test_ulp_system_config_keeps_physical_port1_latency(at110_soc_file: Path, tmp_path: Path) -> None:
    """At ULP the NPU is slower than the fabric: caching weights in SRAM is free and saves port-1 energy."""
    out = tmp_path / "vela.ini"
    write_system_config(load_soc(at110_soc_file), "ULP", out)
    lines = _lines(out)
    assert "Dram_read_latency=9" in lines and "Sram_read_latency=7" in lines
    assert not any(line.startswith("; Port 1") for line in lines)


def test_an_explicit_port1_override_is_not_clamped(at110_soc_file: Path, tmp_path: Path) -> None:
    """A sweep that sets port-1 timing on purpose must reach Vela as written, even at HP."""
    from helia_npu.soc import MemoryTimingOverride, TimingOverrides

    out = tmp_path / "vela.ini"
    overrides = TimingOverrides(
        memories={"hblram": MemoryTimingOverride(read_latency_cycles=96, write_latency_cycles=96)}
    )
    write_system_config(_single_port(load_soc(at110_soc_file)), "HP", out, timing_overrides=overrides)
    lines = _lines(out)
    assert "Dram_read_latency=96" in lines and "Dram_write_latency=96" in lines
    assert not any(line.startswith("; Port 1") for line in lines)


def test_packaged_soc_hp_section_clamps_port1_to_port0_and_is_single_port(tmp_path: Path) -> None:
    """v10: SSRAM at 32 cycles is faster than HBLRAM at 44, port 0 is bandwidth-bound (0.5) and single-port, so the
    HP section presents port 1 at port 0's latency. That is the ini the FPGA correlation streams were compiled with
    (Sram 32 / 8, Dram 32 / 12, one port), and helx's kws stream is byte-identical to the FPGA's."""
    out = tmp_path / "vela.ini"
    write_system_config(load_soc("atomiq110"), "HP", out)
    lines = _lines(out)
    assert "Sram_ports_used=1" in lines and "Sram_max_reads=8" in lines and "Sram_burst_length=128" in lines
    assert "Sram_read_latency=32" in lines and "Dram_read_latency=32" in lines and "Dram_max_reads=12" in lines
    assert any(line.startswith("; Port 1 (hblram) latency is presented as no worse than port 0") for line in lines)


def test_custom_placement_adds_a_memory_mode_section(at110_soc_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "vela.ini"
    odd = MemoryPlacement(const="hblram", arena="ssram", cache="hblram")
    write_system_config(load_soc(at110_soc_file), "ULP", out, extra_placements=[odd])
    text = out.read_text(encoding="utf-8")
    section = text[text.index("[Memory_Mode.const_hblram__arena_ssram__cache_hblram]") :]
    assert "const_mem_area=Axi1" in section and "arena_mem_area=Axi0" in section and "cache_mem_area=Axi1" in section


def test_header_names_the_soc_and_its_origin(at110_soc_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "vela.ini"
    write_system_config(load_soc(at110_soc_file), "ULP", out)
    assert _lines(out)[1] == "; SoC: AT110 (atomiq110, golden)  Mode: ULP"


def test_a_placement_given_only_as_an_extra_decides_the_port_occupant(tmp_path: Path) -> None:
    """compile_for_target's documented use passes the run's placement in extra_placements; its memory must win."""
    soc = load_soc("atomiq110")
    psram = MemoryPlacement(const="psram", arena="ssram", cache="ssram")
    out = tmp_path / "vela.ini"
    write_system_config(soc, "ULP", out, extra_placements=[psram])
    text = out.read_text(encoding="utf-8")
    assert "carries psram" in text and "Dram_read_latency=16" in text


def test_extras_that_disagree_about_a_port_need_an_explicit_placement(tmp_path: Path) -> None:
    import pytest

    from helia_npu.soc import SocError

    soc = load_soc("atomiq110")
    extras = [MemoryPlacement(const=m, arena="ssram", cache="ssram") for m in ("psram", "rram")]
    with pytest.raises(SocError, match="pass placement="):
        write_system_config(soc, "ULP", tmp_path / "vela.ini", extra_placements=extras)
    write_system_config(soc, "ULP", tmp_path / "vela.ini", extra_placements=extras, placement=extras[1])
    assert "carries rram" in (tmp_path / "vela.ini").read_text(encoding="utf-8")


def test_a_derived_port1_override_keeps_the_clamp(at110_soc_file: Path, tmp_path: Path) -> None:
    """An override a tool derived (deliberate=False), such as emulated application traffic, is no sweep of port 1."""
    from helia_npu.soc import MemoryTimingOverride, TimingOverrides

    out = tmp_path / "vela.ini"
    overrides = TimingOverrides(memories={"hblram": MemoryTimingOverride(bandwidth=0.375, deliberate=False)})
    write_system_config(_single_port(load_soc(at110_soc_file)), "HP", out, timing_overrides=overrides)
    lines = _lines(out)
    assert "Dram_read_latency=32" in lines and "Dram_write_latency=32" in lines  # physical 44, still clamped
    assert any(line.startswith("; Port 1") for line in lines)

"""docs/port_mapping.md is checked against the code it describes, so the page cannot drift."""

from __future__ import annotations

import re
from pathlib import Path

from helia_ethos.fvp import FvpSubsystem, NpuFamily, resolve_fvp_target_profile
from helia_ethos.pmu import list_programs, program_for
from helia_ethos.soc import AXI_PATTERN_TO_VELA_MODE, VELA_AXI_NAME, VELA_PORT_TYPE
from helia_ethos.soc_schema import load_soc

DOC = Path(__file__).resolve().parents[1] / "docs" / "port_mapping.md"


def _tables() -> list[list[list[str]]]:
    """Every markdown table in the page as rows of cells, header first, separator dropped."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in DOC.read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r"-+", c) for c in cells):
                continue
            current.append(cells)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def _table_with_header(first_cell: str) -> list[list[str]]:
    for table in _tables():
        if table[0][0] == first_cell:
            return table
    raise AssertionError(f"no table whose first header cell is {first_cell!r}")


def test_port_class_rows_match_soc_and_fvp_constants() -> None:
    header, *rows = _table_with_header("NPU port class (Arm)")
    col = {name: i for i, name in enumerate(header)}
    assert len(rows) == 2
    for row in rows:
        port = int(row[col["SoC `axi_port`"]])
        assert row[col["Vela memory name"]] == f"`{VELA_PORT_TYPE[port]}`"
        assert row[col["Vela AXI name"]] == f"`{VELA_AXI_NAME[port]}`"
        prefix = VELA_PORT_TYPE[port]
        keys = re.findall(r"`([A-Za-z_]+)`", row[col["Vela ini keys"]])
        assert keys == [
            f"{prefix}_clock_scale",
            f"{prefix}_ports_used",
            f"{prefix}_burst_length",
            f"{prefix}_read_latency",
            f"{prefix}_write_latency",
            f"{prefix}_max_reads",
            f"{prefix}_max_writes",
        ]
        assert row[col["PMU host field prefix"]] == ("`npu_sram_`" if port == 0 else "`npu_ext_`")
        for subsystem, column in (
            (FvpSubsystem.CORSTONE_320, "FVP timing adapter (Corstone-320)"),
            (FvpSubsystem.CORSTONE_300, "FVP timing adapter (Corstone-300)"),
        ):
            base = resolve_fvp_target_profile(subsystem).timing_adapter_bases[port]
            assert row[col[column]] == f"TA{port} at `{base}`"


def test_port_class_rows_match_the_packaged_soc() -> None:
    header, *rows = _table_with_header("NPU port class (Arm)")
    col = {name: i for i, name in enumerate(header)}
    soc = load_soc("atomiq110")
    assert [m.name for m in soc.memories_on_axi(0)] == ["ssram"]
    assert [m.name for m in soc.memories_on_axi(1)] == ["hblram", "psram", "rram"]
    physical = {int(r[col["SoC `axi_port`"]]): r[col["AT110 physical memory"]] for r in rows}
    assert physical[0].startswith("SSRAM")
    assert physical[1].startswith("HBLRAM")


def test_memory_mode_rows_match_the_port_patterns() -> None:
    header, *rows = _table_with_header("Vela memory mode")
    assert header == ["Vela memory mode", "const", "arena", "cache"]
    seen = {}
    for row in rows:
        mode = row[0].strip("`")
        pattern = tuple(int(cell.removeprefix("port ")) for cell in row[1:])
        seen[pattern] = mode
    assert seen == AXI_PATTERN_TO_VELA_MODE


def test_pmu_rows_match_the_programs() -> None:
    _header, *rows = _table_with_header("NPU family")
    by_family = {row[0]: row for row in rows}
    u85_fields = {
        slot.field
        for name in list_programs(NpuFamily.ETHOS_U85)
        for slot in program_for(NpuFamily.ETHOS_U85, name).slots
    }
    u55_fields = {
        slot.field
        for name in list_programs(NpuFamily.ETHOS_U55)
        for slot in program_for(NpuFamily.ETHOS_U55, name).slots
    }
    u85_events = {
        slot.event
        for name in list_programs(NpuFamily.ETHOS_U85)
        for slot in program_for(NpuFamily.ETHOS_U85, name).slots
    }
    u55_events = {
        slot.event
        for name in list_programs(NpuFamily.ETHOS_U55)
        for slot in program_for(NpuFamily.ETHOS_U55, name).slots
    }
    # U55/U65 programs name ports AXI0/AXI1 and land in the sram/ext host fields.
    assert any(e.startswith("ETHOSU_PMU_AXI0_") for e in u55_events)
    assert any(e.startswith("ETHOSU_PMU_AXI1_") for e in u55_events)
    assert not any(e.startswith("ETHOSU_PMU_SRAM") or e.startswith("ETHOSU_PMU_EXT") for e in u55_events)
    assert {f.rsplit("_", 3)[0] for f in u55_fields if f.startswith("npu_")} >= {"npu_sram", "npu_ext"}
    # U85 programs name them SRAM/EXT with SRAM0/EXT0/EXT1 sub-counters.
    for prefix in ("ETHOSU_PMU_SRAM_", "ETHOSU_PMU_SRAM0_", "ETHOSU_PMU_EXT_", "ETHOSU_PMU_EXT0_", "ETHOSU_PMU_EXT1_"):
        assert any(e.startswith(prefix) for e in u85_events), prefix
    assert not any(e.startswith("ETHOSU_PMU_AXI") for e in u85_events)
    for field_prefix in ("npu_sram_", "npu_sram0_", "npu_ext_", "npu_ext0_", "npu_ext1_"):
        assert any(f.startswith(field_prefix) for f in u85_fields), field_prefix
        assert f"`{field_prefix}*`" in by_family["Ethos-U85"][3]
    assert "`ETHOSU_PMU_AXI0_*`" in by_family["Ethos-U55, U65"][1]
    assert "`ETHOSU_PMU_AXI1_*`" in by_family["Ethos-U55, U65"][2]
    assert "`ETHOSU_PMU_SRAM_*`" in by_family["Ethos-U85"][1]
    assert "`ETHOSU_PMU_EXT_*`" in by_family["Ethos-U85"][2]
    assert "`SRAM0_*`" in by_family["Ethos-U85"][1] and "SRAM1" not in by_family["Ethos-U85"][1]
    assert "`EXT0_*`" in by_family["Ethos-U85"][2] and "`EXT1_*`" in by_family["Ethos-U85"][2]
    # The prose about SRAM1 must match what helx programs: SRAM0 read, SRAM1 never.
    assert not any(e.startswith("ETHOSU_PMU_SRAM1_") for e in u85_events)
    text = " ".join(DOC.read_text(encoding="utf-8").split())
    assert "helx's programs read `SRAM0_*` but not `SRAM1_*`" in text


def test_modes_paragraph_matches_the_packaged_clocks() -> None:
    soc = load_soc("atomiq110")
    text = DOC.read_text(encoding="utf-8")
    assert set(soc.modes) == {"ULP", "HP"}
    for mode, mhz in (("ULP", 100), ("HP", 500)):
        assert soc.modes[mode].npu_clock_hz == mhz * 1e6
    assert "100 MHz and 500 MHz" in text
    assert "`LP` never appears as a platform mode" in text
    fabric = {name: m.params.axi_clock_hz for name, m in soc.memories.items()}
    assert fabric == {"ssram": 250e6, "hblram": 250e6, "rram": 250e6, "psram": 62.5e6}
    assert "fixed at 250 MHz" in text and "PSRAM's XSPI bridge at 62.5 MHz" in text

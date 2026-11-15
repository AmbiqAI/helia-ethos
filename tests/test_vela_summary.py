"""Reading Vela's summary CSV."""

from __future__ import annotations

from pathlib import Path

import pytest

from helia_ethos.vela import (
    REPORTED_COLUMNS,
    VelaColumn,
    VelaSummary,
    VelaSummaryError,
    parse_vela_summary,
)

# A realistic row: units are Vela's, so the *_memory_used and arena_cache_size
# columns are KiB while sram_total_bytes is bytes.
_HEADERS = "experiment,network,system_config,memory_mode,arena_cache_size,nn_macs,inference_time,cycles_total,sram_memory_used,sram_total_bytes,arena_size,feature_map_storage_area"
_VALUES = "demo,demo_model,atomiq110_ULP,Shared_Sram,1024.0,2649600.00,0.00065,64300.00,512.00,1638400.00,589824,Sram"


def _write(tmp_path: Path, headers: str = _HEADERS, values: str = _VALUES) -> Path:
    path = tmp_path / "summary.csv"
    path.write_text(f"{headers}\n{values}\n", encoding="utf-8")
    return path


class TestParsing:
    def test_windows_line_endings_are_read(self, tmp_path: Path) -> None:
        """Vela on Windows writes \\r\\r\\n, which reads back with a blank row between header and values."""
        path = tmp_path / "summary.csv"
        path.write_bytes(f"{_HEADERS}\r\r\n{_VALUES}\r\r\n".encode())
        assert parse_vela_summary(path).nn_macs == 2649600.0

    def test_named_accessors(self, tmp_path: Path) -> None:
        summary = parse_vela_summary(_write(tmp_path))
        assert summary.nn_macs == 2649600.0
        assert summary.inference_time_s == 0.00065
        assert summary.cycles_total == 64300.0
        assert summary.sram_total_bytes == 1638400.0
        assert summary.arena_size_bytes == 589824.0

    def test_raw_keeps_every_column_verbatim(self, tmp_path: Path) -> None:
        """Vela's own strings survive: downstream CSVs republish them unchanged."""
        summary = parse_vela_summary(_write(tmp_path))
        assert summary.raw["nn_macs"] == "2649600.00"
        assert summary.raw["experiment"] == "demo"

    def test_feature_map_storage_area_is_normalised(self, tmp_path: Path) -> None:
        assert parse_vela_summary(_write(tmp_path)).feature_map_storage_area == "sram"

    def test_memory_used_kib_by_area(self, tmp_path: Path) -> None:
        summary = parse_vela_summary(_write(tmp_path))
        assert summary.memory_used_kib("sram") == 512.0
        assert summary.memory_used_kib("dram") is None  # column absent
        assert summary.memory_used_kib("nonsense") is None  # not a known column

    def test_absent_column_is_none_not_an_error(self, tmp_path: Path) -> None:
        summary = parse_vela_summary(_write(tmp_path))
        assert summary.number(VelaColumn.DRAM_TOTAL_BYTES) is None
        assert summary.text(VelaColumn.DRAM_TOTAL_BYTES) is None

    def test_non_numeric_value_is_none(self, tmp_path: Path) -> None:
        summary = parse_vela_summary(_write(tmp_path, "nn_macs", "n/a"))
        assert summary.number(VelaColumn.NN_MACS) is None
        assert summary.text(VelaColumn.NN_MACS) == "n/a"

    def test_empty_value_is_none(self, tmp_path: Path) -> None:
        """Vela writes an empty cell for a column that does not apply."""
        summary = parse_vela_summary(_write(tmp_path, "network,nn_macs", "demo,"))
        assert summary.text(VelaColumn.NN_MACS) is None
        assert summary.number(VelaColumn.NN_MACS) is None


class TestReported:
    def test_reported_is_the_declared_subset(self, tmp_path: Path) -> None:
        reported = parse_vela_summary(_write(tmp_path)).reported
        assert set(reported) <= {column.value for column in REPORTED_COLUMNS}
        assert reported["nn_macs"] == "2649600.00"
        # Present in raw, deliberately not republished.
        assert "experiment" not in reported

    def test_reported_omits_absent_columns(self, tmp_path: Path) -> None:
        assert "dram_bandwidth" not in parse_vela_summary(_write(tmp_path)).reported


class TestEmptySummary:
    def test_empty_summary_is_falsey(self) -> None:
        """Vela was bypassed: callers check truthiness rather than a sentinel."""
        assert not VelaSummary()
        assert VelaSummary().reported == {}
        assert VelaSummary().nn_macs is None

    def test_parsed_summary_is_truthy(self, tmp_path: Path) -> None:
        assert parse_vela_summary(_write(tmp_path))


class TestErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(VelaSummaryError, match="Could not read"):
            parse_vela_summary(tmp_path / "absent.csv")

    def test_header_only(self, tmp_path: Path) -> None:
        path = tmp_path / "summary.csv"
        path.write_text("nn_macs\n", encoding="utf-8")
        with pytest.raises(VelaSummaryError, match="expected header and data rows"):
            parse_vela_summary(path)

    def test_column_count_mismatch(self, tmp_path: Path) -> None:
        with pytest.raises(VelaSummaryError, match="did not match value count"):
            parse_vela_summary(_write(tmp_path, "a,b,c", "1,2"))

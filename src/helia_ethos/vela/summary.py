"""The summary CSV Vela writes beside a compiled model.

Vela emits one header row and one data row. Which columns appear depends on
the Vela version and the memory mode, so every column is optional and the
whole row is kept in :attr:`VelaSummary.raw`; the named accessors cover the
columns the helia toolchain actually reads.

Units are Vela's, not ours, and the names here say so: ``*_memory_used`` and
``arena_cache_size`` are **KiB** even though the compiler is invoked with
``--arena-cache-size`` in bytes.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType


class VelaSummaryError(ValueError):
    """A Vela summary CSV could not be read."""


class VelaColumn(StrEnum):
    """Summary columns the toolchain depends on.

    Declared once so no consumer has to spell a column name, and so the set
    that must survive a Vela upgrade is visible in one place.
    """

    SYSTEM_CONFIG = "system_config"
    ACCELERATOR_CONFIGURATION = "accelerator_configuration"
    CORE_CLOCK = "core_clock"
    MEMORY_MODE = "memory_mode"
    NN_MACS = "nn_macs"
    INFERENCE_TIME = "inference_time"
    CYCLES_TOTAL = "cycles_total"
    CYCLES_NPU = "cycles_npu"
    CYCLES_SRAM_ACCESS = "cycles_sram_access"
    CYCLES_DRAM_ACCESS = "cycles_dram_access"
    SRAM_TOTAL_BYTES = "sram_total_bytes"
    DRAM_TOTAL_BYTES = "dram_total_bytes"
    SRAM_BANDWIDTH = "sram_bandwidth"
    DRAM_BANDWIDTH = "dram_bandwidth"
    SRAM_MEMORY_USED = "sram_memory_used"
    DRAM_MEMORY_USED = "dram_memory_used"
    ON_CHIP_FLASH_MEMORY_USED = "on_chip_flash_memory_used"
    OFF_CHIP_FLASH_MEMORY_USED = "off_chip_flash_memory_used"
    ARENA_CACHE_SIZE = "arena_cache_size"
    ARENA_SIZE = "arena_size"
    FEATURE_MAP_STORAGE_AREA = "feature_map_storage_area"
    TOTAL_ORIGINAL_WEIGHTS = "total_original_weights"
    TOTAL_NPU_ENCODED_WEIGHTS = "total_npu_encoded_weights"


#: Columns surfaced verbatim in downstream reports. A subset of VelaColumn:
#: the rest are read but not republished.
REPORTED_COLUMNS: tuple[VelaColumn, ...] = (
    VelaColumn.ARENA_CACHE_SIZE,
    VelaColumn.SRAM_BANDWIDTH,
    VelaColumn.DRAM_BANDWIDTH,
    VelaColumn.SRAM_MEMORY_USED,
    VelaColumn.DRAM_MEMORY_USED,
    VelaColumn.TOTAL_ORIGINAL_WEIGHTS,
    VelaColumn.TOTAL_NPU_ENCODED_WEIGHTS,
    VelaColumn.NN_MACS,
    VelaColumn.INFERENCE_TIME,
    VelaColumn.CYCLES_NPU,
    VelaColumn.CYCLES_SRAM_ACCESS,
    VelaColumn.CYCLES_DRAM_ACCESS,
    VelaColumn.CYCLES_TOTAL,
    VelaColumn.SRAM_TOTAL_BYTES,
    VelaColumn.DRAM_TOTAL_BYTES,
)


@dataclass(frozen=True)
class VelaSummary:
    """One parsed summary row.

    An empty summary means Vela was bypassed; it is falsey, so callers can
    write ``if summary:`` rather than checking a sentinel.
    """

    raw: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __bool__(self) -> bool:
        return bool(self.raw)

    def text(self, column: VelaColumn) -> str | None:
        value = self.raw.get(column.value)
        return value if value not in (None, "") else None

    def number(self, column: VelaColumn) -> float | None:
        """Column as a float, or None when absent or not numeric."""
        value = self.text(column)
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @property
    def reported(self) -> dict[str, str]:
        """The REPORTED_COLUMNS subset, values exactly as Vela wrote them."""
        return {column.value: self.raw[column.value] for column in REPORTED_COLUMNS if column.value in self.raw}

    # Named accessors for the columns consumers read by name.
    @property
    def nn_macs(self) -> float | None:
        return self.number(VelaColumn.NN_MACS)

    @property
    def inference_time_s(self) -> float | None:
        return self.number(VelaColumn.INFERENCE_TIME)

    @property
    def cycles_total(self) -> float | None:
        """Vela's total: per operation, the slowest of its NPU, SRAM-access and DRAM-access estimates, summed.
        It carries Vela's memory-access model, which can be far above the hardware (helia-estimator #57)."""
        return self.number(VelaColumn.CYCLES_TOTAL)

    @property
    def cycles_npu(self) -> float | None:
        """Vela's estimate of the cycles the NPU is busy; the figure to compare with the FVP's NPU_ACTIVE."""
        return self.number(VelaColumn.CYCLES_NPU)

    @property
    def core_clock_hz(self) -> float | None:
        return self.number(VelaColumn.CORE_CLOCK)

    @property
    def sram_total_bytes(self) -> float | None:
        return self.number(VelaColumn.SRAM_TOTAL_BYTES)

    @property
    def dram_total_bytes(self) -> float | None:
        return self.number(VelaColumn.DRAM_TOTAL_BYTES)

    @property
    def arena_size_bytes(self) -> float | None:
        """Vela's explicit arena size, in bytes (unlike the KiB columns)."""
        return self.number(VelaColumn.ARENA_SIZE)

    @property
    def feature_map_storage_area(self) -> str | None:
        area = self.text(VelaColumn.FEATURE_MAP_STORAGE_AREA)
        return area.strip().lower() if area else None

    def memory_used_kib(self, area: str) -> float | None:
        """KiB used in a named storage area, e.g. ``sram`` or ``on_chip_flash``."""
        try:
            column = VelaColumn(f"{area}_memory_used")
        except ValueError:
            return None
        return self.number(column)


def parse_vela_summary(path: str | Path) -> VelaSummary:
    """Read the single data row from a Vela summary CSV."""
    summary_path = Path(path)
    try:
        # Vela on Windows writes its CSV with "\r\r\n" line endings, which read back as a blank row between
        # the header and the values; blank rows carry nothing, so they are dropped.
        rows = [row for row in csv.reader(summary_path.read_text(encoding="utf-8").splitlines()) if row]
    except OSError as exc:
        raise VelaSummaryError(f"Could not read Vela summary CSV {summary_path}: {exc}") from exc

    if len(rows) < 2:
        raise VelaSummaryError(f"Malformed Vela summary CSV {summary_path}: expected header and data rows")
    headers, values = rows[0], rows[1]
    if len(headers) != len(values):
        raise VelaSummaryError(
            f"Malformed Vela summary CSV {summary_path}: header count {len(headers)} "
            f"did not match value count {len(values)}"
        )
    return VelaSummary(raw=MappingProxyType(dict(zip(headers, values, strict=True))))

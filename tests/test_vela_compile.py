"""compile_for_target end to end: real Vela on a synthetic model."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from helia_ethos.soc_schema import load_soc
from helia_ethos.vela import VelaColumn, VelaError, compile_for_target, find_vela_binary


def _no_vela() -> bool:
    if shutil.which("vela") is not None:
        return False
    try:
        find_vela_binary()
    except VelaError:
        return True
    return False


pytestmark = pytest.mark.skipif(_no_vela(), reason="vela not installed")


def test_compile_for_target(at110_soc_file: Path, add_chain_model: Path, tmp_path: Path) -> None:
    soc = load_soc(at110_soc_file)
    out = tmp_path / "out"
    compiled = compile_for_target(
        add_chain_model, soc, "ULP", memory_mode="Shared_Sram", arena_cache_size=1 << 20, output_dir=out
    )

    assert compiled.system_config == "atomiq110_ULP"
    assert compiled.config_file == out / "vela_config.ini"
    assert compiled.tflite == out / "add_chain_vela.tflite" and compiled.tflite.is_file()
    assert compiled.summary_path.is_file()
    assert compiled.summary.text(VelaColumn.SYSTEM_CONFIG) == "atomiq110_ULP"
    assert compiled.summary.text(VelaColumn.MEMORY_MODE) == "Shared_Sram"
    assert compiled.summary.cycles_total is not None and compiled.summary.cycles_total > 0
    assert compiled.per_layer, "the --verbose-performance table should have been parsed"
    assert all("Add" in row.nng_operator or "Add" in row.original_operator for row in compiled.per_layer)

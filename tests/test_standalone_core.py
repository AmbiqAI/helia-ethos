"""The core imports and loads a SoC without the optional extras; the extras say what to install."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, check=False)


def test_core_needs_no_flatbuffer_library() -> None:
    result = _run(
        """
        import sys
        sys.modules["ai_edge_litert"] = None  # make the import fail as if the extra were absent
        import helia_ethos
        from helia_ethos.vela import compile_for_target, write_system_config, LayerRow, parse_verbose_performance
        soc = helia_ethos.load_soc("atomiq110")
        timing = helia_ethos.derive_timing(soc, "ULP")
        print(soc.id, timing.memories["ssram"].read_latency, helia_ethos.VelaMemoryMode("shared_sram").value)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "atomiq110 7 Shared_Sram"


def test_the_models_extra_is_named_when_missing() -> None:
    result = _run(
        """
        import sys
        sys.modules["ai_edge_litert"] = None
        import helia_ethos
        from pathlib import Path
        try:
            from helia_ethos.vela import read_tflite_graph
            read_tflite_graph(Path("x.tflite"))
        except ImportError as exc:
            print(exc)
        for name, call in (
            ("command_stream_bytes", lambda: helia_ethos.command_stream_bytes(Path("x.tflite"))),
            ("characterize_model", lambda: helia_ethos.characterize_model(Path("x.tflite"))),
            ("fence_model", lambda: helia_ethos.fence_model(Path("x.tflite"), [], Path("out"))),
        ):
            try:
                call()
            except ImportError as exc:
                print(name, "names the extra:", "helia-ethos[models]" in str(exc))
        print("dir:", "fence_model" in dir(helia_ethos), "star:", "fence_model" not in helia_ethos.__all__)
        """
    )
    assert result.returncode == 0, result.stderr
    assert "pip install 'helia-ethos[models]'" in result.stdout
    for name in ("command_stream_bytes", "characterize_model", "fence_model"):
        assert f"{name} names the extra: True" in result.stdout, result.stdout
    assert "dir: True star: True" in result.stdout


def test_version_is_single_sourced() -> None:
    from importlib.metadata import version

    import helia_ethos

    assert version("helia-ethos") == helia_ethos.__version__


def test_the_suite_collects_on_a_core_only_install(tmp_path) -> None:
    """With the library made unimportable for a whole pytest process, the model tests skip and the core tests pass."""
    from pathlib import Path

    (tmp_path / "sitecustomize.py").write_text("import sys\nsys.modules['ai_edge_litert'] = None\n", encoding="utf-8")
    tests = Path(__file__).resolve().parent
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(tests / "test_model.py"),
            str(tests / "test_soc.py"),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        cwd=tests.parent,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "skipped" in result.stdout and "passed" in result.stdout

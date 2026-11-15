"""The core imports and loads a SoC without the optional extras; the extras say what to install."""

from __future__ import annotations

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
        import helia_npu
        from helia_npu.vela import compile_for_target, write_system_config, LayerRow, parse_verbose_performance
        soc = helia_npu.load_soc("atomiq110")
        timing = helia_npu.derive_timing(soc, "ULP")
        print(soc.id, timing.memories["ssram"].read_latency, helia_npu.VelaMemoryMode("shared_sram").value)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "atomiq110 7 Shared_Sram"


def test_the_models_extra_is_named_when_missing() -> None:
    result = _run(
        """
        import sys
        sys.modules["ai_edge_litert"] = None
        import helia_npu
        from pathlib import Path
        try:
            from helia_npu.vela import read_tflite_graph
            read_tflite_graph(Path("x.tflite"))
        except ImportError as exc:
            print(exc)
        for name, call in (
            ("command_stream_bytes", lambda: helia_npu.command_stream_bytes(Path("x.tflite"))),
            ("characterize_model", lambda: helia_npu.characterize_model(Path("x.tflite"))),
            ("fence_model", lambda: helia_npu.fence_model(Path("x.tflite"), [], Path("out"))),
        ):
            try:
                call()
            except ImportError as exc:
                print(name, "names the extra:", "helia-npu[models]" in str(exc))
        print("dir:", "fence_model" in dir(helia_npu), "star:", "fence_model" not in helia_npu.__all__)
        """
    )
    assert result.returncode == 0, result.stderr
    assert "pip install 'helia-npu[models]'" in result.stdout
    for name in ("command_stream_bytes", "characterize_model", "fence_model"):
        assert f"{name} names the extra: True" in result.stdout, result.stdout
    assert "dir: True star: True" in result.stdout


def test_version_is_single_sourced() -> None:
    from importlib.metadata import version

    import helia_npu

    assert version("helia-npu") == helia_npu.__version__


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
        env={"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin"},
        cwd=tests.parent,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "skipped" in result.stdout and "passed" in result.stdout

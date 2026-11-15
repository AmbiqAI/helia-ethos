"""Locating Vela and building its command line."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from helia_ethos.fvp import NpuFamily
from helia_ethos.vela import VelaError, VelaInvocation, accelerator_config, find_vela_binary, invoke_vela, vela_command


class TestAcceleratorConfig:
    def test_family_and_macs(self) -> None:
        assert accelerator_config("ethos-u85", 256) == "ethos-u85-256"
        assert accelerator_config(NpuFamily.ETHOS_U55, 32) == "ethos-u55-32"

    def test_unknown_family(self) -> None:
        with pytest.raises(VelaError, match="Unsupported NPU family"):
            accelerator_config("ethos-u99", 256)

    def test_bad_mac_count(self) -> None:
        with pytest.raises(VelaError, match="positive"):
            accelerator_config("ethos-u85", 0)


class TestFindVelaBinary:
    """Every search location is injected so the machine's real Vela cannot leak in."""

    @staticmethod
    def _search(
        tmp_path: Path,
        *,
        which: Callable[[str], str | None] = lambda _name: None,
        env: Mapping[str, str] | None = None,
        executable: Path | None = None,
        extra_dirs: Iterable[Path] = (),
        windows: bool = False,
    ) -> str:
        bare = tmp_path / "bare"
        bare.mkdir(exist_ok=True)
        return find_vela_binary(
            which=which,
            env={} if env is None else env,
            executable=bare / "python" if executable is None else executable,
            script=bare / "script.py",
            extra_dirs=extra_dirs,
            windows=windows,
        )

    def test_path_wins(self, tmp_path: Path) -> None:
        on_path = tmp_path / "bin" / "vela"
        on_path.parent.mkdir()
        on_path.write_text("#!/bin/sh\n")
        assert self._search(tmp_path, which=lambda _name: str(on_path)) == str(on_path)

    def test_next_to_the_interpreter(self, tmp_path: Path) -> None:
        venv_vela = tmp_path / "venv" / "bin" / "vela"
        venv_vela.parent.mkdir(parents=True)
        venv_vela.write_text("#!/bin/sh\n")
        assert self._search(tmp_path, executable=venv_vela.parent / "python") == str(venv_vela)

    def test_virtual_env_bin_is_searched(self, tmp_path: Path) -> None:
        venv_vela = tmp_path / "venv" / "bin" / "vela"
        venv_vela.parent.mkdir(parents=True)
        venv_vela.write_text("#!/bin/sh\n")
        assert self._search(tmp_path, env={"VIRTUAL_ENV": str(tmp_path / "venv")}) == str(venv_vela)

    def test_windows_exe_next_to_the_interpreter(self, tmp_path: Path) -> None:
        """A Windows venv holds vela.exe in Scripts, which is off PATH when helx.exe is called directly (#108)."""
        venv_vela = tmp_path / "venv" / "Scripts" / "vela.exe"
        venv_vela.parent.mkdir(parents=True)
        venv_vela.write_text("")
        found = self._search(tmp_path, executable=venv_vela.parent / "python.exe", windows=True)
        assert found == str(venv_vela)

    def test_windows_virtual_env_scripts_is_searched(self, tmp_path: Path) -> None:
        venv_vela = tmp_path / "venv" / "Scripts" / "vela.exe"
        venv_vela.parent.mkdir(parents=True)
        venv_vela.write_text("")
        found = self._search(tmp_path, env={"VIRTUAL_ENV": str(tmp_path / "venv")}, windows=True)
        assert found == str(venv_vela)

    def test_falls_back_to_extra_dirs(self, tmp_path: Path) -> None:
        local = tmp_path / "tools" / "vela"
        local.parent.mkdir()
        local.write_text("#!/bin/sh\n")
        assert self._search(tmp_path, extra_dirs=[local.parent]) == str(local)

    def test_missing_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(VelaError, match="Could not locate"):
            self._search(tmp_path)


_BASE = VelaInvocation(
    accelerator_config="ethos-u85-256",
    config_file=Path("/cfg/vela_config.ini"),
    system_config="atomiq110_ULP",
    memory_mode="Shared_Sram",
    arena_cache_size=1048576,
)


def _invocation(**overrides: object) -> VelaInvocation:
    return replace(_BASE, **overrides)


class TestCommandLine:
    def test_full_command(self) -> None:
        argv = vela_command("vela", _invocation(), Path("/m/kws.tflite"), Path("/out"))
        assert argv == [
            "vela",
            "--accelerator-config=ethos-u85-256",
            "--config",
            str(Path("/cfg/vela_config.ini")),
            "--system-config",
            "atomiq110_ULP",
            "--memory-mode",
            "Shared_Sram",
            str(Path("/m/kws.tflite")),
            "--output-dir",
            str(Path("/out")),
            "--arena-cache-size",
            "1048576",
            "--enable-debug-db",
            "--verbose-performance",
        ]

    def test_optional_flags_can_be_dropped(self) -> None:
        argv = vela_command(
            "vela", _invocation(verbose_performance=False, enable_debug_db=False), Path("m.tflite"), Path("o")
        )
        assert "--verbose-performance" not in argv
        assert "--enable-debug-db" not in argv

    def test_invoke_returns_stdout_and_stderr(self) -> None:
        commands: list[list[str]] = []
        options: list[dict[str, object]] = []

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            commands.append(command)
            options.append(dict(kwargs))
            return subprocess.CompletedProcess(command, 0, stdout="table\n", stderr="warning\n")

        out = invoke_vela("vela", _invocation(), Path("m.tflite"), Path("o"), runner=fake_run)
        assert out == "table\nwarning\n"
        assert commands[0][0] == "vela"
        assert options[0] == {"check": True, "capture_output": True, "text": True}


def test_a_failing_vela_is_a_vela_error_with_its_last_lines(tmp_path: Path) -> None:
    import subprocess

    def failing_run(command, **kwargs):
        raise subprocess.CalledProcessError(2, command, output="", stderr="line one\nInvalid memory mode Bogus\n")

    with pytest.raises(VelaError, match=r"exit code 2 on m\.tflite:\nline one\nInvalid memory mode Bogus"):
        invoke_vela("vela", _invocation(), Path("m.tflite"), Path("o"), runner=failing_run)

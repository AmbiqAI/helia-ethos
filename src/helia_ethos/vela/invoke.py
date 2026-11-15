"""Running the Vela compiler.

Binary discovery, the accelerator-config string for an NPU, and the command
line for one compile. Nothing here knows about a pipeline, a platform file or
a result bundle: a caller hands over the system config it already generated
and gets Vela's stdout back.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ..fvp import NpuFamily

logger = logging.getLogger(__name__)


class VelaError(RuntimeError):
    """Vela could not be located or asked to do something it cannot."""


def accelerator_config(npu: str | NpuFamily, macs: int) -> str:
    """Vela's ``--accelerator-config`` value, e.g. ``ethos-u85-256``."""
    try:
        family = NpuFamily(npu)
    except ValueError as exc:
        supported = ", ".join(member.value for member in NpuFamily)
        raise VelaError(f"Unsupported NPU family for Vela accelerator-config: {npu}. Supported: {supported}") from exc
    if macs <= 0:
        raise VelaError(f"MAC count must be positive, got {macs}")
    return f"{family.value}-{macs}"


def find_vela_binary(
    *,
    which: Callable[[str], str | None] = shutil.which,
    env: Mapping[str, str] | None = None,
    executable: Path | None = None,
    script: Path | None = None,
    extra_dirs: Iterable[Path] = (),
    windows: bool = os.name == "nt",
) -> str:
    """Locate the ``vela`` executable.

    Search order: next to the running interpreter (so the environment's own
    Vela wins over a stale one on ``PATH``), then the virtual environment's
    scripts directory (``$VIRTUAL_ENV/bin``, or ``Scripts`` on Windows), then
    next to the running script, then ``extra_dirs``, then ``PATH``. On Windows
    each directory is searched for ``vela.exe`` before ``vela``.
    ``executable``, ``script`` and ``windows`` default to the current
    process's; tests pass their own.
    """
    environment = os.environ if env is None else env
    candidate_dirs: list[Path] = []

    def add(path: Path | None) -> None:
        if path is not None and path not in candidate_dirs:
            candidate_dirs.append(path)

    interpreter = Path(sys.executable) if executable is None else executable
    add(interpreter.parent)
    add(interpreter.resolve().parent)
    virtual_env = environment.get("VIRTUAL_ENV")
    if virtual_env:
        add(Path(virtual_env) / ("Scripts" if windows else "bin"))
    if script is None and sys.argv and sys.argv[0]:
        script = Path(sys.argv[0])
    if script is not None:
        add(script.parent)
        add(script.resolve().parent)
    for extra in extra_dirs:
        add(Path(extra))

    # A pip-installed Vela on Windows is the console-script launcher vela.exe.
    names = ("vela.exe", "vela") if windows else ("vela",)
    candidates: list[str | None] = [str(candidate_dir / name) for candidate_dir in candidate_dirs for name in names]
    candidates.append(which("vela"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise VelaError("Could not locate the Vela executable in PATH or alongside the active Python interpreter")


@dataclass(frozen=True)
class VelaInvocation:
    """One Vela compile, fully specified.

    ``config_file`` and ``system_config`` come from whoever generated the Vela
    ``.ini`` for the target; ``memory_mode`` is a ``[Memory_Mode.*]`` section
    of that file. ``arena_cache_size`` is in bytes.
    """

    accelerator_config: str
    config_file: Path
    system_config: str
    memory_mode: str
    arena_cache_size: int
    verbose_performance: bool = True
    enable_debug_db: bool = True


def vela_command(vela_bin: str, invocation: VelaInvocation, model_source: Path, output_dir: Path) -> list[str]:
    """The argv for one compile; separated from running it so it can be inspected."""
    command = [
        vela_bin,
        f"--accelerator-config={invocation.accelerator_config}",
        "--config",
        str(invocation.config_file),
        "--system-config",
        invocation.system_config,
        "--memory-mode",
        invocation.memory_mode,
        str(model_source),
        "--output-dir",
        str(output_dir),
        "--arena-cache-size",
        str(invocation.arena_cache_size),
    ]
    if invocation.enable_debug_db:
        command.append("--enable-debug-db")
    if invocation.verbose_performance:
        command.append("--verbose-performance")
    return command


def invoke_vela(
    vela_bin: str,
    invocation: VelaInvocation,
    model_source: Path,
    output_dir: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Run Vela and return its combined stdout and stderr.

    With ``verbose_performance`` the per-layer table is in that text; the
    summary CSV and the compiled ``*_vela.tflite`` land in ``output_dir``.
    """
    command = vela_command(vela_bin, invocation, model_source, output_dir)
    try:
        result = runner(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        tail = "\n".join(str(exc.stderr or exc.stdout or "").strip().splitlines()[-5:])
        raise VelaError(f"Vela failed with exit code {exc.returncode} on {model_source.name}:\n{tail}") from exc
    if result.stderr:
        logger.debug("Vela stderr:\n%s", result.stderr)
    return result.stdout + result.stderr

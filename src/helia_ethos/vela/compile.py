"""Compile a model for a SoC and operating point, in one call.

This is what every tool in the chain needs and used to reconstruct on its
own: generate the system config for the SoC, run Vela with it, and read back
what Vela produced.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..soc import MemoryPlacement, SocDefinition, TimingOverrides, VelaMemoryMode
from .invoke import VelaInvocation, accelerator_config, find_vela_binary, invoke_vela
from .per_layer import LayerRow, parse_verbose_performance
from .summary import VelaSummary, parse_vela_summary
from .system_config import write_system_config

VelaRunner = Callable[[str, VelaInvocation, Path, Path], str]


@dataclass(frozen=True)
class CompiledModel:
    """Everything a compile produced, with the paths it was produced under."""

    tflite: Path  # the model with its Ethos-U command stream
    summary: VelaSummary
    per_layer: list[LayerRow]
    system_config: str  # the section name Vela ran with
    config_file: Path  # the generated .ini, kept for reproducibility
    summary_path: Path
    stdout: str


def compile_for_target(
    model: Path,
    soc: SocDefinition,
    mode: str,
    *,
    memory_mode: VelaMemoryMode | str = VelaMemoryMode.SHARED_SRAM,
    arena_cache_size: int = 0,
    output_dir: Path,
    extra_placements: Sequence[MemoryPlacement] | None = None,
    timing_overrides: TimingOverrides | None = None,
    placement: MemoryPlacement | None = None,
    vela_bin: str | None = None,
    runner: VelaRunner = invoke_vela,
) -> CompiledModel:
    """Compile *model* for *soc* at *mode*.

    ``memory_mode`` is a standard Vela mode or the label of one of
    ``extra_placements``. ``arena_cache_size`` is in bytes. Vela's outputs and
    the generated ``.ini`` land in ``output_dir``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Absolute: Vela reads a two-segment relative path as one of its own bundled config files.
    config_file = (output_dir / "vela_config.ini").resolve()
    system_config = write_system_config(
        soc,
        mode,
        config_file,
        extra_placements=list(extra_placements) if extra_placements else None,
        timing_overrides=timing_overrides,
        placement=placement,
    )
    invocation = VelaInvocation(
        accelerator_config=accelerator_config(soc.npu, soc.macs),
        config_file=config_file,
        system_config=system_config,
        memory_mode=memory_mode,
        arena_cache_size=arena_cache_size,
    )
    stdout = runner(vela_bin or find_vela_binary(), invocation, model, output_dir)
    summary_path = output_dir / f"{model.stem}_summary_{system_config}.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Vela did not produce the expected summary: {summary_path}")
    tflite = output_dir / f"{model.stem}_vela.tflite"
    return CompiledModel(
        tflite=tflite,
        summary=parse_vela_summary(summary_path),
        per_layer=parse_verbose_performance(stdout),
        system_config=system_config,
        config_file=config_file,
        summary_path=summary_path,
        stdout=stdout,
    )

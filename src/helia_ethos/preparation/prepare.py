"""Prepare an original model by composing the existing Vela and fencing APIs."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from contextlib import suppress
from dataclasses import asdict
from datetime import date
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ..catalog import CatalogNpuFacts
from ..catalog_identity import CatalogContentIdentity
from ..soc import NpuFamily, SocError, derive_timing
from ..vela import VelaColumn, VelaError, compile_for_target, find_vela_binary, read_tflite_graph
from ..vela.invoke import VelaInvocation, invoke_vela, vela_command
from .inspection import inspect_prepared_model, sha256
from .types import (
    Accelerator,
    AcceleratorCompatibility,
    Artifact,
    ArtifactRole,
    BudgetFences,
    CompilationFailed,
    CompileEvidence,
    CompilerUnavailable,
    CustomNpuTarget,
    ExplicitFences,
    FenceEvidence,
    InvalidPreparation,
    InvalidPreparedArtifact,
    PackageVersion,
    PreparationRecord,
    PreparedModel,
    PrepareRequest,
    Segment,
    TargetEvidence,
)


def _json_default(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Cannot retain compiler setting {type(value).__name__}")


def _target(request: PrepareRequest) -> tuple[Accelerator, TargetEvidence]:
    target = request.target
    if isinstance(target, CatalogNpuFacts):
        accelerator = Accelerator(
            NpuFamily(f"{target.npu.npu.family.rsplit('-', 1)[0]}-{target.npu.npu.model}"),
            target.npu.npu.macs_per_cycle,
        )
        evidence = TargetEvidence(
            kind="catalog",
            target_id=target.target_id,
            npu_id=target.npu.id,
            implementation_id=target.implementation_id,
            catalog=CatalogContentIdentity.capture(target.identity),
        )
    elif isinstance(target, CustomNpuTarget) and target.identity.strip():
        accelerator = target.accelerator
        evidence = TargetEvidence(kind="custom", target_id=target.identity)
    else:
        raise InvalidPreparation("target must supply catalog facts or an explicit custom identity")
    settings = request.settings
    if accelerator != Accelerator(NpuFamily(settings.soc.npu), settings.soc.macs):
        raise InvalidPreparation("Compiler accelerator does not match selected NPU facts")
    if not settings.policy_id.strip():
        raise InvalidPreparation("settings.policy_id must identify the explicit compiler assumptions")
    if type(settings.arena_cache_size) is not int or settings.arena_cache_size < 0:
        raise InvalidPreparation("settings.arena_cache_size must be a non-negative byte count")
    clock = settings.soc.resolve_mode(settings.mode).npu_clock_hz
    if not math.isfinite(clock) or clock <= 0:
        raise InvalidPreparation("settings.mode must supply a positive finite NPU clock")
    timing = derive_timing(settings.soc, settings.mode, settings.timing_overrides)
    if any(not math.isfinite(m.bandwidth) or m.bandwidth <= 0 for m in timing.memories.values()):
        raise InvalidPreparation("settings.soc requires positive finite memory bandwidth")
    # Resolve all requested memory names; selection never depends on catalog order.
    settings.soc.placement_axi_ports(settings.placement)
    return accelerator, evidence


def _package_versions() -> tuple[PackageVersion, ...]:
    values = []
    for name in ("helia-ethos", "helia-platforms", "ethos-u-vela", "ai-edge-litert"):
        with suppress(PackageNotFoundError):
            values.append(PackageVersion(name=name, version=version(name)))
    return tuple(values)


def prepare_model(request: PrepareRequest) -> PreparedModel:
    """Compile an uncompiled source with explicit target/settings and optional fences.

    A fresh destination is required. Source bytes are preserved. On compiler or
    mapping failure diagnostics remain in the destination, without a success
    record. No firmware build, FVP/device execution or implicit global cache occurs.
    """
    source, root = Path(request.source).resolve(), Path(request.output_dir).resolve()
    try:
        expected, target = _target(request)
        source_info = inspect_prepared_model(source)
        if source_info.regions:
            raise InvalidPreparation(f"{source}: already compiled; use inspect_prepared_model instead")
        existing = [op for op in source_info.custom_operators if op.name == "ETHOS_U_STREAM_SPLIT"]
        if existing:
            raise InvalidPreparation(
                f"{source}: already contains ETHOS_U_STREAM_SPLIT; importing fences is unsupported"
            )
        ops = read_tflite_graph(source)
        selection = request.fences
        if isinstance(selection, ExplicitFences):
            if len(set(selection.boundaries)) != len(selection.boundaries) or any(
                type(i) is not int or not 0 <= i < len(ops) - 1 for i in selection.boundaries
            ):
                raise InvalidPreparation("fences.boundaries must be unique operator indices before the last operator")
        elif isinstance(selection, BudgetFences):
            if (
                type(selection.cycle_budget) is not int
                or selection.cycle_budget <= 0
                or not math.isfinite(selection.min_coverage)
                or not 0 <= selection.min_coverage <= 1
            ):
                raise InvalidPreparation("fences require positive cycle_budget and min_coverage in [0, 1]")
        elif selection is not None:
            raise InvalidPreparation("Unsupported fence selection")
        settings = request.settings
        executable = settings.vela_bin or find_vela_binary()
        located = shutil.which(executable) or executable
        binary = Path(located).resolve()
        identity = subprocess.run([str(binary), "--version"], check=True, capture_output=True, text=True)
    except VelaError as exc:
        raise CompilerUnavailable(str(exc)) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise CompilerUnavailable(f"Cannot identify Vela executable: {exc}") from exc
    except (SocError, InvalidPreparedArtifact, ValueError, TypeError, ZeroDivisionError) as exc:
        raise InvalidPreparation(str(exc)) from exc
    try:
        root.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise InvalidPreparation(f"{root}: preparation requires a fresh writable destination ({exc})") from exc
    try:
        with tempfile.TemporaryDirectory(prefix="scratch-", dir=root) as directory:
            snapshot = Path(directory) / "source.tflite"
            shutil.copyfile(source, snapshot)
            if sha256(snapshot) != source_info.sha256:
                raise InvalidPreparation("Source changed while preparation was starting")
            artifacts: list[Artifact] = []

            def retain(path: Path, role: ArtifactRole, required: bool) -> None:
                artifacts.append(
                    Artifact(role=role, path=path.relative_to(root).as_posix(), sha256=sha256(path), required=required)
                )

            arguments: tuple[str, ...] = ()

            def compile_one(model: Path, destination: Path):
                def run(vela: str, invocation: VelaInvocation, input_path: Path, output_path: Path) -> str:
                    nonlocal arguments
                    argv = vela_command("vela", invocation, input_path, output_path)
                    replacements = {
                        str(input_path): "<source>"
                        if input_path == snapshot
                        else input_path.relative_to(root).as_posix(),
                        str(output_path): output_path.relative_to(root).as_posix(),
                        str(invocation.config_file): invocation.config_file.relative_to(root).as_posix(),
                    }
                    arguments = tuple(replacements.get(arg, arg) for arg in argv)
                    return invoke_vela(vela, invocation, input_path, output_path)

                compiled = compile_for_target(
                    model,
                    settings.soc,
                    settings.mode,
                    placement=settings.placement,
                    memory_mode=settings.soc.placement_label(settings.placement),
                    extra_placements=(settings.placement,),
                    arena_cache_size=settings.arena_cache_size,
                    timing_overrides=settings.timing_overrides,
                    output_dir=destination,
                    vela_bin=str(binary),
                    runner=run,
                )
                (destination / "stdout.txt").write_text(compiled.stdout, encoding="utf-8")
                return compiled

            baseline = None
            boundaries: tuple[int, ...] = ()
            coverage = None
            if isinstance(selection, BudgetFences):
                from ..fencing import FencePlanError, plan_fences

                baseline = compile_one(snapshot, root / "estimate")
                try:
                    plan = plan_fences(
                        baseline.per_layer,
                        ops,
                        cycle_budget=selection.cycle_budget,
                        min_coverage=selection.min_coverage,
                        model=snapshot,
                    )
                except FencePlanError as exc:
                    raise InvalidPreparation(f"Cannot plan fences: {exc}") from exc
                boundaries, coverage = plan.after_ops, plan.coverage
            elif isinstance(selection, ExplicitFences):
                boundaries = tuple(sorted(selection.boundaries))
            compiler_input = snapshot
            sidecar = None
            if boundaries:
                from ..fencing import fence_model, load_fence_boundaries

                try:
                    rewritten = fence_model(snapshot, boundaries, root / "rewrite")
                except ValueError as exc:
                    raise InvalidPreparation(f"Cannot insert requested fences: {exc}") from exc
                if rewritten is None:
                    raise InvalidPreparation("Requested fences did not produce a rewritten graph")
                compiler_input = rewritten.fenced_model_path
                sidecar = load_fence_boundaries(rewritten.boundaries_path)
                retain(rewritten.boundaries_path, ArtifactRole.BOUNDARIES, True)
                retain(compiler_input, ArtifactRole.REWRITTEN, False)
            compiled = (
                baseline if baseline is not None and not boundaries else compile_one(compiler_input, root / "compiled")
            )
            info = inspect_prepared_model(compiled.tflite)
            if not info.regions:
                raise InvalidPreparedArtifact("Vela output contains no compiled Ethos-U regions")
            if info.compatibility(expected) == AcceleratorCompatibility.INCOMPATIBLE:
                raise InvalidPreparedArtifact("Compiled accelerator differs from the requested accelerator")
            fence_evidence = None
            if selection is not None:
                spans = ()
                if sidecar is not None:
                    from ..fencing import npu_invocation_spans

                    try:
                        spans = tuple(
                            Segment(first=s.first_segment, last=s.last_segment)
                            for s in npu_invocation_spans(compiled.tflite, sidecar)
                        )
                    except ValueError as exc:
                        raise InvalidPreparedArtifact(f"Invalid final fence mapping: {exc}") from exc
                    if len(spans) != len(info.regions):
                        raise InvalidPreparedArtifact("Fence mapping does not cover every compiled NPU region")
                fence_evidence = FenceEvidence(
                    kind="budget" if isinstance(selection, BudgetFences) else "explicit",
                    boundaries=boundaries,
                    cycle_budget=selection.cycle_budget if isinstance(selection, BudgetFences) else None,
                    min_coverage=selection.min_coverage if isinstance(selection, BudgetFences) else None,
                    matched_cycles=coverage.matched_cycles if coverage else None,
                    unmatched_cycles=coverage.unmatched_cycles if coverage else None,
                    spans=spans,
                )
            retain(compiled.tflite, ArtifactRole.MODEL, True)
            retain(compiled.config_file, ArtifactRole.CONFIG, True)
            if request.retain_source:
                retained_source = root / "source.tflite"
                shutil.copyfile(snapshot, retained_source)
                retain(retained_source, ArtifactRole.SOURCE, False)
            known_paths = {a.path for a in artifacts}
            for folder in (root / "compiled", root / "estimate"):
                if folder.is_dir():
                    for path in sorted(folder.iterdir()):
                        if path.is_file() and path.relative_to(root).as_posix() not in known_paths:
                            retain(path, ArtifactRole.DIAGNOSTIC, False)
            effective = json.dumps(
                {
                    "soc": asdict(settings.soc),
                    "timing_overrides": asdict(settings.timing_overrides) if settings.timing_overrides else None,
                },
                default=_json_default,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            record = PreparationRecord(
                source_sha256=source_info.sha256,
                target=target,
                compiler=CompileEvidence(
                    accelerator=expected,
                    mode=settings.mode,
                    placement=settings.placement,
                    arena_cache_size=settings.arena_cache_size,
                    policy_id=settings.policy_id,
                    effective_model_json=effective,
                    effective_model_sha256=hashlib.sha256(effective.encode()).hexdigest(),
                    arguments=arguments,
                    version_output=(identity.stdout + identity.stderr).strip(),
                    executable_sha256=sha256(binary),
                    reported_accelerator=compiled.summary.text(VelaColumn.ACCELERATOR_CONFIGURATION),
                    reported_system_config=compiled.summary.text(VelaColumn.SYSTEM_CONFIG),
                    reported_memory_mode=compiled.summary.text(VelaColumn.MEMORY_MODE),
                ),
                artifacts=tuple(artifacts),
                fences=fence_evidence,
                packages=_package_versions(),
                regions=info.regions,
                custom_operators=info.custom_operators,
                inputs=info.inputs,
                outputs=info.outputs,
            )
            # Validate through the same reader before publishing the success record.
            pending = root / "pending.json"
            pending.write_text(record.model_dump_json(indent=2), encoding="utf-8")
            validated = inspect_prepared_model(pending)
            record_path = root / "preparation.json"
            pending.rename(record_path)
            return PreparedModel(compiled.tflite, record_path, validated)
    except Exception as exc:
        (root / "pending.json").unlink(missing_ok=True)
        (root / "failure.txt").write_text(str(exc), encoding="utf-8")
        if isinstance(exc, (InvalidPreparation, InvalidPreparedArtifact, ImportError)):
            raise
        if isinstance(exc, (VelaError, OSError, subprocess.SubprocessError)):
            raise CompilationFailed(f"{root}: preparation failed; see failure.txt ({exc})") from exc
        raise

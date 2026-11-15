"""Read compiled models and relocatable preparation records without invoking Vela."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import replace
from pathlib import Path

from pydantic import ValidationError

from .._optional import litert_schema
from ..model import model_io
from ..soc import NpuFamily
from .types import (
    Accelerator,
    ArtifactRole,
    CustomOperator,
    InvalidPreparedArtifact,
    PreparationRecord,
    PreparedModelInfo,
    Region,
    Segment,
)


def sha256(path: Path) -> str:
    """Return a file's SHA-256 without retaining its bytes in memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _accelerator(payload: bytes) -> Accelerator | None:
    # COP1 begins with the driver's Config action. CONFIG[31:28] is product,
    # CONFIG[3:0] is log2(MACs). See Vela driver_actions.py and Ethos CONFIG.
    # Other payload formats are deliberately left unknown.
    if len(payload) < 16 or payload[:4] != b"COP1" or payload[4] != 1:
        return None
    word = struct.unpack_from("<I", payload, 8)[0]
    family = {0: NpuFamily.ETHOS_U55, 1: NpuFamily.ETHOS_U65, 2: NpuFamily.ETHOS_U85}.get(word >> 28)
    exponent = word & 15
    if family is None or exponent == 0:
        return None
    return Accelerator(family, 1 << exponent)


def _inspect_file(path: Path) -> PreparedModelInfo:
    schema = litert_schema()
    data = path.read_bytes()
    model = schema.Model.GetRootAs(data, 0)
    regions = []
    requirements: set[tuple[str, int]] = set()
    for graph_index in range(model.SubgraphsLength()):
        graph = model.Subgraphs(graph_index)
        for operator_index in range(graph.OperatorsLength()):
            op = graph.Operators(operator_index)
            opcode = model.OperatorCodes(op.OpcodeIndex())
            if max(opcode.BuiltinCode(), opcode.DeprecatedBuiltinCode()) != schema.BuiltinOperator.CUSTOM:
                continue
            name = opcode.CustomCode().decode("utf-8")
            requirements.add((name, opcode.Version()))
            if name != "ethos-u":
                continue
            payload = b""
            if op.InputsLength() > 0 and 0 <= op.Inputs(0) < graph.TensorsLength():
                tensor = graph.Tensors(op.Inputs(0))
                buffer = model.Buffers(tensor.Buffer())
                if buffer.Offset():
                    end = buffer.Offset() + buffer.Size()
                    if end <= len(data):
                        payload = data[buffer.Offset() : end]
                else:
                    payload = bytes(buffer.DataAsNumpy())
            accelerator = _accelerator(payload)
            regions.append(
                Region(
                    subgraph=graph_index,
                    operator=operator_index,
                    accelerator=accelerator,
                    unknown_reason=None if accelerator else "Missing or unsupported command payload identity",
                )
            )
    inputs, outputs = model_io(path)
    return PreparedModelInfo(
        path,
        hashlib.sha256(data).hexdigest(),
        tuple(regions),
        tuple(CustomOperator(name=name, version=version) for name, version in sorted(requirements)),
        inputs,
        outputs,
    )


def _artifact_path(root: Path, relative: str) -> Path:
    # Accept only portable record paths, including when read on a different OS.
    parts = relative.split("/")
    if not relative or "\\" in relative or ":" in relative or any(p in ("", ".", "..") for p in parts):
        raise InvalidPreparedArtifact(f"Invalid relative artifact path: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise InvalidPreparedArtifact(f"Artifact escapes record directory: {relative}")
    return path


def _inspect_record(path: Path) -> PreparedModelInfo:
    record = PreparationRecord.model_validate_json(path.read_bytes())
    root = path.parent
    missing = []
    paths: dict[ArtifactRole, list[Path]] = {}
    seen = set()
    for artifact in record.artifacts:
        if artifact.path in seen:
            raise InvalidPreparedArtifact(f"Duplicate artifact: {artifact.path}")
        seen.add(artifact.path)
        resolved = _artifact_path(root, artifact.path)
        paths.setdefault(artifact.role, []).append(resolved)
        if not resolved.is_file():
            if artifact.required:
                raise InvalidPreparedArtifact(f"Missing required {artifact.role}: {artifact.path}")
            missing.append(artifact.path)
        elif sha256(resolved) != artifact.sha256:
            raise InvalidPreparedArtifact(f"Hash mismatch: {artifact.path}")
    required_roles = {ArtifactRole.MODEL, ArtifactRole.CONFIG}
    if record.fences is not None and record.fences.boundaries:
        required_roles.add(ArtifactRole.BOUNDARIES)
    for role in required_roles:
        entries = [a for a in record.artifacts if a.role == role]
        if len(entries) != 1 or not entries[0].required:
            raise InvalidPreparedArtifact(f"Record requires exactly one required {role} artifact")
    if (
        hashlib.sha256(record.compiler.effective_model_json.encode()).hexdigest()
        != record.compiler.effective_model_sha256
    ):
        raise InvalidPreparedArtifact("Effective compiler model hash mismatch")
    info = _inspect_file(paths[ArtifactRole.MODEL][0])
    if (info.regions, info.custom_operators, info.inputs, info.outputs) != (
        record.regions,
        record.custom_operators,
        record.inputs,
        record.outputs,
    ):
        raise InvalidPreparedArtifact("Observed model requirements differ from the preparation record")
    if not info.regions:
        raise InvalidPreparedArtifact("Recorded model contains no Ethos-U regions")
    if record.fences is not None and record.fences.boundaries:
        from ..fencing import load_fence_boundaries, npu_invocation_spans

        boundaries = load_fence_boundaries(paths[ArtifactRole.BOUNDARIES][0])
        if boundaries.source_sha256 != record.source_sha256:
            raise InvalidPreparedArtifact("Fence sidecar is bound to a different source")
        if tuple(boundaries.boundary_after_ops) != record.fences.boundaries:
            raise InvalidPreparedArtifact("Fence boundaries do not match the preparation record")
        spans = tuple(
            Segment(first=s.first_segment, last=s.last_segment) for s in npu_invocation_spans(info.tflite, boundaries)
        )
        if spans != record.fences.spans or len(spans) != len(info.regions):
            raise InvalidPreparedArtifact("Final invocation mapping differs from recorded fence spans")
    return replace(info, record=record, unavailable_optional=tuple(missing))


def inspect_prepared_model(path: Path) -> PreparedModelInfo:
    """Inspect a standalone TFLite or validate a record and its retained files.

    No compiler or consumer is imported. Missing optional diagnostics are reported;
    missing required artifacts or mismatched bytes raise InvalidPreparedArtifact.
    Standalone files provide observed facts only, without invented provenance.
    """
    path = Path(path).resolve()
    try:
        return _inspect_record(path) if path.suffix == ".json" else _inspect_file(path)
    except InvalidPreparedArtifact:
        raise
    except (OSError, ValueError, TypeError, IndexError, AttributeError, struct.error, ValidationError) as exc:
        raise InvalidPreparedArtifact(f"{path}: cannot inspect prepared model ({exc})") from exc

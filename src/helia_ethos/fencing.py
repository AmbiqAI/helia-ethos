"""Cutting a model into separately measurable NPU segments.

Vela ends the current Ethos-U command stream at any custom op it cannot fuse,
hands off to the CPU, and starts a new one. Inserting ``ETHOS_U_STREAM_SPLIT`` (an
identity op the runtime implements as a memcpy) between two operators
therefore forces a command-stream break there, which is what per-segment
measurement, chunked execution and latency bounding all build on.

Three layers, each usable alone:

- **policy**: :class:`FenceConfig` / :func:`compute_boundaries` say which
  operator indices a fence follows (per op, every N ops, or explicit).
- **planning**: :func:`plan_fences` proposes cut points from Vela's own
  per-layer estimate (a cycle budget per chunk, ops too big to share a
  chunk), so a caller gets rough cutoffs with no simulator involved.
- **rewriting**: :func:`rewrite_model_with_fences` / :func:`fence_model` inject
  the ops into the flatbuffer and write a boundaries sidecar.

Fence positions are original TFLite operator indices, the same coordinates
:func:`helia_ethos.vela.match_original_op` produces.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from ._optional import litert_schema

try:
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.tools import flatbuffer_utils
except ImportError:  # the models extra is absent; the entry points say so when called
    schema = None  # type: ignore[assignment]
    flatbuffer_utils = None  # type: ignore[assignment]

from .vela.per_layer import LayerRow, TFLiteOp, index_by_output_name, match_original_op, read_tflite_graph

__all__ = [
    "STREAM_SPLIT_OP_NAME",
    "Fence",
    "FenceBindingError",
    "FenceBoundaries",
    "FenceConfig",
    "FenceCoverage",
    "FenceMode",
    "FencePlan",
    "FencePlanError",
    "FenceSegment",
    "FenceSidecarError",
    "FenceSpan",
    "RewriteResult",
    "attribute_cycles",
    "check_plan_binding",
    "compute_boundaries",
    "fence_config_for",
    "fence_model",
    "load_fence_boundaries",
    "npu_invocation_spans",
    "plan_fences",
    "rewrite_model_with_fences",
]

STREAM_SPLIT_OP_NAME = "ETHOS_U_STREAM_SPLIT"
#: The custom operator Vela compiles each run of NPU operators into; the runtime starts the NPU once per instance.
ETHOS_U_OP_NAME = "ethos-u"
STREAM_SPLIT_OP_VERSION = 1  # contract version; docs/stream_split_contract.md
_CUSTOM_BUILTIN_CODE = 32  # tflite BuiltinOperator.CUSTOM


# ── Policy ──────────────────────────────────────────────────────────────────


class FenceMode(StrEnum):
    """Strategy for choosing fence positions.

    Surfaced in YAML as ``models[].per_layer.mode``.
    """

    OFF = "off"
    PER_OP = "per_op"
    PER_GROUP = "per_group"
    #: Fence exactly at ``boundaries``; what :func:`plan_fences` produces.
    EXPLICIT = "explicit"


@dataclass(frozen=True)
class FenceConfig:
    """Where to insert ETHOS_U_STREAM_SPLIT fences.

    ``boundaries``, when set, wins over *mode*: it names the operator indices
    directly instead of deriving them.
    """

    mode: FenceMode = FenceMode.OFF
    group_size: int = 1
    boundaries: tuple[int, ...] | None = None


def compute_boundaries(num_ops: int, config: FenceConfig) -> list[int]:
    """Return operator indices AFTER which a fence should be inserted.

    Sorted, 0-based. A fence at index ``i`` sits between ``op[i]`` and
    ``op[i + 1]``, so the last operator is never a boundary.
    """
    if config.mode == FenceMode.OFF:
        return []

    if config.boundaries is not None:
        return sorted(b for b in config.boundaries if 0 <= b < num_ops - 1)

    if config.mode == FenceMode.PER_OP:
        return list(range(num_ops - 1))

    if config.mode == FenceMode.PER_GROUP:
        size = max(1, config.group_size)
        return list(range(size - 1, num_ops - 1, size))

    return []


# ── Planning ────────────────────────────────────────────────────────────────


class FencePlanError(ValueError):
    """A plan was asked to reason about cost it does not have."""


class FenceBindingError(ValueError):
    """A plan or fence carries identity that does not match the model it is being applied to."""


@dataclass(frozen=True)
class Fence:
    """A proposed command-stream break after original operator ``after_op``.

    ``after_op`` is the coordinate every consumer agrees on; the rest is identity
    for joining the plan to compiled and runtime artifacts: the output tensor
    the break intercepts (what Vela and the ETHOS_U_STREAM_SPLIT rewrite key on) and the
    index of the segment that ends here.
    """

    after_op: int
    reason: str
    output_tensor_name: str | None = None
    opcode: str | None = None
    segment_index: int | None = None


@dataclass(frozen=True)
class FenceCoverage:
    """How much of Vela's estimated cost the planner could attribute to original operators."""

    matched_cycles: int
    unmatched_cycles: int
    unmatched_rows: tuple[tuple[str, int], ...]  # (vela row name, op_cycles)

    @property
    def total_cycles(self) -> int:
        return self.matched_cycles + self.unmatched_cycles

    @property
    def fraction(self) -> float:
        """Attributed share of the denominator (sum of op_cycles over every Vela row). 1.0 when there are no rows."""
        return 1.0 if self.total_cycles == 0 else self.matched_cycles / self.total_cycles

    def summary(self) -> str:
        if not self.unmatched_rows:
            return f"all {self.matched_cycles:,} estimated cycles attributed"
        names = ", ".join(repr(name) for name, _ in self.unmatched_rows[:3])
        more = f" (+{len(self.unmatched_rows) - 3} more)" if len(self.unmatched_rows) > 3 else ""
        return (
            f"{self.unmatched_cycles:,} of {self.total_cycles:,} estimated cycles "
            f"({1 - self.fraction:.1%}) in {len(self.unmatched_rows)} unmatched Vela rows: {names}{more}"
        )


@dataclass(frozen=True)
class FencePlan:
    """The fences, and what they were planned from."""

    fences: tuple[Fence, ...]
    coverage: FenceCoverage
    num_ops: int
    model_sha256: str | None = None  # of the original .tflite, when the planner was given it
    subgraph: int = 0

    @property
    def after_ops(self) -> tuple[int, ...]:
        return tuple(fence.after_op for fence in self.fences)

    @property
    def num_segments(self) -> int:
        return len(self.fences) + 1


def attribute_cycles(rows: Sequence[LayerRow], tflite_ops: Sequence[TFLiteOp]) -> tuple[dict[int, int], FenceCoverage]:
    """Vela's estimated cycles folded onto original operator indices, plus what could not be placed."""
    index = index_by_output_name(tflite_ops)
    cycles: dict[int, int] = dict.fromkeys((op.index for op in tflite_ops), 0)
    matched = 0
    unmatched: list[tuple[str, int]] = []
    for row in rows:
        target = match_original_op(row.name, index)
        if target is None:
            unmatched.append((row.name, row.op_cycles))
        else:
            cycles[target.index] = cycles.get(target.index, 0) + row.op_cycles
            matched += row.op_cycles
    coverage = FenceCoverage(matched, sum(c for _, c in unmatched), tuple(unmatched))
    return cycles, coverage


def cycles_per_original_op(rows: Sequence[LayerRow], tflite_ops: Sequence[TFLiteOp]) -> dict[int, int]:
    """:func:`attribute_cycles` without the coverage, for callers that only want the map."""
    return attribute_cycles(rows, tflite_ops)[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_fences(
    rows: Sequence[LayerRow],
    tflite_ops: Sequence[TFLiteOp],
    *,
    cycle_budget: int | None = None,
    max_op_cycles: int | None = None,
    group_size: int | None = None,
    min_coverage: float = 1.0,
    model: Path | None = None,
) -> FencePlan:
    """Propose fence positions from Vela's per-layer estimate.

    ``cycle_budget``: start a new chunk before the operator that would push
    the running total past the budget. ``max_op_cycles``: isolate any operator
    estimated above the threshold in a chunk of its own. ``group_size``: a
    fence every N operators. Policies combine; the result is sorted and
    never fences after the last operator.

    The cycle-based policies reason about Vela rows that could be mapped back
    to original operators. Rows that could not (their cost is real but has no
    home) are reported in ``plan.coverage``; when a cycle-based policy is
    requested and the attributed fraction is below ``min_coverage`` the plan
    raises rather than silently budgeting on partial cost. Pass a lower
    ``min_coverage`` to accept that explicitly. Estimates are estimates: a
    plan bounds nothing on hardware.
    """
    num_ops = len(tflite_ops)
    cycles, coverage = attribute_cycles(rows, tflite_ops)
    sha = _sha256(model) if model is not None else None
    if num_ops < 2:
        return FencePlan((), coverage, num_ops, sha)

    uses_cost = (cycle_budget is not None and cycle_budget > 0) or (max_op_cycles is not None and max_op_cycles > 0)
    if uses_cost and not rows:
        raise FencePlanError(
            "cannot plan on estimated cost: no Vela per-layer rows were given (unknown cost is not zero cost)"
        )
    if uses_cost and coverage.fraction < min_coverage:
        raise FencePlanError(
            f"cannot plan on estimated cost: {coverage.summary()}; "
            f"pass min_coverage<={coverage.fraction:.3f} to plan on the attributed part anyway"
        )

    ordered = sorted(tflite_ops, key=lambda op: op.index)
    by_index = {op.index: op for op in ordered}
    fences: dict[int, str] = {}

    def add(after_op: int, reason: str) -> None:
        if 0 <= after_op < num_ops - 1 and after_op not in fences:
            fences[after_op] = reason

    if cycle_budget is not None and cycle_budget > 0:
        running = 0
        for position, op in enumerate(ordered):
            op_cycles = cycles.get(op.index, 0)
            if running and running + op_cycles > cycle_budget:
                add(ordered[position - 1].index, f"chunk reached {running:,} of {cycle_budget:,} estimated cycles")
                running = 0
            running += op_cycles

    if max_op_cycles is not None and max_op_cycles > 0:
        for position, op in enumerate(ordered):
            op_cycles = cycles.get(op.index, 0)
            if op_cycles > max_op_cycles:
                reason = f"{op.opcode} estimated at {op_cycles:,} cycles, above {max_op_cycles:,}"
                if position > 0:
                    add(ordered[position - 1].index, reason)
                add(op.index, reason)

    if group_size is not None and group_size > 0:
        for after_op in compute_boundaries(num_ops, FenceConfig(mode=FenceMode.PER_GROUP, group_size=group_size)):
            add(after_op, f"every {group_size} operators")

    planned = tuple(
        Fence(
            after_op=after_op,
            reason=fences[after_op],
            output_tensor_name=by_index[after_op].output_tensor_name,
            opcode=by_index[after_op].opcode,
            segment_index=segment,
        )
        for segment, after_op in enumerate(sorted(fences))
    )
    return FencePlan(planned, coverage, num_ops, sha)


def fence_config_for(fences: FencePlan | Sequence[Fence] | Sequence[int]) -> FenceConfig:
    """The explicit :class:`FenceConfig` that reproduces a plan's positions.

    This is the deliberate reduction to positions: the plan's model hash and
    the fences' tensor names and opcodes are not carried into the config.
    :func:`fence_model` checks them first; a caller that feeds this config to
    :func:`rewrite_model_with_fences` directly has chosen to skip that.
    """
    items: Sequence[Fence] | Sequence[int] = fences.fences if isinstance(fences, FencePlan) else fences
    indices = tuple(sorted({f.after_op if isinstance(f, Fence) else int(f) for f in items}))
    return FenceConfig(mode=FenceMode.EXPLICIT if indices else FenceMode.OFF, boundaries=indices or None)


# ── Rewriting ───────────────────────────────────────────────────────────────


class FenceSidecarError(ValueError):
    """A fence boundaries sidecar is missing, unreadable or not one."""


class FenceSegment(BaseModel):
    """One command-stream segment of a fenced model: the original operators it spans, inclusive."""

    model_config = ConfigDict(extra="ignore")

    segment_index: int
    op_range_start: int
    op_range_end: int
    #: The output tensor the fence after this segment intercepts; absent on the last segment.
    end_tensor_name: str | None = None
    #: The ETHOS_U_STREAM_SPLIT output that replaces it downstream; absent on the last segment.
    fence_tensor_name: str | None = None
    #: Actual input/output pairs in the final fenced graph; the first output is the join key above.
    tensor_pairs: list[tuple[str, str]] | None = None


class FenceBoundaries(BaseModel):
    """The boundaries sidecar written next to a fenced model.

    It is the join between the fenced compile and the per-segment measurement:
    which original operators each measured segment covers, and which model the
    fences were cut into. A reader ignores fields it does not know, so an
    older reader still accepts a newer writer's file.
    """

    model_config = ConfigDict(extra="ignore")

    source_model: str | None = None
    source_sha256: str | None = None
    subgraph: int = 0
    num_segments: int
    boundary_after_ops: list[int]
    segments: list[FenceSegment]

    def write(self, path: Path) -> None:
        """Write the sidecar as indented JSON, leaving out the fields a segment does not have."""
        path.write_text(json.dumps(self.model_dump(exclude_none=True), indent=2), encoding="utf-8")


def load_fence_boundaries(path: Path) -> FenceBoundaries:
    """Read a boundaries sidecar; a :class:`FenceSidecarError` names the file when it is missing or not one."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FenceSidecarError(f"{path}: cannot read fence boundaries sidecar ({exc.strerror or exc})") from exc
    try:
        return FenceBoundaries.model_validate_json(text)
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        location = ".".join(str(part) for part in first["loc"]) or "document"
        raise FenceSidecarError(f"{path}: not a fence boundaries sidecar ({location}: {first['msg']})") from exc


@dataclass(frozen=True)
class FenceSpan:
    """Consecutive fence segments a compiled model runs as one stretch, inclusive: one segment, unless Vela removed
    the fences between them, as it does when it folds the operators a fence follows into constants."""

    first_segment: int
    last_segment: int


def npu_invocation_spans(compiled_model_path: Path, boundaries: FenceBoundaries) -> tuple[FenceSpan, ...]:
    """The fence segments each NPU invocation of a fenced model Vela compiled covers, in the order the runtime makes
    them.

    Vela compiles each run of NPU operators into one ``ethos-u`` operator, and the runtime starts the NPU once for
    each, so the per-segment PMU prints one ``SEGMENT`` line per ``ethos-u`` operator. Operators Vela leaves on the
    CPU (a QUANTIZE, a SPACE_TO_BATCH_ND) are not invocations: a fence segment holding only those prints no line. The
    fences stay in the compiled model as ``ETHOS_U_STREAM_SPLIT`` operators, each named by the tensor in the
    *boundaries* sidecar, except where Vela folded the operators before one into constants and removed it with them.
    The segments either side of a removed fence then run as one span. The n-th line is therefore not always
    the n-th fence segment.

    Raises ``ValueError`` for an ``ETHOS_U_STREAM_SPLIT`` the sidecar does not name, or fences out of order: the
    model was not compiled from the fences the sidecar describes.
    """
    fence_after = {seg.fence_tensor_name: seg.segment_index for seg in boundaries.segments if seg.fence_tensor_name}
    ops = read_tflite_graph(compiled_model_path)
    kept: list[int] = []  # the fence each ETHOS_U_STREAM_SPLIT is, by the segment it follows
    for op in ops:
        if op.opcode != STREAM_SPLIT_OP_NAME:
            continue
        fence = fence_after.get(op.output_tensor_name)
        if fence is None or (kept and fence <= kept[-1]):
            raise ValueError(
                f"{compiled_model_path.name} has an ETHOS_U_STREAM_SPLIT ({op.output_tensor_name!r}) that "
                "is not the next "
                "fence of its boundaries sidecar: the model was compiled from other fences, the sidecar predates "
                "fence tensor names, or Vela reordered fences on parallel branches"
            )
        kept.append(fence)
    starts = [0, *(fence + 1 for fence in kept)]
    ends = [*kept, boundaries.num_segments - 1]
    span = 0
    invocations: list[FenceSpan] = []
    for op in ops:
        if op.opcode == STREAM_SPLIT_OP_NAME:
            span += 1
        elif op.opcode == ETHOS_U_OP_NAME:
            invocations.append(FenceSpan(starts[span], ends[span]))
    return tuple(invocations)


@dataclass
class RewriteResult:
    """Result of rewriting a TFLite model with fences."""

    fenced_model_path: Path
    boundaries_path: Path
    num_segments: int
    boundary_indices: list[int]
    #: sha256 of the model the fences were cut into; the join key for compiled and runtime artifacts.
    source_sha256: str | None = None
    #: The output tensor each fence intercepts, and the ETHOS_U_STREAM_SPLIT output that replaces it downstream.
    fence_tensor_names: tuple[tuple[str, str], ...] = ()


def rewrite_model_with_fences(
    model_path: Path,
    config: FenceConfig,
    output_dir: Path,
) -> RewriteResult | None:
    """Rewrite a TFLite model, inserting ETHOS_U_STREAM_SPLIT ops at boundaries.

    The rewriter:
    1. Reads the original flatbuffer via TF's mutable object API
    2. Computes fence boundaries
    3. For each boundary, inserts an ETHOS_U_STREAM_SPLIT custom op that takes the
       live tensors produced through op[i] and forwards matching copies to
       downstream consumers, including residual branches and graph outputs
    4. Writes the modified model and a boundaries JSON sidecar

    Returns None if mode is 'off' or no boundaries needed.
    """
    litert_schema()  # a missing extra is reported by name, before any file is touched
    model = flatbuffer_utils.read_model(str(model_path))
    subgraph = _primary_subgraph(model, model_path)
    num_ops = len(subgraph.operators)

    boundaries = compute_boundaries(num_ops, config)
    if not boundaries:
        return None

    pairs = _inject_fences(model, subgraph, boundaries, model_path)
    intercepted = [group[0][0] for group in pairs]
    # Earlier cuts rewire later fence inputs. Resolve actual names only after all cuts are inserted.
    actual_pairs = {
        _tensor_name(subgraph, op.outputs[0]): [
            (_tensor_name(subgraph, source), _tensor_name(subgraph, output))
            for source, output in zip(op.inputs, op.outputs, strict=True)
        ]
        for op in subgraph.operators
        if model.operatorCodes[op.opcodeIndex].customCode == STREAM_SPLIT_OP_NAME.encode()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    fenced_path = output_dir / f"{model_path.stem}.fenced.tflite"
    flatbuffer_utils.write_model(model, str(fenced_path))

    # Write boundaries sidecar
    num_segments = len(boundaries) + 1
    source_sha256 = _sha256(model_path)
    fence_tensor_names = tuple(group[0] for group in pairs)
    segments: list[FenceSegment] = []
    prev = 0
    for i, b in enumerate(boundaries):
        segments.append(
            FenceSegment(
                segment_index=i,
                op_range_start=prev,
                op_range_end=b,
                end_tensor_name=intercepted[i],
                fence_tensor_name=fence_tensor_names[i][1],
                tensor_pairs=actual_pairs[fence_tensor_names[i][1]],
            )
        )
        prev = b + 1
    segments.append(FenceSegment(segment_index=len(boundaries), op_range_start=prev, op_range_end=num_ops - 1))
    sidecar = FenceBoundaries(
        source_model=model_path.name,
        source_sha256=source_sha256,
        subgraph=0,
        num_segments=num_segments,
        boundary_after_ops=boundaries,
        segments=segments,
    )

    boundaries_path = output_dir / f"{model_path.stem}.fenced_boundaries.json"
    sidecar.write(boundaries_path)

    return RewriteResult(
        fenced_model_path=fenced_path,
        boundaries_path=boundaries_path,
        num_segments=num_segments,
        boundary_indices=boundaries,
        source_sha256=source_sha256,
        fence_tensor_names=fence_tensor_names,
    )


def _tensor_name(subgraph: schema.SubGraphT, tensor_index: int) -> str:
    name = subgraph.tensors[tensor_index].name
    if isinstance(name, bytes):
        name = name.decode()
    return name or f"tensor_{tensor_index}"


def _fence_tensor_name(intercepted_name: str) -> str:
    """The name :func:`_inject_fences` gives the ETHOS_U_STREAM_SPLIT output and the sidecar records; consumers join
    on it.
    *intercepted_name* comes from :func:`_tensor_name`; the writer adds a cut suffix when needed."""
    return f"{intercepted_name}/stream_split"


def check_plan_binding(model_path: Path, fences: FencePlan | Sequence[Fence] | Sequence[int]) -> None:
    """Raise :class:`FenceBindingError` unless every piece of identity *fences* carries matches *model_path*.

    A :class:`FencePlan` is bound by its ``model_sha256`` (when the planner was
    given the model), its operator count and its subgraph; each :class:`Fence`
    by its ``output_tensor_name`` and ``opcode`` (when set). Positions alone (a
    sequence of ints, or a :class:`Fence` with no identity) carry nothing to
    check and pass; that is the deliberately unbound path, and the caller is
    asserting the coordinates apply. To apply a bound plan's positions to
    another model, pass ``plan.after_ops`` (or :func:`fence_config_for`, the
    same reduction) rather than the plan.
    """
    plan = fences if isinstance(fences, FencePlan) else None
    items: list[Fence] = list(plan.fences) if plan is not None else [f for f in fences if isinstance(f, Fence)]
    bound = [f for f in items if f.output_tensor_name is not None or f.opcode is not None]
    if plan is None and not bound:
        return
    if plan is not None:
        if plan.subgraph != 0:
            raise FenceBindingError(f"plan targets subgraph {plan.subgraph}; the rewrite supports subgraph 0 only")
        if plan.model_sha256 is not None:
            actual = _sha256(model_path)
            if actual != plan.model_sha256:
                raise FenceBindingError(
                    f"plan was cut for a model with sha256 {plan.model_sha256[:12]}…, but {model_path.name} "
                    f"is {actual[:12]}…; re-plan on this model, or pass plan.after_ops to apply positions unbound"
                )
    ops = read_tflite_graph(model_path)
    if plan is not None and plan.num_ops != len(ops):
        raise FenceBindingError(f"plan counted {plan.num_ops} operators; {model_path.name} has {len(ops)}")
    by_index = {op.index: op for op in ops}
    for fence in bound:
        op = by_index.get(fence.after_op)
        if op is None:
            raise FenceBindingError(
                f"fence after operator {fence.after_op} is outside {model_path.name} ({len(ops)} operators)"
            )
        if fence.output_tensor_name is not None and fence.output_tensor_name != op.output_tensor_name:
            raise FenceBindingError(
                f"fence after operator {fence.after_op} was planned on output tensor {fence.output_tensor_name!r}; "
                f"{model_path.name} has {op.output_tensor_name!r} there"
            )
        if fence.opcode is not None and fence.opcode != op.opcode:
            raise FenceBindingError(
                f"fence after operator {fence.after_op} was planned on a {fence.opcode}; "
                f"{model_path.name} has a {op.opcode} there"
            )
        if fence.after_op >= len(ops) - 1:
            raise FenceBindingError(
                f"fence after operator {fence.after_op} follows the last operator of {model_path.name}; it cuts nothing"
            )


def fence_model(
    model_path: Path,
    fences: FencePlan | Sequence[Fence] | Sequence[int],
    output_dir: Path,
) -> RewriteResult | None:
    """Inject ``ETHOS_U_STREAM_SPLIT`` after each fence position; ``None`` when there is nothing to do.

    Runs :func:`check_plan_binding` first, so a plan cut for a different model
    (or a fence whose tensor name or opcode no longer sits at ``after_op``)
    raises before anything is written.
    """
    check_plan_binding(model_path, fences)
    return rewrite_model_with_fences(model_path, fence_config_for(fences), output_dir)


def _primary_subgraph(model: schema.ModelT, model_path: Path) -> schema.SubGraphT:
    """Return the model's only subgraph, rejecting a structurally empty or multi-subgraph model.

    Every field on the generated flatbuffer classes is optional, so a
    truncated or hand-edited .tflite reaches us with None where a list should
    be. Failing here names the file instead of raising an AttributeError from
    somewhere deep in the rewrite. The ETHOS_U_STREAM_SPLIT contract (v1) covers
    subgraph 0 of a single-subgraph model only; anything else is rejected
    rather than partially rewritten.
    """
    subgraphs = model.subgraphs
    if not subgraphs:
        raise ValueError(f"TFLite model has no subgraphs: {model_path}")
    if len(subgraphs) != 1:
        raise ValueError(
            f"TFLite model has {len(subgraphs)} subgraphs: {model_path}; "
            "the ETHOS_U_STREAM_SPLIT rewrite supports a single subgraph (contract v1)"
        )
    subgraph = subgraphs[0]
    if not subgraph.operators:
        raise ValueError(f"TFLite model subgraph has no operators: {model_path}")
    if subgraph.tensors is None:
        raise ValueError(f"TFLite model subgraph has no tensors: {model_path}")
    return subgraph


def _stream_split_opcode_index(model: schema.ModelT, model_path: Path) -> int:
    """The operator-code index every fence uses: an existing ETHOS_U_STREAM_SPLIT entry, or a new one.

    An existing entry is reused only if it is the contract's wire format: its
    effective builtin code is CUSTOM and its version is
    ``STREAM_SPLIT_OP_VERSION``. The effective code is the larger of
    ``builtin_code`` and ``deprecated_builtin_code``, which is how TFLite's
    ``GetBuiltinCode`` bridges the v3 and v3a schemas, so a legacy entry with
    only the deprecated field set, or a current one with only ``builtin_code``
    set, is the same CUSTOM definition to every consumer (heliaRT's replay on
    person_detect, 2026-09-22). ``builtin_code`` itself must be CUSTOM or
    unset: Vela reads ``builtin_code`` whenever it is non-zero, so a
    hand-edited entry with another builtin there and CUSTOM in the deprecated
    field would compile as that other operator. Anything else is rejected
    rather than reused, since the emitted fences would then carry a
    definition not every consumer of contract v1 reads the same way. A reused
    entry is passed through as written, not rewritten to 32/32.
    """
    if model.operatorCodes is None:
        model.operatorCodes = []
    for index, oc in enumerate(model.operatorCodes):
        if not (oc.customCode and oc.customCode.decode() == STREAM_SPLIT_OP_NAME):
            continue
        builtin, deprecated, version = oc.builtinCode or 0, oc.deprecatedBuiltinCode or 0, oc.version
        effective = max(builtin, deprecated)
        if (
            effective != _CUSTOM_BUILTIN_CODE
            or builtin not in (0, _CUSTOM_BUILTIN_CODE)
            or version != STREAM_SPLIT_OP_VERSION
        ):
            raise ValueError(
                f"{model_path} already defines an ETHOS_U_STREAM_SPLIT operator code with "
                f"(builtin_code, deprecated_builtin_code, version) = {(builtin, deprecated, version)}; the rewrite "
                f"needs an effective builtin code of CUSTOM ({_CUSTOM_BUILTIN_CODE}, the larger of the two fields, "
                f"with builtin_code itself CUSTOM or unset) at version {STREAM_SPLIT_OP_VERSION} "
                f"(contract v{STREAM_SPLIT_OP_VERSION}) and will not reuse or emit an incompatible definition"
            )
        return index
    oc = schema.OperatorCodeT()
    oc.deprecatedBuiltinCode = _CUSTOM_BUILTIN_CODE
    oc.customCode = STREAM_SPLIT_OP_NAME.encode()
    oc.version = STREAM_SPLIT_OP_VERSION
    oc.builtinCode = _CUSTOM_BUILTIN_CODE
    model.operatorCodes.append(oc)
    return len(model.operatorCodes) - 1


def _cut_tensors(subgraph: schema.SubGraphT, boundary: int) -> list[int]:
    """Return produced tensors live across a cut, putting the selected layer's outputs first."""
    produced = {
        int(t)
        for op in subgraph.operators[: boundary + 1]
        for t in (() if op.outputs is None else op.outputs)
        if t >= 0
    }
    used = {
        int(t) for op in subgraph.operators[boundary + 1 :] for t in (() if op.inputs is None else op.inputs) if t >= 0
    }
    used.update(int(t) for t in (() if subgraph.outputs is None else subgraph.outputs))
    crossing = produced & used
    boundary_outputs = subgraph.operators[boundary].outputs
    preferred = [int(t) for t in (() if boundary_outputs is None else boundary_outputs) if t in crossing]
    return preferred + sorted(crossing - set(preferred))


def _inject_fences(
    model: schema.ModelT, subgraph: schema.SubGraphT, boundaries: list[int], model_path: Path
) -> list[list[tuple[str, str]]]:
    """Insert one multi-tensor boundary per cut and return its original/input-to-output names.

    Constants and graph inputs are already available and are not copied. Every
    tensor produced before a cut and consumed after it (or returned) is carried.
    Reverse insertion keeps original operator indices and later cut identities stable.
    """
    cuts = [_cut_tensors(subgraph, boundary) for boundary in boundaries]
    supported = {
        schema.TensorType.INT8,
        schema.TensorType.UINT8,
        schema.TensorType.INT16,
        schema.TensorType.INT32,
        schema.TensorType.INT64,
        schema.TensorType.FLOAT16,
        schema.TensorType.FLOAT32,
        schema.TensorType.BOOL,
    }
    for boundary, tensors in zip(boundaries, cuts, strict=True):
        if not tensors:
            raise ValueError(f"{model_path}: no live produced tensors cross the fence after operator {boundary}")
        for index in tensors:
            tensor = subgraph.tensors[index]
            if tensor.type not in supported or tensor.isVariable or tensor.sparsity is not None:
                raise ValueError(
                    f"{model_path}: fence tensor {_tensor_name(subgraph, index)!r} must be a dense, fixed-width value"
                )
    opcode = _stream_split_opcode_index(model, model_path)
    groups: list[list[tuple[str, str]]] = [[] for _ in boundaries]
    for position in reversed(range(len(boundaries))):
        boundary = boundaries[position]
        replacements: dict[int, int] = {}
        for ordinal, index in enumerate(cuts[position]):
            original_name = _tensor_name(subgraph, index)
            # Preserve the first-output join convention; other carried tensors need cut-specific names.
            output_name = _fence_tensor_name(original_name)
            if ordinal or any(output_name == _tensor_name(subgraph, t) for t in range(len(subgraph.tensors))):
                output_name += f"_cut_{boundary}_{ordinal}"
            tensor = copy.deepcopy(subgraph.tensors[index])
            tensor.name = output_name.encode()
            tensor.isVariable = False
            if model.buffers is None:
                model.buffers = []
            tensor.buffer = len(model.buffers)
            model.buffers.append(schema.BufferT())
            replacements[index] = len(subgraph.tensors)
            subgraph.tensors.append(tensor)
            groups[position].append((original_name, output_name))
        fence = schema.OperatorT()
        fence.opcodeIndex = opcode
        fence.inputs = list(replacements)
        fence.outputs = list(replacements.values())
        subgraph.operators.insert(boundary + 1, fence)
        for op in subgraph.operators[boundary + 2 :]:
            if op.inputs is not None:
                op.inputs = [replacements.get(int(t), int(t)) for t in op.inputs]
        if subgraph.outputs is not None:
            subgraph.outputs = [replacements.get(int(t), int(t)) for t in subgraph.outputs]
        # Named entry points must expose the same retained outputs as the subgraph.
        for signature in model.signatureDefs or []:
            if signature.subgraphIndex == 0:
                for output in signature.outputs or []:
                    output.tensorIndex = replacements.get(output.tensorIndex, output.tensorIndex)
    return groups

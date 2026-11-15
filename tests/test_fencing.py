"""Fence policy, fence planning from Vela's estimate, and the ETHOS_U_STREAM_SPLIT rewrite."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("ai_edge_litert", reason="needs the models extra")
from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.tools import flatbuffer_utils

from conftest import INT8, _buffer, _tensor, write_add_chain_model
from helia_ethos.fencing import (
    Fence,
    FenceBindingError,
    FenceBoundaries,
    FenceConfig,
    FenceMode,
    FencePlan,
    FencePlanError,
    FenceSidecarError,
    FenceSpan,
    _primary_subgraph,
    attribute_cycles,
    check_plan_binding,
    compute_boundaries,
    fence_config_for,
    fence_model,
    load_fence_boundaries,
    npu_invocation_spans,
    plan_fences,
    rewrite_model_with_fences,
)
from helia_ethos.vela import LayerRow, TFLiteOp, read_tflite_graph


def _row(name: str, op_cycles: int) -> LayerRow:
    return LayerRow(
        original_operator="Conv2D",
        nng_operator="Conv2D",
        target="NPU",
        op_cycles=op_cycles,
        npu_cycles=op_cycles,
        sram_ac=0,
        dram_ac=0,
        on_flash_ac=0,
        off_flash_ac=0,
        mac_count=0,
        network_pct_cycles=0.0,
        network_pct_mac=0.0,
        util_pct_mac=0.0,
        name=name,
    )


OPS = [TFLiteOp(index=i, opcode="CONV_2D", output_tensor_name=f"conv{i}/out") for i in range(4)]
# Vela split conv2 into two NNG ops; both fold back onto original op 2.
ROWS = [
    _row("conv0/out", 100),
    _row("conv1/out", 250),
    _row("conv2/out_lut_0", 200),
    _row("conv2/out", 100),
    _row("conv3/out", 50),
]


class TestPlanFences:
    def test_cycle_budget_starts_a_new_chunk_before_overflow(self) -> None:
        plan = plan_fences(ROWS, OPS, cycle_budget=400)
        assert plan.after_ops == (1,)
        assert "350 of 400" in plan.fences[0].reason
        assert plan.num_segments == 2

    def test_fences_carry_identity(self) -> None:
        plan = plan_fences(ROWS, OPS, cycle_budget=400, max_op_cycles=280)
        assert [(f.after_op, f.output_tensor_name, f.opcode, f.segment_index) for f in plan.fences] == [
            (1, "conv1/out", "CONV_2D", 0),
            (2, "conv2/out", "CONV_2D", 1),
        ]

    def test_large_op_is_isolated(self) -> None:
        plan = plan_fences(ROWS, OPS, max_op_cycles=280)
        assert plan.after_ops == (1, 2)
        assert "300" in plan.fences[0].reason

    def test_policies_combine_and_never_fence_after_the_last_op(self) -> None:
        plan = plan_fences(ROWS, OPS, cycle_budget=400, max_op_cycles=280, group_size=1)
        assert plan.after_ops == (0, 1, 2)

    def test_single_op_has_nothing_to_cut(self) -> None:
        assert plan_fences(ROWS[:1], OPS[:1], cycle_budget=1).fences == ()

    def test_full_coverage_is_reported(self) -> None:
        plan = plan_fences(ROWS, OPS, cycle_budget=400)
        assert plan.coverage.fraction == 1.0
        assert plan.coverage.matched_cycles == 700
        assert plan.coverage.summary() == "all 700 estimated cycles attributed"

    def test_unmatched_cost_blocks_budget_planning_by_default(self) -> None:
        rows = [*ROWS, _row("softmax_lut_that_matches_nothing", 90)]
        cycles, coverage = attribute_cycles(rows, OPS)
        assert cycles[2] == 300 and coverage.unmatched_cycles == 90
        assert coverage.fraction == pytest.approx(700 / 790)
        with pytest.raises(FencePlanError, match=r"90 of 790 estimated cycles .* 1 unmatched Vela rows"):
            plan_fences(rows, OPS, cycle_budget=400)

    def test_unknown_cost_can_be_accepted_explicitly(self) -> None:
        rows = [*ROWS, _row("softmax_lut_that_matches_nothing", 90)]
        plan = plan_fences(rows, OPS, cycle_budget=400, min_coverage=0.8)
        assert plan.after_ops == (1,)
        assert plan.coverage.unmatched_rows == (("softmax_lut_that_matches_nothing", 90),)

    def test_no_rows_is_unknown_cost_not_zero(self) -> None:
        with pytest.raises(FencePlanError, match="no Vela per-layer rows"):
            plan_fences([], OPS, cycle_budget=400)

    def test_group_policy_needs_no_cost(self) -> None:
        rows = [_row("nothing_matches", 999)]
        assert plan_fences(rows, OPS, group_size=2).after_ops == (1,)

    def test_plan_records_the_model_hash(self, tmp_path: Path) -> None:
        model_path = write_add_chain_model(tmp_path / "chain.tflite", num_ops=4)
        plan = plan_fences([], OPS, group_size=2, model=model_path)
        assert plan.model_sha256 is not None and len(plan.model_sha256) == 64

    def test_plan_round_trips_into_an_explicit_config(self) -> None:
        plan = plan_fences(ROWS, OPS, max_op_cycles=280)
        assert fence_config_for(plan) == FenceConfig(mode=FenceMode.EXPLICIT, boundaries=(1, 2))
        assert fence_config_for([Fence(after_op=2, reason="x"), Fence(after_op=0, reason="y")]).boundaries == (0, 2)
        assert fence_config_for([]) == FenceConfig(mode=FenceMode.OFF)


def test_fence_model_applies_a_plan_and_records_the_join_keys(tmp_path: Path) -> None:
    model_path = write_add_chain_model(tmp_path / "chain.tflite", num_ops=4)
    result = fence_model(model_path, [Fence(after_op=1, reason="test")], tmp_path / "out")
    assert result is not None
    assert result.boundary_indices == [1]
    assert result.num_segments == 2
    assert result.source_sha256 is not None and len(result.source_sha256) == 64
    assert result.fence_tensor_names == (("add_2", "add_2/stream_split"),)
    fenced = flatbuffer_utils.read_model(str(result.fenced_model_path))
    assert len(fenced.subgraphs[0].operators) == 5
    names = [t.name.decode() for t in fenced.subgraphs[0].tensors]
    assert "add_2/stream_split" in names
    sidecar = json.loads(result.boundaries_path.read_text())
    assert sidecar["source_sha256"] == result.source_sha256
    assert sidecar["segments"][0]["end_tensor_name"] == "add_2"
    assert sidecar["segments"][0]["fence_tensor_name"] == "add_2/stream_split"
    assert "end_tensor_name" not in sidecar["segments"][1]


class TestPlanBinding:
    """A plan carries the identity of the model it was cut for; the rewrite refuses to apply it elsewhere."""

    def test_a_bound_plan_applies_to_its_own_model(self, tmp_path: Path) -> None:
        model_path = write_add_chain_model(tmp_path / "chain.tflite", num_ops=4)
        plan = plan_fences([], read_tflite_graph(model_path), group_size=2, model=model_path)
        assert plan.model_sha256 is not None and plan.fences[0].output_tensor_name == "add_2"
        result = fence_model(model_path, plan, tmp_path / "out")
        assert result is not None and result.source_sha256 == plan.model_sha256

    def test_a_plan_cut_for_another_model_is_rejected_before_writing(self, tmp_path: Path) -> None:
        planned_on = write_add_chain_model(tmp_path / "planned.tflite", num_ops=4)
        plan = plan_fences([], read_tflite_graph(planned_on), group_size=2, model=planned_on)
        other = flatbuffer_utils.read_model(str(planned_on))
        other.subgraphs[0].tensors[2].name = b"add_2_renamed"  # same shape, different source bytes and boundary name
        other_path = tmp_path / "other.tflite"
        flatbuffer_utils.write_model(other, str(other_path))
        with pytest.raises(FenceBindingError, match="cut for a model with sha256"):
            fence_model(other_path, plan, tmp_path / "out")
        assert not (tmp_path / "out").exists()
        # Positions alone are the explicit unbound path.
        unbound = fence_model(other_path, plan.after_ops, tmp_path / "out")
        assert unbound is not None and unbound.fence_tensor_names[0][0] == "add_2_renamed"

    def test_a_plan_with_the_wrong_operator_count_is_rejected(self, tmp_path: Path) -> None:
        model_path = write_add_chain_model(tmp_path / "chain.tflite", num_ops=4)
        plan = plan_fences([], read_tflite_graph(model_path), group_size=2, model=None)
        shorter = FencePlan(plan.fences, plan.coverage, num_ops=3)
        with pytest.raises(FenceBindingError, match=r"plan counted 3 operators; chain\.tflite has 4"):
            check_plan_binding(model_path, shorter)

    def test_a_fence_whose_tensor_name_moved_is_rejected(self, tmp_path: Path) -> None:
        model_path = write_add_chain_model(tmp_path / "chain.tflite", num_ops=4)
        with pytest.raises(FenceBindingError, match=r"planned on output tensor 'add_3'; chain\.tflite has 'add_2'"):
            fence_model(model_path, [Fence(after_op=1, reason="x", output_tensor_name="add_3")], tmp_path / "out")

    def test_a_fence_planned_on_another_opcode_is_rejected(self, tmp_path: Path) -> None:
        model_path = write_add_chain_model(tmp_path / "chain.tflite", num_ops=4)
        with pytest.raises(FenceBindingError, match=r"planned on a CONV_2D; chain\.tflite has a ADD"):
            fence_model(model_path, [Fence(after_op=1, reason="x", opcode="CONV_2D")], tmp_path / "out")

    def test_a_fence_outside_the_model_is_rejected(self, tmp_path: Path) -> None:
        model_path = write_add_chain_model(tmp_path / "chain.tflite", num_ops=2)
        with pytest.raises(FenceBindingError, match=r"outside chain\.tflite"):
            check_plan_binding(model_path, [Fence(after_op=7, reason="x", output_tensor_name="add_8")])

    def test_a_bound_fence_after_the_last_operator_is_an_error_not_a_silent_drop(self, tmp_path: Path) -> None:
        model_path = write_add_chain_model(tmp_path / "chain.tflite", num_ops=4)
        with pytest.raises(FenceBindingError, match="follows the last operator"):
            fence_model(model_path, [Fence(after_op=3, reason="x", output_tensor_name="add_4", opcode="ADD")], tmp_path)
        # Positions alone keep the old behaviour: out-of-range positions are filtered, nothing to do.
        assert fence_model(model_path, [3], tmp_path / "out") is None

    def test_positions_alone_carry_nothing_to_check(self, tmp_path: Path) -> None:
        model_path = write_add_chain_model(tmp_path / "chain.tflite", num_ops=2)
        check_plan_binding(model_path, [0])
        check_plan_binding(model_path, [Fence(after_op=0, reason="x")])


def _with_existing_stream_split_opcode(path: Path, *, builtin: int, deprecated: int, version: int) -> Path:
    # Own the bytes: these fixture edits may overwrite the file on Windows.
    model = flatbuffer_utils.read_model_from_bytearray(path.read_bytes())
    oc = schema.OperatorCodeT()
    oc.customCode = b"ETHOS_U_STREAM_SPLIT"
    oc.builtinCode, oc.deprecatedBuiltinCode, oc.version = builtin, deprecated, version
    model.operatorCodes.append(oc)
    out = path.with_name(f"{path.stem}_v{version}_{builtin}_{deprecated}.tflite")
    flatbuffer_utils.write_model(model, str(out))
    return out


class TestExistingStreamSplitOpcode:
    """heliaRT's control: a same-name operator code is reused only if it is the contract's wire format."""

    @pytest.mark.parametrize(("builtin", "deprecated"), [(32, 32), (32, 0), (0, 32)])
    def test_a_contract_v1_custom_entry_is_reused(self, tmp_path: Path, builtin: int, deprecated: int) -> None:
        """Effective builtin code CUSTOM (the larger of the two schema fields, as TFLM's GetBuiltinCode) at version 1."""
        source = _with_existing_stream_split_opcode(
            write_add_chain_model(tmp_path / "chain.tflite", num_ops=3),
            builtin=builtin,
            deprecated=deprecated,
            version=1,
        )
        result = fence_model(source, [0], tmp_path / "out")
        assert result is not None
        fenced = flatbuffer_utils.read_model(str(result.fenced_model_path))
        codes = [oc for oc in fenced.operatorCodes if oc.customCode == b"ETHOS_U_STREAM_SPLIT"]
        assert len(codes) == 1
        assert (codes[0].builtinCode, codes[0].deprecatedBuiltinCode, codes[0].version) == (builtin, deprecated, 1)

    @pytest.mark.parametrize(
        ("builtin", "deprecated", "version", "shown"),
        [
            (32, 32, 2, "(32, 32, 2)"),
            (0, 0, 1, "(0, 0, 1)"),
            (0, 32, 2, "(0, 32, 2)"),
            (3, 0, 1, "(3, 0, 1)"),
            (
                5,
                32,
                1,
                "(5, 32, 1)",
            ),  # TFLM would read CUSTOM, Vela would read builtin 5: not the same op to every consumer
        ],
    )
    def test_an_incompatible_entry_is_rejected_before_writing(
        self, tmp_path: Path, builtin: int, deprecated: int, version: int, shown: str
    ) -> None:
        source = _with_existing_stream_split_opcode(
            write_add_chain_model(tmp_path / "chain.tflite", num_ops=3),
            builtin=builtin,
            deprecated=deprecated,
            version=version,
        )
        expected = re.escape(f"= {shown}; the rewrite needs an effective builtin code of CUSTOM (32")
        with pytest.raises(ValueError, match=f"already defines an ETHOS_U_STREAM_SPLIT operator code .* {expected}"):
            fence_model(source, [0], tmp_path / "out")
        assert not (tmp_path / "out").exists()


def test_rewrite_rejects_a_model_with_more_than_one_subgraph(tmp_path: Path) -> None:
    """Contract v1 covers subgraph 0 of a single-subgraph model; nothing is written for anything else."""
    model_path = write_add_chain_model(tmp_path / "chain.tflite", num_ops=3)
    # Own the bytes: these fixture edits may overwrite the file on Windows.
    model = flatbuffer_utils.read_model_from_bytearray(model_path.read_bytes())
    model.subgraphs.append(copy.deepcopy(model.subgraphs[0]))
    multi_path = tmp_path / "multi.tflite"
    flatbuffer_utils.write_model(model, str(multi_path))
    with pytest.raises(
        ValueError, match=r"has 2 subgraphs: .*multi\.tflite; the ETHOS_U_STREAM_SPLIT rewrite supports a single"
    ):
        fence_model(multi_path, [1], tmp_path / "out")
    assert not (tmp_path / "out").exists()


class TestComputeBoundaries:
    def test_off_returns_empty(self):
        assert compute_boundaries(5, FenceConfig(mode="off")) == []

    def test_per_op(self):
        # 5 ops -> fences after 0,1,2,3 (not after last)
        assert compute_boundaries(5, FenceConfig(mode="per_op")) == [0, 1, 2, 3]

    def test_per_group_size_2(self):
        # 6 ops, group_size=2 -> fences after ops 1, 3 (not after 5)
        assert compute_boundaries(6, FenceConfig(mode="per_group", group_size=2)) == [1, 3]

    def test_per_group_size_3(self):
        # 7 ops, group_size=3 -> fences after ops 2, 5
        assert compute_boundaries(7, FenceConfig(mode="per_group", group_size=3)) == [2, 5]

    def test_explicit_boundaries(self):
        config = FenceConfig(mode="per_op", boundaries=[1, 4, 10])
        # 6 ops -> boundary 10 is out of range (>= num_ops-1=5), filtered out
        assert compute_boundaries(6, config) == [1, 4]

    def test_single_op_no_boundaries(self):
        assert compute_boundaries(1, FenceConfig(mode="per_op")) == []


class TestRewriteModel:
    def test_rewrite_per_op(self, tmp_path: Path):
        model_path = tmp_path / "test.tflite"
        write_add_chain_model(model_path, num_ops=3)

        config = FenceConfig(mode="per_op")
        result = rewrite_model_with_fences(model_path, config, tmp_path / "out")

        assert result is not None
        assert result.num_segments == 3
        assert result.boundary_indices == [0, 1]
        assert result.fenced_model_path.exists()
        assert result.boundaries_path.exists()

        # Verify the fenced model is valid
        fenced = flatbuffer_utils.read_model(str(result.fenced_model_path))
        sg = fenced.subgraphs[0]

        # Original: 3 ops. With 2 fences: 3 + 2 = 5 ops total
        assert len(sg.operators) == 5

        # Check that ETHOS_U_STREAM_SPLIT opcode exists
        has_stream_split = False
        for oc in fenced.operatorCodes:
            if oc.customCode and oc.customCode.decode() == "ETHOS_U_STREAM_SPLIT":
                has_stream_split = True
                break
        assert has_stream_split

        # Check tensors increased (2 new tensors for 2 fences)
        # Original: 4 tensors (3+1 chain). Fenced: 4 + 2 = 6
        assert len(sg.tensors) == 6

    def test_rewrite_per_group(self, tmp_path: Path):
        model_path = tmp_path / "test.tflite"
        write_add_chain_model(model_path, num_ops=6)

        config = FenceConfig(mode="per_group", group_size=3)
        result = rewrite_model_with_fences(model_path, config, tmp_path / "out")

        assert result is not None
        # 6 ops, group_size=3 -> fence after op 2 -> 2 segments
        assert result.num_segments == 2
        assert result.boundary_indices == [2]

    def test_rewrite_off_returns_none(self, tmp_path: Path):
        model_path = tmp_path / "test.tflite"
        write_add_chain_model(model_path, num_ops=3)

        config = FenceConfig(mode="off")
        result = rewrite_model_with_fences(model_path, config, tmp_path / "out")
        assert result is None

    def test_boundaries_sidecar(self, tmp_path: Path):
        model_path = tmp_path / "test.tflite"
        write_add_chain_model(model_path, num_ops=4)

        config = FenceConfig(mode="per_op")
        result = rewrite_model_with_fences(model_path, config, tmp_path / "out")

        sidecar = json.loads(result.boundaries_path.read_text())
        assert sidecar["num_segments"] == 4
        assert len(sidecar["segments"]) == 4
        # First segment: ops 0-0
        assert sidecar["segments"][0]["op_range_start"] == 0
        assert sidecar["segments"][0]["op_range_end"] == 0
        # Last segment: ops 3-3
        assert sidecar["segments"][3]["op_range_start"] == 3
        assert sidecar["segments"][3]["op_range_end"] == 3

    def test_tensor_wiring_integrity(self, tmp_path: Path):
        """Verify that after rewriting, the tensor chain is unbroken."""
        model_path = tmp_path / "test.tflite"
        write_add_chain_model(model_path, num_ops=3)

        config = FenceConfig(mode="per_op")
        result = rewrite_model_with_fences(model_path, config, tmp_path / "out")

        fenced = flatbuffer_utils.read_model(str(result.fenced_model_path))
        sg = fenced.subgraphs[0]

        # Walk the operator chain and verify connectivity
        # Each op's output should be the next op's input (for this simple chain)
        for i in range(len(sg.operators) - 1):
            current_op = sg.operators[i]
            next_op = sg.operators[i + 1]
            current_output = current_op.outputs[0]
            next_inputs = list(next_op.inputs)
            assert current_output in next_inputs, (
                f"Op {i} output tensor {current_output} not found in op {i + 1} inputs {next_inputs}"
            )


def test_boundaries_sidecar_round_trips_through_its_model(tmp_path: Path) -> None:
    model_path = write_add_chain_model(tmp_path / "chain.tflite", num_ops=4)
    result = rewrite_model_with_fences(
        model_path, FenceConfig(mode=FenceMode.PER_GROUP, group_size=2), tmp_path / "out"
    )
    assert result is not None

    boundaries = load_fence_boundaries(result.boundaries_path)

    assert boundaries.source_model == model_path.name
    assert boundaries.source_sha256 == result.source_sha256
    assert boundaries.num_segments == result.num_segments
    assert boundaries.boundary_after_ops == result.boundary_indices
    assert [(s.op_range_start, s.op_range_end) for s in boundaries.segments] == [(0, 1), (2, 3)]
    assert boundaries.segments[0].fence_tensor_name == result.fence_tensor_names[0][1]
    assert boundaries.segments[-1].end_tensor_name is None
    assert json.loads(result.boundaries_path.read_text(encoding="utf-8")) == boundaries.model_dump(
        exclude_none=True, mode="json"
    )


def test_load_fence_boundaries_ignores_fields_a_newer_writer_adds(tmp_path: Path) -> None:
    path = tmp_path / "model.fenced_boundaries.json"
    path.write_text(
        json.dumps(
            {
                "num_segments": 1,
                "boundary_after_ops": [],
                "segments": [{"segment_index": 0, "op_range_start": 0, "op_range_end": 0, "later": 1}],
                "later": True,
            }
        )
    )

    assert load_fence_boundaries(path).segments[0].op_range_end == 0


def test_load_fence_boundaries_names_a_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "absent.json"
    with pytest.raises(FenceSidecarError, match=re.escape(f"{path}: cannot read fence boundaries sidecar")):
        load_fence_boundaries(path)


def test_load_fence_boundaries_names_the_file_and_the_field(tmp_path: Path) -> None:
    path = tmp_path / "model.fenced_boundaries.json"
    path.write_text(json.dumps({"num_segments": 1, "boundary_after_ops": [], "segments": [{"segment_index": 0}]}))

    with pytest.raises(
        ValueError, match=re.escape(f"{path}: not a fence boundaries sidecar (segments.0.op_range_start")
    ):
        load_fence_boundaries(path)


def test_primary_subgraph_rejects_a_model_with_no_subgraphs() -> None:
    """A structurally empty model must fail by name, not by AttributeError."""
    model = schema.ModelT()
    model.subgraphs = None
    with pytest.raises(ValueError, match="no subgraphs"):
        _primary_subgraph(model, Path("empty.tflite"))


def test_primary_subgraph_rejects_a_subgraph_with_no_operators() -> None:
    model = schema.ModelT()
    subgraph = schema.SubGraphT()
    subgraph.operators = None
    model.subgraphs = [subgraph]
    with pytest.raises(ValueError, match="no operators"):
        _primary_subgraph(model, Path("empty.tflite"))


def test_primary_subgraph_rejects_a_subgraph_with_no_tensors() -> None:
    model = schema.ModelT()
    subgraph = schema.SubGraphT()
    subgraph.operators = [schema.OperatorT()]
    subgraph.tensors = None
    model.subgraphs = [subgraph]
    with pytest.raises(ValueError, match="no tensors"):
        _primary_subgraph(model, Path("empty.tflite"))


def _write_compiled_model(path: Path, opcodes: list[str]) -> Path:
    """A model shaped like Vela's output: ``ethos-u`` and ``ETHOS_U_STREAM_SPLIT`` custom operators among CPU builtins. An
    opcode ``ETHOS_U_STREAM_SPLIT:<name>`` is a fence whose output tensor is ``<name>``."""
    model = schema.ModelT()
    model.version = 3
    kinds = [name.partition(":")[0] for name in opcodes]
    names = list(dict.fromkeys(kinds))
    model.operatorCodes = []
    for name in names:
        oc = schema.OperatorCodeT()
        if name in ("ethos-u", "ETHOS_U_STREAM_SPLIT"):
            oc.builtinCode = oc.deprecatedBuiltinCode = schema.BuiltinOperator.CUSTOM
            oc.customCode = name.encode()
        else:
            oc.builtinCode = oc.deprecatedBuiltinCode = getattr(schema.BuiltinOperator, name)
        oc.version = 1
        model.operatorCodes.append(oc)
    model.buffers = [_buffer(None) for _ in range(len(opcodes) + 2)]
    subgraph = schema.SubGraphT()
    subgraph.tensors = [_tensor("input", [1, 4], INT8, 1)] + [
        _tensor(name.partition(":")[2] or f"t{i}", [1, 4], INT8, i + 2) for i, name in enumerate(opcodes)
    ]
    subgraph.operators = []
    for index, kind in enumerate(kinds):
        op = schema.OperatorT()
        op.opcodeIndex = names.index(kind)
        op.inputs, op.outputs = [index], [index + 1]
        subgraph.operators.append(op)
    subgraph.inputs, subgraph.outputs = [0], [len(opcodes)]
    model.subgraphs = [subgraph]
    flatbuffer_utils.write_model(model, str(path))
    return path


def _boundaries(num_segments: int) -> FenceBoundaries:
    """One operator per segment, the fence after segment i named f<i>."""
    segments = [
        {"segment_index": i, "op_range_start": i, "op_range_end": i, "fence_tensor_name": f"f{i}"}
        for i in range(num_segments - 1)
    ]
    segments.append(
        {"segment_index": num_segments - 1, "op_range_start": num_segments - 1, "op_range_end": num_segments - 1}
    )
    return FenceBoundaries.model_validate(
        {"num_segments": num_segments, "boundary_after_ops": list(range(num_segments - 1)), "segments": segments}
    )


def test_npu_invocations_skip_fence_segments_left_on_the_cpu(tmp_path: Path) -> None:
    # A CPU QUANTIZE first, a CPU op mid-model, two NPU runs in the last segment around a CPU DEQUANTIZE.
    compiled = _write_compiled_model(
        tmp_path / "model.fenced_vela.tflite",
        ["QUANTIZE", "ETHOS_U_STREAM_SPLIT:f0", "ethos-u", "ETHOS_U_STREAM_SPLIT:f1", "SPACE_TO_BATCH_ND", "ETHOS_U_STREAM_SPLIT:f2",
         "ethos-u", "ETHOS_U_STREAM_SPLIT:f3", "ethos-u", "DEQUANTIZE", "ethos-u"],
    )  # fmt: skip
    assert npu_invocation_spans(compiled, _boundaries(5)) == (
        FenceSpan(1, 1),
        FenceSpan(3, 3),
        FenceSpan(4, 4),
        FenceSpan(4, 4),
    )


def test_segments_either_side_of_a_fence_vela_removed_are_one_span(tmp_path: Path) -> None:
    # The TCN seen on the FVP: Vela folded a SHAPE, STRIDED_SLICE, PACK chain into constants and removed their fences.
    compiled = _write_compiled_model(
        tmp_path / "model.fenced_vela.tflite",
        ["QUANTIZE", "ETHOS_U_STREAM_SPLIT:f0", "ethos-u", "ETHOS_U_STREAM_SPLIT:f4", "ethos-u"],
    )
    assert npu_invocation_spans(compiled, _boundaries(6)) == (FenceSpan(1, 4), FenceSpan(5, 5))


def test_a_model_with_no_npu_operators_has_no_invocations(tmp_path: Path) -> None:
    compiled = _write_compiled_model(
        tmp_path / "cpu_only.tflite", ["QUANTIZE", "ETHOS_U_STREAM_SPLIT:f0", "DEQUANTIZE"]
    )
    assert npu_invocation_spans(compiled, _boundaries(2)) == ()


@pytest.mark.parametrize(
    "opcodes",
    [
        ["ethos-u", "ETHOS_U_STREAM_SPLIT:elsewhere", "ethos-u"],  # a fence the sidecar does not name
        ["ethos-u", "ETHOS_U_STREAM_SPLIT:f1", "ethos-u", "ETHOS_U_STREAM_SPLIT:f0", "ethos-u"],  # fences out of order
    ],
)
def test_a_compiled_model_not_cut_by_the_sidecar_is_refused(tmp_path: Path, opcodes: list[str]) -> None:
    compiled = _write_compiled_model(tmp_path / "other.tflite", opcodes)
    with pytest.raises(ValueError, match="not the next fence"):
        npu_invocation_spans(compiled, _boundaries(3))


def test_a_fence_after_an_unnamed_tensor_carries_the_name_the_sidecar_records(tmp_path: Path) -> None:
    # Two unnamed outputs: each fence must still be findable by name, and the two names must differ.
    model_path = write_add_chain_model(tmp_path / "unnamed.tflite", num_ops=3)
    # Own the bytes: these fixture edits may overwrite the file on Windows.
    model = flatbuffer_utils.read_model_from_bytearray(model_path.read_bytes())
    model.subgraphs[0].tensors[1].name = b""
    model.subgraphs[0].tensors[2].name = b""
    flatbuffer_utils.write_model(model, str(model_path))

    result = rewrite_model_with_fences(model_path, FenceConfig(mode="per_op"), tmp_path / "out")
    assert result is not None
    recorded = [s.fence_tensor_name for s in load_fence_boundaries(result.boundaries_path).segments][:2]
    written = [
        op.output_tensor_name
        for op in read_tflite_graph(result.fenced_model_path)
        if op.opcode == "ETHOS_U_STREAM_SPLIT"
    ]
    assert recorded == written == ["tensor_1/stream_split", "tensor_2/stream_split"]
    assert npu_invocation_spans(result.fenced_model_path, load_fence_boundaries(result.boundaries_path)) == ()


def test_cut_carries_residual_and_multiple_outputs_across_successive_fences(tmp_path: Path) -> None:
    """All live produced values cross each cut; constants and graph inputs remain available directly."""
    path = write_add_chain_model(tmp_path / "residual.tflite", num_ops=4)
    # Own the bytes: these fixture edits may overwrite the file on Windows.
    model = flatbuffer_utils.read_model_from_bytearray(path.read_bytes())
    graph = model.subgraphs[0]
    early = int(graph.operators[0].outputs[0])
    middle = int(graph.operators[1].outputs[0])
    extra = copy.deepcopy(graph.tensors[middle])
    extra.name = b"second_output"
    extra.type = schema.TensorType.FLOAT16
    graph.tensors.append(extra)
    extra_id = len(graph.tensors) - 1
    graph.operators[1].outputs = [middle, extra_id]
    graph.operators[3].inputs = [int(graph.operators[2].outputs[0]), early, extra_id]
    flatbuffer_utils.write_model(model, str(path))
    result = fence_model(path, [1, 2], tmp_path / "out")
    assert result is not None
    rewritten = flatbuffer_utils.read_model(str(result.fenced_model_path)).subgraphs[0]
    first, second = rewritten.operators[2], rewritten.operators[4]
    assert list(first.inputs) == [middle, extra_id, early]
    assert len(second.inputs) == 3
    assert int(first.outputs[1]) in second.inputs and int(first.outputs[2]) in second.inputs
    assert set(rewritten.operators[-1].inputs) == set(second.outputs)
    for fence in (first, second):
        for source, output in zip(fence.inputs, fence.outputs, strict=True):
            a, b = rewritten.tensors[source], rewritten.tensors[output]
            assert a.type == b.type
            assert list(a.shape) == list(b.shape)
    sidecar = load_fence_boundaries(result.boundaries_path)
    assert all(len(segment.tensor_pairs or []) == 3 for segment in sidecar.segments[:-1])
    for segment, fence in zip(sidecar.segments[:-1], (first, second), strict=True):
        assert segment.tensor_pairs == [
            (rewritten.tensors[source].name.decode(), rewritten.tensors[output].name.decode())
            for source, output in zip(fence.inputs, fence.outputs, strict=True)
        ]
    assert sidecar.segments[0].end_tensor_name == graph.tensors[middle].name.decode()


@pytest.mark.parametrize("dtype", [schema.TensorType.INT16, schema.TensorType.FLOAT16])
def test_fence_preserves_extended_tensor_storage(tmp_path: Path, dtype: int) -> None:
    path = write_add_chain_model(tmp_path / "typed.tflite", num_ops=3)
    # Own the bytes: these fixture edits may overwrite the file on Windows.
    model = flatbuffer_utils.read_model_from_bytearray(path.read_bytes())
    graph = model.subgraphs[0]
    graph.tensors[graph.operators[0].outputs[0]].type = dtype
    flatbuffer_utils.write_model(model, str(path))
    result = fence_model(path, [0], tmp_path / "out")
    assert result is not None
    graph = flatbuffer_utils.read_model(str(result.fenced_model_path)).subgraphs[0]
    fence = graph.operators[1]
    assert graph.tensors[fence.inputs[0]].type == graph.tensors[fence.outputs[0]].type == dtype


def test_named_signatures_follow_early_outputs_through_multiple_cuts(tmp_path: Path) -> None:
    """Named results and subgraph outputs retain the same values after all fence rewrites."""
    path = write_add_chain_model(tmp_path / "signatures.tflite", num_ops=4)
    # Own the bytes: these fixture edits may overwrite the file on Windows.
    model = flatbuffer_utils.read_model_from_bytearray(path.read_bytes())
    graph = model.subgraphs[0]
    early = int(graph.operators[0].outputs[0])
    final = int(graph.outputs[0])
    graph.outputs = [early, final]
    signature = schema.SignatureDefT()
    signature.signatureKey = b"serve"
    signature.subgraphIndex = 0
    signature.inputs = []
    signature.outputs = []
    for name, index in ((b"early", early), (b"final", final)):
        item = schema.TensorMapT()
        item.name, item.tensorIndex = name, index
        signature.outputs.append(item)
    inp = schema.TensorMapT()
    inp.name, inp.tensorIndex = b"input", int(graph.inputs[0])
    signature.inputs.append(inp)
    model.signatureDefs = [signature, copy.deepcopy(signature)]
    model.signatureDefs[1].signatureKey = b"alias"
    flatbuffer_utils.write_model(model, str(path))
    result = fence_model(path, [0, 2], tmp_path / "out")
    assert result is not None
    rewritten = flatbuffer_utils.read_model(str(result.fenced_model_path))
    outputs = list(rewritten.subgraphs[0].outputs)
    assert outputs[0] != early and outputs[1] == final
    for named in rewritten.signatureDefs:
        assert [item.tensorIndex for item in named.outputs] == outputs
        assert [item.name for item in named.outputs] == [b"early", b"final"]
        assert named.inputs[0].tensorIndex == inp.tensorIndex


@pytest.mark.parametrize("boundary", [0, 2])
@pytest.mark.parametrize("has_graph_outputs", [True, False])
def test_unrelated_resource_ops_with_absent_vectors_do_not_break_cut(
    tmp_path: Path, boundary: int, has_graph_outputs: bool
) -> None:
    """A dense cut before a zero-input handle or after a zero-output assignment remains valid."""
    path = write_add_chain_model(tmp_path / "resource.tflite", num_ops=3)
    # Own the bytes: these fixture edits may overwrite the file on Windows.
    model = flatbuffer_utils.read_model_from_bytearray(path.read_bytes())
    graph = model.subgraphs[0]
    resource = schema.TensorT()
    resource.name, resource.type, resource.shape = b"handle", schema.TensorType.RESOURCE, []
    resource_id = len(graph.tensors)
    graph.tensors.append(resource)
    operations = []
    for builtin in (schema.BuiltinOperator.VAR_HANDLE, schema.BuiltinOperator.ASSIGN_VARIABLE):
        code = schema.OperatorCodeT()
        code.builtinCode = builtin
        code.deprecatedBuiltinCode = schema.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES
        op = schema.OperatorT()
        op.opcodeIndex = len(model.operatorCodes)
        model.operatorCodes.append(code)
        operations.append(op)
    handle, assign = operations
    handle.inputs, handle.outputs = None, [resource_id]
    handle.builtinOptionsType = schema.BuiltinOptions.VarHandleOptions
    handle.builtinOptions = schema.VarHandleOptionsT()
    handle.builtinOptions.container, handle.builtinOptions.sharedName = b"", b"state"
    assign.inputs, assign.outputs = [resource_id, int(graph.inputs[0])], None
    graph.operators[1:1] = operations
    if not has_graph_outputs:
        graph.outputs = None
    flatbuffer_utils.write_model(model, str(path))
    result = fence_model(path, [boundary], tmp_path / "out")
    assert result is not None
    rewritten = flatbuffer_utils.read_model(str(result.fenced_model_path)).subgraphs[0]
    fence = rewritten.operators[boundary + 1]
    assert len(fence.inputs) == len(fence.outputs) == 1
    assert rewritten.tensors[fence.inputs[0]].type == schema.TensorType.INT8
    if not has_graph_outputs:
        assert rewritten.outputs is None or len(rewritten.outputs) == 0

"""Vela's per-layer table, the TFLite operator list, and matching the two."""

from __future__ import annotations

from pathlib import Path

import pytest

from helia_npu.vela import (
    TFLiteOp,
    index_by_output_name,
    match_original_op,
    parse_verbose_performance,
    read_tflite_graph,
)

SAMPLE_VERBOSE_OUTPUT = """\
################################################################################
Performance for NPU Grap /tmp/vela_verbose_test/gelu_test
Original Operator    NNG Operator         Target Staging Usage  Peak% (Staging)  Op Cycles Network% (cycles)        NPU    SRAM AC    DRAM AC OnFlash AC OffFlash AC  MAC Count Network% (MAC)  Util% (MAC) Name
-------------------- -------------------- ------ ------------- ---------------- ---------- ----------------- ---------- ---------- ---------- ---------- ----------- ---------- -------------- ------------ --------------------
Conv2D               Conv2D               NPU                0           100.00       2000            66.67       2000          0        791          0           0     147456          80.00        28.80 layer1/Conv2D
100                  LUT                  NPU                0           100.00       1000            33.33       1000          0          0          0           0      36864          20.00       100.00 layer2/LUT

Network summary
Total cycles: 3000
"""


def test_parse_verbose_performance_basic():
    rows = parse_verbose_performance(SAMPLE_VERBOSE_OUTPUT)
    assert len(rows) == 2

    row0 = rows[0]
    assert row0.original_operator == "Conv2D"
    assert row0.nng_operator == "Conv2D"
    assert row0.target == "NPU"
    assert row0.op_cycles == 2000
    assert row0.npu_cycles == 2000
    assert row0.sram_ac == 0
    assert row0.dram_ac == 791
    assert row0.on_flash_ac == 0
    assert row0.off_flash_ac == 0
    assert row0.mac_count == 147456
    assert row0.network_pct_cycles == 66.67
    assert row0.network_pct_mac == 80.00
    assert row0.util_pct_mac == 28.80
    assert "layer1" in row0.name

    row1 = rows[1]
    assert row1.original_operator == "100"
    assert row1.nng_operator == "LUT"
    assert row1.op_cycles == 1000
    assert row1.mac_count == 36864


def test_parse_verbose_performance_empty():
    rows = parse_verbose_performance("no table here at all")
    assert rows == []


def test_read_tflite_graph_names_ops_by_output_tensor(conv_model: Path) -> None:
    ops = read_tflite_graph(conv_model)
    assert ops == [TFLiteOp(index=0, opcode="CONV_2D", output_tensor_name="conv/output")]


class TestMatchOriginalOp:
    ops: tuple[TFLiteOp, ...] = (
        TFLiteOp(index=0, opcode="CONV_2D", output_tensor_name="conv/output"),
        TFLiteOp(index=1, opcode="SOFTMAX", output_tensor_name="dense/BiasAdd"),
        TFLiteOp(index=2, opcode="ADD", output_tensor_name="dense/BiasAdd/extra"),
    )
    index = index_by_output_name(ops)

    def test_exact(self) -> None:
        assert match_original_op("conv/output", self.index) is self.ops[0]

    def test_known_vela_suffix_is_stripped(self) -> None:
        assert match_original_op("dense/BiasAdd_softmax_1", self.index) is self.ops[1]
        assert match_original_op("conv/output_lut_0", self.index) is self.ops[0]

    def test_longest_prefix_wins(self) -> None:
        assert match_original_op("dense/BiasAdd/extra/whatever_vela_added", self.index) is self.ops[2]
        assert match_original_op("dense/BiasAdd_something", self.index) is self.ops[1]

    def test_no_match(self) -> None:
        assert match_original_op("unrelated", self.index) is None


@pytest.mark.parametrize("count", [0, 2])
def test_read_tflite_graph_rejects_unsupported_subgraph_scope(conv_model: Path, count: int) -> None:
    from ai_edge_litert.tools import flatbuffer_utils

    model = flatbuffer_utils.read_model(str(conv_model))
    assert model.subgraphs is not None
    model.subgraphs = model.subgraphs * count
    flatbuffer_utils.write_model(model, str(conv_model))
    with pytest.raises(ValueError, match="single-subgraph"):
        read_tflite_graph(conv_model)

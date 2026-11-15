"""Reading a TFLite model: parameters, bytes, topology, command stream."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

pytest.importorskip("ai_edge_litert", reason="needs the models extra")

from conftest import COMMAND_STREAM_BYTES, CONV_BIAS_BYTES, CONV_WEIGHT_BYTES
from helia_npu.model import characterize_model, command_stream_bytes


class TestCharacterizeModel:
    def test_counts_follow_from_the_shapes(self, conv_model: Path) -> None:
        mc = characterize_model(conv_model)
        assert mc.weight_params == CONV_WEIGHT_BYTES  # int8: one byte per element
        assert mc.weight_bytes == CONV_WEIGHT_BYTES
        assert mc.bias_params == CONV_BIAS_BYTES // 4
        assert mc.bias_bytes == CONV_BIAS_BYTES
        assert mc.other_const_bytes == COMMAND_STREAM_BYTES
        assert mc.total_params == mc.weight_params + mc.bias_params + mc.other_const_params
        assert mc.total_const_bytes == mc.weight_bytes + mc.bias_bytes + mc.other_const_bytes
        assert mc.weight_dtype == "int8"
        assert mc.weight_dtype_bytes == 1

    def test_topology_and_io(self, conv_model: Path) -> None:
        mc = characterize_model(conv_model)
        assert mc.num_operators == 1
        assert mc.num_tensors == 5
        assert mc.operator_counts == {"CONV_2D": 1}
        assert mc.input_shapes == [[1, 4, 4, 2]]
        assert mc.output_shapes == [[1, 4, 4, 4]]
        assert mc.input_dtypes == ["int8"]
        assert mc.output_dtypes == ["int8"]
        assert mc.file_size_bytes == conv_model.stat().st_size

    def test_summary_dict_uses_the_prefix(self, conv_model: Path) -> None:
        sd = characterize_model(conv_model).as_summary_dict(prefix="tflite_")
        assert all(key.startswith("tflite_") for key in sd)
        assert sd["tflite_input_shapes"] == "[1,4,4,2]"
        assert sd["tflite_weight_bytes"] == CONV_WEIGHT_BYTES

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            characterize_model(tmp_path / "nope.tflite")

    def test_not_a_flatbuffer(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.tflite"
        bad.write_bytes(b"not a flatbuffer")
        with pytest.raises(struct.error):
            characterize_model(bad)


class TestCommandStreamBytes:
    def test_reports_the_command_stream_buffer(self, conv_model: Path) -> None:
        assert command_stream_bytes(conv_model) == COMMAND_STREAM_BYTES

    def test_zero_without_a_command_stream(self, conv_model_without_command_stream: Path) -> None:
        assert command_stream_bytes(conv_model_without_command_stream) == 0

    def test_zero_for_garbage(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.tflite"
        bad.write_bytes(b"garbage")
        assert command_stream_bytes(bad) == 0


def test_model_io_preserves_concrete_shapes_and_storage(conv_model: Path):
    from helia_npu.model import model_io

    inputs, outputs = model_io(conv_model)
    assert [(t.index, t.dtype, t.shape) for t in inputs] == [(0, "int8", (1, 4, 4, 2))]
    assert [(t.index, t.dtype, t.shape) for t in outputs] == [(0, "int8", (1, 4, 4, 4))]
    assert inputs[0].name and outputs[0].name

"""AOT specialization cannot change command bytes or unrelated CPU signatures."""

from pathlib import Path

import pytest

from conftest import flatbuffer_utils, schema
from helia_ethos.vela.static_shapes import has_ethosu_stream, specialize_ethosu_output_shapes


@pytest.mark.parametrize("legacy", [False, True])
def test_only_compiled_ethosu_output_signatures_are_specialized(conv_model: Path, tmp_path: Path, legacy: bool) -> None:
    assert flatbuffer_utils is not None and schema is not None
    model = flatbuffer_utils.read_model_from_bytearray(conv_model.read_bytes())
    code = model.operatorCodes[0]
    code.builtinCode = 0 if legacy else schema.BuiltinOperator.CUSTOM
    code.deprecatedBuiltinCode = schema.BuiltinOperator.CUSTOM
    code.customCode = b"ethos-u"
    model.subgraphs[0].tensors[3].shapeSignature = [-1, 4, 4, 4]
    model.subgraphs[0].tensors[0].shapeSignature = [-1, 4, 4, 2]
    flatbuffer_utils.write_model(model, str(conv_model))
    model = flatbuffer_utils.read_model_from_bytearray(conv_model.read_bytes())
    assert has_ethosu_stream(conv_model)
    original = conv_model.read_bytes()
    output = tmp_path / "static.tflite"
    assert specialize_ethosu_output_shapes(conv_model, output) == output
    assert conv_model.read_bytes() == original
    actual = flatbuffer_utils.read_model(str(output))
    assert list(actual.subgraphs[0].tensors[3].shapeSignature) == [1, 4, 4, 4]
    assert list(actual.subgraphs[0].tensors[0].shapeSignature) == [-1, 4, 4, 2]
    for before, after in zip(model.buffers, actual.buffers, strict=True):
        assert (before.data is None and after.data is None) or list(before.data) == list(after.data)


def test_cpu_model_is_returned_without_rewriting(conv_model: Path, tmp_path: Path) -> None:
    assert flatbuffer_utils is not None
    model = flatbuffer_utils.read_model_from_bytearray(conv_model.read_bytes())
    model.subgraphs[0].tensors[3].shapeSignature = [-1, 4, 4, 4]
    flatbuffer_utils.write_model(model, str(conv_model))
    output = tmp_path / "static.tflite"
    assert specialize_ethosu_output_shapes(conv_model, output) == conv_model
    assert not output.exists()


def test_inconsistent_npu_shape_is_rejected(conv_model: Path, tmp_path: Path) -> None:
    assert flatbuffer_utils is not None and schema is not None
    model = flatbuffer_utils.read_model_from_bytearray(conv_model.read_bytes())
    model.operatorCodes[0].builtinCode = schema.BuiltinOperator.CUSTOM
    model.operatorCodes[0].customCode = b"ethos-u"
    model.subgraphs[0].tensors[3].shapeSignature = [-1, 99, 4, 4]
    flatbuffer_utils.write_model(model, str(conv_model))
    with pytest.raises(ValueError, match="no consistent concrete shape"):
        specialize_ethosu_output_shapes(conv_model, tmp_path / "static.tflite")


@pytest.mark.parametrize("custom", [b"ethos-u", b"ETHOS_U_STREAM_SPLIT"])
def test_stream_detection_uses_invoked_operator_in_any_subgraph(conv_model: Path, custom: bytes) -> None:
    from copy import deepcopy

    assert flatbuffer_utils is not None and schema is not None
    assert not has_ethosu_stream(conv_model)
    model = flatbuffer_utils.read_model_from_bytearray(conv_model.read_bytes())
    model.operatorCodes.append(deepcopy(model.operatorCodes[0]))
    model.operatorCodes[1].builtinCode = schema.BuiltinOperator.CUSTOM
    model.operatorCodes[1].deprecatedBuiltinCode = schema.BuiltinOperator.CUSTOM
    model.operatorCodes[1].customCode = custom
    flatbuffer_utils.write_model(model, str(conv_model))
    assert not has_ethosu_stream(conv_model)  # Unused opcode entries are not invocations.
    model.subgraphs.append(deepcopy(model.subgraphs[0]))
    model.subgraphs[1].operators[0].opcodeIndex = 1
    flatbuffer_utils.write_model(model, str(conv_model))
    assert has_ethosu_stream(conv_model) is (custom == b"ethos-u")

"""Synthetic TFLite models for the package tests: no model files, exact expectations."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import SkipTest

import pytest

# The model-writing helpers need the models extra; the core tests do not, so the
# suite collects without it and the helpers skip the tests that call them.
try:
    from ai_edge_litert import schema_py_generated as _schema
    from ai_edge_litert.tools import flatbuffer_utils as _flatbuffer_utils

    schema: Any = _schema
    flatbuffer_utils: Any = _flatbuffer_utils
except ImportError:  # pragma: no cover - exercised only on a core-only install
    schema = None
    flatbuffer_utils = None

# The schema constants the model writers use; zero on a core-only install, where no writer runs.
INT8 = schema.TensorType.INT8 if schema else 0
INT32 = schema.TensorType.INT32 if schema else 0
UINT8 = schema.TensorType.UINT8 if schema else 0
CONV_2D = schema.BuiltinOperator.CONV_2D if schema else 0

# One CONV_2D: input [1,4,4,2] -> output [1,4,4,4] with a 3x3x2x4 int8 kernel and
# an int32 bias, plus an unattached uint8 constant named like Vela's command
# stream. Every byte count below follows from these shapes.
CONV_WEIGHT_BYTES = 4 * 3 * 3 * 2  # 72 int8 elements
CONV_BIAS_BYTES = 4 * 4  # 4 int32 elements
COMMAND_STREAM_BYTES = 100


def _tensor(name: str, shape: list[int], ttype: int, buffer: int) -> schema.TensorT:
    t = schema.TensorT()
    t.name = name.encode()
    t.shape = shape
    t.type = ttype
    t.buffer = buffer
    t.isVariable = False
    q = schema.QuantizationParametersT()
    q.scale = [0.00390625]
    q.zeroPoint = [0]
    t.quantization = q
    return t


def _buffer(data: bytes | None) -> schema.BufferT:
    b = schema.BufferT()
    b.data = list(data) if data is not None else None
    return b


def _need_models_extra() -> None:
    if schema is None:  # pytest reports unittest's SkipTest as a skip
        raise SkipTest("needs the models extra (ai-edge-litert)")


def write_conv_model(path: Path, *, with_command_stream: bool = True) -> Path:
    _need_models_extra()
    model = schema.ModelT()
    model.version = 3
    model.description = b"helia-npu test model"
    model.buffers = [
        _buffer(None),  # 0: the empty sentinel buffer
        _buffer(None),  # 1: input (no data)
        _buffer(bytes(range(CONV_WEIGHT_BYTES))),  # 2: weights
        _buffer(bytes(CONV_BIAS_BYTES)),  # 3: bias
        _buffer(None),  # 4: output
        _buffer(bytes(COMMAND_STREAM_BYTES)),  # 5: command stream
    ]
    oc = schema.OperatorCodeT()
    oc.builtinCode = CONV_2D
    oc.deprecatedBuiltinCode = CONV_2D
    oc.version = 1
    model.operatorCodes = [oc]

    sg = schema.SubGraphT()
    sg.name = b"main"
    sg.tensors = [
        _tensor("input", [1, 4, 4, 2], INT8, 1),
        _tensor("conv/weights", [4, 3, 3, 2], INT8, 2),
        _tensor("conv/bias", [4], INT32, 3),
        _tensor("conv/output", [1, 4, 4, 4], INT8, 4),
    ]
    if with_command_stream:
        sg.tensors.append(_tensor("ethos_u_command_stream", [COMMAND_STREAM_BYTES], UINT8, 5))
    op = schema.OperatorT()
    op.opcodeIndex = 0
    op.inputs = [0, 1, 2]
    op.outputs = [3]
    op.builtinOptionsType = 0
    sg.operators = [op]
    sg.inputs = [0]
    sg.outputs = [3]
    model.subgraphs = [sg]
    flatbuffer_utils.write_model(model, str(path))
    return path


@pytest.fixture
def conv_model(tmp_path: Path) -> Path:
    return write_conv_model(tmp_path / "conv.tflite")


@pytest.fixture
def conv_model_without_command_stream(tmp_path: Path) -> Path:
    return write_conv_model(tmp_path / "conv_plain.tflite", with_command_stream=False)


# ---------------------------------------------------------------------------
# A SoC that is AT110 as the NPU sees it with the spec's timing (platform v7 values:
# SSRAM 64 ns / 8 outstanding). Kept at those numbers so the tests have a port 1 that
# is nominally slower and deeper than port 0, which the HP ini clamp acts on.
# ---------------------------------------------------------------------------

AT110_SOC_YAML = """
name: AT110
id: atomiq110
origin: golden
npu: ethos-u85
macs: 256
axi_bus_width_bits: 128
modes:
  ULP: {npu_clock_hz: 100e6}
  HP: {npu_clock_hz: 500e6}
memories:
  ssram:
    axi_port: 0
    axi_clock_hz: 250e6
    read_latency_ns: 64
    write_latency_ns: 64
    max_outstanding: 8
    burst_length: 256
    ports_used: 2
    provenance: {status: confirmed, source: Atomiq110 SoC specification}
  hblram:
    axi_port: 1
    axi_clock_hz: 250e6
    read_latency_ns: 88
    write_latency_ns: 88
    max_outstanding: 12
    burst_length: 256
    ports_used: 1
    provenance: {status: unverified, source: architect estimate}
"""


@pytest.fixture
def at110_soc_file(tmp_path: Path) -> Path:
    path = tmp_path / "at110_soc.yaml"
    path.write_text(AT110_SOC_YAML, encoding="utf-8")
    return path


def write_add_chain_model(path: Path, num_ops: int = 2) -> Path:
    _need_models_extra()
    """A chain of int8 ADDs Vela compiles for the Ethos-U (no weights, elementwise)."""
    model = schema.ModelT()
    model.version = 3
    model.description = b"helia-npu add chain"
    model.buffers = [_buffer(None) for _ in range(num_ops + 2)]
    oc = schema.OperatorCodeT()
    oc.builtinCode = schema.BuiltinOperator.ADD
    oc.deprecatedBuiltinCode = schema.BuiltinOperator.ADD
    oc.version = 1
    model.operatorCodes = [oc]
    sg = schema.SubGraphT()
    sg.name = b"main"
    sg.tensors = [_tensor(f"add_{i}", [1, 4, 4, 2], INT8, i + 1) for i in range(num_ops + 1)]
    sg.operators = []
    for i in range(num_ops):
        op = schema.OperatorT()
        op.opcodeIndex = 0
        op.inputs = [i, i]
        op.outputs = [i + 1]
        op.builtinOptionsType = schema.BuiltinOptions.AddOptions
        options = schema.AddOptionsT()
        options.fusedActivationFunction = 0
        op.builtinOptions = options
        sg.operators.append(op)
    sg.inputs = [0]
    sg.outputs = [num_ops]
    model.subgraphs = [sg]
    flatbuffer_utils.write_model(model, str(path))
    return path


@pytest.fixture
def add_chain_model(tmp_path: Path) -> Path:
    return write_add_chain_model(tmp_path / "add_chain.tflite")

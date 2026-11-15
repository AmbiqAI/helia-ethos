"""Read what a TFLite model is, straight from the flatbuffer.

Parameter counts, weight / bias / constant byte breakdowns, operator
summaries, input and output tensor metadata, and the size of the Ethos-U
command stream Vela embedded. Independent of model cards or documentation, and
of any tool that consumes the numbers.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from ._optional import litert_schema


@cache
def _weight_bearing_ops() -> frozenset[int]:
    builtin = litert_schema().BuiltinOperator
    return frozenset({builtin.CONV_2D, builtin.DEPTHWISE_CONV_2D, builtin.FULLY_CONNECTED, builtin.TRANSPOSE_CONV})


@cache
def _type_byte_width() -> dict[int, int]:
    tensor_type = litert_schema().TensorType
    return {
        tensor_type.INT8: 1,
        tensor_type.UINT8: 1,
        tensor_type.INT16: 2,
        tensor_type.INT32: 4,
        tensor_type.INT64: 8,
        tensor_type.FLOAT16: 2,
        tensor_type.FLOAT32: 4,
        tensor_type.FLOAT64: 8,
        tensor_type.BOOL: 1,
    }


@cache
def _type_names() -> dict[int, str]:
    tensor_type = litert_schema().TensorType
    return {
        tensor_type.INT8: "int8",
        tensor_type.UINT8: "uint8",
        tensor_type.INT16: "int16",
        tensor_type.INT32: "int32",
        tensor_type.INT64: "int64",
        tensor_type.FLOAT16: "float16",
        tensor_type.FLOAT32: "float32",
        tensor_type.FLOAT64: "float64",
        tensor_type.BOOL: "bool",
    }


@dataclass(frozen=True)
class ModelCharacterization:
    """Accurate model metrics derived directly from the TFLite flatbuffer."""

    # Identity
    file_path: str
    file_size_bytes: int

    # Parameter counts (element counts, NOT byte counts)
    weight_params: int  # Conv/FC weight tensor elements
    bias_params: int  # Conv/FC bias tensor elements
    other_const_params: int  # All other constant tensor elements
    total_params: int  # weight + bias + other_const

    # Byte counts
    weight_bytes: int  # Raw weight data bytes in flatbuffer
    bias_bytes: int  # Raw bias data bytes
    other_const_bytes: int  # Other constant data bytes
    total_const_bytes: int  # weight + bias + other_const

    # Quantization info
    weight_dtype: str  # Dominant weight data type (e.g. "int8")
    weight_dtype_bytes: int  # Bytes per weight element

    # Topology
    num_operators: int
    num_tensors: int
    operator_counts: dict[str, int] = field(default_factory=dict)

    # I/O
    input_shapes: list[list[int]] = field(default_factory=list)
    output_shapes: list[list[int]] = field(default_factory=list)
    input_dtypes: list[str] = field(default_factory=list)
    output_dtypes: list[str] = field(default_factory=list)

    def as_summary_dict(self, prefix: str = "model_") -> dict[str, Any]:
        """Return a flat dict suitable for merging into a pipeline summary row."""
        return {
            f"{prefix}file_size_bytes": self.file_size_bytes,
            f"{prefix}total_params": self.total_params,
            f"{prefix}weight_params": self.weight_params,
            f"{prefix}bias_params": self.bias_params,
            f"{prefix}other_const_params": self.other_const_params,
            f"{prefix}weight_bytes": self.weight_bytes,
            f"{prefix}bias_bytes": self.bias_bytes,
            f"{prefix}other_const_bytes": self.other_const_bytes,
            f"{prefix}total_const_bytes": self.total_const_bytes,
            f"{prefix}weight_dtype": self.weight_dtype,
            f"{prefix}num_operators": self.num_operators,
            f"{prefix}num_tensors": self.num_tensors,
            f"{prefix}input_shapes": _format_shapes(self.input_shapes),
            f"{prefix}output_shapes": _format_shapes(self.output_shapes),
            f"{prefix}input_dtypes": ";".join(self.input_dtypes),
            f"{prefix}output_dtypes": ";".join(self.output_dtypes),
        }


def characterize_model(tflite_path: Path) -> ModelCharacterization:
    """Analyze a TFLite model file and return accurate characterization metrics.

    Iterates all subgraphs to ensure complete parameter counts for models with
    control flow or separate pre/post-processing subgraphs.
    """
    schema = litert_schema()  # before the read, so a missing extra is reported ahead of a missing file
    buf = tflite_path.read_bytes()
    file_size = len(buf)
    model = schema.Model.GetRootAs(buf, 0)

    # Resolve builtin operator codes (shared across all subgraphs)
    op_codes: list[int] = []
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
        op_codes.append(code)

    # Accumulate across ALL subgraphs
    weight_bytes = 0
    weight_params = 0
    bias_bytes = 0
    bias_params = 0
    other_const_bytes = 0
    other_const_params = 0
    weight_type_counts: dict[int, int] = {}
    operator_counts: dict[str, int] = {}
    total_operators = 0
    total_tensors = 0
    # Track buffer indices already counted (buffers can be shared across subgraphs)
    counted_buffers: set[int] = set()

    for sg_idx in range(model.SubgraphsLength()):
        sg = model.Subgraphs(sg_idx)
        total_operators += sg.OperatorsLength()
        total_tensors += sg.TensorsLength()

        # Identify weight and bias tensor indices from weight-bearing ops
        weight_indices: set[int] = set()
        bias_indices: set[int] = set()

        for op_idx in range(sg.OperatorsLength()):
            op = sg.Operators(op_idx)
            opcode = op_codes[op.OpcodeIndex()]

            op_name = _opcode_name(opcode)
            operator_counts[op_name] = operator_counts.get(op_name, 0) + 1

            if opcode in _weight_bearing_ops():
                if op.InputsLength() >= 2:
                    weight_indices.add(op.Inputs(1))
                # TRANSPOSE_CONV inserts output_shape before weights/input/bias.
                bias_slot = 3 if opcode == schema.BuiltinOperator.TRANSPOSE_CONV else 2
                if op.InputsLength() > bias_slot and op.Inputs(bias_slot) != -1:
                    bias_indices.add(op.Inputs(bias_slot))

        # I/O tensor indices for this subgraph (excluded from constant counting)
        input_indices = set(sg.Inputs(i) for i in range(sg.InputsLength()))
        output_indices = set(sg.Outputs(i) for i in range(sg.OutputsLength()))

        # Categorize constant tensors
        for ti in range(sg.TensorsLength()):
            t = sg.Tensors(ti)
            buf_idx = t.Buffer()
            if buf_idx in counted_buffers:
                continue
            buf_obj = model.Buffers(buf_idx)
            data_len = buf_obj.DataLength()
            if data_len == 0 or ti in input_indices or ti in output_indices:
                continue

            counted_buffers.add(buf_idx)
            ttype = t.Type()
            elem_size = _type_byte_width().get(ttype, 1)
            num_elems = data_len // elem_size if elem_size > 0 else data_len

            if ti in weight_indices:
                weight_bytes += data_len
                weight_params += num_elems
                weight_type_counts[ttype] = weight_type_counts.get(ttype, 0) + num_elems
            elif ti in bias_indices:
                bias_bytes += data_len
                bias_params += num_elems
            else:
                other_const_bytes += data_len
                other_const_params += num_elems

    # Determine dominant weight dtype
    dominant_type = (
        max(weight_type_counts, key=lambda t: weight_type_counts[t])
        if weight_type_counts
        else litert_schema().TensorType.INT8
    )
    weight_dtype = _type_names().get(dominant_type, f"type_{dominant_type}")
    weight_dtype_bytes = _type_byte_width().get(dominant_type, 1)

    # I/O shapes and dtypes from subgraph 0 (the primary/entry subgraph)
    sg = model.Subgraphs(0)
    input_shapes = []
    input_dtypes = []
    for i in range(sg.InputsLength()):
        t = sg.Tensors(sg.Inputs(i))
        shape = [t.Shape(j) for j in range(t.ShapeLength())]
        input_shapes.append(shape)
        input_dtypes.append(_type_names().get(t.Type(), f"type_{t.Type()}"))

    output_shapes = []
    output_dtypes = []
    for i in range(sg.OutputsLength()):
        t = sg.Tensors(sg.Outputs(i))
        shape = [t.Shape(j) for j in range(t.ShapeLength())]
        output_shapes.append(shape)
        output_dtypes.append(_type_names().get(t.Type(), f"type_{t.Type()}"))

    total_params = weight_params + bias_params + other_const_params
    total_const_bytes = weight_bytes + bias_bytes + other_const_bytes

    return ModelCharacterization(
        file_path=str(tflite_path),
        file_size_bytes=file_size,
        weight_params=weight_params,
        bias_params=bias_params,
        other_const_params=other_const_params,
        total_params=total_params,
        weight_bytes=weight_bytes,
        bias_bytes=bias_bytes,
        other_const_bytes=other_const_bytes,
        total_const_bytes=total_const_bytes,
        weight_dtype=weight_dtype,
        weight_dtype_bytes=weight_dtype_bytes,
        num_operators=total_operators,
        num_tensors=total_tensors,
        operator_counts=operator_counts,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        input_dtypes=input_dtypes,
        output_dtypes=output_dtypes,
    )


def command_stream_bytes(tflite_path: Path) -> int:
    """Return the Vela Ethos-U command-stream buffer size in bytes."""
    try:
        schema = litert_schema()  # before the read, so a missing extra is reported ahead of a missing file
        model = schema.Model.GetRootAs(tflite_path.read_bytes(), 0)
        for subgraph_index in range(model.SubgraphsLength()):
            subgraph = model.Subgraphs(subgraph_index)
            for tensor_index in range(subgraph.TensorsLength()):
                tensor = subgraph.Tensors(tensor_index)
                if tensor.Name() == b"ethos_u_command_stream":
                    return model.Buffers(tensor.Buffer()).DataLength()
    except (OSError, IndexError, struct.error):
        return 0
    return 0


def _opcode_name(code: int) -> str:
    """Best-effort human-readable name for a builtin opcode."""
    # Reverse lookup from the BuiltinOperator constants
    builtin = litert_schema().BuiltinOperator
    for attr in dir(builtin):
        if attr.startswith("_"):
            continue
        if getattr(builtin, attr) == code:
            return attr
    return f"CUSTOM_{code}"


def _format_shapes(shapes: list[list[int]]) -> str:
    """Format list of shapes for CSV: e.g. '[1,49,10];[1,12]'"""
    return ";".join("[" + ",".join(str(d) for d in s) + "]" for s in shapes)


@dataclass(frozen=True)
class ModelTensor:
    """An entry-subgraph tensor's ordinal, name, scalar storage type and fixed dimensions."""

    index: int
    name: str
    dtype: str
    shape: tuple[int, ...]


def model_io(tflite_path: Path) -> tuple[tuple[ModelTensor, ...], tuple[ModelTensor, ...]]:
    """Return concrete entry-subgraph tensor shapes; a shape signature never resizes this capture."""
    schema = litert_schema()
    model = schema.Model.GetRootAs(tflite_path.read_bytes(), 0)
    if model.SubgraphsLength() == 0:
        raise ValueError("model has no entry subgraph")
    graph = model.Subgraphs(0)
    groups = []
    for label in ("Inputs", "Outputs"):
        tensors = []
        for index in range(getattr(graph, label + "Length")()):
            tensor = graph.Tensors(getattr(graph, label)(index))
            shape = tuple(int(tensor.Shape(i)) for i in range(tensor.ShapeLength()))
            if any(n <= 0 for n in shape):
                raise ValueError(f"{label.lower()}[{index}] has dynamic or empty dimensions")
            dtype = _type_names().get(tensor.Type())
            if dtype is None:
                raise ValueError(f"{label.lower()}[{index}] uses unsupported storage type {tensor.Type()}")
            name = tensor.Name()
            tensors.append(
                ModelTensor(index, name.decode("utf-8") if name else f"{label.lower()}_{index}", dtype, shape)
            )
        if not tensors:
            raise ValueError(f"model has no {label.lower()}")
        groups.append(tuple(tensors))
    return groups[0], groups[1]

"""Specialize the fixed tensor envelopes of already-compiled Vela command streams."""

from __future__ import annotations

from pathlib import Path

from .._optional import litert_schema


def specialize_ethosu_output_shapes(model_path: Path, output_path: Path) -> Path:
    """Return an AOT handoff copy with static Ethos-U output signatures, or the unchanged input path.

    Vela compiles a command stream for the concrete tensor shapes but can retain
    dynamic shape signatures from the original graph. Those signatures cannot
    describe a resizable compiled NPU invocation. Only Ethos-U outputs are
    specialized; CPU operators and all tensor/command-stream payloads are preserved.
    """
    schema = litert_schema()
    data = bytearray(model_path.read_bytes())
    model = schema.Model.GetRootAs(data)
    changed = False
    for graph_index in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(graph_index)
        if subgraph is None:
            continue
        for operation_index in range(subgraph.OperatorsLength()):
            operation = subgraph.Operators(operation_index)
            if operation is None:
                continue
            code = model.OperatorCodes(operation.OpcodeIndex())
            if (
                code is None
                or (code.BuiltinCode() or code.DeprecatedBuiltinCode()) != schema.BuiltinOperator.CUSTOM
                or code.CustomCode() != b"ethos-u"
            ):
                continue
            for output_index in range(operation.OutputsLength()):
                index = operation.Outputs(output_index)
                tensor = subgraph.Tensors(index)
                if tensor is None:
                    raise ValueError(f"Vela output tensor {index} is missing")
                if tensor.ShapeSignatureLength() == 0:
                    continue
                signature = tensor.ShapeSignatureAsNumpy()
                if not any(dim < 0 for dim in signature):
                    continue
                shape = tensor.ShapeAsNumpy()
                if (
                    tensor.ShapeLength() != tensor.ShapeSignatureLength()
                    or any(dim <= 0 for dim in shape)
                    or any(old >= 0 and old != new for old, new in zip(signature, shape, strict=True))
                ):
                    raise ValueError(f"Vela output tensor {index} has no consistent concrete shape")
                # Patch only the existing signature vector; all flatbuffer offsets and payloads stay intact.
                signature[:] = shape
                changed = True
    if not changed:
        return model_path
    if output_path.resolve() == model_path.resolve():
        raise ValueError("The specialized Vela handoff must use a separate output path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return output_path


def has_ethosu_stream(model_path: Path) -> bool:
    """Return whether any subgraph invokes a compiled Ethos-U command stream."""
    schema = litert_schema()
    model = schema.Model.GetRootAs(model_path.read_bytes())
    for graph_index in range(model.SubgraphsLength()):
        graph = model.Subgraphs(graph_index)
        for index in range(graph.OperatorsLength()):
            code = model.OperatorCodes(graph.Operators(index).OpcodeIndex())
            if (
                code.BuiltinCode() or code.DeprecatedBuiltinCode()
            ) == schema.BuiltinOperator.CUSTOM and code.CustomCode() == b"ethos-u":
                return True
    return False

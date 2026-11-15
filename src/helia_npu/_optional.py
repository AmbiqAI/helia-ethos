"""Imports behind an extra, with an error that says which extra to install."""

from __future__ import annotations

from types import ModuleType


def litert_schema() -> ModuleType:
    """``ai_edge_litert.schema_py_generated``, or an ImportError naming the ``models`` extra."""
    try:
        from ai_edge_litert import schema_py_generated
    except ImportError as exc:
        raise ImportError("reading .tflite files needs the 'models' extra: pip install 'helia-npu[models]'") from exc
    return schema_py_generated

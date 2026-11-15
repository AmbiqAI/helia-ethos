"""Vela's per-layer view of a model, and how it maps back to the TFLite graph.

Parses the ``--verbose-performance`` table Vela prints, reads the operator
list of the original ``.tflite``, and matches the two by the naming rules Vela
uses. Nothing here knows about measured cycles; calibrating Vela's estimate
against a simulator or silicon is the caller's business.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .._optional import litert_schema


@dataclass
class LayerRow:
    original_operator: str
    nng_operator: str
    target: str
    op_cycles: int
    npu_cycles: int
    sram_ac: int
    dram_ac: int
    on_flash_ac: int
    off_flash_ac: int
    mac_count: int
    network_pct_cycles: float
    network_pct_mac: float
    util_pct_mac: float
    name: str


_PERF_TABLE_HEADER_RE = re.compile(r"^Original Operator\s+NNG Operator\s+Target\s+Staging Usage")
_SEPARATOR_RE = re.compile(r"^-{10,}")

# Canonical column names used for header-driven lookup.
# Keys = normalized header text, Values = field name used in LayerRow.
_COLUMN_MAP: dict[str, str] = {
    "original operator": "original_operator",
    "nng operator": "nng_operator",
    "target": "target",
    "staging usage": "staging_usage",
    "peak% (staging)": "peak_pct_staging",
    "op cycles": "op_cycles",
    "network% (cycles)": "network_pct_cycles",
    "npu": "npu_cycles",
    "sram ac": "sram_ac",
    "dram ac": "dram_ac",
    "onflash ac": "on_flash_ac",
    "offflash ac": "off_flash_ac",
    "mac count": "mac_count",
    "network% (mac)": "network_pct_mac",
    "util% (mac)": "util_pct_mac",
    "name": "name",
}


def parse_verbose_performance(text: str) -> list[LayerRow]:
    """Parse the verbose-performance table from Vela stdout text.

    Returns a list of LayerRow for each operator row in the table.
    """
    lines = text.splitlines()
    rows: list[LayerRow] = []

    # Find the header line and separator to determine column positions.
    header_idx = None
    for i, line in enumerate(lines):
        if _PERF_TABLE_HEADER_RE.match(line):
            header_idx = i
            break

    if header_idx is None:
        return rows

    # The separator line (dashes) is one line after the header.
    sep_idx = header_idx + 1
    if sep_idx >= len(lines) or not _SEPARATOR_RE.match(lines[sep_idx]):
        return rows

    # Determine column boundaries from the separator line.
    sep_line = lines[sep_idx]
    col_spans = _parse_column_spans(sep_line)

    # Parse header text within each column span to get field names.
    header_line = lines[header_idx]
    col_names = _parse_header_names(header_line, col_spans)

    # Parse data rows (lines after separator until blank or non-matching).
    for i in range(sep_idx + 1, len(lines)):
        line = lines[i]
        if not line.strip() or line.startswith("Network summary") or line.startswith("#"):
            break
        row = _parse_data_row(line, col_spans, col_names)
        if row is not None:
            rows.append(row)

    return rows


def _parse_column_spans(sep_line: str) -> list[tuple[int, int]]:
    """Parse dash groups to find (start, end) of each column."""
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(sep_line):
        if sep_line[i] == "-":
            start = i
            while i < len(sep_line) and sep_line[i] == "-":
                i += 1
            spans.append((start, i))
        else:
            i += 1
    return spans


def _parse_header_names(header_line: str, col_spans: list[tuple[int, int]]) -> list[str]:
    """Extract and normalize column names from the header line.

    Returns a list of canonical field names (from _COLUMN_MAP) for each column.
    Unknown columns get their raw header text as the name.
    """
    names: list[str] = []
    for start, end in col_spans:
        raw = header_line[start : min(end, len(header_line))].strip().lower()
        canonical = _COLUMN_MAP.get(raw, raw)
        names.append(canonical)
    return names


def _parse_data_row(
    line: str,
    col_spans: list[tuple[int, int]],
    col_names: list[str],
) -> LayerRow | None:
    """Extract fields from a data line using column boundaries and named columns."""
    # Extract raw field values by position.
    raw_fields: list[str] = []
    for start, end in col_spans:
        if start < len(line):
            raw_fields.append(line[start : min(end, len(line))].strip())
        else:
            raw_fields.append("")

    # The last column (Name) extends to end of line — Vela's separator
    # for the final column is shorter than the actual data, so we take
    # everything from the end of the penultimate column's separator.
    if len(col_spans) >= 2:
        last_start = col_spans[-2][1]
        raw_fields[-1] = line[last_start:].strip()
    elif col_spans:
        raw_fields[-1] = line[col_spans[-1][0] :].strip()

    # Build a named dict from col_names → raw_fields.
    if len(raw_fields) != len(col_names):
        return None
    fields: dict[str, str] = dict(zip(col_names, raw_fields, strict=True))

    try:
        return LayerRow(
            original_operator=fields.get("original_operator", ""),
            nng_operator=fields.get("nng_operator", ""),
            target=fields.get("target", ""),
            op_cycles=_safe_int(fields.get("op_cycles", "0")),
            npu_cycles=_safe_int(fields.get("npu_cycles", "0")),
            sram_ac=_safe_int(fields.get("sram_ac", "0")),
            dram_ac=_safe_int(fields.get("dram_ac", "0")),
            on_flash_ac=_safe_int(fields.get("on_flash_ac", "0")),
            off_flash_ac=_safe_int(fields.get("off_flash_ac", "0")),
            mac_count=_safe_int(fields.get("mac_count", "0")),
            network_pct_cycles=_safe_float(fields.get("network_pct_cycles", "0")),
            network_pct_mac=_safe_float(fields.get("network_pct_mac", "0")),
            util_pct_mac=_safe_float(fields.get("util_pct_mac", "0")),
            name=fields.get("name", ""),
        )
    except (ValueError, IndexError):
        return None


def _safe_int(s: str) -> int:
    s = s.strip()
    if not s:
        return 0
    return int(s)


def _safe_float(s: str) -> float:
    s = s.strip()
    if not s:
        return 0.0
    return float(s)


def _builtin_opcode_names() -> dict[int, str]:
    """Builtin operator code to name, from the flatbuffer schema (needs the ``models`` extra)."""
    builtin = litert_schema().BuiltinOperator
    return {value: name for name, value in vars(builtin).items() if name.isupper() and isinstance(value, int)}


@dataclass
class TFLiteOp:
    """Metadata for a single operator in the original TFLite graph."""

    index: int
    opcode: str
    output_tensor_name: str


def read_tflite_graph(model_path: Path) -> list[TFLiteOp]:
    """Return a single-subgraph TFLite model's operator list.

    Each operator is annotated with its output tensor name, which is what
    Vela uses in the verbose-performance 'Name' column.
    Raises ValueError for other graph scopes, which cannot be represented by
    this flat operator list or a Model Explorer overlay for graph ``main``.
    """
    schema = litert_schema()  # before the read, so a missing extra is reported ahead of a missing file
    opcode_names = _builtin_opcode_names()
    model = schema.Model.GetRootAs(model_path.read_bytes())
    if model.SubgraphsLength() != 1:
        raise ValueError(
            "Per-layer attribution supports only single-subgraph models; executed subgraph scope is unknown"
        )
    subgraph = model.Subgraphs(0)

    ops: list[TFLiteOp] = []
    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        # Use the first output tensor name — Vela uses this as the op name.
        # For multi-output ops, Vela still keys on the first output.
        out_idx = op.Outputs(0) if op.OutputsLength() > 0 else -1
        if out_idx >= 0:
            tensor = subgraph.Tensors(out_idx)
            name = tensor.Name().decode() if tensor.Name() else f"tensor_{out_idx}"
        else:
            name = f"op_{i}"

        opcode_obj = model.OperatorCodes(op.OpcodeIndex())
        # Extended codes use a placeholder in the legacy int8 field.
        builtin_code = max(opcode_obj.BuiltinCode(), opcode_obj.DeprecatedBuiltinCode())
        custom_code = opcode_obj.CustomCode()
        opcode_name = custom_code.decode() if custom_code else opcode_names.get(builtin_code, f"BUILTIN_{builtin_code}")

        ops.append(TFLiteOp(index=i, opcode=opcode_name, output_tensor_name=name))

    return ops


def index_by_output_name(tflite_ops: Sequence[TFLiteOp]) -> dict[str, TFLiteOp]:
    """The lookup :func:`match_original_op` works from: output tensor name -> op."""
    return {op.output_tensor_name: op for op in tflite_ops}


def match_original_op(vela_name: str, name_to_op: Mapping[str, TFLiteOp]) -> TFLiteOp | None:
    """Match a Vela verbose-performance name to the original TFLite op it came from.

    Vela names each NNG op after the output tensor of the TFLite op it was cut
    from, sometimes with a suffix when one op became several. Strategy:

    1. Exact match on the full name.
    2. Strip a known Vela suffix (``_softmax_0``, ``_lut_1``, ...) and retry.
    3. Longest tensor name that is a prefix of the Vela name.
    """
    # 1. Exact match
    if vela_name in name_to_op:
        return name_to_op[vela_name]

    # 2. Strip common Vela suffixes: _softmax_N, _lut_N, _0, _1, etc.
    stripped = re.sub(r"_(?:softmax|lut|add|sub|mul|reshape)_\d+$", "", vela_name)
    if stripped in name_to_op:
        return name_to_op[stripped]

    # 3. Longest prefix match — Vela may append suffixes we don't know about
    best_match: TFLiteOp | None = None
    best_len = 0
    for tensor_name, op in name_to_op.items():
        if vela_name.startswith(tensor_name) and len(tensor_name) > best_len:
            best_match = op
            best_len = len(tensor_name)

    return best_match

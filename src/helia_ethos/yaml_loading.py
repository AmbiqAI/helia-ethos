"""YAML loading that refuses what PyYAML would silently accept.

PyYAML keeps the last of two identical keys, so a comment line that loses its
``#`` and turns into a second ``energy:`` mapping overwrites the real one
without a word. Every SoC, target, platform and energy file goes through
:func:`load_yaml_mapping` so that case is an error naming the key.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml


class DuplicateKeyError(ValueError):
    """A mapping in the file repeats a key."""


class _StrictLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:  # type: ignore[override]
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                mark = key_node.start_mark
                raise DuplicateKeyError(f"duplicate key {key!r} at line {mark.line + 1}, column {mark.column + 1}")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_yaml_text(text: str) -> Any:
    """Parse *text* with duplicate keys rejected."""
    return yaml.load(text, Loader=_StrictLoader)


def load_yaml_mapping(
    path: Path, *, what: str = "file", on_read: Callable[[Path, bytes], None] | None = None
) -> dict[str, Any]:
    """Load *path* as a top-level mapping; duplicate keys and non-mappings are errors that name the file."""
    try:
        payload = path.read_bytes()
        raw = load_yaml_text(payload.decode("utf-8"))
    except DuplicateKeyError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark is not None else ""
        problem = getattr(exc, "problem", None) or str(exc).splitlines()[0]
        raise ValueError(f"{path}: not valid YAML ({problem}{where})") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: not UTF-8 text ({exc.reason} at byte {exc.start})") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{what} must be a YAML mapping: {path}")
    if on_read is not None:
        on_read(path, payload)
    return raw

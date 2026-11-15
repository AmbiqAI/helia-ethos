"""Explicit catalog inputs and serializable identity for reproducible simulations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import date
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory

from helia_platforms import PlatformError, builtin, load_overlay

from .catalog import read_catalog_npu
from .catalog_identity import CatalogContentIdentity
from .catalog_identity import CatalogModelIdentity as CatalogModelIdentity
from .catalog_policy import at110_model_policy, project_catalog_soc
from .soc import SocDefinition, SocError


class CatalogPolicy(StrEnum):
    """Available consumer-owned modeling policies; selection is explicit."""

    AT110 = "atomiq110"


@dataclass(frozen=True)
class CatalogModel:
    """Executable SoC settings and the identity of their source inputs."""

    soc: SocDefinition
    identity: CatalogModelIdentity


def _json_date(value: object) -> str:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Unsupported model identity value: {type(value).__name__}")


def _model_digest(soc: SocDefinition) -> str:
    payload = {"version": 1, "soc": asdict(replace(soc, catalog_identity=None)), "memory_order": list(soc.memories)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_date)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_catalog_model(
    target: str,
    *,
    policy: CatalogPolicy,
    overlays: tuple[Path, ...] = (),
    comparison_memories: tuple[str, ...] = (),
    npu: str | None = None,
    on_read: Callable[[Path, bytes], None] | None = None,
) -> CatalogModel:
    """Resolve built-in facts plus explicit overlays, without environment/plugin discovery."""
    if policy is not CatalogPolicy.AT110:
        raise SocError(f"Unsupported catalog modeling policy: {policy}")
    hashes: list[str] = []

    def capture_overlay(path: Path, data: bytes) -> None:
        hashes.append(hashlib.sha256(data).hexdigest())
        if on_read is not None:
            on_read(path, data)

    policy_hashes: list[str] = []

    def capture_policy(path: Path, data: bytes) -> None:
        policy_hashes.append(hashlib.sha256(data).hexdigest())
        if on_read is not None:
            on_read(path, data)

    try:
        catalog = builtin(include_plugins=False)
        for path in overlays:
            # The public catalog loader owns YAML 1.2 parsing. A snapshot keeps
            # parsed bytes identical to retained bytes; its stable basename keeps
            # native source labels independent of the caller's directory.
            payload = path.read_bytes()
            with TemporaryDirectory(prefix="helx-catalog-") as directory:
                snapshot = Path(directory) / "overlay.yaml"
                snapshot.write_bytes(payload)
                try:
                    catalog = load_overlay(snapshot, base=catalog)
                except PlatformError as exc:
                    message = " ".join(str(exc).replace(str(snapshot), str(path)).splitlines())
                    raise SocError(message) from exc
            capture_overlay(path, payload)
        facts = read_catalog_npu(target, catalog=catalog, npu_id=npu)
        selected = at110_model_policy(comparison_memories=comparison_memories, on_read=capture_policy)
        projection = project_catalog_soc(facts, policy=selected)
        identity = CatalogModelIdentity(
            target=target,
            npu=facts.npu.id,
            catalog=CatalogContentIdentity.capture(facts.identity),
            policy=selected.id,
            policy_sha256=policy_hashes[0],
            model_sha256=_model_digest(projection.soc),
            overlay_sha256=tuple(hashes),
        )
    except (PlatformError, OSError, ValueError) as exc:
        raise SocError(str(exc)) from exc
    return CatalogModel(replace(projection.soc, catalog_identity=identity), identity)

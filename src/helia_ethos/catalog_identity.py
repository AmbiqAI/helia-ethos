"""Closed, versioned catalog identity records shared by producer and retained readers."""

from __future__ import annotations

from typing import Annotated, Literal

from helia_platforms import CatalogIdentity
from pydantic import ConfigDict, Field, StrictStr
from pydantic.dataclasses import dataclass

Sha256 = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
CatalogDigest = Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
_IDENTITY_CONFIG = ConfigDict(extra="forbid", strict=True)


@dataclass(frozen=True, config=_IDENTITY_CONFIG)
class CatalogSourceIdentity:
    """One explicitly applied source in catalog order."""

    label: StrictStr
    schema_version: Literal[1]
    sha256: CatalogDigest
    distribution_version: StrictStr | None = None


@dataclass(frozen=True, config=_IDENTITY_CONFIG)
class CatalogContentIdentity:
    """The supported catalog identity, isolated from upstream schema additions."""

    package_version: StrictStr
    schema_version: Literal[1]
    catalog_digest: CatalogDigest
    sources: tuple[CatalogSourceIdentity, ...] = ()

    @classmethod
    def capture(cls, identity: CatalogIdentity) -> CatalogContentIdentity:
        """Copy supported native identity fields without dropping any source."""
        if identity.schema_version != 1:
            raise ValueError(f"Unsupported catalog identity schema: {identity.schema_version}")
        return cls(
            package_version=identity.package_version,
            schema_version=1,
            catalog_digest=identity.catalog_digest,
            sources=tuple(CatalogSourceIdentity(**source.to_dict()) for source in identity.sources),
        )


@dataclass(frozen=True, config=_IDENTITY_CONFIG)
class CatalogModelIdentity:
    """Catalog content, exact overlay bytes, policy source and effective model settings.

    The model digest includes resolved values and memory order, so changes to
    policy/projection code cannot silently reuse a different model on replay.
    This identity does not assert equivalence to a physical device.
    """

    target: StrictStr
    npu: StrictStr
    catalog: CatalogContentIdentity
    policy: StrictStr
    policy_sha256: Sha256
    model_sha256: Sha256
    overlay_sha256: tuple[Sha256, ...] = ()
    version: Literal[1] = 1

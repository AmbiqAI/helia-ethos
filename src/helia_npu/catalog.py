"""Catalog facts for an NPU, independent of Vela placement and FVP timing policy.

No catalog access-port number or sequence order selects a compiler port or a
placement default. Raw catalog provenance stays attached to the facts; reading
an unverified catalog value does not turn it into a confirmed HELX value.
"""

from __future__ import annotations

from dataclasses import dataclass

from helia_platforms import Catalog, CatalogIdentity, PlatformError, resolve
from helia_platforms.model import Access, DomainPoint, MemoryRegion, NpuInstance, Soc

from .soc import SocError


@dataclass(frozen=True)
class CatalogMemoryPath:
    """A memory and its reach record, including absent or explicitly denied reach."""

    region: MemoryRegion
    access: Access | None

    @property
    def reachable(self) -> bool:
        """Whether the catalog explicitly records direct reach from the selected NPU."""
        return self.access is not None and self.access.access == "direct"


@dataclass(frozen=True)
class CatalogNpuFacts:
    """Resolved NPU facts with source identity, before applying tool modeling policy."""

    target_id: str
    implementation_id: str | None
    soc: Soc
    npu: NpuInstance
    clock_points: tuple[DomainPoint, ...]
    memory_paths: tuple[CatalogMemoryPath, ...]
    identity: CatalogIdentity


def read_catalog_npu(target_id: str, *, catalog: Catalog, npu_id: str | None = None) -> CatalogNpuFacts:
    """Read one NPU's facts and catalog identity; require selection on multi-NPU targets.

    The caller supplies the catalog so side-loaded sources cannot be accidentally
    replaced by an implicit built-in lookup. Missing reach stays unknown. This
    reader does not claim that reachable memory has an available FVP model.
    """
    try:
        target = resolve(target_id, catalog=catalog)
        instances = target.npus()
        if npu_id is None:
            if len(instances) != 1:
                raise SocError(f"{target_id}: expected one NPU; found {len(instances)}; select an NPU explicitly")
            npu = instances[0]
        else:
            selected = next((instance for instance in instances if instance.id == npu_id), None)
            if selected is None:
                raise SocError(f"{target_id}: no NPU named '{npu_id}'")
            npu = selected
        domain = next((domain for domain in target.soc.clock_domains if domain.id == npu.clock_domain), None)
        if domain is None:
            raise SocError(f"{target_id}: no clock domain '{npu.clock_domain}' for NPU '{npu.id}'")
        regions = (*target.soc.memories, *target.external_memories())
        return CatalogNpuFacts(
            target_id=target.id,
            implementation_id=target.implementation.id if target.implementation is not None else None,
            soc=target.soc,
            npu=npu,
            clock_points=tuple(sorted(domain.points, key=lambda point: point.name)),
            memory_paths=tuple(
                CatalogMemoryPath(region, target.reachable(npu.id, region.name))
                for region in sorted(regions, key=lambda region: region.name)
            ),
            identity=catalog.identity(),
        )
    except PlatformError as exc:
        raise SocError(f"{target_id}: {exc}") from exc

"""Arm's reference energy per NPU cycle, one row per NPU configuration.

The value is Arm's data for the NPU at Arm's reference process node and
voltage; a platform scales it per mode to its own silicon. It lives here so
every platform that models the same NPU configuration reads the same number
instead of carrying a copy.
"""

from __future__ import annotations

from dataclasses import dataclass

from helia_ethos.soc import Provenance, ProvenanceStatus


@dataclass(frozen=True)
class NpuReferenceEnergy:
    npu: str
    macs: int
    pj_per_cycle: float
    provenance: Provenance


ARM_REFERENCE_ENERGY: dict[tuple[str, int], NpuReferenceEnergy] = {
    ("ethos-u85", 256): NpuReferenceEnergy(
        npu="ethos-u85",
        macs=256,
        pj_per_cycle=116.749,
        provenance=Provenance(
            status=ProvenanceStatus.CONFIRMED,
            source=(
                "Arm reference data for Ethos-U85-256 at Arm's reference node and 0.8 V, "
                "~25% MAC utilisation (MobileNetV2)"
            ),
            reviewed_by="Arm",
        ),
    ),
}


def reference_energy(npu: str, macs: int) -> NpuReferenceEnergy | None:
    """Arm's reference row for this NPU configuration, or ``None`` when there is none."""
    return ARM_REFERENCE_ENERGY.get((npu.strip().lower(), macs))

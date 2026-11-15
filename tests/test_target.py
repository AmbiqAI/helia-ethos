"""Targets: realizations of a SoC, what they implement, and how they are recorded."""

from __future__ import annotations

from pathlib import Path

import pytest

from helia_npu.soc_schema import load_soc
from helia_npu.target import Feature, TargetError, TargetKind, list_targets, load_target


class TestPackagedTargets:
    def test_every_packaged_target_realizes_a_packaged_soc(self) -> None:
        ids = list_targets()
        assert ids == list_targets("atomiq110")
        for target_id in ids:
            target = load_target(target_id)
            assert target.id == target_id
            assert load_soc(target.soc).id == target.soc

    def test_fpga_drop_is_a_subset_at_one_clock(self) -> None:
        fpga = load_target("atomiq110-fpga-turbo-npu")
        assert fpga.kind is TargetKind.FPGA
        assert fpga.board == "atomiq110_fpga_turbo"
        assert fpga.implements_all(Feature.CPU, Feature.NPU, Feature.SSRAM)
        assert not fpga.implements_all(Feature.HBLRAM)
        hblram = fpga.emulation_for(Feature.HBLRAM)
        assert hblram is not None and hblram.stand_in == "absent"
        assert fpga.clocks.npu_to_fabric_ratio == 1.0
        assert fpga.timing_is_representative is False

    def test_simulator_names_its_fidelity_limits(self) -> None:
        fvp = load_target("atomiq110-fvp-corstone-320")
        assert fvp.kind is TargetKind.SIMULATOR
        assert fvp.clocks.fabric_hz == 250e6 and fvp.clocks.npu_hz is None  # set per run from the SoC mode
        assert any("timing adapters" in note for note in fvp.fidelity_notes)
        assert fvp.label() == "atomiq110-fvp-corstone-320 (simulator)"
        assert fvp.fvp_subsystem == "corstone_320"  # helx refuses to attribute a Corstone-300 run to it

    def test_only_a_simulator_names_a_corstone(self) -> None:
        from helia_npu.target import validate_target_document

        document = {"id": "x", "soc": "atomiq110", "kind": "fpga", "name": "X", "fvp_subsystem": "corstone_320"}
        with pytest.raises(TargetError, match="only a simulator target has; this one is fpga"):
            validate_target_document(document)
        assert validate_target_document({**document, "kind": "simulator"}).fvp_subsystem == "corstone_320"

    def test_silicon_is_the_only_representative_target(self) -> None:
        representative = [t for t in list_targets() if load_target(t).timing_is_representative]
        assert representative == ["atomiq110-rtl", "atomiq110-silicon"]

    def test_require_names_what_is_missing(self) -> None:
        with pytest.raises(TargetError, match="does not implement: hblram, gpu"):
            load_target("atomiq110-fpga-turbo-npu").require(Feature.NPU, Feature.HBLRAM, Feature.GPU)

    def test_unknown_id_lists_the_alternatives(self) -> None:
        with pytest.raises(TargetError, match="Available: atomiq110-fpga-turbo-npu"):
            load_target("atomiq110-evb")


class TestTargetFiles:
    def test_absent_and_implemented_conflict_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(
            "id: x\nsoc: atomiq110\nkind: fpga\nname: x\nimplements: [hblram]\n"
            "emulation: [{feature: hblram, stand_in: absent}]\n"
        )
        with pytest.raises(TargetError, match="hblram is listed both as implemented and as absent"):
            load_target(path)

    def test_build_id_records_the_drop(self, tmp_path: Path) -> None:
        path = tmp_path / "drop.yaml"
        path.write_text(
            "id: atomiq110-fpga-turbo-npu-2026-10\nsoc: atomiq110\nkind: fpga\nname: a later drop\n"
            "soc_revision: rtl-2026-10\nbuild_id: bitstream-7f3a\nimplements: [cpu, npu]\n"
            "clocks: {cpu_hz: 25e6, npu_hz: 25e6, fabric_hz: 25e6}\n"
        )
        target = load_target(path)
        assert (target.soc_revision, target.build_id) == ("rtl-2026-10", "bitstream-7f3a")
        assert target.label() == "atomiq110-fpga-turbo-npu-2026-10 (fpga, rtl-2026-10)"


def test_catalog_policy_is_not_a_silicon_revision() -> None:
    soc = load_soc("atomiq110")
    assert soc.revision is None
    assert soc.catalog_identity is not None
    assert soc.catalog_identity.policy == "atomiq110-model-v10"

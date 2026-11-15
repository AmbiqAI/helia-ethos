"""Catalog boundary controls using the pinned upstream catalog and public overlays."""

import pytest
from helia_platforms import builtin, load_overlay

from helia_npu.catalog import CatalogMemoryPath, read_catalog_npu
from helia_npu.soc import SocError


def test_at110_preserves_facts_and_identity():
    catalog = builtin()
    facts = read_catalog_npu("atomiq110", catalog=catalog)
    assert facts.npu.npu.accelerator_config == "ethos-u85-256"
    assert facts.npu.axi_width_bits == 128
    assert {p.name: p.mhz for p in facts.clock_points} == {"ulp": 100, "hp": 500}
    assert facts.identity == catalog.identity()
    ssram = next(p for p in facts.memory_paths if p.region.name == "ssram")
    assert ssram.reachable
    assert ssram.access is not None
    assert ssram.access.characterization is None
    assert ssram.access.provenance.status == "unverified"
    assert facts.npu.axi_ports["sram"].provenance.status == "datasheet"
    assert {"itcm", "dtcm", "ssram", "hblram", "rram"} <= {
        path.region.name for path in facts.memory_paths if path.reachable
    }
    # No translation of catalog ceilings into the model's fitted queue depths.
    assert not hasattr(facts, "max_outstanding")


def test_overlay_identity_is_retained_without_inventing_policy_ancestry():
    catalog = load_overlay({"socs": {"lab_at110": {"based_on": "atomiq110"}}}, base=builtin())
    facts = read_catalog_npu("lab_at110", catalog=catalog)
    assert facts.soc.id == "lab_at110"
    # Catalog #134: based_on is not the built-in derives_from relation.
    assert facts.soc.derives_from is None
    assert facts.identity.sources
    assert facts.identity == catalog.identity()
    assert facts.identity != builtin().identity()
    assert facts.identity.sources[-1].sha256


def test_fpga_keeps_implementation_clocks_and_memory_subset():
    facts = read_catalog_npu("atomiq110_fpga", catalog=builtin())
    assert {p.name: p.mhz for p in facts.clock_points} == {"ulp": 10, "hp": 50}
    assert "hblram" not in {p.region.name for p in facts.memory_paths}


def test_memory_order_is_canonical_not_a_default_selection():
    facts = read_catalog_npu("atomiq110", catalog=builtin())
    names = [p.region.name for p in facts.memory_paths]
    assert names == sorted(names)
    assert not hasattr(facts, "default_memory")


def test_absent_reach_is_not_inferred_from_address():
    facts = read_catalog_npu("atomiq110", catalog=builtin())
    region = next(p.region for p in facts.memory_paths if p.region.name == "ssram")
    assert region.base is not None
    assert not CatalogMemoryPath(region=region, access=None).reachable


@pytest.mark.parametrize(
    "target,npu,message",
    [
        ("apollo510", None, "expected one NPU"),
        ("atomiq110", "nonexistent", "no NPU named"),
        ("no-such-target", None, "no-such-target"),
    ],
)
def test_selection_errors_use_soc_error(target, npu, message):
    with pytest.raises(SocError, match=message):
        read_catalog_npu(target, catalog=builtin(), npu_id=npu)

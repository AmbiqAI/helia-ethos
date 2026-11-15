"""AT110 adoption parity and policy isolation, without running compilers or FVPs."""

import configparser
from dataclasses import replace

import pytest
from helia_platforms import builtin, load_overlay
from helia_platforms.model import Provenance as CatalogProvenance

from helia_npu.catalog import read_catalog_npu
from helia_npu.catalog_policy import PortDefault, VelaPortClass, at110_model_policy, project_catalog_soc
from helia_npu.soc import MemoryPlacement, SocError, SocOrigin, derive_timing
from helia_npu.soc_schema import load_soc
from helia_npu.vela import write_system_config


@pytest.fixture
def facts():
    return read_catalog_npu("atomiq110", catalog=builtin())


@pytest.mark.parametrize("mode", ["ULP", "HP"])
@pytest.mark.parametrize(
    "placement",
    [
        MemoryPlacement("ssram", "ssram", "ssram"),
        MemoryPlacement("hblram", "ssram", "ssram"),
        MemoryPlacement("hblram", "hblram", "ssram"),
        MemoryPlacement("rram", "ssram", "ssram"),
        MemoryPlacement("rram", "rram", "ssram"),
    ],
)
def test_generated_vela_and_fvp_settings_retain_parity(facts, mode, placement, tmp_path):
    old = load_soc("atomiq110")
    projected = project_catalog_soc(facts, policy=at110_model_policy())
    new = projected.soc
    sections = []
    for index, soc in enumerate((old, new)):
        path = tmp_path / f"{index}.ini"
        write_system_config(soc, mode, path, placement=placement)
        parser = configparser.ConfigParser()
        parser.read(path)
        sections.append({section: dict(parser[section]) for section in parser.sections()})
    assert sections[0] == sections[1]
    old_timing, new_timing = (derive_timing(soc, mode) for soc in (old, new))
    assert old_timing.port_timing(old.occupants(placement)) == new_timing.port_timing(new.occupants(placement))
    assert old_timing.as_ta_profile(old.occupants(placement)) == new_timing.as_ta_profile(new.occupants(placement))
    assert old.sram_port_hash == new.sram_port_hash
    assert new.origin is SocOrigin.DERIVED
    assert projected.facts.identity == facts.identity
    assert "psram" not in new.memories


def test_order_and_hardware_port_ceilings_do_not_choose_policy(facts):
    policy = at110_model_policy()
    # Hardware ceilings may change independently of the fitted path queues.
    ports = {
        name: replace(port, max_outstanding_reads=1, max_outstanding_writes=1)
        for name, port in facts.npu.axi_ports.items()
    }
    shuffled = replace(facts, memory_paths=tuple(reversed(facts.memory_paths)), npu=replace(facts.npu, axi_ports=ports))
    projected = project_catalog_soc(shuffled, policy=replace(policy, memories=tuple(reversed(policy.memories))))
    expected = load_soc("atomiq110")
    assert projected.soc.standard_placements() == expected.standard_placements()
    assert projected.soc.occupants() == expected.occupants()
    # Hardware transaction ceilings never replace calibrated path queues.
    assert projected.soc.memories["ssram"].params.max_outstanding == expected.memories["ssram"].params.max_outstanding
    assert projected.soc.memories["ssram"].params.max_outstanding_writes == (
        expected.memories["ssram"].params.max_outstanding_writes
    )


def test_explicit_default_can_select_rram(facts):
    policy = at110_model_policy()
    policy = replace(policy, defaults=(policy.defaults[0], PortDefault(VelaPortClass.EXTERNAL, "rram")))
    assert project_catalog_soc(facts, policy=policy).soc.occupants()[1] == "rram"


def test_derived_soc_uses_catalog_clocks_with_explicit_policy():
    catalog = load_overlay(
        {
            "socs": {
                "lab_at110": {
                    "based_on": "atomiq110",
                    "clock_domains": {"npu": {"hp": {"mhz": 400}}},
                }
            }
        },
        base=builtin(),
    )
    facts = read_catalog_npu("lab_at110", catalog=catalog)
    result = project_catalog_soc(facts, policy=at110_model_policy())
    assert result.soc.id == "lab_at110"
    assert result.soc.resolve_mode("HP").npu_clock_hz == 400e6
    assert result.facts.identity.sources
    assert result.soc.provenance["sram_ports"].status.value == "unverified"


@pytest.mark.parametrize("reach", [None, "none"])
def test_missing_or_denied_reach_is_actionable(facts, reach):
    paths = tuple(
        replace(p, access=None if reach is None else replace(p.access, access="none"))
        if p.region.name == "ssram" and p.access is not None
        else p
        for p in facts.memory_paths
    )
    with pytest.raises(SocError, match="requires catalog NPU reach to 'ssram'"):
        project_catalog_soc(replace(facts, memory_paths=paths), policy=at110_model_policy())


def test_missing_width_is_not_filled_from_legacy(facts):
    with pytest.raises(SocError, match="catalog AXI bus width"):
        project_catalog_soc(replace(facts, npu=replace(facts.npu, axi_width_bits=None)), policy=at110_model_policy())


def test_different_accelerator_needs_appropriate_policy(facts):
    npu = replace(facts.npu, npu=replace(facts.npu.npu, macs_per_cycle=128))
    with pytest.raises(SocError, match="does not support"):
        project_catalog_soc(replace(facts, npu=npu), policy=at110_model_policy())


def test_fpga_cannot_inherit_silicon_timings():
    facts = read_catalog_npu("atomiq110_fpga", catalog=builtin())
    with pytest.raises(SocError, match="implementation timing needs a separate policy"):
        project_catalog_soc(facts, policy=at110_model_policy())


def test_invalid_default_is_actionable(facts):
    policy = replace(at110_model_policy(), defaults=())
    with pytest.raises(SocError, match="explicit default"):
        project_catalog_soc(facts, policy=policy)


def test_projection_preserves_specific_clock_and_reach_provenance(facts):
    measured = CatalogProvenance(status="measured", source=("meas:lab/2026-11-06",))
    uncertain = CatalogProvenance(status="unverified", source=("doc:estimate@v1:p1",))
    points = tuple(
        replace(point, provenance=measured, field_provenance={"mhz": uncertain}) for point in facts.clock_points
    )
    paths = tuple(
        replace(path, access=replace(path.access, provenance=measured, field_provenance={"access": uncertain}))
        if path.access is not None
        else path
        for path in facts.memory_paths
    )
    projected = project_catalog_soc(
        replace(facts, clock_points=points, memory_paths=paths), policy=at110_model_policy()
    )
    for key, record in projected.soc.provenance.items():
        if key.startswith(("catalog.clock.", "catalog.reach.")):
            assert record.status.value == "unverified"
            assert record.source is not None
            assert "doc:estimate@v1:p1" in record.source

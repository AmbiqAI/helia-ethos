"""SoC definitions: loading, identity, placements, derived timing."""

from __future__ import annotations

from pathlib import Path

import pytest

from helia_ethos.soc import (
    MemoryPlacement,
    MemoryTimingOverride,
    ProvenanceStatus,
    SocError,
    SocOrigin,
    TimingOverrides,
    VelaMemoryMode,
    derive_timing,
    parse_memory_placement,
    standard_vela_mode,
)
from helia_ethos.soc_schema import list_socs, load_soc


class TestLoad:
    def test_identity_and_origin(self, at110_soc_file: Path) -> None:
        soc = load_soc(at110_soc_file)
        assert (soc.id, soc.name, soc.origin) == ("atomiq110", "AT110", SocOrigin.GOLDEN)
        assert soc.npu == "ethos-u85"
        assert soc.macs == 256
        assert sorted(soc.memories) == ["hblram", "ssram"]
        assert soc.memory_for_axi(1) is soc.memories["hblram"]

    def test_origin_defaults_to_custom_and_id_to_the_name(self, tmp_path: Path, at110_soc_file: Path) -> None:
        text = at110_soc_file.read_text().replace("id: atomiq110\n", "").replace("origin: golden\n", "")
        path = tmp_path / "anon.yaml"
        path.write_text(text)
        soc = load_soc(path)
        assert soc.origin is SocOrigin.CUSTOM
        assert soc.id == "at110"

    def test_provenance_is_kept_per_memory(self, at110_soc_file: Path) -> None:
        soc = load_soc(at110_soc_file)
        assert soc.memories["ssram"].provenance is not None
        assert soc.memories["ssram"].provenance.status is ProvenanceStatus.CONFIRMED
        assert soc.provenance_label("memories.ssram").startswith("Confirmed")
        assert soc.provenance_label("memories.hblram").startswith("Unverified")

    def test_several_memories_may_share_a_port_and_the_first_is_its_default(
        self, tmp_path: Path, at110_soc_file: Path
    ) -> None:
        text = at110_soc_file.read_text().replace("axi_port: 1", "axi_port: 0")
        path = tmp_path / "both_on_port0.yaml"
        path.write_text(text)
        soc = load_soc(path)
        assert [m.name for m in soc.memories_on_axi(0)] == ["ssram", "hblram"]
        assert soc.memory_for_axi(0) is soc.memories["ssram"] and soc.memory_for_axi(1) is None

    def test_port_0_must_carry_a_memory(self, tmp_path: Path, at110_soc_file: Path) -> None:
        text = at110_soc_file.read_text().replace("axi_port: 0", "axi_port: 1")
        path = tmp_path / "no_port0.yaml"
        path.write_text(text)
        with pytest.raises(SocError, match="axi_port 0"):
            load_soc(path)

    def test_extra_keys_are_rejected(self, tmp_path: Path, at110_soc_file: Path) -> None:
        path = tmp_path / "extra.yaml"
        path.write_text(at110_soc_file.read_text() + "energy: {}\n")
        with pytest.raises(SocError, match="energy"):
            load_soc(path)


class TestPlacements:
    def test_standard_modes_resolve_to_memories(self, at110_soc_file: Path) -> None:
        soc = load_soc(at110_soc_file)
        shared = parse_memory_placement("Shared_Sram", soc)
        assert shared == MemoryPlacement(const="hblram", arena="ssram", cache="ssram")
        assert soc.vela_memory_mode_label(shared) == "Shared_Sram"
        assert soc.fvp_memory_mode(shared) == "Shared_Sram"
        assert parse_memory_placement("sram_only", soc) == MemoryPlacement("ssram", "ssram", "ssram")

    def test_custom_placement_gets_a_descriptive_label(self, at110_soc_file: Path) -> None:
        soc = load_soc(at110_soc_file)
        odd = parse_memory_placement({"const": "hblram", "arena": "ssram", "cache": "hblram"}, soc)
        assert soc.vela_memory_mode_label(odd) == "const_hblram__arena_ssram__cache_hblram"
        assert soc.fvp_memory_mode(odd) == "Shared_Sram"

    def test_unknown_memory(self, at110_soc_file: Path) -> None:
        soc = load_soc(at110_soc_file)
        with pytest.raises(SocError, match="Memory 'psram' not found"):
            parse_memory_placement({"const": "psram", "arena": "ssram", "cache": "ssram"}, soc)


class TestDerivedTiming:
    """The AT110 numbers every helx result was built on: ceil(ns x clock), bandwidth = min(1, AXI / NPU)."""

    def test_ulp(self, at110_soc_file: Path) -> None:
        timing = derive_timing(load_soc(at110_soc_file), "ULP")
        assert timing.npu_clock_hz == 100e6
        ssram, hblram = timing.memories["ssram"], timing.memories["hblram"]
        assert (ssram.read_latency, ssram.write_latency, ssram.bandwidth, ssram.max_outstanding) == (7, 7, 1.0, 8)
        assert (hblram.read_latency, hblram.write_latency, hblram.bandwidth, hblram.max_outstanding) == (9, 9, 1.0, 12)

    def test_hp(self, at110_soc_file: Path) -> None:
        timing = derive_timing(load_soc(at110_soc_file), "hp")  # case-insensitive
        assert (timing.memories["ssram"].read_latency, timing.memories["hblram"].read_latency) == (32, 44)
        assert timing.memories["ssram"].bandwidth == pytest.approx(0.5)
        assert timing.memories["hblram"].bandwidth == pytest.approx(0.5)

    def test_vela_memory_mode_spells_velas_section_names_and_reads_any_case(self) -> None:
        assert VelaMemoryMode("shared_sram") is VelaMemoryMode.SHARED_SRAM
        assert VelaMemoryMode.DEDICATED_SRAM == "Dedicated_Sram"
        assert f"{VelaMemoryMode.SRAM_ONLY}" == "Sram_Only"
        assert standard_vela_mode("SRAM_ONLY") is VelaMemoryMode.SRAM_ONLY
        assert standard_vela_mode("Sram_Only_512KB") is None

    def test_timing_adapter_profile_is_keyed_by_port(self, at110_soc_file: Path) -> None:
        profile = derive_timing(load_soc(at110_soc_file), "ULP").as_ta_profile()
        assert profile[0].read_latency == 7
        assert profile[1].max_transactions == 12

    def test_overrides_replace_derived_values(self, at110_soc_file: Path) -> None:
        overrides = TimingOverrides(memories={"hblram": MemoryTimingOverride(read_latency_cycles=55, bandwidth=0.4)})
        timing = derive_timing(load_soc(at110_soc_file), "HP", overrides)
        assert timing.memories["hblram"].read_latency == 55
        assert timing.memories["hblram"].write_latency == 44
        assert timing.memories["hblram"].bandwidth == pytest.approx(0.4)

    def test_unknown_mode(self, at110_soc_file: Path) -> None:
        with pytest.raises(SocError, match="Mode 'LP' not defined"):
            derive_timing(load_soc(at110_soc_file), "LP")


class TestPackagedSocs:
    def test_atomiq110_resolves_catalog_and_policy(self) -> None:
        assert list_socs() == ["atomiq110"]
        soc = load_soc("atomiq110")
        assert (soc.id, soc.name, soc.origin) == ("atomiq110", "AT110", SocOrigin.DERIVED)
        assert soc.revision is None
        assert soc.catalog_identity is not None
        assert soc.catalog_identity.policy == "atomiq110-model-v10"
        assert soc.macs == 256 and soc.npu == "ethos-u85"
        assert sorted(soc.memories) == ["hblram", "psram", "rram", "ssram"]
        assert sorted(soc.modes) == ["HP", "ULP"]

    def test_lookup_is_case_insensitive_and_names_the_alternatives(self) -> None:
        assert load_soc("ATOMIQ110").id == "atomiq110"
        with pytest.raises(SocError, match="Available: atomiq110"):
            load_soc("apollo510")

    def test_packaged_values_are_the_v10_at110_numbers(self) -> None:
        """v10: SSRAM is the spec figure (64 ns, 8 outstanding reads, 1 port, 128 B bursts) with 4 outstanding writes
        fitted to the FPGA; HBLRAM is the spec figure, writes as reads."""
        hp = derive_timing(load_soc("atomiq110"), "HP")
        ssram, hblram = hp.memories["ssram"], hp.memories["hblram"]
        assert (ssram.read_latency, ssram.max_outstanding, ssram.ports_used, ssram.burst_length) == (32, 8, 1, 128)
        assert (ssram.max_outstanding_writes, ssram.write_outstanding) == (4, 4)
        assert (hblram.read_latency, hblram.max_outstanding, hblram.ports_used) == (44, 12, 1)
        assert (hblram.max_outstanding_writes, hblram.write_outstanding) == (None, 12)
        ta = hp.as_ta_profile()
        assert (ta[0].max_transactions, ta[0].write_transactions) == (8, 4)
        assert (ta[1].max_transactions, ta[1].write_transactions) == (12, 12)
        ulp = derive_timing(load_soc("atomiq110"), "ULP")
        assert (ulp.memories["ssram"].read_latency, ulp.memories["hblram"].read_latency) == (7, 9)
        assert load_soc("atomiq110").sram_port_hash == 0x80


class TestDeriveSoc:
    """A what-if or variant is the golden definition plus a diff, never a copy."""

    def test_identity_only_keeps_the_origin(self) -> None:
        from helia_ethos.soc_schema import SocOverridesModel, derive_soc

        base = load_soc("atomiq110")
        soc = derive_soc(base, SocOverridesModel(id="atomiq110-alias", name="Alias"))
        assert soc.origin is base.origin and soc.memories == base.memories and soc.id == "atomiq110-alias"

    def test_hardware_change_makes_it_derived(self) -> None:
        from helia_ethos.soc_schema import SocOverridesModel, derive_soc

        soc = derive_soc(load_soc("atomiq110"), SocOverridesModel(id="x", macs=1024))
        assert soc.origin is SocOrigin.DERIVED and soc.macs == 1024

    def test_memory_replace_and_remove(self) -> None:
        from helia_ethos.soc_schema import MemoryModel, SocOverridesModel, derive_soc

        psram = MemoryModel(
            axi_port=1,
            axi_clock_hz=62.5e6,
            read_latency_ns=152,
            write_latency_ns=152,
            max_outstanding=10,
            burst_length=256,
        )
        soc = derive_soc(load_soc("atomiq110"), SocOverridesModel(id="p", memories={"hblram": None, "psram": psram}))
        assert sorted(soc.memories) == ["psram", "rram", "ssram"]
        assert soc.memory_for_axi(1) is soc.memories["psram"]  # hblram removed: the next port-1 memory is the default
        assert soc.provenance_label("memories.hblram") == "No provenance recorded"

    def test_removing_an_unknown_memory_fails(self) -> None:
        from helia_ethos.soc_schema import SocOverridesModel, derive_soc

        with pytest.raises(SocError, match="no such memory"):
            derive_soc(load_soc("atomiq110"), SocOverridesModel(memories={"mram": None}))

    def test_a_second_memory_on_a_port_is_allowed_but_port_0_is_required(self) -> None:
        from helia_ethos.soc_schema import MemoryModel, SocOverridesModel, derive_soc

        clash = MemoryModel(
            axi_port=0, axi_clock_hz=1e6, read_latency_ns=1, write_latency_ns=1, max_outstanding=1, burst_length=1
        )
        derived = derive_soc(load_soc("atomiq110"), SocOverridesModel(memories={"extra": clash}))
        assert [m.name for m in derived.memories_on_axi(0)] == ["ssram", "extra"]
        with pytest.raises(SocError, match="no memory on AXI port 0"):
            derive_soc(load_soc("atomiq110"), SocOverridesModel(memories={"ssram": None}))


class TestProvenanceVocabulary:
    """confirmed / fitted / unverified, per block and per value; a result reports the least trusted input."""

    def test_the_packaged_soc_marks_fits_and_estimates_per_value(self) -> None:
        soc = load_soc("atomiq110")
        ssram = soc.memories["ssram"].provenance
        assert ssram is not None and ssram.status is ProvenanceStatus.CONFIRMED
        assert ssram.for_field("read_latency_ns").status is ProvenanceStatus.CONFIRMED
        assert ssram.for_field("max_outstanding").status is ProvenanceStatus.CONFIRMED
        assert (
            ssram.for_field("burst_length").status is ProvenanceStatus.FITTED
            and ssram.for_field("ports_used").status is ProvenanceStatus.FITTED
            and ssram.for_field("max_outstanding_writes").status is ProvenanceStatus.FITTED
        )  # the driver's split, Vela's port count and the write limit are fits, not the memory's specification
        assert ssram.for_field("write_latency_ns").status is ProvenanceStatus.UNVERIFIED
        assert soc.sram_port_hash == 0x80 and soc.provenance["sram_ports"].status is ProvenanceStatus.UNVERIFIED
        hblram = soc.memories["hblram"].provenance
        assert hblram is not None and hblram.status is ProvenanceStatus.CONFIRMED
        assert hblram.for_field("read_latency_ns").status is ProvenanceStatus.UNVERIFIED
        assert hblram.for_field("max_outstanding").status is ProvenanceStatus.CONFIRMED
        assert soc.provenance_label("memories.ssram").startswith("Confirmed (")
        assert soc.provenance_label("memories.hblram").startswith("Confirmed (")

    def test_retired_and_old_spellings_are_refused(self, tmp_path: Path, at110_soc_file: Path) -> None:
        for spelling, hint in (
            ("derived", "'derived' is retired"),
            ("reviewed", "'reviewed' is retired: write 'confirmed'"),
        ):
            text = at110_soc_file.read_text().replace("status: confirmed", f"status: {spelling}")
            path = tmp_path / f"{spelling}.yaml"
            path.write_text(text)
            with pytest.raises(SocError, match=hint):
                load_soc(path)

    def test_field_records_must_name_values_the_block_has(self, tmp_path: Path, at110_soc_file: Path) -> None:
        text = at110_soc_file.read_text().replace(
            "provenance: {status: confirmed, source: Atomiq110 SoC specification}",
            "provenance: {status: confirmed, fields: {latency: {status: fitted}}}",
        )
        path = tmp_path / "bad_field.yaml"
        path.write_text(text)
        with pytest.raises(SocError, match="names unknown values: latency"):
            load_soc(path)

    def test_least_trusted_orders_the_vocabulary(self) -> None:
        from helia_ethos.soc import least_trusted

        assert least_trusted([ProvenanceStatus.CONFIRMED, ProvenanceStatus.FITTED]) is ProvenanceStatus.FITTED
        assert least_trusted([ProvenanceStatus.UNVERIFIED, ProvenanceStatus.CONFIRMED]) is ProvenanceStatus.UNVERIFIED
        assert least_trusted([]) is None


class TestStrictYaml:
    def test_a_duplicate_key_is_an_error_naming_the_line(self, tmp_path: Path, at110_soc_file: Path) -> None:
        text = at110_soc_file.read_text() + "\nmodes:\n  ULP: {npu_clock_hz: 1e6}\n"
        path = tmp_path / "dup.yaml"
        path.write_text(text)
        with pytest.raises(SocError, match=r"duplicate key 'modes' at line \d+"):
            load_soc(path)

    def test_every_packaged_soc_and_target_loads_strictly(self) -> None:
        from helia_ethos.target import list_targets, load_target

        for soc_id in list_socs():
            load_soc(soc_id)
        for target_id in list_targets():
            load_target(target_id)


class TestAxiBeatWidth:
    """The beat width belongs to the NPU (Vela: 8 bytes a cycle on the U55, 16 on the U65 and U85)."""

    def test_unset_it_is_the_npus(self, tmp_path: Path, at110_soc_file: Path) -> None:
        text = at110_soc_file.read_text(encoding="utf-8").replace("axi_bus_width_bits: 128\n", "")
        for npu, macs, bits in (("ethos-u55", 128, 64), ("ethos-u65", 256, 128), ("ethos-u85", 256, 128)):
            path = tmp_path / f"{npu}.yaml"
            path.write_text(text.replace("npu: ethos-u85", f"npu: {npu}").replace("macs: 256", f"macs: {macs}"))
            assert load_soc(path).axi_bus_width_bits == bits

    def test_another_width_is_refused(self, tmp_path: Path, at110_soc_file: Path) -> None:
        path = tmp_path / "u55.yaml"
        path.write_text(at110_soc_file.read_text(encoding="utf-8").replace("npu: ethos-u85", "npu: ethos-u55"))
        with pytest.raises(SocError, match="an ethos-u55 moves 64-bit AXI data beats, but axi_bus_width_bits is 128"):
            load_soc(path)

    def test_an_override_that_changes_the_npu_takes_its_width(self) -> None:
        from helia_ethos.soc_schema import SocOverridesModel, derive_soc

        base = load_soc("atomiq110")
        assert derive_soc(base, SocOverridesModel(id="u55", npu="ethos-u55", macs=128)).axi_bus_width_bits == 64
        assert derive_soc(base, SocOverridesModel(id="u85", macs=1024)).axi_bus_width_bits == 128
        with pytest.raises(SocError, match="moves 64-bit"):
            derive_soc(base, SocOverridesModel(id="u55", npu="ethos-u55", macs=128, axi_bus_width_bits=128))


def test_axi_max_beats_code_maps_the_ethos_u_bursts_and_refuses_others() -> None:
    from helia_ethos.soc import axi_max_beats_code

    assert [axi_max_beats_code(b) for b in (64, 128, 256)] == [0, 1, 2]
    with pytest.raises(ValueError, match="512 B is not a split the Ethos-U NPUs support"):
        axi_max_beats_code(512)


class TestOutstandingWrites:
    """max_outstanding_writes: optional, defaults to the read limit, must be at least 1, reaches the adapter and Vela."""

    def _soc(self, tmp_path, ssram_extra: str):
        text = f"""name: T
id: t
origin: custom
npu: ethos-u85
macs: 256
modes:
  HP: {{npu_clock_hz: 500e6}}
memories:
  ssram:
    axi_port: 0
    axi_clock_hz: 250e6
    read_latency_ns: 64
    write_latency_ns: 64
    max_outstanding: 8
    burst_length: 128
{ssram_extra}
  hblram:
    axi_port: 1
    axi_clock_hz: 250e6
    read_latency_ns: 88
    write_latency_ns: 88
    max_outstanding: 12
    burst_length: 256
"""
        path = tmp_path / "t.yaml"
        path.write_text(text, encoding="utf-8")
        return load_soc(path)

    def test_absent_means_writes_equal_reads(self, tmp_path) -> None:
        soc = self._soc(tmp_path, "")
        assert soc.memories["ssram"].params.max_outstanding_writes is None
        ta = derive_timing(soc, "HP").as_ta_profile()
        assert (ta[0].max_transactions, ta[0].write_transactions) == (8, 8)

    def test_a_write_limit_reaches_the_adapter_and_vela(self, tmp_path) -> None:
        from helia_ethos.vela.system_config import write_system_config

        soc = self._soc(tmp_path, "    max_outstanding_writes: 4")
        hp = derive_timing(soc, "HP")
        assert hp.memories["ssram"].write_outstanding == 4
        assert (hp.as_ta_profile()[0].max_transactions, hp.as_ta_profile()[0].write_transactions) == (8, 4)
        out = tmp_path / "vela.ini"
        write_system_config(soc, "HP", out)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert "Sram_max_reads=8" in lines and "Sram_max_writes=4" in lines
        assert "Dram_max_reads=12" in lines and "Dram_max_writes=12" in lines

    @pytest.mark.parametrize("bad", ["0", "-1", "true", "2.5", "many"])
    def test_an_invalid_write_limit_is_refused(self, tmp_path, bad: str) -> None:
        with pytest.raises(SocError, match="max_outstanding_writes"):
            self._soc(tmp_path, f"    max_outstanding_writes: {bad}")

    def test_an_override_replaces_only_the_write_limit(self, tmp_path) -> None:
        soc = self._soc(tmp_path, "    max_outstanding_writes: 4")
        over = TimingOverrides(memories={"ssram": MemoryTimingOverride(max_outstanding_writes=6)})
        hp = derive_timing(soc, "HP", over)
        assert (hp.memories["ssram"].max_outstanding, hp.memories["ssram"].write_outstanding) == (8, 6)
        assert MemoryTimingOverride(max_outstanding_writes=6).reaches_vela
        reads_only = TimingOverrides(memories={"ssram": MemoryTimingOverride(max_outstanding=16)})
        hp2 = derive_timing(soc, "HP", reads_only)
        assert (hp2.memories["ssram"].max_outstanding, hp2.memories["ssram"].write_outstanding) == (16, 4)

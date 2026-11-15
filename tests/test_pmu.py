"""PMU programs: the event-per-slot contract behind every profile name."""

from __future__ import annotations

import pytest

from helia_ethos.fvp import NpuFamily, required_pmu_fields, resolve_fvp_target_profile
from helia_ethos.pmu import PmuUnit, list_programs, program_for, render_c_header, slot_count


class TestPrograms:
    def test_every_family_has_a_default_and_the_right_slot_count(self) -> None:
        for family in NpuFamily:
            program = program_for(family)
            assert program.name == "default"
            assert len(program.slots) == slot_count(family)
            assert program.slots[0].event == "ETHOSU_PMU_NPU_ACTIVE"

    def test_u85_programs(self) -> None:
        assert list_programs(NpuFamily.ETHOS_U85) == (
            "default",
            "default_ext_rd_stall",
            "ext_stall",
            "port_stall",
            "sram_stall",
            "sram_transactions",
            "dv_align",
            "hpx_default_v1",
        )

    def test_port_stall_reads_both_ports_stall_counters(self) -> None:
        """What the RTL testbench and the FPGA report when locating a memory stall: where the NPU waits, per port."""
        program = program_for(NpuFamily.ETHOS_U85, "port_stall")
        assert program.events == (
            "ETHOSU_PMU_NPU_ACTIVE",
            "ETHOSU_PMU_MAC_ACTIVE",
            "ETHOSU_PMU_SRAM_RD_STALL_LIMIT",
            "ETHOSU_PMU_SRAM_RD_TRAN_REQ_STALLED",
            "ETHOSU_PMU_EXT_RD_STALL_LIMIT",
            "ETHOSU_PMU_EXT_RD_TRAN_REQ_STALLED",
            "ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED",
            "ETHOSU_PMU_EXT_RD_DATA_BEAT_RECEIVED",
        )
        assert program.acquisition is not None and program.acquisition.consumer == "helx-fvp-runner"

    def test_dv_align_is_u85_only(self) -> None:
        with pytest.raises(ValueError, match="No PMU program 'dv_align' for ethos-u55"):
            program_for(NpuFamily.ETHOS_U55, "dv_align")

    def test_same_fields_different_events_are_distinguishable(self) -> None:
        """Two profiles can require the same fields while programming different events."""
        default = program_for(NpuFamily.ETHOS_U85, "default")
        stall = program_for(NpuFamily.ETHOS_U85, "default_ext_rd_stall")
        assert default.required_fields == stall.required_fields
        assert default.events != stall.events
        assert stall.events[6] == "ETHOSU_PMU_EXT_RD_TRAN_REQ_STALLED"
        assert default.slots[6].unit is PmuUnit.BEATS and stall.slots[6].unit is PmuUnit.CYCLES

    def test_dv_align_matches_the_fpga_and_dv_datasets(self) -> None:
        program = program_for(NpuFamily.ETHOS_U85, "dv_align")
        assert program.events == (
            "ETHOSU_PMU_NPU_ACTIVE",
            "ETHOSU_PMU_MAC_ACTIVE",
            "ETHOSU_PMU_MAC_STALLED_BY_IB",
            "ETHOSU_PMU_MAC_STALLED_BY_W_OR_ACC",
            "ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED",
            "ETHOSU_PMU_SRAM_WR_DATA_BEAT_WRITTEN",
            "ETHOSU_PMU_EXT_RD_DATA_BEAT_RECEIVED",
            "ETHOSU_PMU_EXT_WR_DATA_BEAT_WRITTEN",
        )

    def test_cycle_counter_is_the_dedicated_ungated_ccnt(self) -> None:
        program = program_for(NpuFamily.ETHOS_U85)
        assert program.cycle_counter.gated_by_event is None
        assert program.cycle_counter.field == "npu_total_cycles"
        assert [d.formula for d in program.derived] == ["npu_total_cycles - npu_active_cycles"]


class TestIdentityAndAcquisition:
    def test_program_id_is_stable_and_versioned(self) -> None:
        assert program_for(NpuFamily.ETHOS_U85, "dv_align").id == "helia-ethos/ethos-u85/dv_align/v1"

    def test_acquisition_is_recorded_per_consumer(self) -> None:
        """Width, place and unit of accumulation and the overflow scope belong to the consumer, not the events."""
        from helia_ethos.pmu import AccumulationSite, OverflowScope

        program = program_for(NpuFamily.ETHOS_U85)
        assert (program.event_counter_bits, program.cycle_counter.bits) == (32, 64)
        assert program.measured_fields[-1] == "npu_total_cycles"  # CCNT is measured
        assert "npu_idle_cycles" not in program.measured_fields  # derived, not measured
        helx = program.acquisition
        assert helx is not None and helx.consumer == "helx-fvp-runner"
        assert (helx.accumulation_bits, helx.accumulation_site, helx.accumulation_unit) == (
            64,
            AccumulationSite.FIRMWARE,
            "dispatch",
        )
        assert helx.overflow_scope is OverflowScope.ANY_COUNTER and helx.reset_per_unit
        hpx = program_for(NpuFamily.ETHOS_U85, "hpx_default_v1").acquisition
        assert hpx is not None and hpx.consumer == "hpx-firmware"
        assert (hpx.accumulation_bits, hpx.accumulation_unit, hpx.overflow_scope) == (
            32,
            "layer",
            OverflowScope.SELECTED_EVENT_SLOTS,
        )
        assert "heliaPROFILER" in hpx.source

    def test_same_events_do_not_make_captures_comparable(self) -> None:
        """Equal ordered events identify the program; comparable numbers also need the same acquisition."""
        from dataclasses import replace

        from helia_ethos.pmu import HELX_RUNNER_ACQUISITION

        hpx = program_for(NpuFamily.ETHOS_U85, "hpx_default_v1")
        as_helx_would_read_it = replace(hpx, acquisition=HELX_RUNNER_ACQUISITION)
        assert hpx.same_program_as(as_helx_would_read_it)
        assert not hpx.comparable_with(as_helx_would_read_it)
        assert hpx.comparable_with(hpx)
        assert not replace(hpx, acquisition=None).comparable_with(hpx)  # unknown acquisition is not equal
        # Provenance text is not a semantic difference.
        assert hpx.acquisition is not None
        reworded = replace(hpx, acquisition=replace(hpx.acquisition, source="elsewhere"))
        assert hpx.comparable_with(reworded)
        default = program_for(NpuFamily.ETHOS_U85)
        assert default.comparable_with(program_for(NpuFamily.ETHOS_U85, "default"))
        # A different printed label is not a different program.
        relabelled = replace(default, cycle_counter=replace(default.cycle_counter, report_name="CYCLES"))
        assert default.same_program_as(relabelled)
        assert not default.same_program_as(program_for(NpuFamily.ETHOS_U85, "default_ext_rd_stall"))

    def test_hpx_default_is_a_named_program_with_an_event_slot_cycle_counter(self) -> None:
        from helia_ethos.pmu import CycleSource

        hpx = program_for(NpuFamily.ETHOS_U85, "hpx_default_v1")
        assert hpx.id == "hpx/ethos-u85/hpx_default_v1/v1"
        assert hpx.events == (
            "ETHOSU_PMU_CYCLE",
            "ETHOSU_PMU_NPU_ACTIVE",
            "ETHOSU_PMU_MAC_ACTIVE",
            "ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED",
        )
        assert hpx.cycle_counter.source is CycleSource.EVENT_SLOT and hpx.cycle_counter.bits == 32
        assert hpx.derived == ()
        assert hpx.required_fields == (
            "npu_total_cycles",
            "npu_active_cycles",
            "npu_mac_active_cycles",
            "npu_sram_rd_data_beats",
        )
        # A four-slot program on an eight-slot part is a program, not a pass of something larger.
        assert len(hpx.slots) == 4

    def test_a_cycle_slot_must_declare_its_source(self) -> None:
        from helia_ethos.pmu import PmuProgram, PmuSlot, PmuUnit

        with pytest.raises(ValueError, match=r"cycle_counter\.source is not event_slot"):
            PmuProgram(
                "bad", NpuFamily.ETHOS_U85, (PmuSlot("ETHOSU_PMU_CYCLE", "npu_total_cycles", PmuUnit.CYCLES, "x"),)
            )


class TestRequiredFields:
    """What the host validates is derived from the program, not maintained separately."""

    def test_u55_default(self) -> None:
        profile = resolve_fvp_target_profile("corstone_300")
        assert required_pmu_fields(profile, "default") == (
            "npu_active_cycles",
            "npu_idle_cycles",
            "npu_total_cycles",
            "npu_sram_rd_data_beats",
            "npu_sram_wr_data_beats",
            "npu_ext_rd_data_beats",
        )

    def test_u85_ext_stall_requires_both_request_stall_counters(self) -> None:
        fields = required_pmu_fields(resolve_fvp_target_profile("corstone_320"), "ext_stall")
        assert "npu_ext_rd_tran_req_stalled" in fields and "npu_ext_wr_tran_req_stalled" in fields

    def test_none_means_default(self) -> None:
        profile = resolve_fvp_target_profile("corstone_320")
        assert required_pmu_fields(profile, None) == required_pmu_fields(profile, "default")


class TestCHeader:
    def test_header_carries_the_table_the_firmware_compiles(self) -> None:
        text = render_c_header(program_for(NpuFamily.ETHOS_U85, "dv_align"))
        assert '#define HELX_PMU_PROGRAM_NAME "ethos-u85/dv_align"' in text
        assert "#define HELX_PMU_NSLOTS 8U" in text
        assert '{ETHOSU_PMU_NPU_ACTIVE, "NPU ACTIVE", "cycles"}' in text
        assert '{ETHOSU_PMU_MAC_ACTIVE, "NPU ETHOSU_PMU_MAC_ACTIVE", "cycles"}' in text
        assert '{ETHOSU_PMU_EXT_WR_DATA_BEAT_WRITTEN, "NPU ETHOSU_PMU_EXT_WR_DATA_BEAT_WRITTEN", "beats"}' in text
        assert '#define HELX_PMU_DERIVED_NPU_IDLE_CYCLES_REPORT_NAME "NPU IDLE"' in text
        assert "#define HELX_PMU_CCNT_GATED 0" in text
        assert "#define HELX_PMU_CCNT_SOURCE_CCNT 1" in text
        assert "#define HELX_PMU_CCNT_BITS 64" in text
        assert "#define HELX_PMU_SLOT0_IS_NPU_ACTIVE 1" in text
        assert "#define HELX_PMU_NDERIVED 1U" in text

    def test_hpx_header_declares_its_shape_instead_of_pretending(self) -> None:
        """An event-slot cycle count and no derived field: the consumer sees that and decides, not a dummy macro."""
        text = render_c_header(program_for(NpuFamily.ETHOS_U85, "hpx_default_v1"))
        assert "#define HELX_PMU_CCNT_SOURCE_EVENT_SLOT 1" in text
        assert "HELX_PMU_CCNT_SOURCE_CCNT" not in text
        assert "#define HELX_PMU_CCNT_BITS 32" in text
        assert "#define HELX_PMU_SLOT0_IS_NPU_ACTIVE 0" in text
        assert "#define HELX_PMU_NDERIVED 0U" in text
        assert "HELX_PMU_DERIVED_" not in text

    def test_u55_header_has_four_slots(self) -> None:
        text = render_c_header(program_for(NpuFamily.ETHOS_U55))
        assert "#define HELX_PMU_NSLOTS 4U" in text
        assert text.count("{ETHOSU_PMU_") == 4


def test_sram_stall_profile_captures_both_directions_in_one_pass() -> None:
    program = program_for(NpuFamily.ETHOS_U85, "sram_stall")
    assert program.events == (
        "ETHOSU_PMU_NPU_ACTIVE",
        "ETHOSU_PMU_MAC_ACTIVE",
        "ETHOSU_PMU_SRAM_RD_STALL_LIMIT",
        "ETHOSU_PMU_SRAM_RD_TRAN_REQ_STALLED",
        "ETHOSU_PMU_SRAM_WR_STALL_LIMIT",
        "ETHOSU_PMU_SRAM_WR_TRAN_REQ_STALLED",
        "ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED",
        "ETHOSU_PMU_SRAM_WR_DATA_BEAT_WRITTEN",
    )
    assert len(program.slots) == slot_count(NpuFamily.ETHOS_U85)
    for family in (NpuFamily.ETHOS_U55, NpuFamily.ETHOS_U65):
        with pytest.raises(ValueError, match="No PMU program"):
            program_for(family, "sram_stall")


def test_sram_transactions_pair_accepted_requests_with_all_energy_beats() -> None:
    program = program_for(NpuFamily.ETHOS_U85, "sram_transactions")
    assert program.events == (
        "ETHOSU_PMU_NPU_ACTIVE",
        "ETHOSU_PMU_MAC_ACTIVE",
        "ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED",
        "ETHOSU_PMU_SRAM_WR_DATA_BEAT_WRITTEN",
        "ETHOSU_PMU_EXT_RD_DATA_BEAT_RECEIVED",
        "ETHOSU_PMU_EXT_WR_DATA_BEAT_WRITTEN",
        "ETHOSU_PMU_SRAM_RD_TRANS_ACCEPTED",
        "ETHOSU_PMU_SRAM_WR_TRANS_ACCEPTED",
    )
    assert all(slot.unit is PmuUnit.TRANSACTIONS for slot in program.slots[-2:])
    for family in (NpuFamily.ETHOS_U55, NpuFamily.ETHOS_U65):
        with pytest.raises(ValueError, match="sram_transactions"):
            program_for(family, "sram_transactions")

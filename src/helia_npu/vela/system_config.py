"""Vela's system configuration for a SoC: the ``.ini`` nobody should have to write by hand."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..soc import (
    NPU_PORT_CLASS,
    VELA_AXI_NAME,
    VELA_PORT_TYPE,
    MemoryPlacement,
    MemoryTiming,
    SocDefinition,
    SocError,
    TimingOverrides,
    derive_timing,
)


def system_config_name_for(soc: SocDefinition, mode: str) -> str:
    """The ``[System_Config.*]`` section name for a SoC and mode, e.g. ``atomiq110_ULP``.

    Built from the canonical ``id``, never the human-facing ``name``: product
    names change for product reasons and the strings recorded in results,
    summary CSV names and generated ``.ini`` files must not. Bundles written
    before 2026-09-20 carry ``AT110_ULP``; nothing re-derives a name from a
    bundle, so they stay readable.
    """
    return f"{soc.id}_{mode}"


def _port1_latency_clamp(ports: Mapping[int, MemoryTiming], overrides: TimingOverrides | None) -> str | None:
    """Why, and whether, the generated section presents port 1 as no slower than port 0.

    Vela decides per layer whether to copy weights into the port-0 memory
    (the SRAM cache) or stream them from port 1, by comparing the two ports'
    latency numbers; it has no model of a port saturating. When the NPU clock
    exceeds port 0's AXI clock the SRAM port is bandwidth-bound and every
    cached weight crosses it twice, so the copy that Vela's model calls
    cheaper is what stalls the NPU (FPGA and RTL). The clamp steers
    Vela's placement; it does not describe port 1, whose timing the FVP still
    receives unchanged. Modes where the NPU is no faster than the fabric keep
    the physical numbers, since caching costs nothing there and saves port-1
    energy. Vela also weighs ``ports_used``: with two port-0 ports declared it
    caches regardless of latency, so the clamp applies to single-port SoCs
    only. A run that overrides the port-1 memory's timing explicitly is
    sweeping it on purpose, so the override reaches Vela as written; an
    override of the FVP's rate window alone does not count, nor does one a
    tool derived (``MemoryTimingOverride.deliberate`` false). The
    clamp also needs port 1 to be able to hide its latency, i.e. a deeper
    outstanding-read queue than port 0: a port-1 memory that is itself
    rate-limited (the RRAM variant, 2 in flight like SSRAM) keeps its physical
    numbers, since streaming from it is not known to be better. Returns the
    ini comment, or ``None`` when no clamp applies.
    """
    port0, port1 = ports.get(0), ports.get(1)
    if port0 is None or port1 is None:
        return None
    port1_override = overrides.memories.get(port1.name) if overrides is not None else None
    # A timing-adapter-only override is no sweep, and neither is one a tool derived (deliberate=False).
    if port1_override is not None and port1_override.reaches_vela and port1_override.deliberate:
        return None
    if port0.bandwidth >= 1.0:  # port 0 keeps up with the NPU in this mode (derived timing, overrides included)
        return None
    if port0.ports_used != 1:  # Vela weighs ports_used too; with two ports declared, latency alone does not steer it
        return None
    if port1.max_outstanding <= port0.max_outstanding:
        return None
    if port1.read_latency <= port0.read_latency and port1.write_latency <= port0.write_latency:
        return None
    return (
        f"; Port 1 ({port1.name}) latency is presented as no worse than port 0 ({port0.name}): port 0 delivers "
        f"{port0.bandwidth:g} of the NPU's demand in this mode, so it is bandwidth-bound and latency alone should not "
        "steer Vela toward caching weights through it. See helia_npu.vela.system_config."
    )


def write_system_config(
    soc: SocDefinition,
    mode: str,
    output_path: Path,
    extra_placements: list[MemoryPlacement] | None = None,
    timing_overrides: TimingOverrides | None = None,
    placement: MemoryPlacement | None = None,
) -> str:
    """Write the Vela ``.ini`` for one SoC and operating point; return the system-config section name.

    The file carries a ``[System_Config.<name>_<mode>]`` section with the
    derived timing, then one ``[Memory_Mode.*]`` section per placement the
    SoC's ports can express, named by physical memory
    (``const_hblram__arena_ssram__cache_ssram``), with Vela's three mode names
    (``Sram_Only``, ``Shared_Sram``, ``Dedicated_Sram``) as ``inherit`` aliases
    of them. Custom placements in *extra_placements* get their own section.
    *placement* is the run's: it decides which of the memories wired to a port
    the port carries (the port's first memory when not given), and that
    memory's timing is what the section holds.
    """
    if placement is None and extra_placements:
        # Callers that name the run's placement only among the extras (compile_for_target's documented use)
        # still get its memories; extras that disagree about a port's memory need placement= to decide.
        occupancies = {tuple(sorted(soc.occupants(extra).items())) for extra in extra_placements}
        if len(occupancies) > 1:
            raise SocError(
                "extra_placements put different memories on one port; pass placement= to say which one this "
                "run's ports carry"
            )
        placement = extra_placements[0]
    timing = derive_timing(soc, mode, timing_overrides)
    occupancy = soc.occupants(placement)
    ports = timing.port_timing(occupancy)
    system_config_name = system_config_name_for(soc, mode)
    clamp_note = _port1_latency_clamp(ports, timing_overrides)

    lines = [
        "; Auto-generated from the SoC definition — do not edit",
        f"; SoC: {soc.name} ({soc.id}, {soc.origin.value})  Mode: {mode}",
        "",
        "; System Configuration",
    ]
    if clamp_note:
        lines.append(clamp_note)
    lines.extend(
        [
            f"[System_Config.{system_config_name}]",
            f"core_clock={timing.npu_clock_hz:g}",
        ]
    )

    # AXI port declarations, each with the physical memory it carries on this SoC
    for port_idx in sorted(VELA_PORT_TYPE):
        carried = occupancy.get(port_idx, "nothing")
        others = [m.name for m in soc.memories_on_axi(port_idx) if m.name != carried]
        if others:
            carried = f"{carried} (also wired: {', '.join(others)})"
        lines.append(f"; axi{port_idx}: Arm's {NPU_PORT_CLASS[port_idx]} port class; on this SoC it carries {carried}")
        lines.append(f"axi{port_idx}_port={VELA_PORT_TYPE[port_idx]}")

    # Per-port memory parameters
    for port_idx in sorted(VELA_PORT_TYPE):
        port_type = VELA_PORT_TYPE[port_idx]
        mt = ports.get(port_idx)
        if mt is None:
            continue
        read_latency, write_latency = mt.read_latency, mt.write_latency
        if clamp_note and port_idx == 1:
            port0 = ports[0]
            read_latency = min(read_latency, port0.read_latency)
            write_latency = min(write_latency, port0.write_latency)
        lines.extend(
            [
                f"{port_type}_clock_scale={mt.bandwidth}",
                f"{port_type}_ports_used={mt.ports_used}",
                f"{port_type}_burst_length={mt.burst_length}",
                f"{port_type}_read_latency={read_latency}",
                f"{port_type}_write_latency={write_latency}",
                f"{port_type}_max_reads={mt.max_outstanding}",
                f"{port_type}_max_writes={mt.write_outstanding}",
            ]
        )

    lines.extend(
        [
            "",
            "; Memory Modes: where const (weights), arena (activations) and cache go, by physical memory.",
            "; Vela's own mode names are aliases of the placement they mean on this SoC.",
        ]
    )
    written: set[str] = set()

    def section(label: str, axi_ports: tuple[int, int, int]) -> None:
        const_axi, arena_axi, cache_axi = axi_ports
        lines.extend(
            [
                "",
                f"[Memory_Mode.{label}]",
                f"const_mem_area={VELA_AXI_NAME[const_axi]}",
                f"arena_mem_area={VELA_AXI_NAME[arena_axi]}",
                f"cache_mem_area={VELA_AXI_NAME[cache_axi]}",
            ]
        )
        written.add(label)

    for mode, standard in soc.standard_placements(occupancy).items():
        label = soc.placement_label(standard)
        section(label, soc.placement_axi_ports(standard))
        lines.extend(["", f"[Memory_Mode.{mode.value}]", f"inherit=Memory_Mode.{label}"])
        written.add(mode.value)

    # Custom placements: anything the standard names do not cover.
    for extra in extra_placements or ():
        label = soc.vela_memory_mode_label(extra)
        if label in written:
            continue
        section(label, soc.placement_axi_ports(extra))
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return system_config_name

# NPU ports, SoC ports, Vela names and physical memories

The same two memory paths carry a different name at every layer between the
Ethos-U85 and a result file. This page is the single place that joins them.
Arm calls the two port classes `AXI_SRAM` and `AXI_EXT`; "SRAM-class" and
"EXT-class" below mean those. `tests/test_port_mapping_doc.py`
parses the first three tables and checks their rows against the constants in
`helia_ethos.soc`, `helia_ethos.pmu` and `helia_ethos.fvp`, so those tables cannot
drift from the code; the prose and the last table are documentation.

## The two port classes

| NPU port class (Arm) | SoC `axi_port` | Vela memory name | Vela AXI name | Vela ini keys | What lives there in each memory mode | AT110 physical memory | PMU host field prefix | FVP timing adapter (Corstone-320) | FVP timing adapter (Corstone-300) |
|---|---|---|---|---|---|---|---|---|---|
| `AXI_SRAM` | 0 | `Sram` | `Axi0` | `Sram_clock_scale`, `Sram_ports_used`, `Sram_burst_length`, `Sram_read_latency`, `Sram_write_latency`, `Sram_max_reads`, `Sram_max_writes` | arena and cache always; constants too in `Sram_Only` | SSRAM, 3 MB | `npu_sram_` | TA0 at `0x50006000` | TA0 at `0x48103000` |
| `AXI_EXT` | 1 | `Dram` | `Axi1` | `Dram_clock_scale`, `Dram_ports_used`, `Dram_burst_length`, `Dram_read_latency`, `Dram_write_latency`, `Dram_max_reads`, `Dram_max_writes` | constants in `Shared_Sram` and `Dedicated_Sram`; arena too in `Dedicated_Sram` | HBLRAM by default, or PSRAM or RRAM when the placement names it; one at a time | `npu_ext_` | TA1 at `0x50006800` | TA1 at `0x48103200` |

Vela also knows `OnChipFlash` and `OffChipFlash`. helx never assigns a memory
to them; they appear only as zero columns (`on_chip_flash_memory_used`,
`off_chip_flash_memory_used`) in Vela's summary.

Port numbers are the SoC file's `axi_port` field. Every memory in a SoC
definition sits on exactly one of the two ports. A port may have several
memories wired (AT110's port 1 has HBLRAM, PSRAM and RRAM) but carries one of
them in a run: the first declared unless the run's `memory_placement` names
another. That memory's timing is what Vela and the timing adapter get and its
pJ/bit is what the energy terms use.

## Memory modes are port patterns

A helx `memory_placement` names where constants, arena and cache go. The three
standard Vela mode names are the three patterns below; any other pattern is
labelled `const_X__arena_Y__cache_Z` and run as the executable mode its arena
and constants imply (arena on port 1 means `Dedicated_Sram`, otherwise
constants on port 1 means `Shared_Sram`, otherwise `Sram_Only`; the cache port
does not change the choice).

| Vela memory mode | const | arena | cache |
|---|---|---|---|
| `Sram_Only` | port 0 | port 0 | port 0 |
| `Shared_Sram` | port 1 | port 0 | port 0 |
| `Dedicated_Sram` | port 1 | port 1 | port 0 |

In `Sram_Only` and `Shared_Sram` the cache and the arena are the same memory,
so there is no separate cache region; a separate one exists only in
`Dedicated_Sram`. `vela.arena_cache_size` is passed in every mode: it bounds
the port-0 bytes Vela may use for the arena and its weight cache together.

## PMU counters name the ports a third way

The driver's event symbols differ by NPU family; the host field names do not.

| NPU family | Port 0 events | Port 1 events | Host fields |
|---|---|---|---|
| Ethos-U55, U65 | `ETHOSU_PMU_AXI0_*` | `ETHOSU_PMU_AXI1_*` | `npu_sram_*`, `npu_ext_*` |
| Ethos-U85 | `ETHOSU_PMU_SRAM_*` (and `SRAM0_*` for one of the two SRAM-class interfaces) | `ETHOSU_PMU_EXT_*` (and `EXT0_*`, `EXT1_*` sub-counters) | `npu_sram_*`, `npu_sram0_*`, `npu_ext_*`, `npu_ext0_*`, `npu_ext1_*` |

The driver also defines `SRAM1_*` and the matching `EXT0_*` / `EXT1_*`
families, one per interface of each class. helx's programs read `SRAM0_*` but
not `SRAM1_*`, so in a helx bundle the second SRAM-class interface's traffic is
`SRAM_RD` minus `SRAM0_RD`; FPGA captures with heliaPROFILER read both.

## Physical masters (derived, not confirmed)

The Ethos-U85 exposes three AXI masters on AT110. Two of them form the
SRAM class and stripe 128-byte interleaved traffic between them; the third
carries EXT-class traffic (the driver's `EXT0`/`EXT1` sub-counters exist for
configurations with two EXT interfaces). On AT110 the SRAM-class masters reach
two of the three SSRAM ports and the EXT-class master reaches the HBLRAM
controller. Which numbered master carries which class is derived from the
SoC's AXI performance study and from FPGA counters, not from an integration
table; the AT110 architecture doc in helia-estimator carries that caveat on
its interconnect row.

`ports_used` in the SoC file is what Vela is told, not what is wired. AT110
ships `ports_used: 1` for SSRAM because FPGA counters show one SRAM-class
interface carrying about 90% of SRAM reads. Vela has no model of a port
saturating; what keeps it from caching weights through the busy port at HP is
the generated ini presenting port 1 as no slower than port 0, and Vela 5.2
honours that only for a single-port declaration (with two ports declared it
caches regardless of latency).

## Modes are NPU clocks

`ULP` and `HP` name the NPU clock, 100 MHz and 500 MHz. The CPU has its own
`ULP` / `LP` / `HP` clocks (100 / 250 / 500 MHz) selected independently; helx
does not model the CPU, so `LP` never appears as a platform mode. Memory AXI
clocks are fixed at 250 MHz (PSRAM's XSPI bridge at 62.5 MHz), so at HP a port delivers half of what the NPU can
consume per cycle.

## Where each name shows up

| Name | Appears in |
|---|---|
| `axi_port: 0 / 1` | SoC and platform YAML |
| `Sram` / `Dram`, `Axi0` / `Axi1` | generated Vela ini; lowercase `sram_*` / `dram_*` in Vela's summary columns |
| `const` / `arena` / `cache` by physical memory | config `memory_placement` (the idiom), results `memory.*_placement`, Vela ini sections named `const_X__arena_Y__cache_Z` |
| `Sram_Only` / `Shared_Sram` / `Dedicated_Sram` | config `memory_modes` (an alias for the placement it means on the platform), results `run.memory_mode` when the placement is one of the three, Vela ini alias sections, the runtime build's `HELX_MEMORY_MODE` |
| `npu_sram_*` / `npu_ext_*` | result bundles (`run_summary.*`), energy terms `sram_uj` / `ext_uj` |
| `AXI0` / `AXI1`, `SRAM` / `EXT` | PMU driver symbols in the FVP runner and captures |
| TA0 / TA1 | FVP timing-adapter registers, `ETHOSU_TA_*_0` / `_1` build defines |
| SSRAM / HBLRAM / PSRAM / RRAM | physical memories, platform names, energy YAML |

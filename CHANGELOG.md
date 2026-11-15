# Changelog

All notable changes to helia-npu. The package follows semantic versioning once
it reaches 1.0; until then a minor bump may change an API.

## [0.2.1](https://github.com/AmbiqAI/helia-ethos/compare/v0.2.0...v0.2.1) (2026-11-15)


### Bug Fixes

* keep public preparation imports offline ([#6](https://github.com/AmbiqAI/helia-ethos/issues/6)) ([984889c](https://github.com/AmbiqAI/helia-ethos/commit/984889c160d0718e9a0da5fefd8a829bcf89100c))

## [0.2.0](https://github.com/AmbiqAI/helia-ethos/compare/v0.1.0...v0.2.0) (2026-11-15)


### Features

* add shared model preparation and retained artifact inspection ([#5](https://github.com/AmbiqAI/helia-ethos/issues/5)) ([fc27143](https://github.com/AmbiqAI/helia-ethos/commit/fc27143d2e0cbdb24d41b8a606be2fe9d06f9a63))
* extract standalone helia-ethos package ([#1](https://github.com/AmbiqAI/helia-ethos/issues/1)) ([39b17d3](https://github.com/AmbiqAI/helia-ethos/commit/39b17d3302d3815e3be27d9ae15649f465ec631d))


### Bug Fixes

* correct model inspection and reject invalid timing clocks ([#3](https://github.com/AmbiqAI/helia-ethos/issues/3)) ([bdae235](https://github.com/AmbiqAI/helia-ethos/commit/bdae23593b072400761d8c342c4d9179f83b4ee5))

## 0.1.0 (unreleased, developed inside helia-estimator)

- `MemoryTimingOverride.deliberate` (default true): an override a tool derives sets it false, so it reaches
  Vela without lifting the port-1 latency clamp that a deliberate port-1 sweep lifts.
- SoC definitions with per-value provenance (confirmed, fitted, unverified),
  operating modes, memories with AXI port bindings, timing derivation in NPU
  cycles, and standard and custom memory placements (`helia_npu.soc`,
  `soc_schema`); the golden `atomiq110` ships with the package.
- Targets: which realization of a SoC a measurement is attributed to and
  whether its timing stands for the part (`helia_npu.target`).
- Vela: the system config for a SoC and operating point, `compile_for_target`,
  the typed summary and the per-layer table (`helia_npu.vela`).
- The Corstone FVP target matrix and the PMU programs behind each profile
  (`helia_npu.fvp`, `helia_npu.pmu`). `FVP_TARGET_PROFILES` is a tuple of
  FVP models (the Corstone-300's U55 and U65, the Corstone-320's U85), and
  `resolve_fvp_target_profile(subsystem, npu)` picks one by NPU; it refuses a
  subsystem that does not simulate the NPU, which `validate_fvp_target_for_npu`
  used to check. A simulator target may name its `fvp_subsystem`.
- `NpuFamily` lives in `helia_npu.soc` (still importable from `helia_npu.fvp`)
  and gives each NPU's AXI beat width; a SoC's `axi_bus_width_bits` defaults to
  it and must equal it.
- Model inspection and STREAM_SPLIT fencing with a boundaries sidecar
  (`helia_npu.model`, `helia_npu.fencing`), behind the `models` extra.
- `VelaMemoryMode`, `TimingAdapterPort`, `FenceBoundaries`, `FenceSidecarError`.
- Packaging: the core depends on pydantic and PyYAML only; the `models` extra
  adds LiteRT, the `vela` extra the compiler; every public name is importable
  from `helia_npu` or `helia_npu.vela`, the ones behind `models` on first use;
  `py.typed` ships.

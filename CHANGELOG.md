# Changelog

All notable changes to helia-npu. The package follows semantic versioning once
it reaches 1.0; until then a minor bump may change an API.

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

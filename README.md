# helia-ethos

Ethos-U NPU support for the helia toolchain: what a SoC looks like to the NPU,
how to compile a model for it with Vela, which Corstone FVP can run the
result, and how to read the PMU. One package so that every tool in the chain
(helia-estimator, heliaAOT, heliaRT, heliaPROFILER) computes against the same
silicon facts and spells them the same way.

This standalone package was extracted from heliaEstimator. Hardware catalog facts
come from helia-platforms; this package owns Ethos-specific compilation and timing
policy. Experiment execution, runtime integration and result bundles remain with
its consumers.

The proposed [extraction and preparation contract](docs/extraction_contract.md)
defines ownership, the planned consumer API and the implementation sequence.
The shared API is described in [model preparation and inspection](docs/preparation.md).
Use it for retained compilation evidence and optional fences; the low-level APIs
below remain available.

## Install

```bash
pip install helia-ethos            # the core: SoC and target definitions, timing, system config, PMU programs
pip install 'helia-ethos[vela]'    # plus the Vela compiler, for compile_for_target
pip install 'helia-ethos[models]'  # plus LiteRT, for reading and fencing .tflite files
pip install 'helia-ethos[all]'
```

The core uses pydantic, PyYAML and helia-platforms. A function behind an extra says so
when the extra is missing (`pip install 'helia-ethos[models]'`). Python 3.11–3.13 are tested; optional extras also depend on upstream wheel
availability. The package ships `py.typed`.

## What is in it

| Module | Owns |
|---|---|
| `helia_ethos.soc`, `soc_schema` | The SoC as the NPU sees it: identity (`id`, `origin`, `revision`), NPU instance, memories with AXI bindings, operating modes, per-value provenance (confirmed, fitted, unverified); `load_soc`, `derive_timing`, placements, `VelaMemoryMode`, `TimingAdapterPort`. The catalog-backed `atomiq110` reference ships with the package |
| `helia_ethos.target` | Realizations of a SoC (silicon, FPGA drop, RTL, simulator): what each implements, its clocks, whether its timing stands for the part |
| `helia_ethos.vela` | The Vela `.ini` for a SoC and operating point, `compile_for_target`, the typed summary (`VelaSummary`, `VelaColumn`) and the per-layer table |
| `helia_ethos.fvp` | The Corstone FVP target matrix: which subsystem executes which engine and memory mode, base addresses, timing-adapter layout, required PMU fields |
| `helia_ethos.pmu` | The PMU program behind each profile name: events per counter slot, units, derived fields; `render_c_header` for the firmware, the host's required fields from the same object |
| `helia_ethos.npu_reference` | Arm's reference energy per cycle per NPU configuration, with provenance |
| `helia_ethos.model` (`models` extra) | What a `.tflite` contains: parameters, constant bytes, operators, I/O, the embedded command-stream size |
| `helia_ethos.fencing` (`models` extra) | `ETHOS_U_STREAM_SPLIT` injection to break a command stream at chosen operators, `plan_fences` from Vela's own estimate, and the boundaries sidecar (`FenceBoundaries`) |

Everything a consumer needs is importable from `helia_ethos`; the names behind
the `models` extra are imported on first use.

The two memory ports and every name they carry (SoC `axi_port`, Vela
`Sram`/`Dram`, PMU `AXI0`/`SRAM`, FVP TA0/TA1, the physical memories) are in
[docs/port_mapping.md](https://github.com/AmbiqAI/helia-ethos/blob/main/docs/port_mapping.md), which a test keeps in step with
the code. Changes are in [CHANGELOG.md](https://github.com/AmbiqAI/helia-ethos/blob/main/CHANGELOG.md).

## Using it

Everything below runs with the package alone, no helia-estimator, no config
files. The `vela` extra is needed where a compile happens.

### Compile for a SoC and operating point (heliaAOT, heliaPROFILER)

```python
from pathlib import Path
from helia_ethos import load_soc
from helia_ethos.vela import compile_for_target

soc = load_soc("atomiq110")  # catalog-backed; or load_soc(Path("my_soc.yaml"))
compiled = compile_for_target(
    Path("kws.tflite"),
    soc,
    "ULP",  # or "HP": the NPU clock, not the CPU's
    memory_mode="Shared_Sram",  # weights on port 1 (HBLRAM), arena + cache on port 0 (SSRAM)
    arena_cache_size=1 << 20,  # bytes of SSRAM Vela may use for the arena and weight cache
    output_dir=Path("out"),
)
compiled.tflite  # the model with its Ethos-U command stream
compiled.summary  # VelaSummary: nn_macs, cycles_total, ...
compiled.per_layer  # list[LayerRow] from --verbose-performance
compiled.config_file  # the generated .ini, for reproducibility
```

`memory_mode` can also name a custom placement passed in `extra_placements`;
`timing_overrides` changes latency, bandwidth or outstanding transactions for
one compile.

### Ask where to cut a graph, then cut it (heliaAOT chunking, heliaPROFILER per-segment)

```python
from helia_ethos.fencing import fence_model, plan_fences
from helia_ethos.vela import read_tflite_graph

ops = read_tflite_graph(Path("kws.tflite"))
plan = plan_fences(compiled.per_layer, ops, cycle_budget=50_000, max_op_cycles=200_000, model=Path("kws.tflite"))
plan.coverage.summary()  # what could and could not be attributed, and why
fenced = fence_model(Path("kws.tflite"), plan, Path("out/fenced"))
# fenced.fenced_model_path: ETHOS_U_STREAM_SPLIT after each fence; compile it the same way.
# fenced.source_sha256 and fenced.fence_tensor_names are the join keys; the sidecar carries them per segment.
```

Cycle-based policies plan on Vela's estimate, attributed to original operators
by name; rows the planner cannot place are reported in `plan.coverage`, and a
budget plan raises below full coverage unless `min_coverage=` accepts the
gap. A plan bounds latency on nothing; measure it. `ETHOS_U_STREAM_SPLIT` runtime
support is the consumer's: the fenced model needs an identity (memcpy)
custom-op kernel registered in the runtime.

### Say which realization a number came from

```python
from helia_ethos import load_target

target = load_target("atomiq110-fvp-corstone-320")
target.timing_is_representative  # False: the FVP's timing does not stand for the part
target.label()  # what a result records next to the SoC id and revision
```

### Program the PMU the same way as everyone else (heliaPROFILER)

```python
from helia_ethos import NpuFamily, program_for, render_c_header

program = program_for(NpuFamily.ETHOS_U85, "port_stall")
render_c_header(program)  # the table the firmware compiles
program.required_fields  # what the host must find in the report
```

### Read the SoC (heliaRT, neuralSPOT-X, anything that wants the memory system)

```python
from helia_ethos import derive_timing, load_soc

soc = load_soc("atomiq110")
soc.memories["hblram"].axi_port  # 1
soc.provenance_label("memories.hblram")  # the block's status
soc.provenance["memories.hblram"].for_field("read_latency_ns").status  # one value's status
derive_timing(soc, "HP").memories["ssram"].read_latency  # NPU cycles at the HP clock
```

## Naming

The two memory ports and every name they carry (SoC `axi_port`, Vela `Sram`/`Dram`, PMU `AXI0`/`SRAM`, FVP TA0/TA1, the physical memories) are joined in [docs/port_mapping.md](https://github.com/AmbiqAI/helia-ethos/blob/main/docs/port_mapping.md), which a test keeps in step with the code.

Aligned with heliaPROFILER and heliaAOT, which both already use `atomiq110`:

| Concept | Spelling | Why |
|---|---|---|
| Canonical SoC id | `atomiq110` | Generates identifiers; must be stable |
| Official name | `Atomiq110` | Human-facing text only |
| Engineering name | `AT110` | A valid alias, matched case-insensitively |

The canonical id, never the official name, is what should generate Vela
system-config sections, filenames and config keys, because product names
change for product reasons and identifiers must not. Since 2026-09-20 the section
is `atomiq110_ULP` (id + mode); bundles written earlier carry `AT110_ULP`,
and nothing re-derives a name from a bundle, so both read.

The constraints the SoC model is built to are in [docs/soc_design_notes.md](https://github.com/AmbiqAI/helia-ethos/blob/main/docs/soc_design_notes.md).

## helia-platforms integration

The adapter supports the released helia-platforms 0.3 series from PyPI.
The development lock selects 0.3.0; installed wheels declare `>=0.3,<0.4`.
Refresh deliberately with `uv lock --upgrade-package helia-platforms` and
validate the catalog integration before shipping the updated lock.

`helia_ethos.catalog.read_catalog_npu` accepts an explicit `Catalog`, target name
and optional NPU instance id. It returns frozen facts with the full
`Catalog.identity()`, including source-overlay digests. It retains native
provenance and unresolved reach. Multiple NPUs require explicit selection.

`helia_ethos.catalog_policy.project_catalog_soc` joins those facts to an explicit
`SocModelPolicy`. The result carries the executable `SocDefinition`, original
catalog facts and policy identity together:

```python
from helia_platforms import builtin
from helia_ethos.catalog import read_catalog_npu
from helia_ethos.catalog_policy import at110_model_policy, project_catalog_soc
from helia_ethos.vela import write_system_config
from pathlib import Path

facts = read_catalog_npu("atomiq110", catalog=builtin())
projection = project_catalog_soc(facts, policy=at110_model_policy())
write_system_config(projection.soc, "HP", Path("at110-catalog.ini"))
print(projection.facts.identity.to_dict())
print(projection.policy_id)
```

### Facts and policy boundary

| Input | Owner in this slice |
|---|---|
| NPU family, MAC count, bus width, clock points, SRAM hash | Catalog; policy checks accelerator compatibility |
| Memory identities, capacities, base addresses, reach and native provenance | Catalog; full facts retained beside the projection |
| Compiler port class and default occupant | Explicit helia-ethos policy, independent of catalog order and access-port fields |
| Read/write latencies, fabric-rate assumption, effective queues, burst selection, Vela ports-used | Existing helia-ethos YAML, consumed as model inputs until citable hardware facts are settled in platforms #93 |
| Hardware transaction ceilings | Retained as catalog facts, never substituted for effective model queues |
| Energy coefficients | HELX; this adapter does not read or move them |

The AT110 policy reads `model_policies/atomiq110.yaml`; hardware facts
come from the catalog. `at110_model_policy()` excludes comparison-only PSRAM
unless explicitly selected. The shipped `load_soc("atomiq110")` reference
selects it to preserve existing placement experiments, with unverified reach.
The bare-SoC catalog has XSPI apertures but no populated PSRAM device.

AT110 checks cover ULP/HP, SRAM-only, shared and dedicated HBLRAM/RRAM placements:
generated Vela section values and FVP timing-adapter profiles must match. Expected
non-numerical differences are the derived origin, catalog provenance, retained
catalog identity and omission of the PSRAM alternative in comments. Physical
catalog capacity is not silently used as a Corstone linker capacity. The FPGA
implementation is refused by this silicon policy rather than inheriting silicon
timings. Missing bus width/hash or modeled-memory reach is actionable.

### Default AT110 loading

The standard reference now uses catalog hardware facts and the packaged
`model_policies/atomiq110.yaml` for timing and placement. PSRAM is an explicit
unverified comparison assumption; it does not imply a populated catalog device.
FPGA implementation policies and package extraction remain separate work.

helia-platforms 0.3 removed the deprecated access `port` key (helia-platforms #159), so an
overlay that still writes it no longer loads. The adapter never read that field
as placement policy. No helia-platforms repository changes are required for the
current SoC projection.

HELX exposes both the standard reference and explicit catalog selection through its shared platform loader. See
[the catalog integration guide](https://github.com/AmbiqAI/helia-estimator/blob/main/docs/platform_catalog.md) for configuration,
identity, energy-profile compatibility and replay behavior.

## Development and releases

Run `uv sync --frozen`, then `uv run --frozen python scripts/gates.py`.
The gates check the lock, lint, types and package tests, including real Vela
compilation of generated models. No model download or FVP setup is required.
Run `uv run --frozen python scripts/check_distribution.py` to build distributions
and verify core-only and full-extra wheel installations outside the checkout.

CI covers Python 3.11–3.13 on Linux, plus Python 3.12 on macOS and Windows.
Release Please owns version and changelog changes. Release publication requires
the PyPI trusted publisher and protected `pypi` environment to be configured.
See [release setup](docs/releasing.md).

The inherited 0.1.0 changelog describes the pre-extraction `helia-npu` package;
its old import names are historical. Use `helia_ethos` and the installation
examples above. PMU program identifiers retain their `helia-npu/` prefix so
existing and new captures identify the same counter program.

The source is publicly available under the Ambiq Evaluation Source License;
see [LICENSE](LICENSE) for permitted use and redistribution restrictions.

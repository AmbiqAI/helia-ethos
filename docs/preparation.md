# Model preparation and inspection

`prepare_model()` compiles an original LiteRT model for an explicit Ethos-U
configuration. It can insert stream-split fences and returns the compiled model
with a small, relocatable evidence record. It does not build firmware, execute
inference, choose runtime kernels, or estimate energy. Consumers own those steps.

Install `helia-ethos[all]` for preparation. Inspection needs only `[models]`;
request/record types remain importable with the core package alone.

## Prepare a model

```python
from pathlib import Path
from helia_platforms import builtin
from helia_ethos import MemoryPlacement, NpuCompileSettings, PrepareRequest, prepare_model
from helia_ethos.catalog import read_catalog_npu
from helia_ethos.catalog_policy import at110_model_policy, project_catalog_soc

# The application resolves its catalog once, including any overlays.
facts = read_catalog_npu("atomiq110", catalog=builtin())
projection = project_catalog_soc(facts, policy=at110_model_policy())
request = PrepareRequest(
    source=Path("model.tflite"),
    target=facts,
    settings=NpuCompileSettings(
        soc=projection.soc,
        mode="ULP",
        placement=MemoryPlacement(const="hblram", arena="ssram", cache="ssram"),
        policy_id=projection.policy_id,
        arena_cache_size=0,
    ),
    output_dir=Path("prepared"),  # must not already exist
)
prepared = prepare_model(request)
print(prepared.tflite)
print(prepared.record_path)
print(prepared.info.custom_operators)
print(prepared.compilation.summary)  # typed VelaSummary, already parsed
print(prepared.compilation.summary_path)
print(prepared.compiler_log)
```

The policy in this example is an explicit AT110 **simulation/compiler assumption**,
not hardware qualification or a default for FPGA preparation. For other targets,
pass their resolved `CatalogNpuFacts` and an explicit compiler-facing `SocDefinition`
with the required timings, mode and placement. Preparation does not reload the
catalog or weaken the existing AT110 projection's implementation-target refusal.

What-if experiments use `CustomNpuTarget(identity, Accelerator(family, macs))`
instead of catalog facts. For example, `Accelerator(NpuFamily.ETHOS_U85, 128)`
requires a compiler model with that same family and MAC count. The resulting
record identifies the target as custom, without claiming catalog hardware facts.

`NpuCompileSettings` also accepts existing typed `TimingOverrides` and an explicit
`vela_bin`. The policy ID, full effective compiler model, generated configuration,
normalized arguments, compiler version output and executable hash are retained.
Compiler-reported accelerator/system-config/memory-mode fields are retained
separately from the requested settings and decoded per-region identities.
The executable hash identifies the launcher bytes; recorded Python package
versions describe the preparing environment, not a cryptographic identity for an
external compiler's complete installation.

`prepared.compilation` is the final `CompiledModel` from the existing compiler API,
including its generated `config_file`, `summary_path`, typed `summary` and captured
`stdout`. `compiler_log` identifies that compilation's saved output. These paths
refer to the current preparation directory; they do not trigger another compile.
After relocation, use the retained record's artifact paths and inspection API.

## CPU-only compiler output

Preparation requires at least one Ethos-U region by default. Consumers that also
handle CPU-only models can explicitly allow a successful compile without one:

```python
from dataclasses import replace
from helia_ethos import NpuRequirement

request = replace(request, npu_requirement=NpuRequirement.OPTIONAL)
prepared = prepare_model(request)
```

This option allows zero regions; it does not promise NPU coverage or change Vela's
operator support. Compatibility is `unknown` for zero regions. A decoded region
with the wrong accelerator is still refused. The record retains this policy so
inspection applies the same rule without recompilation.

## Optional fences

```python
from dataclasses import replace
from helia_ethos import ExplicitFences, BudgetFences

# Split after original operators 3 and 8, before their successors.
request = replace(request, fences=ExplicitFences((3, 8)))

# Alternatively, derive cuts from Vela's estimated cycles and attribution.
request = replace(request, fences=BudgetFences(cycle_budget=50_000, min_coverage=1.0))
```

Explicit cuts compile once after rewriting. Budget planning compiles a baseline,
records its attributed/unattributed cycle coverage, and recompiles if cuts are
needed. Final NPU invocations are mapped to source segments using the existing
fence sidecar; CPU-only segments need not have an NPU invocation.

A fence is a runtime scheduling boundary, not preemption or a deadline guarantee.
The consumer must register every reported custom operator/version, including
`ETHOS_U_STREAM_SPLIT` where retained. Its paired tensors and runtime implementation
remain the consumer's responsibility.

Already compiled sources and sources already containing stream-split operators
are refused by preparation. Inspect compiled inputs directly. Existing low-level
fencing/compilation APIs remain available for callers that manage their own plans.

## Inspect, retain and relocate

```python
from helia_ethos import Accelerator, NpuFamily, inspect_prepared_model

info = inspect_prepared_model(Path("prepared/preparation.json"))
print(info.compatibility(Accelerator(NpuFamily.ETHOS_U85, 256)))
print(info.unavailable_optional)

# Standalone TFLite inspection returns observed facts without source provenance.
observed = inspect_prepared_model(Path("already_compiled.tflite"))
assert observed.record is None
```

Inspection checks **every used Ethos-U region**, across subgraphs. Supported COP1
headers expose the product/MAC configuration. Other/missing payload identities
stay unknown. `compatibility(expected)` reports incompatible if any decoded
region disagrees, compatible only if every region is decoded and agrees, and
unknown otherwise. A retained compiler request is separate evidence from decoded
payload identity. Inspection does not invoke Vela or imply runtime qualification.

The version-2 `PreparationRecord` records the NPU requirement alongside source identity, target identity,
compiler inputs, model I/O and custom-op requirements, per-region observations,
fence correspondence, package versions and artifact hashes. JSON fields are
closed and typed; unknown record schema versions are refused. Version-1 records
remain readable with their original required-NPU semantics; they cannot opt into
CPU-only output.

Retain `preparation.json` and all artifacts marked `required`: the final model,
compiler configuration and any fence sidecar needed for correspondence. Preserve
the relative directory layout. The original external source path is never needed.
`retain_source=True` adds an optional source copy. Rewritten graphs, compiler logs
and performance/debug outputs are optional; absence is reported, but present
files must match their hashes. The record is compilation evidence, not a signed
attestation or a measurement bundle.

Consumers can copy the whole directory into their own bundle. If they change its
layout, they must rewrite the record's relative artifact paths before sealing
their native manifest. Inspect the relocated record after deleting the original
workspace to verify the retained closure.

## Failure behavior

- `InvalidPreparation`: original model, target, settings, fences or destination
  cannot be used as requested.
- `CompilerUnavailable`: Vela cannot be located or identified.
- `CompilationFailed`: Vela fails or cannot produce its expected files.
- `InvalidPreparedArtifact`: required retained files are missing/mismatched, the
  result has no NPU region when required, its decoded accelerator disagrees, or
  fence mapping fails.

The source is never overwritten. Preparation reserves a fresh destination and
cleans its private scratch directory. Failures after reservation leave
`failure.txt` and available compiler diagnostics, but no `preparation.json`.
Choose a new output directory for a retry; there is no implicit reuse or cache.
Missing optional dependencies keep their actionable extra-installation error.

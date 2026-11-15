# helia-ethos extraction and model-preparation contract

Status: approved design baseline. The standalone extraction is merged; see
[preparation.md](preparation.md) for the implemented preparation surface and examples.
The existing low-level APIs remain available. Approval of this contract does not claim that another product has
implemented or accepted its adoption work.

## Objective

Extract the existing Ethos-U support package so HELX, heliaAOT and HPX can prepare
the same model using explicit target and compiler settings. Each product remains
independently useful and owns its execution. BENCH orchestrates those products;
it does not become the compiler or require a particular producer architecture.

The original HELX contract PR was documentation-only. This document preserves
its ownership and adoption plan; subsequent implementation is tracked separately.

## Ownership

| Owner | Responsibilities |
| --- | --- |
| helia-platforms | Hardware facts and identities for SoCs, implementations, parts and boards; NPU geometry, reachability, capacities, addresses, clocks and source provenance. |
| helia-ethos | Ethos-U model inspection; compiler-facing projection of resolved facts; explicit Vela configuration and compilation; graph fencing and segment correspondence; NPU PMU definitions; reusable NPU timing/placement policy machinery and static Corstone capability descriptions. |
| HELX | Experiment preparation and orchestration; selection of calibrated simulation assumptions; host setup, runner builds, FVP launch, acquisition, calibration, energy, native results and visualization artifacts. |
| heliaAOT | Conversion to its IR, operator admission, CPU kernels, memory planning, generated modules and Ethos-U invocation. Optional NPU preparation precedes conversion. |
| HPX | Firmware integration, physical builds, flashing, acquisition, instrumentation, hardware-specific admission and native capture results. |
| Runtime adapters | Registration and execution of Ethos-U and stream-split operators, with their buffer and driver contracts. |
| BENCH | Experiment conditions and suites, calls to public producer APIs, evidence accounting, comparison eligibility and reports. |

The dependency direction is `HELX / AOT / HPX -> helia-ethos -> helia-platforms`.
Consumers can also read helia-platforms directly for their non-NPU needs.
helia-ethos never imports a consumer. RT's embedded C/C++ runtime does not acquire
a Python compiler dependency. No shared application scheduler or generic plugin
framework is introduced.

### File disposition

Paths are relative to this repository unless qualified. This contract originated
in HELX PR #194; the preparation API below is planned, not yet implemented.

| Existing area | Extraction disposition |
| --- | --- |
| `src/helia_ethos/catalog.py` | Keep the caller-supplied catalog/NPU-fact reader. Do not create a second physical target catalog. |
| `catalog_loading.py`, `catalog_identity.py`, `catalog_policy.py` | Keep explicit source identity and projection machinery. Separate the existing AT110 simulation convenience path from preparation with caller-resolved facts and explicit compiler settings. |
| `soc.py`, `soc_schema.py`, `socs/` | Keep compiler-facing types, custom what-if definitions and catalog-backed references. A `SocDefinition` is an effective compiler/model view, not a new authority for silicon facts. |
| `model_policies/atomiq110.yaml` | Keep the versioned, explicitly selected policy initially. Its fitted settings remain modeling assumptions; HELX selects it for simulation. It must not become the default policy for FPGA preparation. |
| `target.py`, `targets/` | Replace duplicated physical realization facts with catalog references where the catalog supplies them. Preserve simulator identity, stand-ins and fidelity as tool metadata. Inventory existing callers before retiring any public loader or changing retained identities. |
| `vela/`, `model.py`, `fencing.py`, `pmu.py` | Extract with package tests and documentation. Add the preparation API as composition of these functions, not a second compiler implementation. |
| `fvp.py` | Keep static Corstone/NPU capabilities and register/parameter descriptions. These describe the executable simulator, not the physical SoC. Installed binary discovery, download, build and launch stay with HELX. |
| `npu_reference.py` | Keep the separately attributed Arm reference data. Private AT110 coefficients and calibration history remain in HELX's `energy_profiles/`. |
| `helia_estimator/local_fvp_runtime/` | Remains HELX-owned. In particular, extraction does not automatically move its C++ stream-split kernel, resolver, firmware templates or linker scripts. |

A narrow compatibility adapter may preserve HELX's existing target API during
cutover; it must delegate facts rather than preserve a competing data copy.
Do not silently change old saved identities when a target adapter changes.
Use the normal explicit rebuild/migration boundary for affected retained builds.
A broader legacy-result preservation framework is outside this work.

## Current gaps this contract addresses

- `compile_for_target()` returns paths, Vela summary/per-layer data and stdout.
  It does not yet provide a complete retained preparation record with source and
  output hashes, compiler identity, fence correspondence and runtime requirements.
- `load_catalog_model()` currently selects only the AT110 modeling policy, and
  `project_catalog_soc()` rejects implementation targets. Keep that guard for
  simulation; do not weaken it to admit FPGA preparation with silicon assumptions.
- HELX's analysis stage composes fencing and recompilation itself. Only the
  model-preparation portion becomes shared; segment execution and attribution
  against measured FVP counters remain HELX's responsibility.
- AOT already handles the version-1 stream-split operator and Vela-relative
  tensor bindings. Those remain AOT responsibilities. HPX currently asks for
  precompiled Vela models; its preparation option is follow-up adoption work.
- HELX registers a stream-split kernel in its LiteRT/RT runner. That does not
  establish that a standalone RT or HPX build registers it. Consumer adoption
  must supply and test the required operator registration.

These observations were checked against HELX `f05b97c`, platforms `61eebb5`,
AOT `34c853aa`, HPX `4c01584a`, RT `3d704a71` and BENCH `93428bda`. They describe
those revisions, not permanent product limitations.

## Proposed Python surface

The following signatures and types are design notation, not executable examples:

```python
prepare_model(request: PrepareRequest) -> PreparedModel
inspect_prepared_model(path: Path) -> PreparedModelInfo
```

Implement the public surface in `helia_ethos.preparation`, re-exporting its public
names from `helia_ethos`. Use frozen dataclasses, enums and tuples for domain data;
use a versioned serialization boundary for the preparation record. Existing
`compile_for_target`, `plan_fences`, `fence_model` and inspection functions remain
available to callers that need individual steps.

### Inputs and outputs

| Type | Required contract |
| --- | --- |
| `PrepareRequest` | Source LiteRT path, resolved NPU facts, explicit compiler settings, optional fence selection, caller-owned output directory. |
| `NpuCompileSettings` | Accelerator family/MAC configuration, operating point, named memory placement/port mapping, arena-cache bytes and effective Vela cost-model settings. Settings carry the selected policy identity where one supplied them. No physical-memory order implies a default placement. |
| `FenceSelection` | Either explicit operator boundaries or a cycle-budget planning policy; existing typed fence inputs may implement this. Planning records the Vela estimate and coverage used, without promising a runtime deadline. |
| `PreparedModel` | Output `.tflite`, preparation-record path, source/output identities, effective target/compiler settings, Vela artifacts, optional fence sidecar/segment correspondence, runtime requirements and diagnostics. |
| `PreparedModelInfo` | Observed compiled-model properties and, when supplied, validated preparation-record fields. Absent provenance remains unknown. Inspection does not execute Vela, install dependencies or modify the input. |

Resolved NPU facts should reuse `CatalogNpuFacts` where appropriate. Preserve an
explicit custom/what-if path for current HELX experiments; it records its own
identity and makes no catalog-backed hardware claim. Do not require all targets
to be catalog entries merely to extract the package.

For a catalog-backed request, the consumer loads its catalog once and passes the
selected facts. Preparation must not reload `builtin()` and lose its overlays.
The existing explicit HELX loader can remain a convenience wrapper. AOT may keep
its own source-discovery policy, but must pass the resulting target to preparation
and use that same effective selection for conversion. HPX does likewise for its
selected implementation. A shared name alone is not sufficient correspondence.

Compiler configuration may require latency/bandwidth estimates even for a real
board. Record those as compiler assumptions, separately from the implementation's
reported/configured clocks and observed capture facts. Do not require a calibrated
FVP policy to compile for hardware. Unknown required compiler settings produce an
actionable error or require an explicit caller assumption; never silently borrow
AT110 silicon settings.

### Workflow and prepared inputs

1. Inspect and validate the source and the selected preparation settings.
2. Without fences, compile once. With explicit boundaries, insert fences before
   compilation. With budget planning, first compile for the estimate, plan with
   recorded coverage, insert fences, then compile the fenced model.
3. Inspect the final artifact, validate its expected accelerator and requested
   split correspondence where the format supports inspection, and write the
   preparation record only on success. Compiler-reported settings and decoded
   artifact properties remain separately identified evidence.
4. Return typed artifacts and requirements. Never build firmware or execute a
   simulator/device here.

`prepare_model()` accepts an uncompiled source. It refuses an already compiled
Ethos-U model rather than silently compiling it again or trying to fence opaque
command streams. The first preparation API also refuses a source that already
contains `ETHOS_U_STREAM_SPLIT`, naming the existing fences. It must not add a
second plan or invent correspondence for earlier cuts. Existing low-level
`fence_model()` followed by `compile_for_target()` remains available; support for
importing a pre-fenced source with a validated sidecar is separate follow-up.
An already compiled model uses `inspect_prepared_model()` and
can continue through a consumer's existing prepared-input route. A source model
without compiled NPU regions must not be described as NPU-prepared merely because
Vela exited successfully; return a clear unsupported/no-NPU diagnostic.

For inspection, `path` may name a preparation record or a standalone compiled
`.tflite`. A record resolves its artifact paths relative to its own directory and
checks the recorded hashes. A standalone file supplies only observable facts;
original source identity, placement and compiler settings cannot be invented.
Check every embedded NPU region, not just the first, when reporting accelerator
identity. Inspection is observational: `PreparedModelInfo` returns the decoded
properties and their evidence, not a compatibility verdict against an unstated
request. The preparation call or consumer admission compares those properties
with its explicit requested settings and reports known-compatible, known-
incompatible or unknown. Decode only supported command payload formats; missing/
undecodable identity remains unknown, not a match or a mismatch. A hash-checked record can establish which compiler settings produced
those bytes; it is not independent proof that their payload encodes those settings.
No current HELX inspection API is claimed to implement this new capability.
The consumer decides whether missing facts suffice for its task; a wrong-
accelerator error requires evidence of a mismatch.

Preparation preserves source files. Output is explicit and caller-owned; an
occupied destination is refused rather than overwritten or silently reused.
Temporary scratch is owned and cleaned by the preparation call. On failure,
retain useful diagnostics in the destination, but no success record. A caller
chooses a fresh directory for retry. No implicit global cache or background work.

### Preparation record

Use one small versioned record for compilation evidence, not a new measurement
bundle. At minimum retain:

- Source, optional rewritten graph and final model SHA-256; paths are relative
  to the record for relocation. No claim that matching names imply matching bytes.
- Catalog target and NPU instance, catalog/source identity where available,
  explicit custom-target identity otherwise, policy identity and effective settings.
- Vela version and available compiler implementation identity, normalized arguments,
  configuration bytes/hash, relevant package versions and diagnostic artifacts.
- Fence settings, source binding, coverage and verified final region/segment mapping.
- Required custom operators and versions, required NPU configuration, and observable
  model I/O metadata. Runtime support is a requirement, not a claim of qualification.

The source hash is required; retaining a source copy is optional. A relocated
prepared artifact never requires the original external source path. Required
retained artifacts are the final model, effective compiler configuration, and
any sidecar needed for the claimed fence correspondence. The preparation record
hashes them and uses paths confined beneath its directory (no absolute paths or
parent escapes). Missing or mismatched required artifacts invalidate the record.
Optional source copies, stdout and performance/debug logs are marked optional;
missing optional diagnostics are reported as unavailable without preventing
execution. If present, their recorded hashes must match. Inspection reports this
scope explicitly rather than claiming all original evidence is retained.

Use existing typed identities and sidecars where possible. Do not copy a full
catalog into every record or introduce cryptographic authenticity claims.
Normal safeguards against accidental mismatch are sufficient.

HELX and HPX retain the record and its complete required artifact set in their
existing native bundles, preserving relative layout. They must not retain only
the record and then delete its referenced workspace. If a producer repackages
artifacts, it rewrites paths and updates affected identities before sealing its
manifest; model-byte hashes remain unchanged. Optional artifacts intentionally
omitted are recorded as unavailable. Verify inspection from the sealed, relocated
bundle after the original workspace has been removed.
BENCH consumes producer APIs and identities; no BENCH-owned Vela parser is needed.
Existing native result schemas remain unchanged in PR 1.

### Errors and dependencies

Public preparation errors distinguish invalid input/settings, unavailable compiler,
compiler failure and incompatible/incomplete prepared artifacts. Errors identify
the relevant model or field and retain diagnostic paths. Consumer adapters map
these errors into their existing CLI/API contracts; the library does not print,
exit the process, acquire devices or make installation/licence decisions.

Keep model-reading and Vela dependencies optional, as the current extras do.
`prepare_model()` requires both model and compiler capabilities, installed as
`helia-ethos[models,vela]` (or the existing `all` extra). Inspection requires only
`models`; core types/catalog reads remain importable without either extra.
Missing capabilities produce an actionable extra-installation error on use.
AOT/HPX CPU-only use must not require installing the compiler. Test the actual
Python/dependency intersection for each extra rather than promising that Vela
supports every Python version supported by the core package.

## Consumer examples

These are proposed call sequences, not instructions for the currently released
APIs. Each product exposes its own public CLI and Python entry points over the
same internal flow.

### HELX: simulation

```text
resolve catalog + explicit modeling policy + energy separately
make PrepareRequest in the case-owned preparation directory
prepare_model(request)
build selected LiteRT / RT / AOT runner from PreparedModel.tflite
execute FVP, acquire samples, write HELX result bundle
```

HELX owns the experiment workspace and retains the preparation record with its
artifacts. For AOT execution, pass the prepared model to conversion without
requesting a second preparation. HELX alone owns FVP linker/backing-memory choices:
catalog capacity is not a Corstone allocation. The local U55 arena relocation
experiment is evidence for a separate runtime change, not part of extraction.

### AOT: generated application

```text
resolve AOT target and catalog sources once
if NPU preparation is explicitly requested:
    prepare_model(request using that resolved target and explicit settings)
    convert the returned model
else:
    convert the original or already prepared model as today
emit application module using AOT's memory plan and runtime bindings
```

AOT owns the conversion workspace and preserves preparation evidence with its
conversion artifacts. It preserves Vela's required arena-relative bindings;
helia-ethos does not assign the final application's tensor/code addresses.
A catalog region being NPU-reachable does not establish CPU access or placement
support in AOT. Current AOT excludes AT110 HBLRAM from its catalog memory binding
pending clarification; preparation must not silently override that restriction.

### HPX: physical capture

```text
resolve the selected board/implementation and runtime
prepare an original model when explicitly requested,
    or inspect the user's already prepared model
admit accelerator and runtime requirements against the selected capture route
build firmware, acquire hardware measurements, write HPX result bundle
```

HPX owns the build workspace, hardware admission and retained artifacts. The
prepared-model route remains useful even before HPX adds preparation. It must
register every required custom operator; supporting `ethos-u` alone does not
establish support for `ETHOS_U_STREAM_SPLIT`.

## Fence and scheduling boundary

The version-1 stream-split op is a paired-tensor identity boundary. It preserves
matching supported dtype, shape and quantization per pair, including multiple
live tensors at a cut. Runtime implementations preserve its scheduling point and
required buffer relationships. The actual Ethos-U operators invoke the driver.

A fence does not by itself yield a thread, preempt a command stream, select an
NPU, implement an RTOS semaphore or guarantee a maximum blocking time. Those are
application/runtime scheduling concerns. Vela cycle estimates used to select a
cut remain estimates; HELX/HPX can evaluate the resulting segments separately.

AOT already has its own generated implementation. Keep HELX's LiteRT/RT adapter
in place for extraction. If HPX needs the same C++ support, adopt it deliberately
through the runtime or a small reusable source adapter, with appropriate tests;
do not make RT depend on the Python package to execute inference.

## Implementation sequence and acceptance

Open each HELX PR against main after its predecessor merges, following this
repository's contribution policy. Cross-repository consumer adoption can proceed
independently once the package contract/release is available.

| PR | Scope | Completion evidence |
| --- | --- | --- |
| 1: this contract | Ownership, API proposal, file disposition, stale-description correction | Source review, documentation/reference checks; no behavior change. |
| 2: package extraction | Preserve package history; implement the agreed preparation interface; establish standalone CI/docs/release, metadata and dependency bounds | Core-only imports; installed-wheel tests without HELX importable; actual Vela compile; explicit and budgeted multiple-fence preparation; record relocation/mismatch checks, unknown accelerator inspection and pre-fenced input refusal; source remains unchanged. Publish a new independent package version. |
| 3: HELX adoption | Replace workspace dependency; use shared preparation where appropriate; retire duplicated target facts through the agreed adapter | Normal gates, installed wheel, reference runs, all three runtimes and multiple-fence output checks. Verify preparation-record inspection after cleanup and bundle relocation. Preserve native result semantics and explain any retained-build invalidation. |
| 4: AOT adoption | Optional preparation before conversion; keep prepared input and CPU-only routes | Installed optional dependency, no double compilation, multiple fences/multiple live tensors, Vela-relative binding integrity, numerical output comparison. |
| 5: HPX adoption | Explicit preparation/inspection and supported runtime operator registration | Public API/CLI agreement; wrong accelerator and missing-runtime-op errors; original/prepared paths; physical qualification on an available supported target under HPX's normal hardware gates. |

PR 2 may be split into a mechanical extraction and a preparation implementation
if its review would otherwise mix move-only changes with substantial behavior.
Do not silently broaden it into a multi-product migration. AOT and HPX owners
review their proposed interfaces before their adoption work is implemented;
BENCH reviews producer-facing changes only where its consumed contract changes.

The current package is `0.1.0`; HELX's Release Please configuration versions HELX,
not this package independently. Give the extracted implementation a new version
and its own immutable releases. HELX pins a tested release in its lockfile and
uses an explicit supported dependency range; do not publish changed code again
under the existing version or make consumers follow an unpinned main branch.

The first extraction does not require finishing FPGA calibration, resolving all
hardware-source questions, making HELX public, introducing a common result schema,
or adding another runtime. Repository visibility/licensing and package publication
mechanism need an explicit owner decision before creating/publishing the new repo;
this contract grants no visibility change to existing code or private profiles.

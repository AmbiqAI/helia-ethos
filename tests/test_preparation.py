"""Public preparation/inspection through real Vela and relocatable records."""

from __future__ import annotations

import importlib
import json
import shutil
import struct
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from helia_ethos import (
    Accelerator,
    AcceleratorCompatibility,
    BudgetFences,
    CustomNpuTarget,
    ExplicitFences,
    InvalidPreparation,
    InvalidPreparedArtifact,
    MemoryPlacement,
    NpuCompileSettings,
    NpuFamily,
    NpuRequirement,
    PrepareRequest,
    inspect_prepared_model,
    load_soc,
    prepare_model,
)
from helia_ethos.preparation.types import ArtifactRole, CompilationFailed, CompilerUnavailable


@pytest.fixture
def request_model(add_chain_model: Path, tmp_path: Path) -> PrepareRequest:
    return PrepareRequest(
        source=add_chain_model,
        target=CustomNpuTarget("test-what-if", Accelerator(NpuFamily.ETHOS_U85, 256)),
        settings=NpuCompileSettings(
            load_soc("atomiq110"), "ULP", MemoryPlacement("hblram", "ssram", "ssram"), "explicit-test-policy"
        ),
        output_dir=tmp_path / "prepared",
    )


def test_compile_inspect_and_relocate(request_model: PrepareRequest, tmp_path: Path) -> None:
    original = request_model.source.read_bytes()
    prepared = prepare_model(request_model)
    assert request_model.source.read_bytes() == original
    assert prepared.info.compatibility(Accelerator(NpuFamily.ETHOS_U85, 256)) is AcceleratorCompatibility.COMPATIBLE
    record = prepared.info.record
    assert record is not None and record.target.kind == "custom"
    assert record.compiler.policy_id == "explicit-test-policy"
    assert record.compiler.reported_accelerator
    assert record.compiler.reported_system_config
    assert record.compiler.reported_memory_mode
    assert record.compiler.version_output
    assert "<source>" in record.compiler.arguments
    assert all(not Path(a.path).is_absolute() for a in record.artifacts)
    assert not list(request_model.output_dir.glob("scratch-*"))
    destination = tmp_path / "relocated with spaces"
    shutil.copytree(request_model.output_dir, destination)
    shutil.rmtree(request_model.output_dir)
    request_model.source.unlink()
    relocated = inspect_prepared_model(destination / "preparation.json")
    assert relocated.sha256 == prepared.info.sha256
    standalone = inspect_prepared_model(relocated.tflite)
    assert standalone.record is None
    assert standalone.regions == relocated.regions


@pytest.mark.parametrize("role", [ArtifactRole.CONFIG, ArtifactRole.MODEL, ArtifactRole.DIAGNOSTIC])
def test_missing_and_corrupt_artifacts(request_model: PrepareRequest, role: ArtifactRole) -> None:
    prepared = prepare_model(request_model)
    assert prepared.info.record is not None
    artifact = next(a for a in prepared.info.record.artifacts if a.role == role)
    path = request_model.output_dir / artifact.path
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(InvalidPreparedArtifact, match="Hash mismatch"):
        inspect_prepared_model(prepared.record_path)
    path.unlink()
    if artifact.required:
        with pytest.raises(InvalidPreparedArtifact, match="Missing required"):
            inspect_prepared_model(prepared.record_path)
    else:
        assert artifact.path in inspect_prepared_model(prepared.record_path).unavailable_optional


@pytest.mark.parametrize("relative", ["../escape", "/absolute", "C:/windows", "x\\y"])
def test_record_paths_are_confined(request_model: PrepareRequest, relative: str) -> None:
    prepared = prepare_model(request_model)
    record = json.loads(prepared.record_path.read_text())
    record["artifacts"][0]["path"] = relative
    prepared.record_path.write_text(json.dumps(record))
    with pytest.raises(InvalidPreparedArtifact, match="artifact path"):
        inspect_prepared_model(prepared.record_path)


def test_refuses_occupied_destination_and_prepared_source(request_model: PrepareRequest) -> None:
    prepared = prepare_model(request_model)
    before = prepared.record_path.read_bytes()
    with pytest.raises(InvalidPreparation, match="fresh"):
        prepare_model(request_model)
    assert prepared.record_path.read_bytes() == before
    with pytest.raises(InvalidPreparation, match="already compiled"):
        prepare_model(replace(request_model, source=prepared.tflite, output_dir=request_model.output_dir / "second"))


def test_multiple_fences_and_source_binding(request_model: PrepareRequest, tmp_path: Path) -> None:
    from conftest import write_add_chain_model

    source = write_add_chain_model(tmp_path / "four.tflite", num_ops=4)
    prepared = prepare_model(replace(request_model, source=source, fences=ExplicitFences((0, 2))))
    assert len(prepared.info.regions) == 3
    record = prepared.info.record
    assert record is not None and record.fences is not None
    assert [(s.first, s.last) for s in record.fences.spans] == [(0, 0), (1, 1), (2, 2)]
    assert ("ETHOS_U_STREAM_SPLIT", 1) in [(op.name, op.version) for op in prepared.info.custom_operators]
    rewrite = next(a for a in record.artifacts if a.role == ArtifactRole.REWRITTEN)
    with pytest.raises(InvalidPreparation, match="already contains ETHOS_U_STREAM_SPLIT"):
        prepare_model(
            replace(request_model, source=request_model.output_dir / rewrite.path, output_dir=tmp_path / "pre-fenced")
        )
    sidecar = next(a for a in record.artifacts if a.role == ArtifactRole.BOUNDARIES)
    (request_model.output_dir / sidecar.path).unlink()
    with pytest.raises(InvalidPreparedArtifact, match="Missing required boundaries"):
        inspect_prepared_model(prepared.record_path)


def test_budget_plan_and_compile_count(request_model: PrepareRequest, tmp_path: Path, monkeypatch) -> None:
    from conftest import write_add_chain_model

    implementation = importlib.import_module("helia_ethos.preparation.prepare")
    original = implementation.compile_for_target
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(implementation, "compile_for_target", counted)
    source = write_add_chain_model(tmp_path / "four.tflite", num_ops=4)
    prepared = prepare_model(replace(request_model, source=source, fences=BudgetFences(1)))
    assert len(calls) == 2
    assert prepared.compilation.tflite == prepared.tflite
    assert prepared.compiler_log.read_text() == prepared.compilation.stdout
    assert len(prepared.info.regions) == 4
    assert prepared.info.record is not None and prepared.info.record.fences is not None
    assert prepared.info.record.fences.matched_cycles is not None
    calls.clear()
    uncut = prepare_model(
        replace(request_model, source=source, fences=BudgetFences(10**12), output_dir=tmp_path / "one")
    )
    assert uncut.compilation.tflite == uncut.tflite
    assert uncut.compiler_log.read_text() == uncut.compilation.stdout
    assert len(calls) == 1


def test_catalog_identity_keeps_callers_selection(request_model: PrepareRequest, monkeypatch) -> None:
    from helia_platforms import builtin, load_overlay

    from helia_ethos.catalog import read_catalog_npu

    catalog = load_overlay({"socs": {"lab_at110": {"based_on": "atomiq110"}}}, base=builtin())
    facts = read_catalog_npu("lab_at110", catalog=catalog)

    def unexpected_reload(*args, **kwargs):
        raise AssertionError("Preparation must not reload a catalog")

    monkeypatch.setattr("helia_platforms.builtin", unexpected_reload)
    prepared = prepare_model(replace(request_model, target=facts))
    assert prepared.info.record is not None and prepared.info.record.target.catalog is not None
    assert prepared.info.record.target.catalog.catalog_digest == facts.identity.catalog_digest
    assert prepared.info.record.target.npu_id == facts.npu.id
    assert prepared.info.record.target.catalog.sources[-1].sha256 == facts.identity.sources[-1].sha256


def test_invalid_settings_before_compilation(request_model: PrepareRequest) -> None:
    wrong = CustomNpuTarget("wrong", Accelerator(NpuFamily.ETHOS_U55, 64))
    with pytest.raises(InvalidPreparation, match="accelerator"):
        prepare_model(replace(request_model, target=wrong))
    with pytest.raises(InvalidPreparation, match="boundaries"):
        prepare_model(replace(request_model, fences=ExplicitFences((100,))))
    assert not request_model.output_dir.exists()


def test_failure_leaves_diagnostics_without_success(request_model: PrepareRequest, monkeypatch) -> None:
    implementation = importlib.import_module("helia_ethos.preparation.prepare")

    def failed(*args, **kwargs):
        from helia_ethos.vela import VelaError

        raise VelaError("compiler failed deliberately")

    monkeypatch.setattr(implementation, "compile_for_target", failed)
    with pytest.raises(CompilationFailed, match=r"failure\.txt"):
        prepare_model(request_model)
    assert "deliberately" in (request_model.output_dir / "failure.txt").read_text()
    assert not (request_model.output_dir / "preparation.json").exists()
    assert not list(request_model.output_dir.glob("scratch-*"))


def test_missing_compiler_is_distinct(request_model: PrepareRequest, tmp_path: Path) -> None:
    with pytest.raises(CompilerUnavailable):
        prepare_model(
            replace(request_model, settings=replace(request_model.settings, vela_bin=str(tmp_path / "missing")))
        )
    assert not request_model.output_dir.exists()


@pytest.mark.parametrize("product,macs", [(0, 64), (1, 256), (2, 128)])
def test_inspection_all_regions_and_unknown_payload(add_chain_model: Path, product: int, macs: int) -> None:
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.tools import flatbuffer_utils

    model: Any = flatbuffer_utils.read_model_from_bytearray(add_chain_model.read_bytes())
    code = model.operatorCodes[0]
    code.builtinCode = code.deprecatedBuiltinCode = schema.BuiltinOperator.CUSTOM
    code.customCode = b"ethos-u"
    graph = model.subgraphs[0]
    for index, op in enumerate(graph.operators):
        buffer = schema.BufferT()
        word = (product << 28) | (macs.bit_length() - 1)
        buffer.data = list(b"COP1" + struct.pack("<III", 1, word, 0)) if index == 0 else list(b"COP9unknown")
        model.buffers.append(buffer)
        tensor = schema.TensorT()
        tensor.name, tensor.shape, tensor.type = b"stream", [len(buffer.data)], schema.TensorType.UINT8
        tensor.buffer = len(model.buffers) - 1
        graph.tensors.append(tensor)
        op.inputs = [len(graph.tensors) - 1]
    flatbuffer_utils.write_model(model, str(add_chain_model))
    info = inspect_prepared_model(add_chain_model)
    assert len(info.regions) == 2
    family = (NpuFamily.ETHOS_U55, NpuFamily.ETHOS_U65, NpuFamily.ETHOS_U85)[product]
    expected = Accelerator(family, macs)
    assert info.regions[0].accelerator == expected
    assert info.regions[1].accelerator is None
    assert info.compatibility(expected) == AcceleratorCompatibility.UNKNOWN
    assert info.compatibility(Accelerator(family, macs * 2)) == AcceleratorCompatibility.INCOMPATIBLE


@pytest.mark.parametrize("policy", list(NpuRequirement))
@pytest.mark.parametrize("change", ["no_npu", "wrong_accelerator", "unknown_payload"])
def test_final_model_admission(request_model: PrepareRequest, monkeypatch, change: str, policy: NpuRequirement) -> None:
    request_model = replace(request_model, npu_requirement=policy)
    from ai_edge_litert.tools import flatbuffer_utils

    implementation = importlib.import_module("helia_ethos.preparation.prepare")
    compile_original = implementation.compile_for_target

    def altered(*args, **kwargs):
        compiled = compile_original(*args, **kwargs)
        if change == "no_npu":
            compiled.tflite.write_bytes(Path(args[0]).read_bytes())
        else:
            model: Any = flatbuffer_utils.read_model_from_bytearray(compiled.tflite.read_bytes())
            graph = model.subgraphs[0]
            op = next(op for op in graph.operators if model.operatorCodes[op.opcodeIndex].customCode == b"ethos-u")
            buffer = model.buffers[graph.tensors[op.inputs[0]].buffer]
            data = bytearray(buffer.data)
            if change == "wrong_accelerator":
                data[8] = (data[8] & 0xF0) | 7  # 128 MACs instead of the requested 256.
            else:
                data[:4] = b"COP9"
            buffer.data = list(data)
            flatbuffer_utils.write_model(model, str(compiled.tflite))
        return compiled

    monkeypatch.setattr(implementation, "compile_for_target", altered)
    if change == "no_npu" and policy is NpuRequirement.OPTIONAL:
        assert not prepare_model(request_model).info.regions
    elif change == "unknown_payload":
        prepared = prepare_model(request_model)
        assert prepared.info.compatibility(Accelerator(NpuFamily.ETHOS_U85, 256)) == AcceleratorCompatibility.UNKNOWN
        assert inspect_prepared_model(prepared.record_path).regions[0].accelerator is None
    else:
        with pytest.raises(InvalidPreparedArtifact):
            prepare_model(request_model)
        assert not (request_model.output_dir / "preparation.json").exists()
        assert (request_model.output_dir / "failure.txt").exists()


@pytest.mark.parametrize("family,macs", [(NpuFamily.ETHOS_U55, 64), (NpuFamily.ETHOS_U65, 256)])
def test_other_accelerators_use_explicit_settings(request_model: PrepareRequest, family: NpuFamily, macs: int) -> None:
    soc = replace(request_model.settings.soc, npu=family.value, macs=macs, axi_bus_width_bits=family.axi_beat_bits)
    expected = Accelerator(family, macs)
    prepared = prepare_model(
        replace(
            request_model, target=CustomNpuTarget("other", expected), settings=replace(request_model.settings, soc=soc)
        )
    )
    assert prepared.info.compatibility(expected) == AcceleratorCompatibility.COMPATIBLE


def test_reader_needs_no_vela(request_model: PrepareRequest, monkeypatch) -> None:
    prepared = prepare_model(request_model)

    def forbidden(*args, **kwargs):
        raise AssertionError("Inspection must not execute a subprocess")

    monkeypatch.setattr("subprocess.run", forbidden)
    assert inspect_prepared_model(prepared.record_path).sha256 == prepared.info.sha256


def test_unfenced_multisubgraph_source(request_model: PrepareRequest) -> None:
    import copy

    from ai_edge_litert.tools import flatbuffer_utils

    model: Any = flatbuffer_utils.read_model_from_bytearray(request_model.source.read_bytes())
    additional = copy.deepcopy(model.subgraphs[0])
    additional.name = b"secondary"
    model.subgraphs.append(additional)
    flatbuffer_utils.write_model(model, str(request_model.source))
    prepared = prepare_model(request_model)
    assert {region.subgraph for region in prepared.info.regions} == {0, 1}
    assert prepared.info.compatibility(Accelerator(NpuFamily.ETHOS_U85, 256)) == AcceleratorCompatibility.COMPATIBLE


def test_cleanup_failure_does_not_publish_success(request_model: PrepareRequest, monkeypatch) -> None:
    import tempfile
    from contextlib import contextmanager

    original = tempfile.TemporaryDirectory

    @contextmanager
    def failed_cleanup(*args, **kwargs):
        with original(*args, **kwargs) as directory:
            yield directory
        raise OSError("cleanup failed deliberately")

    monkeypatch.setattr("tempfile.TemporaryDirectory", failed_cleanup)
    with pytest.raises(CompilationFailed, match="cleanup failed"):
        prepare_model(request_model)
    assert not (request_model.output_dir / "preparation.json").exists()
    assert not (request_model.output_dir / "pending.json").exists()
    assert (request_model.output_dir / "failure.txt").exists()


def test_conflicting_placement_is_an_input_error(request_model: PrepareRequest) -> None:
    settings = replace(request_model.settings, placement=MemoryPlacement("hblram", "rram", "ssram"))
    with pytest.raises(InvalidPreparation, match="memory_placement"):
        prepare_model(replace(request_model, settings=settings))
    assert not request_model.output_dir.exists()


@pytest.mark.parametrize("memory", ["srram", "SSRAM"])
def test_unknown_timing_override_is_refused(request_model: PrepareRequest, memory: str) -> None:
    from helia_ethos import MemoryTimingOverride, TimingOverrides

    settings = replace(
        request_model.settings, timing_overrides=TimingOverrides({memory: MemoryTimingOverride(bandwidth=0.5)})
    )
    with pytest.raises(InvalidPreparation, match="unknown keys"):
        prepare_model(replace(request_model, settings=settings))
    assert not request_model.output_dir.exists()


@pytest.mark.parametrize("failure", ["exit", "missing_summary", "invalid_summary"])
def test_compiler_diagnostics_survive_failure(request_model: PrepareRequest, monkeypatch, failure: str) -> None:
    import subprocess

    implementation = importlib.import_module("helia_ethos.preparation.prepare")
    invoke_original = implementation.invoke_vela

    def failed(vela, invocation, input_path, output_path):
        def runner(*args, **kwargs):
            if failure == "exit":
                raise subprocess.CalledProcessError(
                    2, args[0], output="early stdout\n", stderr="first error\n" + "later\n" * 10
                )
            if failure == "invalid_summary":
                summary = output_path / f"{input_path.stem}_summary_{invocation.system_config}.csv"
                summary.write_text("", encoding="utf-8")
            return subprocess.CompletedProcess(args[0], 0, "early stdout\n", "first error\n")

        return invoke_original(vela, invocation, input_path, output_path, runner=runner)

    monkeypatch.setattr(implementation, "invoke_vela", failed)
    with pytest.raises(CompilationFailed):
        prepare_model(request_model)
    log = (request_model.output_dir / "compiled/compiler.log").read_text()
    assert "early stdout" in log and "first error" in log
    assert not (request_model.output_dir / "preparation.json").exists()


def test_compile_result_is_reusable_without_another_compile(request_model: PrepareRequest, monkeypatch) -> None:
    """Consumers receive the final typed result, including retained diagnostic paths."""
    from helia_ethos.vela import parse_vela_summary

    implementation = importlib.import_module("helia_ethos.preparation.prepare")
    original = implementation.compile_for_target
    results = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(implementation, "compile_for_target", capture)
    prepared = prepare_model(request_model)
    assert results == [prepared.compilation]
    assert prepared.compilation.tflite == prepared.tflite
    assert prepared.compilation.summary == parse_vela_summary(prepared.compilation.summary_path)
    assert prepared.compiler_log.read_text() == prepared.compilation.stdout
    assert prepared.compilation.config_file.is_file()
    assert prepared.compilation.summary_path.is_relative_to(request_model.output_dir)


def test_cpu_only_policy_and_retained_record(request_model: PrepareRequest, tmp_path: Path, monkeypatch) -> None:
    """A real CPU-only compile succeeds only when opted in and remains honest after relocation."""
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.tools import flatbuffer_utils

    from conftest import write_add_chain_model
    from helia_ethos import NpuRequirement

    source = write_add_chain_model(tmp_path / "cpu.tflite", num_ops=1)
    model: Any = flatbuffer_utils.read_model_from_bytearray(source.read_bytes())
    model.operatorCodes[0].builtinCode = schema.BuiltinOperator.SIN
    model.operatorCodes[0].deprecatedBuiltinCode = schema.BuiltinOperator.SIN
    for tensor in model.subgraphs[0].tensors:
        tensor.type = schema.TensorType.FLOAT32
        tensor.quantization = None
    op = model.subgraphs[0].operators[0]
    op.inputs = [0]
    op.builtinOptionsType = schema.BuiltinOptions.NONE
    op.builtinOptions = None
    flatbuffer_utils.write_model(model, str(source))
    with pytest.raises(InvalidPreparedArtifact, match="no compiled Ethos-U"):
        prepare_model(replace(request_model, source=source))

    implementation = importlib.import_module("helia_ethos.preparation.prepare")
    original = implementation.compile_for_target
    calls = []

    def capture(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(implementation, "compile_for_target", capture)
    output = tmp_path / "cpu-prepared"
    prepared = prepare_model(
        replace(request_model, source=source, output_dir=output, npu_requirement=NpuRequirement.OPTIONAL)
    )
    assert len(calls) == 1
    assert prepared.info.regions == ()
    assert prepared.info.compatibility(Accelerator(NpuFamily.ETHOS_U85, 256)) is AcceleratorCompatibility.UNKNOWN
    assert prepared.info.record is not None and prepared.info.record.schema_version == 2
    assert prepared.compilation.summary_path.is_file()
    relocated = tmp_path / "moved"
    shutil.copytree(output, relocated)
    shutil.rmtree(output)
    source.unlink()
    info = inspect_prepared_model(relocated / "preparation.json")
    assert info.regions == ()
    record = json.loads((relocated / "preparation.json").read_text())
    record["npu_requirement"] = "required"
    (relocated / "preparation.json").write_text(json.dumps(record))
    with pytest.raises(InvalidPreparedArtifact, match="no Ethos-U"):
        inspect_prepared_model(relocated / "preparation.json")


@pytest.mark.parametrize("omit_version", [False, True])
def test_version_one_record_remains_npu_required(request_model: PrepareRequest, omit_version: bool) -> None:
    """Version-one records remain readable under their original strict NPU contract."""
    prepared = prepare_model(request_model)
    data = json.loads(prepared.record_path.read_text())
    data["schema_version"] = 1
    del data["npu_requirement"]
    if omit_version:
        del data["schema_version"]
    prepared.record_path.write_text(json.dumps(data))
    inspected = inspect_prepared_model(prepared.record_path)
    assert inspected.regions and inspected.record is not None
    assert "npu_requirement" not in inspected.record.model_dump()
    assert json.loads(inspected.record.model_dump_json()) == {**data, "schema_version": 1}
    data["npu_requirement"] = "optional"
    prepared.record_path.write_text(json.dumps(data))
    with pytest.raises(InvalidPreparedArtifact, match="version 1 requires"):
        inspect_prepared_model(prepared.record_path)


def test_cpu_only_unsigned_output_is_observed_not_rejected(request_model, tmp_path):
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.tools import flatbuffer_utils

    from conftest import write_add_chain_model

    source = write_add_chain_model(tmp_path / "unsigned.tflite", num_ops=1)
    model: Any = flatbuffer_utils.read_model_from_bytearray(source.read_bytes())
    code = model.operatorCodes[0]
    code.builtinCode = code.deprecatedBuiltinCode = schema.BuiltinOperator.CAST
    graph = model.subgraphs[0]
    graph.inputs = [0]
    operator = graph.operators[0]
    operator.inputs = [0]
    operator.builtinOptionsType = schema.BuiltinOptions.CastOptions
    operator.builtinOptions = schema.CastOptionsT()
    operator.builtinOptions.inDataType = schema.TensorType.INT32
    operator.builtinOptions.outDataType = schema.TensorType.UINT32
    for tensor in graph.tensors:
        tensor.type = schema.TensorType.INT32
        tensor.quantization = None
    graph.tensors[graph.outputs[0]].type = schema.TensorType.UINT32
    flatbuffer_utils.write_model(model, str(source))
    prepared = prepare_model(replace(request_model, source=source, npu_requirement=NpuRequirement.OPTIONAL))
    observed = inspect_prepared_model(prepared.record_path)
    assert not observed.regions
    assert observed.inputs[0].dtype == "int32"
    assert observed.outputs[0].dtype == "uint32"


def test_explicit_memory_mode_preserves_compiler_label(request_model):
    settings = replace(request_model.settings, memory_mode="Dedicated_Sram")
    prepared = prepare_model(replace(request_model, settings=settings))
    assert prepared.compilation.summary.raw["memory_mode"] == "Dedicated_Sram"
    assert prepared.info.record is not None
    assert prepared.info.record.compiler.reported_memory_mode == "Dedicated_Sram"


def test_memory_mode_must_agree_with_placement(request_model):
    settings = replace(request_model.settings, memory_mode="Sram_Only")
    with pytest.raises(InvalidPreparation, match=r"memory_mode.*placement"):
        prepare_model(replace(request_model, settings=settings))
    assert not request_model.output_dir.exists()

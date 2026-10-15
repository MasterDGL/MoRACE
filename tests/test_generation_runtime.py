"""Isolation regressions for repeated and concurrent scientific configurations."""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys
from threading import Barrier

import pytest

from morace.generation_runtime import (
    GenerationRuntime, current_generation_runtime, generation_overrides,
    science_env_get,
)


def test_capture_is_immutable_and_excludes_unrelated_environment(monkeypatch):
    source = {"LATE_SELECTION_KEEP_TOPK": "2", "UNRELATED_SECRET": "not-captured"}
    runtime = GenerationRuntime.from_environment(source)
    source["LATE_SELECTION_KEEP_TOPK"] = "7"
    assert dict(runtime.environment) == {"LATE_SELECTION_KEEP_TOPK": "2"}
    with pytest.raises(TypeError):
        runtime.environment["LATE_SELECTION_KEEP_TOPK"] = "4"
    with pytest.raises(FrozenInstanceError):
        runtime.environment = {}
    monkeypatch.setenv("LATE_SELECTION_KEEP_TOPK", "9")
    with runtime.activate():
        assert science_env_get("LATE_SELECTION_KEEP_TOPK") == "2"


@pytest.mark.parametrize("key,value", [
    ("LATE_SELECTION_KEEP_TOPK", "bad"),
    ("LATE_SELECTION_KEEP_TOPK", "0"),
    ("LATE_SELECTION_MAX_SCORE_DELTA", "nan"),
    ("LITERATURE_RECALL_BONUS", "inf"),
    ("LITERATURE_RECALL_RETRY_ENABLE", "2"),
])
def test_bad_settings_fail_at_explicit_capture(key, value):
    with pytest.raises(ValueError, match=key):
        GenerationRuntime({key: value})


def test_import_does_not_parse_bad_process_settings():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-B", "-c", "import morace.pipeline_runner; print('imported')"],
        cwd=root, capture_output=True, text=True, timeout=45,
        env={**os.environ, "LATE_SELECTION_KEEP_TOPK": "broken", "LITERATURE_RECALL_BONUS": "broken",
             "M7_LIT_PRESERVING_SWAP_ENABLE": "broken", "MULTI_CONSENSUS_PROMOTION_ENABLE": "broken"},
    )
    assert completed.returncode == 0, completed.stderr
    assert "imported" in completed.stdout


@pytest.mark.parametrize("key", ["MORACE_GALACTICA_PLAIN_PROMPT", "LLM4SD_GALACTICA_PLAIN_PROMPT"])
def test_galactica_prompt_switch_is_captured_and_isolated(key):
    from morace.model_backend import LocalTextGenerator

    generator = LocalTextGenerator("galactica-6.7b", "unused", seed=42)
    before = dict(os.environ)
    for runtime, expected_plain in [(GenerationRuntime(), True), (GenerationRuntime({key: "0"}), False),
                                     (GenerationRuntime(), True)]:
        with runtime.activate():
            text = generator._build_prompt("system", "user")
            assert (text == "system\n\nuser\n") == expected_plain
            if not expected_plain:
                assert "### Instruction:" in text
    assert dict(os.environ) == before


def test_public_metrics_api_uses_fixed_defaults_and_explicit_runtime(monkeypatch):
    from morace import pipeline_runner

    monkeypatch.setenv("LATE_SELECTION_KEEP_TOPK", "broken")
    monkeypatch.setenv("LITERATURE_RECALL_BONUS", "broken")
    monkeypatch.setattr(pipeline_runner, "LATE_SELECTION_KEEP_TOPK", 19)
    monkeypatch.setattr(pipeline_runner, "LITERATURE_RECALL_BONUS", 99.0)
    before = dict(os.environ)
    configured = GenerationRuntime({"LATE_SELECTION_KEEP_TOPK": "2", "LITERATURE_RECALL_BONUS": "0.4"})
    for runtime, expected in [(None, 1), (configured, 2), (None, 1), (configured, 2)]:
        metrics = pipeline_runner.summarize_projection_metrics([], runtime=runtime)
        assert metrics["late_selection_config"]["keep_topk"] == expected
        assert metrics["literature_recall_config"]["bonus"] == (0.4 if runtime else 0.3)
    assert dict(os.environ) == before
    assert current_generation_runtime() is None


def test_nested_repair_scope_restores_settings_even_on_error(monkeypatch):
    monkeypatch.setenv("_MORACE_STAGE2_PROTOCOL_PROFILE", "external")
    runtime = GenerationRuntime({"_MORACE_STAGE2_PROTOCOL_PROFILE": "llama_native"})
    before = dict(os.environ)
    with runtime.activate():
        with pytest.raises(RuntimeError, match="repair failed"):
            with generation_overrides(_MORACE_STAGE2_PROTOCOL_PROFILE="chat_native_core"):
                assert science_env_get("_MORACE_STAGE2_PROTOCOL_PROFILE") == "chat_native_core"
                raise RuntimeError("repair failed")
        assert science_env_get("_MORACE_STAGE2_PROTOCOL_PROFILE") == "llama_native"
    assert dict(os.environ) == before
    assert current_generation_runtime() is None


def test_parallel_contexts_keep_protocol_and_selection_independent(monkeypatch):
    from morace import pipeline_runner
    from morace.stage2_protocol import infer_stage2_protocol_profile, chat_native_core_extraction_policy

    monkeypatch.setenv("STAGE2_PROTOCOL_PROFILE", "invalid-external-profile")
    barrier = Barrier(2)

    def inspect(profile, extraction, topk):
        runtime = GenerationRuntime({
            "STAGE2_PROTOCOL_PROFILE": profile,
            "STAGE2_EXTRACTION_POLICY": extraction,
            "LATE_SELECTION_KEEP_TOPK": str(topk),
        })
        with runtime.activate():
            barrier.wait(timeout=10)
            result = (
                infer_stage2_protocol_profile("llama", "llama", False)[0],
                chat_native_core_extraction_policy(),
                pipeline_runner.summarize_projection_metrics([])["late_selection_config"]["keep_topk"],
            )
            barrier.wait(timeout=10)
            assert science_env_get("STAGE2_PROTOCOL_PROFILE") == profile
        assert current_generation_runtime() is None
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(inspect, "llama_native", "strict_only", 1)
        second = executor.submit(inspect, "chat_native_core", "tolerant_first_json_object", 2)
        assert first.result(timeout=15) == ("llama_native", "strict_only", 1)
        assert second.result(timeout=15) == ("chat_native_core", "tolerant_first_json_object", 2)


def test_model_path_and_trust_use_runtime_while_cuda_stays_machine_setting(monkeypatch):
    from morace import model_backend

    monkeypatch.setenv("MORACE_MODEL_PATH", "/ambient/model")
    monkeypatch.setenv("MORACE_TRUST_REMOTE_CODE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    runtime = GenerationRuntime({"MORACE_MODEL_PATH": "/explicit/model", "MORACE_TRUST_REMOTE_CODE": "0"})
    with runtime.activate():
        assert model_backend.resolve_model_path("llama") == "/explicit/model"
        assert model_backend.trust_remote_code_enabled() is False
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    with GenerationRuntime().activate():
        assert model_backend.resolve_model_path("llama") == "llama"
        assert model_backend.trust_remote_code_enabled() is False

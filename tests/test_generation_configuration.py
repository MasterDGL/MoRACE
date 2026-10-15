"""Generation contracts from the pre-finalization source, without model loading."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import warnings

import pytest
from scipy.stats import ConstantInputWarning

from morace import inference, rule_summary, synthesis
from morace.generation_config import InferenceConfig, RuleSummaryConfig, SynthesisConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((Path(__file__).parent / "fixtures/generation_contract.json").read_text(encoding="utf-8"))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


@pytest.mark.parametrize("module,config_type", [(synthesis, SynthesisConfig), (inference, InferenceConfig), (rule_summary, RuleSummaryConfig)])
def test_cli_parameters_and_defaults_keep_captured_contract(module, config_type):
    name = module.__name__.split(".")[-1]
    expected = CONTRACT["cli_defaults"][name]
    assert vars(module.build_parser().parse_args([])) == expected
    config = config_type()
    assert {key: getattr(config, key) for key in expected} == expected
    assert not hasattr(module, "args")
    with pytest.raises(FrozenInstanceError):
        config.run_tag = "modified"


@pytest.mark.parametrize("module", [synthesis, inference, rule_summary])
def test_cli_help_without_loading_model(module):
    completed = subprocess.run([sys.executable, "-m", module.__name__, "--help"], cwd=PROJECT_ROOT,
                               text=True, encoding="utf-8", capture_output=True, timeout=40)
    assert completed.returncode == 0, completed.stderr
    assert "--run_tag RUN_TAG" in completed.stdout
    assert "--dataset DATASET" in completed.stdout


def test_summary_legacy_cli_alias_is_preserved():
    args = rule_summary.build_parser().parse_args(["--scaffold_input_folder", "other/splits"])
    assert args.split_input_folder == "other/splits"


@pytest.mark.parametrize("case", CONTRACT["prompt_batches"])
def test_real_prompt_batches_match_pre_refactor_source(case, monkeypatch):
    config = InferenceConfig(**{name: case[name] for name in (
        "dataset", "seed", "list_num", "max_samples_per_chunk", "mollogp_stability_cue_enabled")},
        input_folder=str(PROJECT_ROOT / "data/splits"))
    monkeypatch.setenv("MORACE_DATASET_ROOT", "must-not-be-read")
    with warnings.catch_warnings():
        # Constant motif vectors are an expected input to the original Spearman summary.
        warnings.simplefilter("ignore", ConstantInputWarning)
        records, spec, batch_size = inference.build_prompt_records(config.dataset, "", "Contract task brief.", config)
    assert _digest(records) == case["record_sha256"]
    assert batch_size == case["batch_size"]
    assert spec.task_type == case["task_type"]


def test_interleaved_inference_configs_do_not_change_prompt_or_path(tmp_path, monkeypatch):
    config_a = InferenceConfig(input_folder=str(PROJECT_ROOT / "data/splits"),
                              output_folder=str(tmp_path / "output"), list_num=2,
                              max_samples_per_chunk=8, seed=42, run_tag="run-a")
    config_b = replace(config_a, list_num=3, seed=1337, run_tag="run-b", mollogp_stability_cue_enabled=True)
    a1, _, _ = inference.build_prompt_records("bbbp", "", "Contract task brief.", config_a)
    b, _, _ = inference.build_prompt_records("bbbp", "", "Contract task brief.", config_b)
    monkeypatch.setenv("INFERENCE_BBBP_MOLLOGP_STABILITY_CUE_ENABLE", "1")
    monkeypatch.setenv("MORACE_RUN_TAG", "ambient")
    a2, _, _ = inference.build_prompt_records("bbbp", "", "Contract task brief.", config_a)
    assert a1 == a2
    assert len(b) == 3 and a1 != b
    assert "Additional stability cue:" not in a1[0]["stage1_user_prompt"]
    assert "Additional stability cue:" in b[0]["stage1_user_prompt"]
    assert "run-a" in Path(inference.output_path("bbbp", "", config_a)).parts
    assert "run-b" in Path(inference.output_path("bbbp", "", config_b)).parts
    unscoped = replace(config_a, run_tag="")
    assert "ambient" not in Path(inference.output_path("bbbp", "", unscoped)).parts


def test_cli_environment_is_captured_once_and_model_fallback_is_unchanged(monkeypatch):
    environment = {
        "MORACE_RUN_TAG": "initial",
        "MORACE_MODEL_PATH_QWEN35_9B": "/models/qwen",
        "MORACE_MODEL_PATH_GEMMA_4_E4B_IT": "/models/gemma",
        "INFERENCE_BBBP_MOLLOGP_STABILITY_CUE_ENABLE": "1",
        "LATE_SELECTION_KEEP_TOPK": "2",
        "STAGE2_PROTOCOL_PROFILE": "qwen_legacy_d2",
        "MULTI_CONSENSUS_PROMOTION_ENABLE": "1",
        "MORACE_DATASET_ROOT": "/data/fixed",
    }
    args = inference.build_parser().parse_args(["--projector_model", "gemma-4-e4b-it", "--extra_draft_models", "gemma-4-e4b-it"])
    config = InferenceConfig.from_namespace(args, environment=environment)
    environment["MORACE_RUN_TAG"] = "later"
    environment["MULTI_CONSENSUS_PROMOTION_ENABLE"] = "0"
    monkeypatch.setenv("MORACE_MODEL_PATH", "must-not-be-read")
    assert config.run_tag == "initial"
    assert config.model_path == "/models/qwen"
    assert config.projector_model_path == "/models/gemma"
    assert config.extra_draft_model_paths == "/models/gemma"
    assert config.input_folder == "/data/fixed"
    assert config.mollogp_stability_cue_enabled
    assert config.multi_consensus_promotion_enable
    assert not InferenceConfig().multi_consensus_promotion_enable
    assert config.runtime.environment["LATE_SELECTION_KEEP_TOPK"] == "2"
    assert config.runtime.environment["STAGE2_PROTOCOL_PROFILE"] == "qwen_legacy_d2"
    assert dict(InferenceConfig().runtime.environment) == {}
    assert [spec.path for spec in inference.build_draft_specs(config)] == ["/models/qwen", "/models/gemma"]
    assert inference.build_draft_specs(InferenceConfig())[0].path == "qwen3.5-9b"


def test_default_api_dataset_folder_does_not_inherit_ambient_root(monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setenv("MORACE_DATASET_ROOT", "must-not-be-read")
    records, spec, batch_size = inference.build_prompt_records(
        "bbbp", "", "Contract task brief.", InferenceConfig(list_num=1, max_samples_per_chunk=8))
    assert len(records) == 1
    assert batch_size == 8
    assert Path(spec.folder) == Path("data/splits/bbbp")


def test_synthesis_two_configs_keep_real_prompts_and_generation_settings(tmp_path, monkeypatch):
    prompt_path = tmp_path / "synthesize_prompt.json"
    prompt_path.write_text(json.dumps({"bbbp_small": "Small-model task brief.", "bbbp_big": "Big-model task brief."}), encoding="utf-8")
    calls = []
    merging = []
    original_merge = synthesis.merge_rules

    def merge(*args, **kwargs):
        merging.append(kwargs["multi_consensus_promotion_enable"])
        return original_merge(*args, **kwargs)

    def draft(**kwargs):
        calls.append(("draft", kwargs))
        return []

    def project(**kwargs):
        calls.append(("project", kwargs))
        return []

    monkeypatch.setattr(synthesis, "generate_stage1_drafts", draft)
    monkeypatch.setattr(synthesis, "project_drafts_to_rules", project)
    monkeypatch.setattr(synthesis, "merge_rules", merge)
    config_a = SynthesisConfig(input_folder=str(tmp_path), output_folder=str(tmp_path / "out"),
                               run_tag="a", seed=42)
    config_b = replace(config_a, run_tag="b", seed=1337, model="llama-3-8b-instruct",
                       literature_stability_cue_enabled=True, stage1_temperature=0.6,
                       multi_consensus_promotion_enable=True)
    artifacts = [synthesis.main(config) for config in (config_a, config_b, config_a)]
    assert artifacts[0] == artifacts[2]
    assert artifacts[0]["base_prompt"] == "Small-model task brief."
    assert artifacts[1]["base_prompt"] == "Big-model task brief."
    assert artifacts[1]["metrics"]["synth_literature_stability_cue_enabled"] is True
    assert [call[1]["seed"] for call in calls] == [42, 1000042, 1337, 1001337, 42, 1000042]
    assert calls[0][1]["prompt_records"] == calls[4][1]["prompt_records"]
    assert calls[1][1]["heuristic_fallback_sources"] == ("literature",)
    assert calls[2][1]["temperature"] == 0.6
    assert merging == [False, True, False]


def test_two_rule_summaries_use_their_own_inputs_and_training_folder(tmp_path, monkeypatch):
    input_folder = tmp_path / "in"
    config_a = RuleSummaryConfig(input_folder=str(input_folder), output_folder=str(tmp_path / "out"),
        split_input_folder=str(PROJECT_ROOT / "data/splits"), run_tag="a", target_rule_count=3)
    config_b = replace(config_a, run_tag="b", target_rule_count=7, multi_consensus_promotion_enable=True)
    for config in (config_a, config_b):
        source = input_folder / config.run_tag / "qwen3.5-9b/bbbp/qwen3.5-9b_bbbp_dk_response_sample_30.json"
        source.parent.mkdir(parents=True)
        source.write_text(json.dumps({"rules": [], "metrics": {"source_config": config.run_tag}}), encoding="utf-8")
    monkeypatch.setenv("MORACE_RUN_TAG", "wrong-tag")
    monkeypatch.setenv("MORACE_DATASET_ROOT", "wrong-data")
    aggregate = rule_summary.aggregate_rule_candidates
    calls = []

    def record_aggregation(*args, **kwargs):
        calls.append(kwargs)
        return aggregate(*args, **kwargs)

    monkeypatch.setattr(rule_summary, "aggregate_rule_candidates", record_aggregation)
    a1, b, a2 = [rule_summary.main(config) for config in (config_a, config_b, config_a)]
    assert a1 == a2
    assert a1["projection_metrics"]["source_config"] == "a"
    assert b["projection_metrics"]["source_config"] == "b"
    assert [call["target_rule_count"] for call in calls] == [3, 7, 3]
    assert [call["multi_consensus_promotion_enable"] for call in calls] == [False, True, False]
    assert all(Path(call["input_folder"]) == PROJECT_ROOT / "data/splits" for call in calls)


@pytest.mark.parametrize("config_type", [SynthesisConfig, InferenceConfig, RuleSummaryConfig])
def test_explicit_configuration_rejects_ambiguous_run_tags(config_type):
    with pytest.raises(ValueError):
        config_type(run_tag="different run")

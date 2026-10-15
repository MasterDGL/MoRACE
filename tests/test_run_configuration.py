import importlib.util
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from morace import run_configuration as config
from morace.dataset_provenance import record_dataset_provenance
from morace.run_summary import append_task_summary


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_identity_test", ROOT / "scripts/run_paper_experiments.py")
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)


def completed_run(root, *, scientific_status="success"):
    """Use real provenance, branch outcomes and run audit, without training a model."""
    shutil.copytree(ROOT / "data/splits/bbbp", root / "data/splits/bbbp")
    environment = {
        "RUN_TAG": "fixture", "MODEL": "qwen3.5-9b", "MODEL_PATH": "/models/fixture",
        "MORACE_DATASET_MANIFEST": str(ROOT / "data/manifests/split_manifest.csv"),
    }
    resolved = config.prepare_run(root, environment, "bbbp")
    artifact = root / "artifacts/fixture"
    record_dataset_provenance(artifact / "run_config.json", "bbbp", project_root=root,
                              input_folder=str(root / "data/splits"),
                              manifest_path=ROOT / "data/manifests/split_manifest.csv")
    (artifact / "logs/bbbp.log").write_text("complete", encoding="utf-8")
    for branch in ("synthesize", "inference", "all"):
        folder = root / "eval_code_generation_repo/fixture/qwen3.5-9b/bbbp" / branch
        if branch != "synthesize":
            folder /= "sample_30"
        folder.mkdir(parents=True)
        config.write_json(folder / f"qwen3.5-9b_bbbp_{branch}_rules.json", {
            "status": scientific_status, "dataset": "bbbp", "run_tag": "fixture", "knowledge_type": branch,
            "rules": [{"fixture_rule": index} for index in range(4)],
            "selected_rule_count": 4, "compiled_rule_count": 4, "invalid_rule_count": 0,
            "valid_mean": 0.8, "test_mean": 0.7, "final_result_source": branch,
        })
        (folder / f"qwen3.5-9b_bbbp_{branch}_rules.txt").write_text("# rendered fixture\n", encoding="utf-8")
        config.write_json(folder / "outcome.json", {
            "schema_version": "outcome_v1_1", "outcome_category": "success_primary" if scientific_status == "success" else scientific_status,
            "final_result_source": branch, "metric_name": "roc_auc", "generation_seed": 42,
            "actual_valid_mean": 0.8, "actual_test_mean": 0.7,
            "selected_rule_count": 4, "compiled_rule_count": 4, "invalid_rule_count": 0,
        })
    status = append_task_summary(summary_path="artifacts/fixture/summary.json", run_tag="fixture",
                                model="qwen3.5-9b", dataset="bbbp", subtask="", batch_num=30,
                                exec_status="success", exit_code=0, log_path="artifacts/fixture/logs/bbbp.log",
                                project_root=root)
    assert status == scientific_status
    code = config.finish_run(artifact, root)
    identity = config.build_identity(root, resolved, "bbbp")
    return artifact, resolved, identity, code


def test_paper_defaults_ignore_all_known_ambient_scientific_controls():
    public = paper.load_config(ROOT / "configs/paper_experiment.json")
    route = public["routes"][0]
    baseline = paper.build_job_environment({}, public["common_environment"], route,
        model_path="/models/gemma", python_path="python", seed=42, run_tag="test")
    dirty = {key: "irrelevant-contamination" for key in config.DEFAULT_ENVIRONMENT}
    dirty.update({key: "irrelevant-contamination" for key in config.ALIASES})
    dirty.update({"CUDA_VISIBLE_DEVICES": "3", "HF_HOME": "/cache", "HTTPS_PROXY": "https://proxy",
                  "PROJECT_ROOT": "/another/checkout", "MORACE_MODEL_PATH_GEMMA_4_E4B_IT": "/wrong"})
    actual = paper.build_job_environment(dirty, public["common_environment"], route,
        model_path="/models/gemma", python_path="python", seed=42, run_tag="test")
    assert {key: actual[key] for key in config.DEFAULT_ENVIRONMENT} == {key: baseline[key] for key in config.DEFAULT_ENVIRONMENT}
    assert actual["CUDA_VISIBLE_DEVICES"] == "3"
    assert actual["HF_HOME"] == "/cache"
    assert actual["HTTPS_PROXY"] == "https://proxy"
    assert actual["PROJECT_ROOT"] == str(ROOT)
    assert all(key not in actual for key in config.ALIASES)


@pytest.mark.parametrize("value", ["trial 1", "../run", "/absolute", "a/b", "a\\b", "_hidden", "", "a" * 129])
def test_invalid_run_tags_fail_without_creating_output(tmp_path, value):
    with pytest.raises(ValueError):
        config.reserve_run(tmp_path, value)
    assert not (tmp_path / "artifacts").exists()


def test_run_reservation_is_atomic_and_preserves_existing_files(tmp_path):
    def reserve():
        try:
            config.reserve_run(tmp_path, "same_run")
            return True
        except FileExistsError:
            return False
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: reserve(), range(16)))
    assert sum(results) == 1
    sentinel = tmp_path / "artifacts/same_run/summary.json"
    sentinel.write_bytes(b"original result")
    with pytest.raises(FileExistsError):
        config.reserve_run(tmp_path, "same_run")
    assert sentinel.read_bytes() == b"original result"
    assert len({config.new_run_tag() for _ in range(100)}) == 100


def test_generic_override_is_effective_recorded_and_json_escaped(tmp_path):
    unusual_path = 'D:\\model\\"quoted"\\$literal`name\nsecond line'
    resolved = config.prepare_run(tmp_path, {"RUN_TAG": "quoted", "MODEL_PATH": unusual_path,
        "DRAFTS_PER_PROMPT": "4", "LIST_NUM": "2", "STAGE1_TEMPERATURE": "0.95"}, "bace")
    payload = json.loads((tmp_path / "artifacts/quoted/run_config.json").read_text(encoding="utf-8"))
    assert payload["model_path"] == unusual_path
    assert payload["effective_environment"]["MODEL_PATH"] == unusual_path
    assert payload["identity"]["payload"]["environment"]["SYN_DRAFTS_PER_PROMPT"] == "4"
    assert payload["expected_tasks"] == [{"dataset": "bace", "subtask": "", "batch_num": 2}]
    assert resolved["STAGE1_TEMPERATURE"] == "0.95"
    assert not (tmp_path / "prompt_file").exists()
    assert config.file_hash(tmp_path / "artifacts/quoted/prompts/inference_prompt.json") == payload["prompt_hashes"]["inference_prompt.json"]


def test_reused_prompt_source_is_snapshotted_per_run(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("synthesize_prompt.json", "inference_prompt.json"):
        (source / name).write_text('{"task":"custom prompt"}', encoding="utf-8")
    env = {"SKIP_PROMPT_REGEN": "1", "MORACE_PROMPT_SOURCE": str(source)}
    config.prepare_run(tmp_path, {**env, "RUN_TAG": "first"}, "bace")
    (source / "inference_prompt.json").write_text('{"task":"changed"}', encoding="utf-8")
    config.prepare_run(tmp_path, {**env, "RUN_TAG": "second"}, "bace")
    assert (tmp_path / "artifacts/first/prompts/inference_prompt.json").read_text() == '{"task":"custom prompt"}'
    assert (tmp_path / "artifacts/second/prompts/inference_prompt.json").read_text() == '{"task":"changed"}'


@pytest.mark.parametrize("status,expected_code", [("success", 0), ("degraded", config.QUALITY_EXIT_CODE)])
def test_resume_requires_real_audit_but_keeps_scientific_non_success(tmp_path, status, expected_code):
    artifact, environment, identity, code = completed_run(tmp_path, scientific_status=status)
    assert code == expected_code
    assert config.reusable_run(artifact, identity, root=tmp_path, run_tag="fixture") == (True, [])
    assert paper.classify_job_result(code, artifact / "summary.json") == (
        "completed_non_success" if code else "completed", [status])
    # A missing branch artifact invalidates reuse even though the summary and
    # identity still match. This exercises run_audit, not a mock audit response.
    (tmp_path / "eval_code_generation_repo/fixture/qwen3.5-9b/bbbp/all/sample_30/outcome.json").unlink()
    okay, errors = config.reusable_run(artifact, identity, root=tmp_path, run_tag="fixture")
    assert not okay
    assert any("missing JSON artifact" in error for error in errors)


@pytest.mark.parametrize("change", ["budget", "dataset", "model", "input", "prompt", "method", "runtime"])
def test_resume_rejects_changed_experiment(tmp_path, change):
    artifact, environment, identity, _ = completed_run(tmp_path)
    target = "bbbp"
    if change == "budget":
        environment["TARGET_RULE_COUNT"] = "99"
    elif change == "dataset":
        target = "bace"
    elif change == "model":
        environment["MODEL_PATH"] = "/models/other"
    elif change == "input":
        with (tmp_path / "data/splits/bbbp/bbbp_test.csv").open("ab") as handle:
            handle.write(b"\n")
    elif change == "prompt":
        (artifact / "prompts/inference_prompt.json").write_text("changed", encoding="utf-8")
    elif change == "method":
        (tmp_path / "morace").mkdir()
        (tmp_path / "morace/algorithm.py").write_text("changed = True", encoding="utf-8")
    elif change == "runtime":
        environment["PYTHON_BIN"] = "/different/environment/bin/python"
    requested = config.build_identity(tmp_path, environment, target)
    assert config.reusable_run(artifact, requested, root=tmp_path, run_tag="fixture")[0] is False


def test_model_registry_change_invalidates_run_identity(tmp_path):
    registry = tmp_path / "configs/model_registry.json"
    registry.parent.mkdir()
    registry.write_text('{"rf_seeds": [0, 1]}', encoding="utf-8")
    artifact, environment, previous_identity, _ = completed_run(tmp_path)
    assert previous_identity["payload"]["evaluation_config_hashes"]["configs/model_registry.json"] == config.file_hash(registry)
    registry.write_text('{"rf_seeds": [0, 1, 2]}', encoding="utf-8")
    requested = config.build_identity(tmp_path, environment, "bbbp")
    assert previous_identity["fingerprint"] != requested["fingerprint"]
    assert config.reusable_run(artifact, requested, root=tmp_path, run_tag="fixture")[0] is False


@pytest.mark.parametrize("returncode", [-9, -15, 1, 7, 137, config.QUALITY_EXIT_CODE])
def test_success_summary_does_not_hide_process_failure(tmp_path, returncode):
    artifact, _, _, _ = completed_run(tmp_path)
    assert paper.classify_job_result(returncode, artifact / "summary.json")[0] == "failed"


def test_all_success_summary_alone_is_not_resumable(tmp_path):
    artifact, _, identity, _ = completed_run(tmp_path)
    (artifact / "completion.json").unlink()
    assert config.reusable_run(artifact, identity, root=tmp_path, run_tag="fixture")[0] is False
    assert paper.classify_job_result(0, artifact / "summary.json")[0] == "failed"


@pytest.mark.parametrize("suffix", ["json", "txt"])
def test_resume_rejects_missing_final_rules_or_rendered_code(tmp_path, suffix):
    artifact, _, identity, _ = completed_run(tmp_path)
    final_path = tmp_path / f"eval_code_generation_repo/fixture/qwen3.5-9b/bbbp/all/sample_30/qwen3.5-9b_bbbp_all_rules.{suffix}"
    assert config.reusable_run(artifact, identity, root=tmp_path, run_tag="fixture") == (True, [])
    final_path.unlink()
    okay, errors = config.reusable_run(artifact, identity, root=tmp_path, run_tag="fixture")
    assert not okay
    assert any("final artifact" in error for error in errors)


@pytest.mark.parametrize("payload", [[], {"tasks": ["malformed"]}, {"tasks": []}])
def test_malformed_completion_data_is_a_process_failure(tmp_path, payload):
    config.write_json(tmp_path / "summary.json", payload)
    assert paper.classify_job_result(0, tmp_path / "summary.json") == ("failed", [])


def test_environment_alias_and_checkpoint_resolution_are_recorded():
    effective = config.resolve_environment({"LLM4SD_DATASET_ROOT": "/legacy/data", "MODEL": "qwen3.5-9b",
                                           "MORACE_MODEL_PATH_QWEN35_9B": "/selected/qwen",
                                           "CHAT_NATIVE_CORE_EXTRACTION_POLICY": "tolerant_first_json_object"})
    assert effective["MORACE_DATASET_ROOT"] == "/legacy/data"
    assert effective["MODEL_PATH"] == "/selected/qwen"
    assert effective["STAGE2_EXTRACTION_POLICY"] == "tolerant_first_json_object"


def test_extra_model_empty_path_slots_keep_backend_fallback_semantics():
    effective = config.resolve_environment({"EXTRA_DRAFT_MODELS": "qwen3.5-9b,gemma-4-e4b-it",
        "EXTRA_DRAFT_MODEL_PATHS": "/explicit/qwen,", "MORACE_MODEL_PATH_GEMMA_4_E4B_IT": "/selected/gemma"})
    assert effective["EXTRA_DRAFT_MODEL_PATHS"] == "/explicit/qwen,/selected/gemma"


def test_unknown_method_names_are_rejected_instead_of_silently_ignored(tmp_path):
    with pytest.raises(ValueError, match="STAGE2_TEMPERATUR"):
        config.prepare_run(tmp_path, {"RUN_TAG": "typo", "STAGE2_TEMPERATUR": "0.9"}, "bace")
    assert not (tmp_path / "artifacts").exists()


def test_remaining_scoped_outputs_prevent_reusing_an_archived_tag(tmp_path):
    (tmp_path / "eval_code_generation_repo/old_run").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already has artifacts"):
        config.prepare_run(tmp_path, {"RUN_TAG": "old_run"}, "bace")


def test_every_literal_method_environment_read_is_declared():
    import ast
    machine_keys = {"CUDA_VISIBLE_DEVICES", "RULE_CONTRIB_SKLEARN_PYTHON"}
    undeclared = []
    for path in (ROOT / "morace").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            name = ast.unparse(node.func)
            if name not in {"os.environ.get", "os.getenv", "_env_value"}:
                continue
            key = node.args[0].value
            if isinstance(key, str) and not config.is_scientific_environment_key(key) and key not in machine_keys:
                undeclared.append((path.name, key))
    assert not undeclared


def test_bash_launcher_escapes_paths_and_refuses_existing_run(tmp_path):
    bash = shutil.which("bash")
    if os.name == "nt":
        git = shutil.which("git")
        candidate = Path(git).parents[1] / "bin/bash.exe" if git else None
        bash = str(candidate) if candidate and candidate.is_file() else None
    if not bash:
        pytest.skip("Bash is not available")
    project = tmp_path / "project with spaces"
    shutil.copytree(ROOT / "morace", project / "morace", ignore=shutil.ignore_patterns("__pycache__"))
    (project / "scripts").mkdir()
    shutil.copyfile(ROOT / "scripts/run_pipeline.sh", project / "scripts/run_pipeline.sh")
    (project / "prompt_file").mkdir()
    sentinel = project / "prompt_file/inference_prompt.json"
    sentinel.write_bytes(b"shared prompt must stay unchanged")
    model_path = 'C:\\models\\"quoted" $literal`name'
    env = {key: value for key, value in os.environ.items() if not config.is_scientific_environment_key(key)}
    env.update({"PYTHON_BIN": str(Path(sys.executable)).replace("\\", "/"), "PYTHONUTF8": "1",
                "RUN_TAG": "bash_fixture", "MODEL_PATH": model_path,
                "PROJECT_ROOT": str(tmp_path / "wrong checkout"),
                # Abort at provenance, before the first generation call.
                "MORACE_DATASET_MANIFEST": "missing_manifest.csv"})
    command = [bash, str(project / "scripts/run_pipeline.sh").replace("\\", "/"), "bace"]
    first = subprocess.run(command, env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=60)
    assert first.returncode == 1, first.stdout + first.stderr
    artifact = project / "artifacts/bash_fixture"
    cfg = json.loads((artifact / "run_config.json").read_text(encoding="utf-8"))
    assert cfg["model_path"] == model_path
    assert cfg["project_root"] == str(project.resolve())
    assert (artifact / "prompts/inference_prompt.json").is_file()
    assert sentinel.read_bytes() == b"shared prompt must stay unchanged"
    assert "step=dataset_provenance" in first.stdout
    assert not (artifact / "completion.json").exists()
    before = {path.relative_to(artifact): path.read_bytes() for path in artifact.rglob("*") if path.is_file()}
    second = subprocess.run(command, env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=60)
    assert second.returncode != 0
    assert before == {path.relative_to(artifact): path.read_bytes() for path in artifact.rglob("*") if path.is_file()}


@pytest.mark.parametrize("override", [{"LIST_NUM": "0"}, {"STAGE1_TOP_P": "2"},
    {"STAGE1_TEMPERATURE": "nan"}, {"ALL_FINALIST_ENABLE": "perhaps"},
    {"STAGE2_PROTOCOL_PROFILE": "typo"}, {"HEURISTIC_POLICY": "disabled"}])
def test_invalid_scientific_settings_fail_early(override):
    with pytest.raises(ValueError):
        config.resolve_environment(override)

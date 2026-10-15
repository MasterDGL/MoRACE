import os
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_source_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


model_backend = load_source_module("morace_model_backend_under_test", "morace/model_backend.py")
datasets = load_source_module("morace_datasets_under_test", "morace/datasets.py")
paths = load_source_module("morace_paths_under_test", "morace/paths.py")
runtime_snapshot = load_source_module("morace_runtime_snapshot_under_test", "morace/runtime_snapshot.py")
checkpoint_manifest = load_source_module("morace_checkpoint_manifest_under_test", "morace/checkpoint_manifest.py")
model_preflight = load_source_module("morace_model_preflight_under_test", "morace/model_preflight.py")
_saved_backend = sys.modules.get("morace.model_backend")
_saved_preflight = sys.modules.get("morace.model_preflight")
sys.modules["morace.model_backend"] = model_backend
sys.modules["morace.model_preflight"] = model_preflight
try:
    model_canary = load_source_module("morace_model_canary_under_test", "morace/model_canary.py")
finally:
    if _saved_backend is None:
        sys.modules.pop("morace.model_backend", None)
    else:
        sys.modules["morace.model_backend"] = _saved_backend
    if _saved_preflight is None:
        sys.modules.pop("morace.model_preflight", None)
    else:
        sys.modules["morace.model_preflight"] = _saved_preflight
paper_runner = load_source_module("morace_paper_runner_under_test", "scripts/run_paper_experiments.py")


def test_resolve_model_path_has_no_private_default(monkeypatch):
    for key in list(os.environ):
        if key.startswith("MORACE_MODEL_PATH") or key.startswith("LLM4SD_MODEL_PATH"):
            monkeypatch.delenv(key, raising=False)

    assert model_backend.resolve_model_path("qwen3.5-9b") == "qwen3.5-9b"


def test_morace_model_path_env_takes_precedence(monkeypatch):
    monkeypatch.setenv("MORACE_MODEL_PATH_QWEN35_9B", "/models/qwen-public")
    monkeypatch.setenv("LLM4SD_MODEL_PATH_QWEN35_9B", "/models/legacy")

    assert model_backend.resolve_model_path("qwen3.5-9b") == "/models/qwen-public"


def test_legacy_model_path_env_still_supported(monkeypatch):
    monkeypatch.delenv("MORACE_MODEL_PATH", raising=False)
    monkeypatch.setenv("LLM4SD_MODEL_PATH", "/models/legacy")

    assert model_backend.resolve_model_path("chemdfm") == "/models/legacy"


def test_morace_dataset_root_takes_precedence(monkeypatch):
    monkeypatch.setenv("MORACE_DATASET_ROOT", "/datasets/morace")
    monkeypatch.setenv("LLM4SD_DATASET_ROOT", "/datasets/legacy")
    monkeypatch.setenv("DATASET_ROOT", "/datasets/generic")

    assert datasets.resolve_input_folder("data/splits") == "/datasets/morace"


def test_morace_run_tag_takes_precedence(monkeypatch):
    monkeypatch.setenv("MORACE_RUN_TAG", "morace-run")
    monkeypatch.setenv("LLM4SD_RUN_TAG", "legacy-run")

    assert paths.resolve_run_tag() == "morace-run"


def test_generation_seed_advances_deterministically():
    class FakeCuda:
        seeded = []

        @staticmethod
        def is_available():
            return True

        @classmethod
        def manual_seed_all(cls, seed):
            cls.seeded.append(seed)

    class FakeTorch:
        cuda = FakeCuda()
        seeded = []

        @classmethod
        def manual_seed(cls, seed):
            cls.seeded.append(seed)

    generator = model_backend.LocalTextGenerator("qwen2.5-0.5b-instruct", seed=43)

    assert generator._next_generation_seed(FakeTorch) == 43
    assert generator._next_generation_seed(FakeTorch) == 44
    assert FakeTorch.seeded == [43, 44]
    assert FakeCuda.seeded == [43, 44]


def test_remote_checkpoint_code_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("MORACE_TRUST_REMOTE_CODE", raising=False)
    monkeypatch.delenv("LLM4SD_TRUST_REMOTE_CODE", raising=False)
    assert model_backend.trust_remote_code_enabled() is False

    monkeypatch.setenv("MORACE_TRUST_REMOTE_CODE", "1")
    assert model_backend.trust_remote_code_enabled() is True


def test_checkpoint_manifest_round_trip_and_detects_drift(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type": "test"}\n', encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "README.md").write_text("ignored", encoding="utf-8")

    record = checkpoint_manifest.build_checkpoint_record(checkpoint)
    checkpoint_manifest.verify_checkpoint_record("test-model", checkpoint, record)
    assert [item["path"] for item in record["files"]] == ["config.json", "model.safetensors"]

    (checkpoint / "model.safetensors").write_bytes(b"changed")
    try:
        checkpoint_manifest.verify_checkpoint_record("test-model", checkpoint, record)
    except RuntimeError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("checkpoint drift was not detected")


def test_model_preflight_version_parsing():
    assert model_preflight._version_tuple("2.5.1+cu121") == (2, 5, 1)
    assert model_preflight._version_tuple("2.6") == (2, 6, 0)


def test_public_launcher_propagates_seed_and_failures():
    launcher = (ROOT / "scripts/run_pipeline.sh").read_text(encoding="utf-8")

    assert '--seed "${GLOBAL_SEED}"' in launcher
    for knowledge_type in ("synthesize", "inference", "all"):
        block_start = launcher.index(f'--knowledge_type "{knowledge_type}"')
        block_end = launcher.index("|| { local step_exit=$?", block_start)
        assert '--generation_seed "${GLOBAL_SEED}"' in launcher[block_start:block_end]
    assert 'export MORACE_SEED="${GLOBAL_SEED}"' in launcher
    assert "PIPELINE_FAILURE_COUNT" in launcher
    assert "else\n    local exit_code=$?" in launcher
    assert 'exit 1' in launcher
    assert 'run_pipeline_logged "tox21" "${SENTINEL_TOX21}"' not in launcher


def test_runtime_snapshot_accepts_dynamic_launcher_values():
    launcher = ROOT / "scripts/run_pipeline.sh"
    mentions = runtime_snapshot.launch_script_mentions(
        str(launcher),
        {
            "MODEL": "a-model-not-hardcoded-in-the-launcher",
            "MODEL_PATH": "/a/private/path/not-hardcoded-in-the-launcher",
        },
    )

    assert mentions == {"MODEL": True, "MODEL_PATH": True}


def test_paper_routes_define_recovered_stage2_protocols():
    config = paper_runner.load_config(ROOT / "configs/paper_experiment.json")
    paper_runner.validate_config(config)
    routes = {route["id"]: route for route in config["routes"]}

    assert routes["gemma_native_clean"]["environment"]["STAGE2_PROTOCOL_PROFILE"] == "gemma_native"
    assert routes["qwen_legacy"]["environment"]["STAGE2_PROTOCOL_PROFILE"] == "qwen_legacy_d2"
    assert routes["llama31_native_clean"]["environment"]["STAGE2_PROTOCOL_PROFILE"] == "chat_native_core"
    assert routes["deepseek_d52"]["environment"]["STAGE2_TARGET_THINNING_POLICY"] == "first_valid_target_rule"
    assert routes["chemdfm_b3"]["environment"]["STAGE2_CANONICALIZATION_POLICY"] == "safe_enum_feature_canonicalization"
    assert routes["gemma_native_clean"]["python_path_env"] == "MORACE_PYTHON_GEMMA_NATIVE_CLEAN"
    assert routes["deepseek_d52"]["python_path_env"] == "MORACE_PYTHON_DEEPSEEK_D52"
    assert routes["deepseek_d52"]["model_path_env"] == "MORACE_MODEL_PATH_DEEPSEEK_LLM_7B_CHAT"
    assert routes["deepseek_d52"]["checkpoint_manifest_model"] == "deepseek-llm-7b-chat-safetensors"


def test_model_generation_canary_uses_deterministic_short_generation(monkeypatch):
    calls = {}

    class FakeGenerator:
        def __init__(self, model_name, model_path, seed):
            calls["init"] = (model_name, model_path, seed)

        def generate_batch(self, prompts, **kwargs):
            calls["generate"] = (prompts, kwargs)
            return ["OK"]

    monkeypatch.setattr(
        model_canary,
        "check_model_runtime",
        lambda model_name, model_path: {"model": model_name, "model_type": "test"},
    )
    monkeypatch.setattr(model_canary, "LocalTextGenerator", FakeGenerator)

    result = model_canary.run_generation_canary("test-model", "/models/test", max_new_tokens=4)

    assert result["output"] == "OK"
    assert calls["init"] == ("test-model", "/models/test", 42)
    assert calls["generate"][1]["temperature"] == 0.0
    assert calls["generate"][1]["max_new_tokens"] == 4


def test_paper_job_environment_removes_ambient_protocol_contamination():
    config = paper_runner.load_config(ROOT / "configs/paper_experiment.json")
    route = next(item for item in config["routes"] if item["id"] == "gemma_native_clean")
    environment = paper_runner.build_job_environment(
        {
            "PATH": "/usr/bin",
            "STAGE2_PROTOCOL_PROFILE": "chat_native_core",
            "CHAT_NATIVE_CORE_TARGET_THINNING_POLICY": "first_valid_target_rule",
        },
        config["common_environment"],
        route,
        model_path="/models/gemma",
        python_path="/envs/gemma/bin/python",
        seed=43,
        run_tag="paper_gemma_bace_seed43",
        checkpoint_manifest="/repo/configs/checkpoint_manifest.json",
    )

    assert environment["PATH"] == "/usr/bin"
    assert environment["STAGE2_PROTOCOL_PROFILE"] == "gemma_native"
    assert "CHAT_NATIVE_CORE_TARGET_THINNING_POLICY" not in environment
    assert environment["STAGE2_TARGET_THINNING_POLICY"] == "none"
    assert environment["GLOBAL_SEED"] == "43"
    assert environment["PYTHON_BIN"] == "/envs/gemma/bin/python"
    assert environment["MORACE_CHECKPOINT_MANIFEST"] == "/repo/configs/checkpoint_manifest.json"


def test_paper_preflight_checks_runtime_before_hashing(monkeypatch, tmp_path):
    commands = []

    class Completed:
        returncode = 0

    def fake_run(command, **kwargs):
        commands.append(command)
        return Completed()

    monkeypatch.setattr(paper_runner.subprocess, "run", fake_run)
    paper_runner.run_route_preflight(
        {
            "id": "deepseek_d52",
            "model": "deepseek-llm-7b-chat",
            "checkpoint_manifest_model": "deepseek-llm-7b-chat-safetensors",
        },
        "/models/deepseek-safe",
        "/env/bin/python",
        tmp_path / "checkpoint_manifest.json",
    )

    assert commands[0][2] == "morace.model_preflight"
    assert commands[1][2] == "morace.checkpoint_manifest"
    assert commands[1][-1] == "deepseek-llm-7b-chat-safetensors=/models/deepseek-safe"


def test_paper_runner_rejects_zero_exit_without_completed_summary(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config_path = tmp_path / "paper.json"
    config_path.write_text(
        json.dumps(
            {
                "datasets": ["bbbp"],
                "seeds": [42],
                "common_environment": {},
                "routes": [
                    {
                        "id": "fixture_route",
                        "model": "fixture-model",
                        "model_path_env": "MORACE_MODEL_PATH_FIXTURE",
                        "python_path_env": "MORACE_PYTHON_FIXTURE",
                        "environment": {"STAGE2_PROTOCOL_PROFILE": "qwen_legacy_d2"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class Completed:
        returncode = 0

    monkeypatch.setattr(paper_runner, "ROOT", tmp_path)
    monkeypatch.setattr(paper_runner, "run_route_preflight", lambda *_args: None)
    runtime_snapshot = paper_runner.probe_runtime(sys.executable)
    monkeypatch.setattr(paper_runner, "probe_runtime", lambda *_args, **_kwargs: runtime_snapshot)
    monkeypatch.setattr(paper_runner.subprocess, "run", lambda *_args, **_kwargs: Completed())
    monkeypatch.setattr(
        paper_runner.sys,
        "argv",
        [
            "run_paper_experiments.py",
            "--config",
            str(config_path),
            "--execute",
            "--model-path",
            f"fixture_route={checkpoint}",
            "--python-path",
            f"fixture_route={Path(sys.executable)}",
        ],
    )

    assert paper_runner.main() == 1


def test_public_launcher_uses_configured_python_executable():
    launcher = (ROOT / "scripts/run_pipeline.sh").read_text(encoding="utf-8")

    assert 'PYTHON_BIN="${PYTHON_BIN:-python}"' in launcher
    assert 'run_step synthesize "${PYTHON_BIN}" -m morace.synthesis' in launcher
    assert 'run_step train_eval_all "${PYTHON_BIN}" -m morace.evaluation_pipeline' in launcher
    assert 'python -m morace.' not in launcher


def test_paper_runner_distinguishes_quality_outcomes_from_process_failures(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "status": "degraded",
                        "pipeline_exit_status": "success",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    from morace.run_configuration import QUALITY_EXIT_CODE, file_hash
    (tmp_path / "completion.json").write_text(json.dumps({
        "kind": "completed_non_success", "exit_code": QUALITY_EXIT_CODE,
        "summary_sha256": file_hash(summary_path),
    }), encoding="utf-8")
    assert paper_runner.classify_job_result(QUALITY_EXIT_CODE, summary_path) == (
        "completed_non_success",
        ["degraded"],
    )

    summary_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "status": "failed",
                        "pipeline_exit_status": "failed",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert paper_runner.classify_job_result(1, summary_path) == ("failed", [])

    summary_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "status": "incomplete",
                        "pipeline_exit_status": "success",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert paper_runner.classify_job_result(1, summary_path) == ("failed", [])
    assert paper_runner.read_completed_task_statuses(summary_path) is None

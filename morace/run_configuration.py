"""Resolve launcher settings and bind artifacts to one explicit experiment identity.

This module uses only CPU-side configuration and hashing. It never loads a model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


CONFIG_VERSION = "effective_run_config_v2"
QUALITY_EXIT_CODE = 20
RUNTIME_PACKAGES = (
    "accelerate", "huggingface-hub", "numpy", "pandas", "pydantic", "rdkit",
    "safetensors", "scikit-learn", "scipy", "tokenizers", "torch", "transformers",
)
# Run this stdlib-only probe in the selected route's interpreter. Distribution
# metadata is read without importing torch/transformers or loading a checkpoint.
RUNTIME_PROBE_SOURCE = """
import importlib.metadata, json, platform, sys
packages = {}
for name in PACKAGE_NAMES:
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps({"schema_version": "runtime_identity_v1", "status": "available",
    "python_executable": sys.executable, "python_version": sys.version,
    "platform": platform.platform(), "packages": packages}))
""".replace("PACKAGE_NAMES", repr(RUNTIME_PACKAGES))
PROTOCOL_ENV_KEYS = {
    "STAGE2_PROTOCOL_PROFILE", "STAGE2_EXTRACTION_POLICY", "STAGE2_CANONICALIZATION_POLICY",
    "STAGE2_FAMILY_FEATURE_POLICY", "STAGE2_FAMILY_TARGET_POLICY", "STAGE2_FAMILY_TARGETS",
    "STAGE2_TARGET_VOCAB_POLICY", "STAGE2_TARGET_THINNING_POLICY", "STAGE2_TERMINATION_POLICY",
    "STAGE2_RULE_COUNT_POLICY",
}
PROTOCOL_ALIASES = {
    key.replace("STAGE2_", "CHAT_NATIVE_CORE_", 1): key
    for key in PROTOCOL_ENV_KEYS
    if key not in {"STAGE2_PROTOCOL_PROFILE", "STAGE2_FAMILY_FEATURE_POLICY"}
}
ALIASES = {
    **PROTOCOL_ALIASES,
    "LLM4SD_DATASET_ROOT": "MORACE_DATASET_ROOT", "DATASET_ROOT": "MORACE_DATASET_ROOT",
    "LLM4SD_TRUST_REMOTE_CODE": "MORACE_TRUST_REMOTE_CODE",
    "LLM4SD_GALACTICA_PLAIN_PROMPT": "MORACE_GALACTICA_PLAIN_PROMPT",
    "LLM4SD_LLAMA_ALLOW_AUTO_DEVICE_MAP": "MORACE_LLAMA_ALLOW_AUTO_DEVICE_MAP",
    "LLM4SD_BAICHUAN_ALLOW_AUTO_DEVICE_MAP": "MORACE_BAICHUAN_ALLOW_AUTO_DEVICE_MAP",
}
# One defaults table for the generic launcher and the paper launcher. Empty Stage-2
# values intentionally delegate to the existing model-specific protocol resolver.
DEFAULT_ENVIRONMENT = {
    "MODEL": "qwen3.5-9b", "MODEL_PATH": "", "PROJECTOR_MODEL": "", "PROJECTOR_MODEL_PATH": "",
    "USE_CROSS_MODEL_PROJECTOR": "0", "PROJECTOR_MODEL_ENFORCE": "1",
    "LAUNCHER_DEFAULT_PROJECTOR_MODEL": "", "LAUNCHER_DEFAULT_PROJECTOR_PATH": "",
    "EXTRA_DRAFT_MODELS": "", "EXTRA_DRAFT_MODEL_PATHS": "",
    "DRAFTS_PER_PROMPT": "1", "SYN_DRAFTS_PER_PROMPT": "", "INF_DRAFTS_PER_PROMPT": "",
    "SYN_MAX_NEW_TOKENS": "512", "INF_MAX_NEW_TOKENS": "512", "PROJECTOR_MAX_NEW_TOKENS": "192",
    "SYN_PROJECTOR_MAX_NEW_TOKENS": "192", "INF_PROJECTOR_MAX_NEW_TOKENS": "",
    "SYN_MAX_PROMPT_TOKENS": "1536", "INF_MAX_PROMPT_TOKENS": "1200", "PROJECTOR_MAX_PROMPT_TOKENS": "1280",
    "STAGE1_TEMPERATURE": "0.45", "STAGE2_TEMPERATURE": "0.0", "STAGE1_TOP_P": "0.92", "STAGE2_TOP_P": "1.0",
    "STAGE1_NUM_BEAMS": "1", "STAGE2_NUM_BEAMS": "1", "MAX_SAMPLES_PER_CHUNK": "0",
    "LIST_NUM": "30", "QM9_LIST_NUM": "50", "GLOBAL_SEED": "42", "TARGET_RULE_COUNT": "30",
    "LATE_SELECTION_KEEP_TOPK": "1", "LATE_SELECTION_MAX_SCORE_DELTA": "0.0",
    "LATE_SELECTION_MAX_SURVIVORS_PER_PROMPT": "1", "LATE_SELECTION_REQUIRE_NOVELTY": "1",
    "LITERATURE_RECALL_RETRY_ENABLE": "1", "LITERATURE_RECALL_BONUS": "0.30",
    "ALL_PROMOTION_VALID_EPSILON": "0.005", "KEEP_ALL_VALID_DELTA": "0.010",
    "REVERT_ALL_VALID_DELTA": "0.015", "THRESHOLD_KEEP_MIN_GAIN": "0.010",
    "MIN_ALL_COMPILED_FEATURES": "3", "CAPACITY_KEEP_VALID_DELTA": "0.010",
    "ALL_FINALIST_ENABLE": "0", "ALL_FINALIST_MAX_CANDIDATES": "3", "ALL_FINALIST_KEEP_LIT_DELTA": "0.004",
    "ALL_FINALIST_MIN_EFFECTIVE_LIT": "2", "ALL_FINALIST_MIN_EFFECTIVE_DATA": "5", "ALL_FINALIST_MIN_ROOTS": "6",
    "ALL_PLUS_ONE_ALIAS_ENABLE": "0", "ALL_PLUS_ONE_ALIAS_MAX_VALID_DROP": "0.002",
    "ALL_PLUS_ONE_ALIAS_MIN_PROMOTED_MULTI_ROOTS": "2", "LIT_ONLY_SLOT_CONTROL_ENABLE": "0",
    "LIT_ONLY_SLOT_MAX_VALID_DROP": "0.001", "LIT_ONLY_SLOT_MIN_PROMOTED_MULTI_ROOTS": "2",
    "M7_LIT_PRESERVING_SWAP_ENABLE": "0", "MULTI_CONSENSUS_PROMOTION_ENABLE": "0",
    "SYNTH_BBBP_LITERATURE_STABILITY_CUE_ENABLE": "0", "INFERENCE_BBBP_MOLLOGP_STABILITY_CUE_ENABLE": "0",
    "STAGE2_BACKEND": "prompt_only", "MORACE_DATASET_ROOT": "data/splits",
    "MORACE_DATASET_MANIFEST": "data/manifests/split_manifest.csv", "MORACE_CHECKPOINT_MANIFEST": "",
    "SKIP_PROMPT_REGEN": "0", "MORACE_PROMPT_SOURCE": "prompt_file",
    "MORACE_TRUST_REMOTE_CODE": "0", "MORACE_GALACTICA_PLAIN_PROMPT": "1",
    "MORACE_LLAMA_ALLOW_AUTO_DEVICE_MAP": "0", "MORACE_BAICHUAN_ALLOW_AUTO_DEVICE_MAP": "0",
    # These describe the implementation; they are not alternate parser switches.
    "SCHEMA_MODE": "strict_then_lenient_with_repair", "HEURISTIC_POLICY": "low_strength_fallback",
    **{key: "" for key in PROTOCOL_ENV_KEYS},
}
BOOLEAN_KEYS = {
    key for key in DEFAULT_ENVIRONMENT if key.endswith("_ENABLE")
} | {"USE_CROSS_MODEL_PROJECTOR", "PROJECTOR_MODEL_ENFORCE", "LATE_SELECTION_REQUIRE_NOVELTY",
     "SKIP_PROMPT_REGEN", "MORACE_TRUST_REMOTE_CODE", "MORACE_GALACTICA_PLAIN_PROMPT",
     "MORACE_LLAMA_ALLOW_AUTO_DEVICE_MAP", "MORACE_BAICHUAN_ALLOW_AUTO_DEVICE_MAP"}
FLOAT_KEYS = {key for key, value in DEFAULT_ENVIRONMENT.items() if re.fullmatch(r"\d+\.\d+", value)}
INTEGER_KEYS = {key for key, value in DEFAULT_ENVIRONMENT.items() if value.isdigit()} - BOOLEAN_KEYS
INTEGER_KEYS |= {"SYN_DRAFTS_PER_PROMPT", "INF_DRAFTS_PER_PROMPT", "INF_PROJECTOR_MAX_NEW_TOKENS"}
ZERO_ALLOWED = {"MAX_SAMPLES_PER_CHUNK", "GLOBAL_SEED"} | {key for key in INTEGER_KEYS if key.startswith("MIN_") or "_MIN_" in key}
SCIENCE_PREFIXES = (
    "MORACE_", "LLM4SD_", "STAGE1_", "STAGE2_", "CHAT_NATIVE_CORE_", "LATE_SELECTION_",
    "LITERATURE_RECALL_", "ALL_FINALIST_", "ALL_PLUS_ONE_", "LIT_ONLY_", "M7_",
    "MULTI_CONSENSUS_", "SYNTH_BBBP_", "INFERENCE_BBBP_",
)


def validate_run_tag(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("run_tag must be 1-128 ASCII letters/digits/._-, starting with a letter or digit; no normalization is performed")
    return value


def new_run_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:10]


def is_scientific_environment_key(key: str) -> bool:
    return key in DEFAULT_ENVIRONMENT or key in ALIASES or key in {
        "PROJECT_ROOT", "RUN_TAG", "MORACE_RUN_TAG", "LLM4SD_RUN_TAG", "MORACE_SEED",
        "_MORACE_STAGE2_PROTOCOL_PROFILE", "_LLM4SD_STAGE2_PROTOCOL_PROFILE",
        "LAUNCH_SCRIPT_PATH", "MORACE_EXPECTED_IDENTITY", "MORACE_MODEL_PATH", "LLM4SD_MODEL_PATH",
    } or key.startswith(("MORACE_MODEL_PATH_", "LLM4SD_MODEL_PATH_"))


def validate_environment_names(environment: Mapping[str, Any]) -> None:
    unknown = sorted(key for key in environment if key.startswith(SCIENCE_PREFIXES)
                     and not is_scientific_environment_key(key)
                     and not key.startswith("MORACE_PYTHON_"))
    if unknown:
        raise ValueError("unknown method environment variables (check spelling or declare them in run_configuration): " + ", ".join(unknown))


def resolve_environment(overrides: Mapping[str, Any]) -> dict[str, str]:
    """Apply explicit values and documented legacy aliases, then validate types."""
    supplied = {key: str(value) for key, value in overrides.items()}
    resolved = dict(DEFAULT_ENVIRONMENT)
    for alias, primary in ALIASES.items():
        if supplied.get(alias) and not supplied.get(primary) and resolved[primary] == DEFAULT_ENVIRONMENT[primary]:
            resolved[primary] = supplied[alias]
    for key in DEFAULT_ENVIRONMENT:
        if key in supplied and supplied[key] != "":
            resolved[key] = supplied[key]
    # Match the backend's path precedence while recording the actual selected path.
    # Explicit paths passed by the paper runner consequently cannot be replaced by
    # a generic checkpoint environment variable in the invoking shell.
    from morace.model_backend import normalize_model_name
    def model_path(model: str, explicit: str) -> str:
        suffix = normalize_model_name(model).upper().replace(".", "").replace("-", "_")
        return (explicit or supplied.get(f"MORACE_MODEL_PATH_{suffix}")
                or supplied.get(f"LLM4SD_MODEL_PATH_{suffix}")
                or supplied.get("MORACE_MODEL_PATH") or supplied.get("LLM4SD_MODEL_PATH") or model)
    resolved["MODEL_PATH"] = model_path(resolved["MODEL"], resolved["MODEL_PATH"])
    for key, fallback in {
        "PROJECTOR_MODEL": "MODEL", "PROJECTOR_MODEL_PATH": "MODEL_PATH",
        "LAUNCHER_DEFAULT_PROJECTOR_MODEL": "PROJECTOR_MODEL",
        "LAUNCHER_DEFAULT_PROJECTOR_PATH": "PROJECTOR_MODEL_PATH",
        "SYN_DRAFTS_PER_PROMPT": "DRAFTS_PER_PROMPT", "INF_DRAFTS_PER_PROMPT": "DRAFTS_PER_PROMPT",
        "INF_PROJECTOR_MAX_NEW_TOKENS": "PROJECTOR_MAX_NEW_TOKENS",
    }.items():
        resolved[key] = resolved[key] or resolved[fallback]
    extra_models = [value.strip() for value in resolved["EXTRA_DRAFT_MODELS"].split(",") if value.strip()]
    extra_paths = [value.strip() for value in resolved["EXTRA_DRAFT_MODEL_PATHS"].split(",")] if resolved["EXTRA_DRAFT_MODEL_PATHS"] else []
    if len(extra_paths) > len(extra_models):
        raise ValueError("EXTRA_DRAFT_MODEL_PATHS has more paths than extra models")
    if extra_models:
        resolved["EXTRA_DRAFT_MODEL_PATHS"] = ",".join(model_path(model, extra_paths[index] if index < len(extra_paths) else "")
                                                       for index, model in enumerate(extra_models))
    for key in BOOLEAN_KEYS:
        value = resolved[key].lower()
        if value not in {"0", "1", "true", "false", "yes", "no", "on", "off"}:
            raise ValueError(f"{key} must be a boolean (0 or 1)")
        resolved[key] = "1" if value in {"1", "true", "yes", "on"} else "0"
    for key in INTEGER_KEYS:
        try:
            number = int(resolved[key])
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer") from exc
        if number < (0 if key in ZERO_ALLOWED else 1):
            raise ValueError(f"{key} is out of range")
        resolved[key] = str(number)
    for key in FLOAT_KEYS:
        number = float(resolved[key])
        if not math.isfinite(number) or number < 0 or (key.endswith("_TOP_P") and not 0 < number <= 1):
            raise ValueError(f"{key} is out of range")
        resolved[key] = str(number)
    for key in ("SCHEMA_MODE", "HEURISTIC_POLICY"):
        if resolved[key] != DEFAULT_ENVIRONMENT[key]:
            raise ValueError(f"{key} describes the implementation and cannot be overridden")
    from morace import stage2_protocol
    policy_sets = {
        "STAGE2_PROTOCOL_PROFILE": "ALLOWED_STAGE2_PROTOCOL_PROFILES",
        "STAGE2_EXTRACTION_POLICY": "ALLOWED_STAGE2_EXTRACTION_POLICIES",
        "STAGE2_CANONICALIZATION_POLICY": "ALLOWED_STAGE2_CANONICALIZATION_POLICIES",
        "STAGE2_FAMILY_FEATURE_POLICY": "ALLOWED_STAGE2_FAMILY_FEATURE_POLICIES",
        "STAGE2_FAMILY_TARGET_POLICY": "ALLOWED_STAGE2_FAMILY_TARGET_POLICIES",
        "STAGE2_TARGET_VOCAB_POLICY": "ALLOWED_STAGE2_TARGET_VOCAB_POLICIES",
        "STAGE2_TARGET_THINNING_POLICY": "ALLOWED_STAGE2_TARGET_THINNING_POLICIES",
        "STAGE2_TERMINATION_POLICY": "CHAT_NATIVE_CORE_TERMINATION_POLICIES",
        "STAGE2_RULE_COUNT_POLICY": "CHAT_NATIVE_CORE_RULE_COUNT_POLICIES",
    }
    for key, constant in policy_sets.items():
        if resolved[key] and resolved[key] not in getattr(stage2_protocol, constant):
            raise ValueError(f"unsupported {key}={resolved[key]!r}")
    if resolved["STAGE2_FAMILY_TARGETS"] and not set(resolved["STAGE2_FAMILY_TARGETS"].split(",")) <= stage2_protocol.ALLOWED_STAGE2_FAMILY_TARGETS:
        raise ValueError("unsupported STAGE2_FAMILY_TARGETS")
    if resolved["PROJECTOR_MODEL_ENFORCE"] == "1" and resolved["PROJECTOR_MODEL"] != resolved["LAUNCHER_DEFAULT_PROJECTOR_MODEL"]:
        raise ValueError("launcher-enforced projector model mismatch")
    return resolved


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_tasks(target: str, environment: Mapping[str, str]) -> list[dict[str, Any]]:
    from morace.prompt_builder import BASE_TASK_DESCRIPTIONS, TOX21_TASK_DESCRIPTIONS, SIDER_TASK_DESCRIPTIONS, QM9_TASK_DESCRIPTIONS
    groups = {"tox21": list(TOX21_TASK_DESCRIPTIONS), "sider": list(SIDER_TASK_DESCRIPTIONS), "qm9": list(QM9_TASK_DESCRIPTIONS)}
    if target in {"others", "all"}:
        pairs = [(dataset, "") for dataset in BASE_TASK_DESCRIPTIONS]
    elif target == "sentinel":
        pairs = [(dataset, "") for dataset in ("bbbp", "bace", "esol")]
    elif target in BASE_TASK_DESCRIPTIONS:
        pairs = [(target, "")]
    elif target in groups:
        pairs = [(target, subtask) for subtask in groups[target]]
    elif ":" in target:
        dataset, subtask = target.split(":", 1)
        if dataset not in groups or subtask not in groups[dataset]:
            raise ValueError(f"unsupported task: {target}")
        pairs = [(dataset, subtask)]
    else:
        raise ValueError(f"unsupported target: {target}")
    return sorted([{"dataset": dataset, "subtask": subtask, "batch_num": int(environment["QM9_LIST_NUM" if dataset == "qm9" else "LIST_NUM"])}
                   for dataset, subtask in pairs], key=lambda item: (item["dataset"], item["subtask"]))


def prompt_contents(root: Path, environment: Mapping[str, str]) -> dict[str, bytes]:
    from morace.prompt_builder import get_synthesize_task_prompt, get_inference_task_prompt
    if environment["SKIP_PROMPT_REGEN"] == "1":
        source = root / environment["MORACE_PROMPT_SOURCE"]
        return {name: (source / name).read_bytes() for name in ("synthesize_prompt.json", "inference_prompt.json")}
    return {name: json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8") for name, payload in (
        ("synthesize_prompt.json", get_synthesize_task_prompt()), ("inference_prompt.json", get_inference_task_prompt()))}


def current_runtime_snapshot() -> dict[str, Any]:
    """Describe the interpreter actually executing prepare, using only metadata."""
    packages = {}
    for name in RUNTIME_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema_version": "runtime_identity_v1", "status": "available",
        "python_executable": sys.executable, "python_version": sys.version,
        "platform": platform.platform(), "packages": packages,
    }


def probe_runtime(python_executable: str, *, environment: Mapping[str, str] | None = None,
                  cwd: Path | None = None, allow_unavailable: bool = True) -> dict[str, Any]:
    """Probe the requested Python once per route; dry plans may lack that Python."""
    same_python = os.path.normcase(os.path.abspath(python_executable)) == os.path.normcase(os.path.abspath(sys.executable))
    python_environment = {key for key in os.environ if key.startswith("PYTHON")}
    python_environment.update(key for key in (environment or {}) if key.startswith("PYTHON") and key != "PYTHON_BIN")
    same_environment = environment is None or all(environment.get(key) == os.environ.get(key) for key in python_environment if key != "PYTHON_BIN")
    try:
        if same_python and same_environment:
            return current_runtime_snapshot()
        completed = subprocess.run(
            [python_executable, "-c", RUNTIME_PROBE_SOURCE], cwd=cwd,
            env=dict(environment) if environment is not None else None,
            capture_output=True, text=True, encoding="utf-8", timeout=30, check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"probe exited with code {completed.returncode}")
        snapshot = json.loads(completed.stdout)
        if (not isinstance(snapshot, dict) or snapshot.get("status") != "available"
                or snapshot.get("schema_version") != "runtime_identity_v1"
                or not all(isinstance(snapshot.get(key), str) and snapshot[key]
                           for key in ("python_executable", "python_version", "platform"))
                or not isinstance(snapshot.get("packages"), dict)
                or set(snapshot["packages"]) != set(RUNTIME_PACKAGES)
                or any(value is not None and not isinstance(value, str) for value in snapshot["packages"].values())):
            raise ValueError("invalid runtime probe response")
        return snapshot
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        if not allow_unavailable:
            raise RuntimeError(f"cannot probe selected Python {python_executable!r}: {exc}") from exc
        return {"schema_version": "runtime_identity_v1", "status": "unavailable",
                "python_executable": python_executable, "reason": f"{type(exc).__name__}: {exc}"}


def build_identity(root: Path, environment: Mapping[str, str], target: str,
                   *, runtime_snapshot: Mapping[str, Any] | None = None) -> dict[str, Any]:
    root = root.resolve()
    effective = resolve_environment(environment)
    tasks = target_tasks(target, effective)
    inputs = {}
    for task in tasks:
        dataset, subtask = task["dataset"], task["subtask"]
        for split in ("train", "valid", "test"):
            path = root / effective["MORACE_DATASET_ROOT"] / dataset / f"{subtask or dataset}_{split}.csv"
            inputs[str(path.resolve())] = file_hash(path)
    manifests = {key: file_hash(root / effective[key]) if effective[key] else None
                 for key in ("MORACE_DATASET_MANIFEST", "MORACE_CHECKPOINT_MANIFEST")}
    method_files = sorted((root / "morace").glob("*.py")) + [root / "scripts/run_pipeline.sh"]
    methods = {path.relative_to(root).as_posix(): file_hash(path) for path in method_files}
    payload = {
        "version": CONFIG_VERSION, "environment": effective, "tasks": tasks,
        "python_executable": str(environment.get("PYTHON_BIN") or sys.executable),
        "runtime": dict(runtime_snapshot) if runtime_snapshot is not None else probe_runtime(
            str(environment.get("PYTHON_BIN") or sys.executable), environment={**os.environ, **environment}, cwd=root),
        "input_hashes": inputs, "manifest_hashes": manifests,
        "evaluation_config_hashes": {"configs/model_registry.json": file_hash(root / "configs/model_registry.json")},
        "prompt_hashes": {name: hashlib.sha256(content).hexdigest() for name, content in prompt_contents(root, effective).items()},
        "method_hashes": methods,
    }
    return {"fingerprint": hashlib.sha256(json_bytes(payload)).hexdigest(), "payload": payload}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def reserve_run(root: Path, run_tag: str) -> Path:
    """Atomic claim; an existing directory is never reset, even after a failed run."""
    run_tag = validate_run_tag(run_tag)
    for namespace in ("synthesize_model_response", "inference_model_response", "summarized_inference_rules",
                      "eval_code_generation_repo", "eval_result"):
        if (root / namespace / run_tag).exists():
            raise FileExistsError(f"run tag already has artifacts in {namespace}: {run_tag}")
    parent = root / "artifacts"
    parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = parent / run_tag
    artifact_dir.mkdir(exist_ok=False)
    return artifact_dir


def prepare_run(root: Path, incoming: Mapping[str, str], target: str) -> dict[str, str]:
    root = root.resolve()
    validate_environment_names(incoming)
    effective = resolve_environment(incoming)
    run_tag = validate_run_tag(incoming.get("RUN_TAG") or incoming.get("MORACE_RUN_TAG") or incoming.get("LLM4SD_RUN_TAG") or new_run_tag())
    # The planner's snapshot is not trusted as evidence of the environment at
    # execution time: collect again inside the interpreter running this prepare.
    runtime = current_runtime_snapshot()
    identity = build_identity(root, {**effective, "PYTHON_BIN": incoming.get("PYTHON_BIN") or sys.executable}, target,
                              runtime_snapshot=runtime)
    expected = incoming.get("MORACE_EXPECTED_IDENTITY")
    if expected and expected != identity["fingerprint"]:
        raise ValueError("effective configuration, inputs, or runtime changed after the paper job was planned")
    prompts = prompt_contents(root, effective)
    if any(hashlib.sha256(data).hexdigest() != identity["payload"]["prompt_hashes"][name] for name, data in prompts.items()):
        raise ValueError("prompt source changed while preparing run")
    artifact_dir = reserve_run(root, run_tag)
    prompt_dir = artifact_dir / "prompts"
    prompt_dir.mkdir()
    (artifact_dir / "logs").mkdir()
    for name, content in prompts.items():
        (prompt_dir / name).write_bytes(content)
    resolved_config = {key.lower(): value for key, value in effective.items()}
    config = {
        **resolved_config, "schema_version": "run_config_v1_1", "run_tag": run_tag,
        "project_root": str(root), "python_bin": incoming.get("PYTHON_BIN", sys.executable),
        "stage2_protocol_profile_override": effective["STAGE2_PROTOCOL_PROFILE"],
        "resolved_config": resolved_config, "effective_environment": effective,
        "environment_overrides": {key: value for key, value in incoming.items() if key in DEFAULT_ENVIRONMENT and str(value) != DEFAULT_ENVIRONMENT[key]},
        "identity": identity, "expected_tasks": identity["payload"]["tasks"],
        "prompt_hashes": identity["payload"]["prompt_hashes"], "file_hashes": identity["payload"]["method_hashes"],
        "runtime": runtime,
    }
    write_json(artifact_dir / "run_config.json", config)
    write_json(artifact_dir / "summary.json", {"schema_version": "run_summary_v1_2", "run_tag": run_tag, "target": target, "tasks": [], "counts": {}})
    return {**effective, "RUN_TAG": run_tag, "MORACE_RUN_TAG": run_tag, "LLM4SD_RUN_TAG": run_tag,
            "MORACE_SEED": effective["GLOBAL_SEED"], "PROJECT_ROOT": str(root), "PROMPT_DIR": str(prompt_dir),
            "DATASET_MANIFEST": effective["MORACE_DATASET_MANIFEST"], "PYTHON_BIN": incoming.get("PYTHON_BIN") or sys.executable}


def task_keys(tasks: list[dict]) -> list[tuple[str, str, int]]:
    return sorted((str(task["dataset"]), str(task.get("subtask", "")), int(task["batch_num"])) for task in tasks)


def finish_run(artifact_dir: Path, root: Path) -> int:
    from morace.run_audit import audit_run
    config = json.loads((artifact_dir / "run_config.json").read_text(encoding="utf-8"))
    summary_path = artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if task_keys(summary["tasks"]) != task_keys(config["expected_tasks"]):
        raise ValueError("completed task set differs from planned tasks")
    report = audit_run(artifact_dir, project_root=root, require_success=False)
    write_json(artifact_dir / "audit.json", report)
    if report["status"] != "pass":
        raise ValueError("run audit failed: " + "; ".join(report["errors"]))
    statuses = [task["status"] for task in summary["tasks"]]
    if not statuses or any(status not in {"success", "degraded", "partial", "skipped"} for status in statuses):
        raise ValueError("run has incomplete or failed tasks")
    code = QUALITY_EXIT_CODE if any(status != "success" for status in statuses) else 0
    write_json(artifact_dir / "completion.json", {
        "kind": "completed_non_success" if code else "completed", "exit_code": code,
        "identity_fingerprint": config["identity"]["fingerprint"], "summary_sha256": file_hash(summary_path),
    })
    return code


def reusable_run(artifact_dir: Path, expected_identity: dict, *, root: Path, run_tag: str) -> tuple[bool, list[str]]:
    from morace.run_audit import audit_run
    try:
        config = json.loads((artifact_dir / "run_config.json").read_text(encoding="utf-8"))
        summary_path = artifact_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        completion = json.loads((artifact_dir / "completion.json").read_text(encoding="utf-8"))
        if config.get("identity") != expected_identity or config.get("run_tag") != run_tag or summary.get("run_tag") != run_tag:
            return False, ["run identity differs from the requested experiment"]
        if task_keys(summary["tasks"]) != task_keys(expected_identity["payload"]["tasks"]):
            return False, ["completed task set differs from requested tasks"]
        statuses = [task["status"] for task in summary["tasks"]]
        expected_code = QUALITY_EXIT_CODE if any(status != "success" for status in statuses) else 0
        if completion.get("exit_code") != expected_code or completion.get("kind") != ("completed_non_success" if expected_code else "completed"):
            return False, ["completion marker does not describe a completed experiment"]
        if completion.get("identity_fingerprint") != expected_identity["fingerprint"] or completion.get("summary_sha256") != file_hash(summary_path):
            return False, ["completion marker does not match the current artifacts"]
        for name, digest in expected_identity["payload"]["prompt_hashes"].items():
            if file_hash(artifact_dir / "prompts" / name) != digest:
                return False, [f"prompt snapshot differs: {name}"]
        report = audit_run(artifact_dir, project_root=root, require_success=False)
        return report["status"] == "pass", report["errors"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return False, [f"incomplete or malformed run: {exc}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "finish"])
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--target", default="sentinel")
    parser.add_argument("--run-tag")
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            environment = prepare_run(args.project_root, os.environ, args.target)
            # Quoting is performed by Python; user paths are never interpolated into JSON or code.
            for key in sorted(set(ALIASES) | {"_MORACE_STAGE2_PROTOCOL_PROFILE", "_LLM4SD_STAGE2_PROTOCOL_PROFILE"}):
                print(f"unset {key}")
            for key, value in environment.items():
                print(f"export {key}={shlex.quote(value)}")
            return 0
        return finish_run(args.project_root / "artifacts" / validate_run_tag(args.run_tag), args.project_root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[Pipeline] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

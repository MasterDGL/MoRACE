#!/usr/bin/env python
"""Plan or execute the configured five-route, seven-dataset, three-seed experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/paper_experiment.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from morace.run_configuration import (
    DEFAULT_ENVIRONMENT, PROTOCOL_ENV_KEYS, QUALITY_EXIT_CODE,
    build_identity, file_hash, is_scientific_environment_key, probe_runtime,
    resolve_environment, reusable_run, validate_run_tag, validate_environment_names,
)



def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_model_paths(values: list[str]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for value in values:
        route_id, separator, path = value.partition("=")
        if not separator or not route_id or not path:
            raise ValueError(f"invalid --model-path {value!r}; expected ROUTE_ID=/path/to/checkpoint")
        paths[route_id] = path
    return paths


def select(values: list, requested: list, label: str) -> list:
    if not requested:
        return values
    selected = [value for value in values if str(value) in requested]
    missing = sorted(set(requested) - {str(value) for value in selected})
    if missing:
        raise ValueError(f"unknown {label}: {', '.join(missing)}")
    return selected


def build_job_environment(
    base_environment: dict[str, str],
    common_environment: dict,
    route: dict,
    *,
    model_path: str,
    python_path: str,
    seed: int,
    run_tag: str,
    checkpoint_manifest: str = "",
) -> dict[str, str]:
    validate_run_tag(run_tag)
    validate_environment_names(base_environment)
    # Preserve CUDA/cache/proxy/runtime environment, never ambient method controls.
    environment = {key: value for key, value in base_environment.items()
                   if not is_scientific_environment_key(key)}
    explicit = {**common_environment, **route.get("environment", {}),
                "MODEL": route["model"], "MODEL_PATH": model_path,
                "PROJECTOR_MODEL": route["model"], "PROJECTOR_MODEL_PATH": model_path,
                "LAUNCHER_DEFAULT_PROJECTOR_MODEL": route["model"],
                "LAUNCHER_DEFAULT_PROJECTOR_PATH": model_path,
                "GLOBAL_SEED": str(seed), "MORACE_CHECKPOINT_MANIFEST": checkpoint_manifest}
    environment.update(resolve_environment(explicit))
    environment.update({"PROJECT_ROOT": str(ROOT), "PYTHON_BIN": python_path,
                        "RUN_TAG": run_tag, "MORACE_RUN_TAG": run_tag, "LLM4SD_RUN_TAG": run_tag})
    return environment


def run_route_preflight(
    route: dict, model_path: str, python_path: str, manifest_path: Path | None,
    environment: dict[str, str] | None = None,
) -> None:
    commands = [
        [
            python_path,
            "-m",
            "morace.model_preflight",
            "--model",
            route["model"],
            "--model-path",
            model_path,
        ]
    ]
    if manifest_path is not None:
        manifest_model = route.get("checkpoint_manifest_model", route["model"])
        commands.append(
            [
                python_path,
                "-m",
                "morace.checkpoint_manifest",
                "--manifest",
                str(manifest_path),
                "--model",
                f"{manifest_model}={model_path}",
            ]
        )
    for command in commands:
        completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"route preflight failed for {route['id']} with exit code {completed.returncode}: "
                + " ".join(command)
            )


def validate_config(config: dict) -> None:
    if not isinstance(config.get("common_environment"), dict):
        raise ValueError("config common_environment must be an object")
    unknown_common = sorted(set(config["common_environment"]) - set(DEFAULT_ENVIRONMENT))
    if unknown_common:
        raise ValueError(f"unsupported common environment keys: {', '.join(unknown_common)}")
    if config.get("configuration_version", "effective_run_config_v1") != "effective_run_config_v1":
        raise ValueError("unsupported paper configuration version")
    for key in ("datasets", "seeds", "routes"):
        if not isinstance(config.get(key), list) or not config[key]:
            raise ValueError(f"config {key} must be a non-empty list")
    for key in ("datasets", "seeds"):
        if len({str(value) for value in config[key]}) != len(config[key]):
            raise ValueError(f"config {key} must not contain duplicates")
    resolve_environment(config["common_environment"])
    route_ids: set[str] = set()
    for route in config.get("routes", []):
        route_id = str(route.get("id", ""))
        if not route_id or route_id in route_ids:
            raise ValueError(f"route id must be non-empty and unique: {route_id!r}")
        route_ids.add(route_id)
        route_environment = route.get("environment")
        if not isinstance(route_environment, dict):
            raise ValueError(f"route {route_id} must define an environment object")
        unknown = sorted(set(route_environment) - PROTOCOL_ENV_KEYS)
        if unknown:
            raise ValueError(f"route {route_id} has unsupported protocol environment keys: {', '.join(unknown)}")
        if not route_environment.get("STAGE2_PROTOCOL_PROFILE"):
            raise ValueError(f"route {route_id} must define STAGE2_PROTOCOL_PROFILE")
        if not route.get("python_path_env"):
            raise ValueError(f"route {route_id} must define python_path_env")


def read_completed_task_statuses(summary_path: Path) -> list[str] | None:
    if not summary_path.is_file():
        return None
    try:
        summary = load_config(summary_path)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(summary, dict):
        return None
    tasks = summary.get("tasks")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(task, dict) for task in tasks):
        return None
    statuses = [str(task.get("status", "unknown")) for task in tasks]
    pipeline_statuses = [str(task.get("pipeline_exit_status", "")) for task in tasks]
    if not all(status == "success" for status in pipeline_statuses):
        return None
    accepted_statuses = {"success", "degraded", "partial", "skipped"}
    if not all(status in accepted_statuses for status in statuses):
        return None
    return statuses


def classify_job_result(returncode: int, summary_path: Path) -> tuple[str, list[str]]:
    statuses = read_completed_task_statuses(summary_path)
    if statuses is None:
        return "failed", []
    # A signal or arbitrary failure cannot masquerade as a scientific outcome.
    if returncode not in {0, QUALITY_EXIT_CODE}:
        return "failed", statuses
    try:
        completion = load_config(summary_path.parent / "completion.json")
    except (OSError, ValueError):
        return "failed", statuses
    if not isinstance(completion, dict):
        return "failed", statuses
    non_success = any(status != "success" for status in statuses)
    expected_code = QUALITY_EXIT_CODE if non_success else 0
    expected_kind = "completed_non_success" if non_success else "completed"
    if (returncode != expected_code or completion.get("exit_code") != expected_code
            or completion.get("kind") != expected_kind
            or completion.get("summary_sha256") != file_hash(summary_path)):
        return "failed", statuses
    return expected_kind, statuses



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--route", action="append", default=[], help="route id; repeat to select multiple")
    parser.add_argument("--dataset", action="append", default=[], help="dataset; repeat to select multiple")
    parser.add_argument("--seed", action="append", default=[], help="seed; repeat to select multiple")
    parser.add_argument(
        "--model-path",
        action="append",
        default=[],
        metavar="ROUTE_ID=PATH",
        help="checkpoint path override; otherwise use the route's model_path_env variable",
    )
    parser.add_argument(
        "--python-path",
        action="append",
        default=[],
        metavar="ROUTE_ID=PATH",
        help="route-specific Python executable; otherwise use python_path_env or the current interpreter",
    )
    parser.add_argument("--execute", action="store_true", help="run commands; default is a dry-run plan")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify selected checkpoints and runtimes without launching experiment jobs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip only identity-matched completed runs whose provenance and artifacts pass audit",
    )
    parser.add_argument("--plan-output", type=Path, help="write the full resolved job plan before preflight or execution")
    parser.add_argument("--run-prefix", default="paper", help="run tag prefix; default: paper")
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    validate_config(config)
    routes = [route for route in config["routes"] if not args.route or route["id"] in args.route]
    missing_routes = sorted(set(args.route) - {route["id"] for route in routes})
    if missing_routes:
        raise ValueError(f"unknown route: {', '.join(missing_routes)}")
    datasets = select(config["datasets"], args.dataset, "dataset")
    seeds = select(config["seeds"], args.seed, "seed")
    model_paths = parse_model_paths(args.model_path)
    python_paths = parse_model_paths(args.python_path)
    unknown_model_paths = sorted(set(model_paths) - {route["id"] for route in config["routes"]})
    if unknown_model_paths:
        raise ValueError(f"model path supplied for unknown route: {', '.join(unknown_model_paths)}")
    unknown_python_paths = sorted(set(python_paths) - {route["id"] for route in config["routes"]})
    if unknown_python_paths:
        raise ValueError(f"Python path supplied for unknown route: {', '.join(unknown_python_paths)}")
    launcher = ROOT / "scripts/run_pipeline.sh"
    manifest_value = str(config.get("checkpoint_manifest", "")).strip()
    manifest_path = (ROOT / manifest_value).resolve() if manifest_value else None
    run_preflight = args.execute or args.preflight_only
    validate_run_tag(args.run_prefix)
    job_count = len(routes) * len(datasets) * len(seeds)
    print(f"[Paper] jobs={job_count} execute={args.execute} preflight_only={args.preflight_only}")

    # Resolve the entire matrix first. A failed preflight or first process must
    # leave a complete, reviewable plan when --plan-output was requested.
    jobs = []
    route_runtimes = {}
    for route in routes:
        model_path = model_paths.get(route["id"]) or os.environ.get(route["model_path_env"], "")
        python_path = (python_paths.get(route["id"]) or os.environ.get(route["python_path_env"], "") or sys.executable)
        for dataset in datasets:
            for seed in seeds:
                run_tag = validate_run_tag(f"{args.run_prefix}_{route['id']}_{dataset}_seed{seed}")
                environment = build_job_environment(
                    os.environ, config["common_environment"], route,
                    model_path=model_path, python_path=python_path, seed=seed, run_tag=run_tag,
                    checkpoint_manifest=str(manifest_path) if manifest_path is not None else "",
                )
                if (run_preflight or args.plan_output) and route["id"] not in route_runtimes:
                    route_runtimes[route["id"]] = probe_runtime(python_path, environment=environment, cwd=ROOT)
                identity = build_identity(ROOT, environment, dataset, runtime_snapshot=route_runtimes[route["id"]]) if args.execute or args.plan_output else None
                jobs.append({"run_tag": run_tag, "route": route["id"], "dataset": dataset,
                             "seed": seed, "python": python_path, "model_path": model_path,
                             "environment": environment, "identity": identity})
                print(f"[Paper] route={route['id']} dataset={dataset} seed={seed} "
                      f"model_path={model_path or '<required-at-execution>'} "
                      f"python={python_path} stage2_profile={route['environment']['STAGE2_PROTOCOL_PROFILE']}")
                print(f"[Paper] run_tag={run_tag} batches={environment['LIST_NUM']} "
                      f"drafts={environment['SYN_DRAFTS_PER_PROMPT']}/{environment['INF_DRAFTS_PER_PROMPT']} "
                      f"rules={environment['TARGET_RULE_COUNT']}")
    if args.plan_output:
        plan = [{key: job[key] for key in ("run_tag", "route", "dataset", "seed", "python", "identity")} for job in jobs]
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(json.dumps({"jobs": plan}, indent=2, ensure_ascii=False), encoding="utf-8")
    if not run_preflight:
        return 0
    if manifest_path is not None and not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest not found: {manifest_path}")

    completed_count = 0
    completed_non_success_count = 0
    for route in routes:
        route_jobs = [job for job in jobs if job["route"] == route["id"]]
        first_job = route_jobs[0]
        model_path, python_path = first_job["model_path"], first_job["python"]
        if route_runtimes[route["id"]]["status"] != "available":
            raise RuntimeError(f"cannot execute route {route['id']}: selected Python runtime is unavailable: "
                               f"{route_runtimes[route['id']].get('reason', python_path)}")
        if not model_path:
            raise RuntimeError(f"missing checkpoint path for {route['id']}; set {route['model_path_env']} "
                               f"or pass --model-path {route['id']}=/path")
        if not Path(model_path).is_dir():
            raise FileNotFoundError(f"checkpoint directory not found for {route['id']}: {model_path}")
        if not Path(python_path).is_file():
            raise FileNotFoundError(f"Python executable not found for {route['id']}: {python_path}")
        preflight_environment = build_job_environment(
            os.environ, config["common_environment"], route, model_path=model_path,
            python_path=python_path, seed=seeds[0],
            run_tag=validate_run_tag(f"preflight_{route['id']}"),
            checkpoint_manifest=str(manifest_path) if manifest_path is not None else "",
        )
        run_route_preflight(route, model_path, python_path, manifest_path, preflight_environment)
        if args.preflight_only:
            print(f"[Paper] PREFLIGHT_PASSED route={route['id']}")
            continue
        for job in route_jobs:
            run_tag, dataset, seed = job["run_tag"], job["dataset"], job["seed"]
            identity = job["identity"]
            artifact_dir = ROOT / "artifacts" / run_tag
            if artifact_dir.exists():
                reasons = []
                if args.resume:
                    compatible, reasons = reusable_run(artifact_dir, identity, root=ROOT, run_tag=run_tag)
                    if compatible:
                        existing_statuses = read_completed_task_statuses(artifact_dir / "summary.json")
                        if existing_statuses is None:
                            raise RuntimeError(f"completed run has invalid task statuses: {artifact_dir}")
                        completed_count += 1
                        if any(status != "success" for status in existing_statuses):
                            completed_non_success_count += 1
                        print(f"[Paper] RESUME_SKIP route={route['id']} dataset={dataset} "
                              f"seed={seed} task_statuses={existing_statuses}")
                        continue
                detail = "; ".join(reasons) or "resume was not requested"
                raise FileExistsError(f"refusing to overwrite or reuse {artifact_dir}: {detail}; "
                                      "use a new --run-prefix for a new experiment")
            environment = dict(job["environment"])
            environment["MORACE_EXPECTED_IDENTITY"] = identity["fingerprint"]
            completed = subprocess.run(["bash", str(launcher), dataset], cwd=ROOT, env=environment, check=False)
            result_kind, task_statuses = classify_job_result(completed.returncode, artifact_dir / "summary.json")
            if result_kind == "failed":
                print(f"[Paper] FAILED route={route['id']} dataset={dataset} seed={seed}", file=sys.stderr)
                return completed.returncode or 1
            completed_count += 1
            if result_kind == "completed_non_success":
                completed_non_success_count += 1
                print(f"[Paper] COMPLETED_NON_SUCCESS route={route['id']} dataset={dataset} "
                      f"seed={seed} task_statuses={task_statuses}")
    if args.execute:
        print(f"[Paper] completed={completed_count}/{job_count} "
              f"completed_non_success={completed_non_success_count} process_failures=0")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())

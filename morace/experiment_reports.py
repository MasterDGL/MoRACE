"""Source-linked experiment summaries without training or score imputation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from morace.artifact_inputs import resolve_run_rules
from morace.metric_policy import metric_direction, normalize_metric_name
from morace.run_audit import audit_run

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "experiment_reports_v1"
SUCCESS = {"success", "success_primary", "success_fallback"}
KEY = ("dataset", "route", "seed", "metric")
MAIN_FIELDS = ["run_dir", "run_tag", "dataset", "subtask", "route", "route_source", "seed", "scored",
               "actual_run_tag", "actual_dataset", "actual_subtask", "actual_route", "actual_seed", "actual_metric",
               "rf_seeds", "metric", "score", "valid_score", "n_rules", "status",
               "scientific_status", "outcome_category", "final_source", "audit_status",
               "outcome_path", "outcome_sha256", "rules_path", "rules_sha256", "reason"]
FEATURE_FIELDS = ["feature_dir", "dataset", "arm", "rf_seed", "split", "metric", "score",
                  "directional_delta_vs_base", "status", "final_source", "protocol_sha256", "reason"]


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _object_or_empty(path):
    try:
        obj = _read(path)
        return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError):
        return {}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _score(value):
    if isinstance(value, bool) or value in (None, ""):
        raise ValueError("Missing numeric score")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite score")
    return number


def _seed(value):
    if value in (None, ""):
        return ""
    number = int(value)
    if str(value).strip() != str(number) or number < 0:
        raise ValueError(f"Invalid generation seed: {value!r}")
    return str(number)


def _route(config):
    resolved = config.get("resolved_config") or {}
    if not isinstance(resolved, dict):
        resolved = {}
    declared = config.get("route_id") or config.get("route") or resolved.get("route_id")
    if declared:
        return str(declared), "run_config"
    model = resolved.get("model") or config.get("model") or resolved.get("projector_model")
    profile = resolved.get("stage2_protocol_profile") or resolved.get("STAGE2_PROTOCOL_PROFILE") or "unspecified_profile"
    return (f"{model}|{profile}" if model else "unknown"), "recorded_model_and_profile"


def _expected_metric(dataset):
    if dataset in {"bace", "bbbp", "clintox", "hiv", "tox21", "sider"}:
        return "roc_auc"
    if dataset in {"esol", "freesolv", "lipophilicity"}:
        return "rmse"
    return ""


def collect_run(run_dir, *, project_root=None, planned=None):
    directory = Path(run_dir).resolve()
    if directory.name == "summary.json":
        directory = directory.parent
    root = Path(project_root).resolve() if project_root else (directory.parent.parent if directory.parent.name == "artifacts" else ROOT)
    summary = _object_or_empty(directory / "summary.json")
    config = _object_or_empty(directory / "run_config.json")
    try:
        audit = audit_run(directory, project_root=root, require_success=False, allow_legacy=False)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        audit = {"status": "fail", "errors": [f"{type(exc).__name__}: {exc}"], "tasks": []}
    actual_route, route_source = _route(config)
    route = actual_route
    if planned:
        route, route_source = str(planned["route"]), "requested_matrix_plan"
    tasks = summary.get("tasks")
    tasks = [task for task in tasks if isinstance(task, dict)] if isinstance(tasks, list) else []
    if planned:
        # One public-matrix job is one requested single-task row. Conflicting
        # disk identities must not create another requested pairing key.
        matches = [task for task in tasks if task.get("dataset") == planned["dataset"] and not task.get("subtask")]
        if len(matches) == 1:
            tasks = matches
        elif not tasks:
            tasks = [{"status": "missing_task", "_missing": True}]
        elif len(tasks) != 1:
            tasks = [{"status": "identity_conflict", "_plan_conflict": "Disk summary does not contain one unique planned task"}]
    else:
        expected_tasks = config.get("expected_tasks") or []
        expected_tasks = expected_tasks if isinstance(expected_tasks, list) else []
        for expected in (item for item in expected_tasks if isinstance(item, dict)):
            if not any((task.get("dataset"), task.get("subtask") or "") == (expected.get("dataset"), expected.get("subtask") or "") for task in tasks):
                tasks.append(dict(expected, status="missing_task", _missing=True))
    if not tasks:
        tasks = [{"dataset": "unknown", "status": "missing_run", "_missing": True}]
    rows = []
    for task in tasks:
        row = {field: "" for field in MAIN_FIELDS}
        result = task.get("result") or {}
        if not isinstance(result, dict):
            result = {}
        try:
            actual_seed = _seed(result.get("generation_seed") if result.get("generation_seed") is not None else config.get("global_seed"))
        except (TypeError, ValueError):
            actual_seed = ""
        actual_tag = summary.get("run_tag") or config.get("run_tag") or ""
        actual_dataset = "" if task.get("_missing") else task.get("dataset") or ""
        actual_subtask = task.get("subtask") or ""
        actual_metric = normalize_metric_name(result.get("metric_name"))
        requested_dataset = planned["dataset"] if planned else task.get("dataset") or "unknown"
        requested_metric = _expected_metric(requested_dataset) if planned else ""
        row.update(run_dir=str(directory), run_tag=planned["run_tag"] if planned else actual_tag or directory.name,
                   dataset=requested_dataset, subtask="" if planned else actual_subtask, route=route,
                   route_source=route_source, seed=_seed(planned["seed"]) if planned else actual_seed,
                   actual_run_tag=actual_tag, actual_dataset=actual_dataset, actual_subtask=actual_subtask,
                   actual_route=actual_route if config else "", actual_seed=actual_seed, actual_metric=actual_metric,
                   metric=requested_metric or actual_metric or _expected_metric(requested_dataset), scored=False,
                   scientific_status=task.get("status", "unknown"), outcome_category=result.get("outcome_category", ""),
                   final_source=result.get("final_result_source", ""), audit_status=audit["status"],
                   status="audit_failed", reason="; ".join(audit.get("errors", [])))
        if task.get("_missing"):
            row["status"] = task["status"]
        elif task.get("_plan_conflict"):
            row.update(status="artifact_failed", reason=task["_plan_conflict"])
        elif audit["status"] == "pass":
            try:
                if planned and (actual_dataset != planned["dataset"] or actual_subtask or actual_seed != _seed(planned["seed"]) or actual_tag != planned["run_tag"]):
                    raise ValueError("Completed run identity differs from requested matrix job")
                if task.get("status") == "success":
                    matches = [item for item in resolve_run_rules(directory, dataset=row["dataset"], project_root=root)
                               if item["subtask"] == row["subtask"]]
                    if len(matches) != 1:
                        raise ValueError("Expected exactly one matching final result")
                    item = matches[0]
                    outcome = item["outcome"]
                else:
                    # The shared rule resolver intentionally requires successful
                    # rules. An audited degraded result can still have an actual
                    # metric; read that outcome without asserting usable rules.
                    outcome_path = (root / result["outcome_path"]).resolve()
                    outcome = _read(outcome_path)
                    for field, expected in (("dataset", row["dataset"]), ("subtask", row["subtask"]), ("run_tag", row["run_tag"]), ("knowledge_type", "all")):
                        if field in outcome and outcome[field] != expected:
                            raise ValueError(f"Outcome {field} differs from requested task")
                    item = {"outcome_path": outcome_path, "final_result_source": outcome.get("final_result_source", "")}
                    linked_rules = (outcome.get("artifact_paths") or {}).get("metadata")
                    if linked_rules and (root / linked_rules).is_file():
                        item["rules_path"] = root / linked_rules
                actual_seed = _seed(outcome.get("generation_seed") if outcome.get("generation_seed") is not None else config.get("global_seed"))
                row["actual_seed"] = actual_seed
                if not actual_seed:
                    raise ValueError("Generation seed is not recorded")
                if planned and (row["dataset"] != planned["dataset"] or actual_seed != _seed(planned["seed"]) or row["run_tag"] != planned["run_tag"]):
                    raise ValueError("Completed run identity differs from requested matrix job")
                if planned and planned.get("identity") and config.get("identity") != planned["identity"]:
                    raise ValueError("Recorded effective identity differs from requested matrix job")
                recorded_route, recorded_route_source = _route(config)
                if planned and recorded_route_source == "run_config" and recorded_route != str(planned["route"]):
                    raise ValueError("Recorded route differs from requested matrix job")
                metric = normalize_metric_name(outcome.get("metric_name"))
                if not metric:
                    raise ValueError("Metric is not recorded")
                if planned and requested_metric and metric != requested_metric:
                    raise ValueError("Recorded metric differs from the planned dataset metric")
                test_score = _score(outcome["actual_test_mean"]) if outcome.get("actual_test_mean") is not None else ""
                valid_score = _score(outcome["actual_valid_mean"]) if outcome.get("actual_valid_mean") is not None else ""
                row.update(seed=_seed(planned["seed"]) if planned else actual_seed,
                           actual_metric=metric, rf_seeds=json.dumps(outcome.get("rf_seeds") or []), metric=metric,
                           score=test_score, valid_score=valid_score, n_rules=outcome.get("selected_rule_count", ""),
                           scored=test_score != "", status=task.get("status", "unknown"), outcome_category=outcome["outcome_category"], final_source=item["final_result_source"],
                           outcome_path=str(item["outcome_path"]), outcome_sha256=_sha(item["outcome_path"]),
                           rules_path=str(item.get("rules_path", "")), rules_sha256=_sha(item["rules_path"]) if item.get("rules_path") else "",
                           reason="" if test_score != "" else "No actual measured test score; null-model/ITT values are not substituted")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                row.update(status="artifact_failed", reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows, audit


def summarize_main(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["route"], row["dataset"], row["subtask"], row["metric"])].append(row)
    result = []
    for (route, dataset, subtask, metric), group in sorted(groups.items()):
        valid = [row for row in group if row["scored"]]
        scores = [row["score"] for row in valid]
        result.append({"route": route, "dataset": dataset, "subtask": subtask, "metric": metric,
                       "n_requested": len(group), "n_scored": len(scores), "n_not_scored": len(group) - len(scores),
                       "n_distinct_generation_seeds": len({row["seed"] for row in valid}),
                       "mean_score": statistics.fmean(scores) if scores else "",
                       "std_ddof0": statistics.pstdev(scores) if scores else "",
                       "status_counts": json.dumps(dict(Counter(row["status"] for row in group)), sort_keys=True)})
    return result


def _key(row):
    return tuple(str(row.get(name, "")) for name in KEY)


def pair_comparator(rows, comparator_path):
    with Path(comparator_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(KEY) | {"score", "status"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Comparator CSV requires columns {sorted(required)}")
        comparator = list(reader)
    external = {}
    for row in comparator:
        row["seed"] = _seed(row["seed"])
        row["metric"] = normalize_metric_name(row["metric"])
        if not all(row.get(name) for name in KEY):
            raise ValueError("Comparator matching keys must be nonempty")
        if row["status"] not in SUCCESS | {"degraded", "partial", "failed", "killed", "skipped", "missing", "audit_failed", "artifact_failed", "missing_run", "missing_task", "incomplete", "unknown"}:
            raise ValueError(f"Unsupported comparator status: {row['status']}")
        if row["status"] in SUCCESS | {"degraded", "partial"} and row["score"] != "":
            row["score"] = _score(row["score"])
            row["scored"] = True
        else:
            row["score"] = ""
            row["scored"] = False
        if _key(row) in external:
            raise ValueError(f"Duplicate comparator key: {_key(row)}")
        external[_key(row)] = row
    seen, pairs = set(), []
    for row in rows:
        if row.get("subtask"):
            raise ValueError("Matched comparator CSV currently supports single-task datasets only")
        key = _key(row)
        if key in seen:
            raise ValueError(f"Ambiguous main-result pairing key: {key}; do not count repeated runs as independent seeds")
        seen.add(key)
        other = external.get(key)
        pair = {name: row[name] for name in KEY}
        pair.update(main_run=row["run_tag"], main_status=row["status"], comparator_status=other["status"] if other else "missing",
                    main_score=row["score"] if row["scored"] else "",
                    comparator_score=other["score"] if other else "", directional_delta="", pair_status="unpaired")
        if row["scored"] and other and other["scored"]:
            direction = metric_direction(row["metric"])
            if direction["defaulted"]:
                raise ValueError(f"Unknown metric direction: {row['metric']}")
            sign = 1 if direction["higher_is_better"] else -1
            pair.update(pair_status="paired", directional_delta=sign * (row["score"] - other["score"]))
        pairs.append(pair)
    for key, other in external.items():
        if key not in seen:
            pair = {name: other[name] for name in KEY}
            pair.update(main_run="", main_status="missing", comparator_status=other["status"], main_score="",
                        comparator_score=other["score"], directional_delta="", pair_status="unpaired")
            pairs.append(pair)
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[(pair["route"], pair["dataset"], pair["metric"])].append(pair)
    summaries = []
    for (route, dataset, metric), group in sorted(grouped.items()):
        deltas = [pair["directional_delta"] for pair in group if pair["pair_status"] == "paired"]
        summaries.append({"route": route, "dataset": dataset, "metric": metric, "n_requested_union": len(group),
                          "n_paired": len(deltas), "n_unpaired": len(group) - len(deltas),
                          "mean_directional_delta": statistics.fmean(deltas) if deltas else "",
                          "std_ddof0": statistics.pstdev(deltas) if deltas else ""})
    return pairs, summaries, len(comparator)


def collect_feature(directory):
    path = Path(directory).resolve()
    base = {field: "" for field in FEATURE_FIELDS}
    base.update(feature_dir=str(path), status="feature_failed")
    try:
        status, protocol = _read(path / "status.json"), _read(path / "protocol.json")
        if not isinstance(status, dict) or not isinstance(protocol, dict):
            raise ValueError("Feature status and protocol must be JSON objects")
        base.update(dataset=protocol.get("dataset", ""), metric=protocol.get("metric", ""),
                    final_source=(protocol.get("source") or {}).get("final_result_source", ""),
                    protocol_sha256=_sha(path / "protocol.json"))
        if status.get("status") != "success" or protocol.get("schema_version") != "feature_experiments_v1":
            raise ValueError("Feature experiment is not a successful supported result")
        from morace.feature_experiments import ARMS
        seeds, arms = protocol["rf_seeds"], protocol["arms"]
        if not seeds or len(seeds) != len(set(seeds)) or tuple(arms) != ARMS:
            raise ValueError("Feature protocol must contain unique RF seeds and the five supported paired arms")
        with (path / "per_seed.csv").open(encoding="utf-8-sig", newline="") as handle:
            input_rows = list(csv.DictReader(handle))
        expected = {(str(seed), arm, split) for seed in protocol["rf_seeds"] for arm in protocol["arms"] for split in ("valid", "test")}
        actual = [(row["rf_seed"], row["arm"], row["split"]) for row in input_rows]
        if len(actual) != len(set(actual)) or set(actual) != expected or not expected:
            raise ValueError("Feature score rows do not match the complete paired protocol")
        rows = []
        for row in input_rows:
            rows.append(dict(base, arm=row["arm"], rf_seed=row["rf_seed"], split=row["split"],
                             score=_score(row["score"]), directional_delta_vs_base=_score(row["directional_delta_vs_base"]), status="success"))
        return rows
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [dict(base, reason=f"{type(exc).__name__}: {exc}")]


def _write_csv(path, rows, fields=None):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_experiments(*, run_dirs=(), feature_dirs=(), output_dir, project_root=None, plan_path=None, comparator_path=None):
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    root = Path(project_root or ROOT).resolve()
    requests, requested_paths = [], set()
    plan = _read(plan_path) if plan_path else None
    if plan is not None:
        if not isinstance(plan, dict) or not isinstance(plan.get("jobs"), list) or not plan["jobs"]:
            raise ValueError("Matrix plan must contain a nonempty jobs array")
        for job in plan["jobs"]:
            if not all(key in job for key in ("run_tag", "route", "dataset", "seed")):
                raise ValueError("Every matrix job requires run_tag/route/dataset/seed")
            path = (root / "artifacts" / job["run_tag"]).resolve()
            if path in requested_paths:
                raise ValueError(f"Duplicate requested run: {path}")
            requested_paths.add(path)
            requests.append((path, job))
    for value in run_dirs:
        path = Path(value).resolve()
        if path.name == "summary.json":
            path = path.parent
        if path in requested_paths:
            raise ValueError(f"Duplicate requested run: {path}")
        requested_paths.add(path)
        requests.append((path, None))
    features = [Path(path).resolve() for path in feature_dirs]
    if len(features) != len(set(features)):
        raise ValueError("Duplicate feature directories")
    if not requests and not features:
        raise ValueError("At least one run, matrix plan, or feature result is required")
    rows, audits = [], []
    for directory, planned in requests:
        collected, audit = collect_run(directory, project_root=project_root, planned=planned)
        rows.extend(collected)
        audits.append(dict(audit, requested_run_dir=str(directory)))
    feature_rows = [row for path in features for row in collect_feature(path)]
    comparator_count = 0
    pairs, pair_summary = [], []
    if comparator_path:
        pairs, pair_summary, comparator_count = pair_comparator(rows, comparator_path)
    totals = {"schema_version": SCHEMA, "n_requested_run_dirs": len(requests), "n_audit_passed_runs": sum(audit["status"] == "pass" for audit in audits),
              "n_main_requested_rows": len(rows), "n_main_scored_rows": sum(row["scored"] for row in rows),
              "main_status_counts": dict(Counter(row["status"] for row in rows)), "n_requested_feature_dirs": len(features),
              "n_failed_feature_dirs": sum(row["status"] != "success" for row in feature_rows),
              "n_comparator_requested_rows": comparator_count, "n_paired_rows": sum(row["pair_status"] == "paired" for row in pairs),
              "std_ddof": 0, "score_scope": "main final rule-only test mean over recorded RF seeds",
              "incomplete_inputs": any(row["status"] != "success" for row in rows + feature_rows),
              "plan": {"path": str(Path(plan_path).resolve()), "sha256": _sha(plan_path)} if plan_path else None,
              "comparator": {"path": str(Path(comparator_path).resolve()), "sha256": _sha(comparator_path)} if comparator_path else None,
              "implementation_sha256": _sha(__file__)}
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "main_runs.csv", rows, MAIN_FIELDS)
    if rows:
        _write_csv(output / "main_summary.csv", summarize_main(rows))
    _write_csv(output / "feature_runs.csv", feature_rows, FEATURE_FIELDS)
    if pairs:
        _write_csv(output / "matched_pairs.csv", pairs)
        _write_csv(output / "matched_summary.csv", pair_summary)
    for name, value in (("totals.json", totals), ("run_audits.json", audits)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit and summarize completed molecular prediction experiments without retraining.")
    parser.add_argument("--run-dir", action="append", type=Path, default=[], help="Repeat for each requested run (including failed/missing runs)")
    parser.add_argument("--feature-dir", action="append", type=Path, default=[], help="Separate frozen-feature result table")
    parser.add_argument("--plan", type=Path, help="Public matrix jobs JSON; preserves missing jobs in the denominator")
    parser.add_argument("--project-root", type=Path, help="Repository containing artifacts and recorded relative paths")
    parser.add_argument("--comparator", type=Path, help="CSV from a real completed matched baseline; never a paper-table substitution")
    parser.add_argument("--output-dir", required=True, type=Path, help="New report directory")
    args = parser.parse_args(argv)
    totals = report_experiments(run_dirs=args.run_dir, feature_dirs=args.feature_dir, output_dir=args.output_dir,
                                project_root=args.project_root, plan_path=args.plan, comparator_path=args.comparator)
    print(f"Scored main rows: {totals['n_main_scored_rows']}/{totals['n_main_requested_rows']}; report: {args.output_dir}")
    return 1 if totals["incomplete_inputs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

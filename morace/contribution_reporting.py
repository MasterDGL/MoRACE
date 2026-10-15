"""Markdown report rendering for rule contribution diagnostics."""

from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Sequence


def write_dry_run_summary(
    path: Path,
    rows: List[Dict[str, object]],
    scope: str,
    matrix_sample_size: int,
) -> None:
    status_counts = Counter(str(row.get("dry_run_status", "")) for row in rows)
    dataset_counts = Counter(str(row.get("dataset", "")) for row in rows)
    all_passed = bool(rows) and all(str(row.get("dry_run_status")) == "passed" for row in rows)

    lines = [
        f"# Rule Contribution Dry Run Summary: {path.parent.name}",
        "",
        "This is a script and dry-run validation step only. It did not train RandomForest, did not execute exact leave-one-rule-out, did not execute family ablation, did not run SHAP, and did not call an LLM.",
        "",
        "## Scope",
        "",
        f"- Scope executed: `{scope}`",
        f"- Matrix dry-run sample size per split: `{matrix_sample_size}`",
        "- Default scope is all inventory candidates; route, dataset and run filters may narrow it.",
        "- `--scope p0` and `--scope p1` are legacy inventory compatibility filters.",
        "",
        "## Status",
        "",
        "| item | value |",
        "|---|---:|",
        f"| candidate rows checked | {len(rows)} |",
        f"| passed | {status_counts.get('passed', 0)} |",
        f"| failed | {status_counts.get('failed', 0)} |",
        f"| all checked rows passed | {str(all_passed).lower()} |",
        "",
        "## Dataset Coverage",
        "",
        "| dataset | rows |",
        "|---|---:|",
    ]
    for dataset, count in sorted(dataset_counts.items()):
        lines.append(f"| {dataset} | {count} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A passed row means the script could find the rule JSON, outcome file, fixed split CSVs, compile the selected rules, build a small rule feature matrix from the fixed splits, and prepare LORO/family/feature-importance variants without training any model.",
            "",
            "This is not contribution evidence. It only says the candidate is technically ready for a later CPU-only exact LORO run.",
            "",
            "## Boundary",
            "",
            "- No source experiment result was modified.",
            "- No artifacts were modified.",
            "- No existing rule JSON was modified.",
            "- No GPU was used.",
            "- No LLM was called.",
            "- No RandomForest training was run.",
            "- No exact LORO, family ablation, feature importance, or SHAP was executed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_protocol(path: Path) -> None:
    text = f"""# Rule Contribution Protocol: {path.parent.name}

## Inputs

`inventory.csv` records the selected completed-run artifacts. The mode-specific
`*_config.json` records the analysis directory, candidate IDs and split manifest.

## Split Protocol

All contribution experiments use the fixed pre-materialized train/valid/test CSV
splits recorded in the selected split manifest. No split assignment is regenerated
or shuffled.

## Modes

- `--check-inputs`: checks candidate rule/outcome files and manifest split CSVs.
- `--dry-run`: compiles existing rules, builds a small rule feature matrix and checks diagnostic readiness, without training a model.
- `--run-loro`: evaluates exact leave-one-rule-out against the reproduced base.
- `--run-family-ablation`: evaluates deletion of each available feature family.
- `--run-feature-importance`: measures RandomForest feature importance and optionally aligns with LORO/family results in the same output directory.

The default scope is all candidates. Explicit `p0`/`p1` scopes retain legacy inventory compatibility.

## Delta Convention

- ROC-AUC: `valid_delta = ablated_valid_score - base_valid_score`; negative means deletion worsened validation performance and is a possible positive contribution signal.
- RMSE: `valid_delta = ablated_valid_score - base_valid_score`; positive means deletion worsened validation performance and is a possible positive contribution signal.

Valid delta is the primary contribution signal. Test delta is supplemental only and must not be used to select rules.

## Interpretation

Dry-run readiness is not contribution evidence. Formal diagnostics reuse selected
rules, recorded RF seeds and fixed data splits; they do not establish causal
chemical mechanisms. No LLM is called and no source artifact is modified.
"""
    path.write_text(text, encoding="utf-8")


def _append_base_reproduction(
    lines: List[str],
    base_rows: List[Dict[str, object]],
    status_counts: Counter[str],
    warning_ids: Sequence[object] = (),
) -> None:
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")
    if warning_ids:
        lines.extend(["", "Base reproduction warnings are cautionary for interpretation:"])
        for pilot_id in warning_ids:
            lines.append(f"- `{pilot_id}`")
    lines.extend(
        [
            "",
            "| pilot_id | metric | outcome valid | reproduced valid | outcome test | reproduced test | status |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in base_rows:
        lines.append(
            "| {pilot_id} | {metric} | {outcome_valid_mean} | {reproduced_valid_mean} | {outcome_test_mean} | {reproduced_test_mean} | {base_reproduction_status} |".format(
                **row
            )
        )


def _append_issues(
    lines: List[str],
    issue_rows: List[Dict[str, object]],
    md: Callable[[object], str],
) -> None:
    if issue_rows:
        lines.extend(
            [
                "",
                "## Issues",
                "",
                "| pilot_id | stage | severity | issue |",
                "|---|---|---|---|",
            ]
        )
        for row in issue_rows:
            lines.append(
                "| {pilot_id} | {stage} | {severity} | {issue} |".format(
                    pilot_id=md(row.get("pilot_id", "")),
                    stage=md(row.get("stage", "")),
                    severity=md(row.get("severity", "")),
                    issue=md(row.get("issue", "")),
                )
            )


def write_feature_importance_summary(
    path: Path,
    feature_rows: List[Dict[str, object]],
    base_rows: List[Dict[str, object]],
    issue_rows: List[Dict[str, object]],
    candidate_count: int,
) -> None:
    alignment_counts = Counter(str(row.get("alignment_label", "")) for row in feature_rows)
    status_counts = Counter(str(row.get("base_reproduction_status", "")) for row in base_rows)
    route_family_counts = Counter(
        (
            str(row.get("route_or_model", "")),
            str(row.get("dataset", "")),
            str(row.get("feature_family", "")),
        )
        for row in feature_rows
    )
    warning_ids = [
        row.get("pilot_id", "")
        for row in base_rows
        if row.get("base_reproduction_status") == "base_reproduction_warning"
    ]

    def md(value: object) -> str:
        return str(value).replace("|", "\\|")

    def importance_value(row: Dict[str, object]) -> float:
        try:
            return float(row.get("importance_mean", 0.0))
        except (TypeError, ValueError):
            return -1.0

    aligned = [
        row
        for row in feature_rows
        if row.get("alignment_label") == "aligned_strong_diagnostic"
    ]
    mixed = [
        row for row in feature_rows if row.get("alignment_label") == "mixed_diagnostic"
    ]
    low = [row for row in feature_rows if row.get("alignment_label") == "low_priority"]
    top_unfiltered = [
        row for row in feature_rows if row.get("filtered_by_sanitizer") == "no"
    ]

    lines = [
        f"# Rule Contribution Feature Importance Summary: {path.parent.name}",
        "",
        "This is CPU RandomForest feature importance across the selected candidate runs.",
        "",
        "## Boundary",
        "",
        f"- Candidate runs executed: {candidate_count}",
        f"- Feature importance rows generated: {len(feature_rows)}",
        "- No LLM was called.",
        "- No GPU was used.",
        "- No rules were regenerated.",
        "- No split was regenerated; fixed pre-materialized train/valid/test CSV splits were reused.",
        "- No new LORO, family ablation, or SHAP was executed in this step.",
        "- Feature importance and test score are not used to select rules.",
        "",
        "## Base Reproduction",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    _append_base_reproduction(lines, base_rows, status_counts, warning_ids)

    lines.extend(
        [
            "",
            "## Alignment Label Counts",
            "",
            "| alignment_label | count |",
            "|---|---:|",
        ]
    )
    for label, count in sorted(alignment_counts.items()):
        lines.append(f"| {label} | {count} |")

    lines.extend(
        [
            "",
            "## Top Rules By Route / Dataset / Feature Family",
            "",
            "| route | dataset | feature_family | top_rule | importance_mean | rank_mean | alignment |",
            "|---|---|---|---|---:|---:|---|",
        ]
    )
    for route, dataset, family in sorted(route_family_counts):
        candidates = [
            row
            for row in top_unfiltered
            if row.get("route_or_model") == route
            and row.get("dataset") == dataset
            and row.get("feature_family") == family
        ]
        if not candidates:
            continue
        row = sorted(candidates, key=importance_value, reverse=True)[0]
        lines.append(
            "| {route} | {dataset} | {family} | {rule} | {importance} | {rank} | {alignment} |".format(
                route=md(route),
                dataset=dataset,
                family=family,
                rule=md(row.get("canonical_rule", "")),
                importance=md(row.get("importance_mean", "")),
                rank=md(row.get("rank_mean", "")),
                alignment=md(row.get("alignment_label", "")),
            )
        )

    lines.extend(
        [
            "",
            "## Aligned Strong Diagnostic Rules",
            "",
            "| pilot_id | canonical_rule | family | importance_mean | rank_mean | loro_signal | family_signal |",
            "|---|---|---|---:|---:|---|---|",
        ]
    )
    for row in sorted(aligned, key=importance_value, reverse=True)[:20]:
        lines.append(
            "| {pilot_id} | {rule} | {family} | {importance} | {rank} | {loro} | {fam_signal} |".format(
                pilot_id=md(row.get("pilot_id", "")),
                rule=md(row.get("canonical_rule", "")),
                family=md(row.get("feature_family", "")),
                importance=md(row.get("importance_mean", "")),
                rank=md(row.get("rank_mean", "")),
                loro=md(row.get("loro_contribution_signal", "")),
                fam_signal=md(row.get("family_contribution_signal", "")),
            )
        )

    lines.extend(
        [
            "",
            "## Mixed Diagnostic Examples",
            "",
            "| pilot_id | canonical_rule | family | importance_mean | rank_mean | loro_signal | family_signal |",
            "|---|---|---|---:|---:|---|---|",
        ]
    )
    for row in sorted(mixed, key=importance_value, reverse=True)[:20]:
        lines.append(
            "| {pilot_id} | {rule} | {family} | {importance} | {rank} | {loro} | {fam_signal} |".format(
                pilot_id=md(row.get("pilot_id", "")),
                rule=md(row.get("canonical_rule", "")),
                family=md(row.get("feature_family", "")),
                importance=md(row.get("importance_mean", "")),
                rank=md(row.get("rank_mean", "")),
                loro=md(row.get("loro_contribution_signal", "")),
                fam_signal=md(row.get("family_contribution_signal", "")),
            )
        )

    lines.extend(
        [
            "",
            "## Low Priority Examples",
            "",
            "| pilot_id | canonical_rule | family | importance_mean | rank_mean | loro_signal |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in sorted(low, key=importance_value)[:20]:
        lines.append(
            "| {pilot_id} | {rule} | {family} | {importance} | {rank} | {loro} |".format(
                pilot_id=md(row.get("pilot_id", "")),
                rule=md(row.get("canonical_rule", "")),
                family=md(row.get("feature_family", "")),
                importance=md(row.get("importance_mean", "")),
                rank=md(row.get("rank_mean", "")),
                loro=md(row.get("loro_contribution_signal", "")),
            )
        )

    lines.extend(
        [
            "",
            "## Consistency With LORO And Family Ablation",
            "",
            "LORO is rule-level deletion evidence. Family ablation is family-level deletion evidence. Feature importance is attribution evidence from the base RandomForest model. When all three agree, the result is stronger diagnostic evidence. When they disagree, that can indicate redundancy, correlated features, RF attribution bias, or instability.",
            "",
            "This consistency should not be interpreted as chemical mechanism causality.",
            "",
            "## Interpretation",
            "",
            "Feature importance is a predictive diagnostic. It is affected by correlated features, feature redundancy, RandomForest bias, and the sanitizer that removes constant or invalid features. It should not be used to select rules or to make test-set based decisions.",
        ]
    )

    _append_issues(lines, issue_rows, md)

    lines.extend(
        [
            "",
            "## Claims To Avoid",
            "",
            "- Do not claim feature importance proves chemical mechanism causality.",
            "- Do not use feature importance or test score to select rules.",
            "- Do not claim all selected/compiled rules are important.",
            "- Do not generalize these diagnostics beyond the selected rule population.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_family_ablation_summary(
    path: Path,
    family_rows: List[Dict[str, object]],
    base_rows: List[Dict[str, object]],
    issue_rows: List[Dict[str, object]],
    candidate_count: int,
    project_root: Path,
    *,
    read_csv_fn,
) -> None:
    signal_counts = Counter(
        str(row.get("family_contribution_signal", "")) for row in family_rows
    )
    status_counts = Counter(
        str(row.get("base_reproduction_status", "")) for row in base_rows
    )
    route_signal = Counter(
        (
            str(row.get("route_or_model", "")),
            str(row.get("family_contribution_signal", "")),
        )
        for row in family_rows
    )
    dataset_signal = Counter(
        (
            str(row.get("dataset", "")),
            str(row.get("family_contribution_signal", "")),
        )
        for row in family_rows
    )
    family_signal = Counter(
        (
            str(row.get("removed_family", "")),
            str(row.get("family_contribution_signal", "")),
        )
        for row in family_rows
    )

    def md(value: object) -> str:
        return str(value).replace("|", "\\|")

    def score_abs(row: Dict[str, object]) -> float:
        try:
            return abs(float(row.get("valid_delta", 0.0)))
        except (TypeError, ValueError):
            return 0.0

    positive = [
        row
        for row in family_rows
        if row.get("family_contribution_signal") == "positive_signal"
    ]
    harmful = [
        row
        for row in family_rows
        if row.get("family_contribution_signal") == "negative_or_harmful_signal"
    ]
    near_zero = [
        row
        for row in family_rows
        if row.get("family_contribution_signal") == "near_zero"
    ]
    warning_ids = [
        row.get("pilot_id", "")
        for row in base_rows
        if row.get("base_reproduction_status") == "base_reproduction_warning"
    ]

    lines = [
        f"# Rule Contribution Family Ablation Summary: {path.parent.name}",
        "",
        "This is family ablation across the selected candidate runs.",
        "",
        "## Boundary",
        "",
        f"- Candidate runs executed: {candidate_count}",
        f"- Family ablation rows generated: {len(family_rows)}",
        "- No LLM was called.",
        "- No GPU was used.",
        "- No rules were regenerated.",
        "- No split was regenerated; fixed pre-materialized train/valid/test CSV splits were reused.",
        "- No new exact LORO, feature importance, or SHAP was executed in this step.",
        "- Valid delta is the primary family contribution signal; test delta is supplemental only.",
        "",
        "## Base Reproduction",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    _append_base_reproduction(lines, base_rows, status_counts, warning_ids)

    lines.extend(
        [
            "",
            "## Family Signal Counts",
            "",
            "| family_contribution_signal | count |",
            "|---|---:|",
        ]
    )
    for signal, count in sorted(signal_counts.items()):
        lines.append(f"| {signal} | {count} |")

    lines.extend(
        [
            "",
            "## Signal Distribution By Route",
            "",
            "| route | positive_signal | negative_or_harmful_signal | near_zero | unknown | total |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for route in sorted({key[0] for key in route_signal}):
        total = sum(
            route_signal.get((route, signal), 0)
            for signal in [
                "positive_signal",
                "negative_or_harmful_signal",
                "near_zero",
                "unknown",
            ]
        )
        lines.append(
            f"| {md(route)} | {route_signal.get((route, 'positive_signal'), 0)} | {route_signal.get((route, 'negative_or_harmful_signal'), 0)} | {route_signal.get((route, 'near_zero'), 0)} | {route_signal.get((route, 'unknown'), 0)} | {total} |"
        )

    lines.extend(
        [
            "",
            "## Signal Distribution By Dataset",
            "",
            "| dataset | positive_signal | negative_or_harmful_signal | near_zero | unknown | total |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in sorted({key[0] for key in dataset_signal}):
        total = sum(
            dataset_signal.get((dataset, signal), 0)
            for signal in [
                "positive_signal",
                "negative_or_harmful_signal",
                "near_zero",
                "unknown",
            ]
        )
        lines.append(
            f"| {dataset} | {dataset_signal.get((dataset, 'positive_signal'), 0)} | {dataset_signal.get((dataset, 'negative_or_harmful_signal'), 0)} | {dataset_signal.get((dataset, 'near_zero'), 0)} | {dataset_signal.get((dataset, 'unknown'), 0)} | {total} |"
        )

    lines.extend(
        [
            "",
            "## Signal Distribution By Removed Family",
            "",
            "| removed_family | positive_signal | negative_or_harmful_signal | near_zero | unknown | total |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for family in sorted({key[0] for key in family_signal}):
        total = sum(
            family_signal.get((family, signal), 0)
            for signal in [
                "positive_signal",
                "negative_or_harmful_signal",
                "near_zero",
                "unknown",
            ]
        )
        lines.append(
            f"| {family} | {family_signal.get((family, 'positive_signal'), 0)} | {family_signal.get((family, 'negative_or_harmful_signal'), 0)} | {family_signal.get((family, 'near_zero'), 0)} | {family_signal.get((family, 'unknown'), 0)} | {total} |"
        )

    lines.extend(
        [
            "",
            "## Families Whose Removal Worsened Valid Performance",
            "",
            "| pilot_id | removed_family | removed_rules | valid_delta | test_delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in sorted(positive, key=score_abs, reverse=True)[:12]:
        lines.append(
            "| {pilot_id} | {removed_family} | {removed_rule_count} | {valid_delta} | {test_delta} |".format(
                pilot_id=md(row.get("pilot_id", "")),
                removed_family=md(row.get("removed_family", "")),
                removed_rule_count=md(row.get("removed_rule_count", "")),
                valid_delta=md(row.get("valid_delta", "")),
                test_delta=md(row.get("test_delta", "")),
            )
        )

    lines.extend(
        [
            "",
            "## Families Whose Removal Improved Valid Performance",
            "",
            "| pilot_id | removed_family | removed_rules | valid_delta | test_delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in sorted(harmful, key=score_abs, reverse=True)[:12]:
        lines.append(
            "| {pilot_id} | {removed_family} | {removed_rule_count} | {valid_delta} | {test_delta} |".format(
                pilot_id=md(row.get("pilot_id", "")),
                removed_family=md(row.get("removed_family", "")),
                removed_rule_count=md(row.get("removed_rule_count", "")),
                valid_delta=md(row.get("valid_delta", "")),
                test_delta=md(row.get("test_delta", "")),
            )
        )

    lines.extend(
        [
            "",
            "## Near-Zero Family Examples",
            "",
            "| pilot_id | removed_family | valid_delta |",
            "|---|---|---:|",
        ]
    )
    for row in sorted(near_zero, key=score_abs)[:12]:
        lines.append(
            "| {pilot_id} | {removed_family} | {valid_delta} |".format(
                pilot_id=md(row.get("pilot_id", "")),
                removed_family=md(row.get("removed_family", "")),
                valid_delta=md(row.get("valid_delta", "")),
            )
        )

    lines.extend(
        [
            "",
            "## Consistency With Exact LORO",
            "",
            "Family ablation is family-level evidence; exact LORO is rule-level evidence. If a family has positive_signal and LORO also shows multiple positive_signal rules from that same family, this is stronger diagnostic evidence. It is still not proof of chemical mechanism causality.",
            "",
            "| route | dataset | family | family_positive_rows | loro_positive_rows | interpretation |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    loro_path = path.parent / "loro.csv"
    loro_positive_counts: Counter = Counter()
    if loro_path.exists():
        for row in read_csv_fn(loro_path):
            if row.get("contribution_signal") == "positive_signal":
                loro_positive_counts[
                    (
                        row.get("route_or_model", ""),
                        row.get("dataset", ""),
                        row.get("feature_family", ""),
                    )
                ] += 1
    family_positive_counts = Counter(
        (
            str(row.get("route_or_model", "")),
            str(row.get("dataset", "")),
            str(row.get("removed_family", "")),
        )
        for row in positive
    )
    consistency_keys = sorted(set(family_positive_counts) | set(loro_positive_counts))
    if consistency_keys:
        for route, dataset, family in consistency_keys:
            fp = family_positive_counts.get((route, dataset, family), 0)
            lp = loro_positive_counts.get((route, dataset, family), 0)
            if fp and lp:
                interpretation = "stronger_diagnostic_alignment"
            elif fp:
                interpretation = "family_positive_without_loro_positive_count"
            else:
                interpretation = "loro_positive_without_family_positive_count"
            lines.append(
                f"| {md(route)} | {dataset} | {family} | {fp} | {lp} | {interpretation} |"
            )
    else:
        lines.append(
            "| none | none | none | 0 | 0 | no_loro_file_or_no_positive_rows |"
        )

    _append_issues(lines, issue_rows, md)

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A positive family signal means removing that feature family worsened validation performance under the metric direction. For ROC-AUC this is a negative valid delta; for RMSE this is a positive valid delta. Test delta is supplemental and must not be used to select a family.",
            "",
            "Families whose removal improved validation performance should be treated as potential redundancy, instability, or harmful/noisy signal candidates for later review. Near-zero family rows should not be over-interpreted.",
            "",
            "## Claims To Avoid",
            "",
            "- Do not claim all feature families contribute in all tasks.",
            "- Do not claim causal chemical mechanism from family ablation deltas.",
            "- Do not use test delta to select families.",
            "- Do not generalize these diagnostics beyond the selected rule population.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_loro_smoke_summary(
    path: Path,
    loro_rows: List[Dict[str, object]],
    base_rows: List[Dict[str, object]],
    issue_rows: List[Dict[str, object]],
    run_label: str = "smoke",
    candidate_count: int | None = None,
    project_root: Path | None = None,
    *,
    read_csv_fn,
) -> None:
    signal_counts = Counter(
        str(row.get("contribution_signal", "")) for row in loro_rows
    )
    status_counts = Counter(
        str(row.get("base_reproduction_status", "")) for row in base_rows
    )
    by_run = Counter(str(row.get("pilot_id", "")) for row in loro_rows)

    def score_abs(row: Dict[str, object]) -> float:
        try:
            return abs(float(row.get("valid_delta", 0.0)))
        except (TypeError, ValueError):
            return 0.0

    def md(value: object) -> str:
        return str(value).replace("|", "\\|")

    positive = [
        row for row in loro_rows if row.get("contribution_signal") == "positive_signal"
    ]
    harmful = [
        row
        for row in loro_rows
        if row.get("contribution_signal") == "negative_or_harmful_signal"
    ]
    near_zero = [
        row for row in loro_rows if row.get("contribution_signal") == "near_zero"
    ]

    title = f"Rule Contribution LORO Summary: {run_label}"
    intro = "This is an exact leave-one-rule-out diagnostic across the selected candidate runs."
    lines = [
        f"# {title}",
        "",
        intro,
        "",
        "## Boundary",
        "",
        f"- Candidate runs executed: {candidate_count if candidate_count is not None else len(base_rows)}",
        "- LORO rows generated: {}".format(len(loro_rows)),
        "- No LLM was called.",
        "- No GPU was used.",
        "- No rules were regenerated.",
        "- No split was regenerated; fixed pre-materialized train/valid/test CSV splits were reused.",
        "- No family ablation, feature importance, or SHAP was executed.",
        "- Valid delta is the primary contribution signal; test delta is supplemental only.",
        "",
        "## Base Reproduction",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    _append_base_reproduction(lines, base_rows, status_counts)

    lines.extend(
        [
            "",
            "## LORO Signal Counts",
            "",
            "| contribution_signal | count |",
            "|---|---:|",
        ]
    )
    for signal, count in sorted(signal_counts.items()):
        lines.append(f"| {signal} | {count} |")
    lines.extend(
        [
            "",
            "## Rows By Run",
            "",
            "| pilot_id | LORO rows |",
            "|---|---:|",
        ]
    )
    for pilot_id, count in sorted(by_run.items()):
        lines.append(f"| {pilot_id} | {count} |")

    route_counts = Counter(str(row.get("route_or_model", "")) for row in loro_rows)
    dataset_signal = Counter(
        (str(row.get("dataset", "")), str(row.get("contribution_signal", "")))
        for row in loro_rows
    )
    family_signal = Counter(
        (
            str(row.get("feature_family", "")),
            str(row.get("contribution_signal", "")),
        )
        for row in loro_rows
    )
    route_signal = Counter(
        (
            str(row.get("route_or_model", "")),
            str(row.get("contribution_signal", "")),
        )
        for row in loro_rows
    )

    lines.extend(
        [
            "",
            "## Signal Distribution By Route",
            "",
            "| route | positive_signal | negative_or_harmful_signal | near_zero | unknown | total |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for route in sorted(route_counts):
        lines.append(
            f"| {md(route)} | {route_signal.get((route, 'positive_signal'), 0)} | {route_signal.get((route, 'negative_or_harmful_signal'), 0)} | {route_signal.get((route, 'near_zero'), 0)} | {route_signal.get((route, 'unknown'), 0)} | {route_counts[route]} |"
        )
    lines.extend(
        [
            "",
            "## Signal Distribution By Dataset",
            "",
            "| dataset | positive_signal | negative_or_harmful_signal | near_zero | unknown | total |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in sorted({key[0] for key in dataset_signal}):
        total = sum(
            dataset_signal.get((dataset, signal), 0)
            for signal in [
                "positive_signal",
                "negative_or_harmful_signal",
                "near_zero",
                "unknown",
            ]
        )
        lines.append(
            f"| {dataset} | {dataset_signal.get((dataset, 'positive_signal'), 0)} | {dataset_signal.get((dataset, 'negative_or_harmful_signal'), 0)} | {dataset_signal.get((dataset, 'near_zero'), 0)} | {dataset_signal.get((dataset, 'unknown'), 0)} | {total} |"
        )
    lines.extend(
        [
            "",
            "## Signal Distribution By Feature Family",
            "",
            "| feature_family | positive_signal | negative_or_harmful_signal | near_zero | unknown | total |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for family in sorted({key[0] for key in family_signal}):
        total = sum(
            family_signal.get((family, signal), 0)
            for signal in [
                "positive_signal",
                "negative_or_harmful_signal",
                "near_zero",
                "unknown",
            ]
        )
        lines.append(
            f"| {family} | {family_signal.get((family, 'positive_signal'), 0)} | {family_signal.get((family, 'negative_or_harmful_signal'), 0)} | {family_signal.get((family, 'near_zero'), 0)} | {family_signal.get((family, 'unknown'), 0)} | {total} |"
        )


    lines.extend(
        [
            "",
            "## Strongest Positive-Signal Examples",
            "",
            "| pilot_id | canonical_rule | family | valid_delta | test_delta |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in sorted(positive, key=score_abs, reverse=True)[:10]:
        lines.append(
            "| {pilot_id} | {canonical_rule} | {feature_family} | {valid_delta} | {test_delta} |".format(
                pilot_id=md(row.get("pilot_id", "")),
                canonical_rule=md(row.get("canonical_rule", "")),
                feature_family=md(row.get("feature_family", "")),
                valid_delta=md(row.get("valid_delta", "")),
                test_delta=md(row.get("test_delta", "")),
            )
        )

    lines.extend(
        [
            "",
            "## Deletion Improved Valid Score",
            "",
            "| pilot_id | canonical_rule | family | valid_delta | test_delta |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in sorted(harmful, key=score_abs, reverse=True)[:10]:
        lines.append(
            "| {pilot_id} | {canonical_rule} | {feature_family} | {valid_delta} | {test_delta} |".format(
                pilot_id=md(row.get("pilot_id", "")),
                canonical_rule=md(row.get("canonical_rule", "")),
                feature_family=md(row.get("feature_family", "")),
                valid_delta=md(row.get("valid_delta", "")),
                test_delta=md(row.get("test_delta", "")),
            )
        )

    lines.extend(
        [
            "",
            "## Near-Zero Examples",
            "",
            "| pilot_id | canonical_rule | family | valid_delta |",
            "|---|---|---|---:|",
        ]
    )
    for row in sorted(near_zero, key=score_abs)[:10]:
        lines.append(
            "| {pilot_id} | {canonical_rule} | {feature_family} | {valid_delta} |".format(
                pilot_id=md(row.get("pilot_id", "")),
                canonical_rule=md(row.get("canonical_rule", "")),
                feature_family=md(row.get("feature_family", "")),
                valid_delta=md(row.get("valid_delta", "")),
            )
        )

    _append_issues(lines, issue_rows, md)

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A positive signal means deletion worsened validation performance under the metric direction. For ROC-AUC this is a negative valid delta; for RMSE this is a positive valid delta. These are predictive diagnostics, not causal mechanism proof.",
            "",
            "Rules whose deletion improved validation performance should be treated as potential redundancy, instability, or harmful/noisy signal candidates for later review. Near-zero rows are low-priority for exact contribution claims.",
            "",
            "## Next Steps",
            "",
            "- Review exact LORO deltas and base reproduction gaps.",
            "- Compare family ablation with rule-level LORO evidence.",
            "- Interpret feature importance together with deletion evidence.",
            "- Keep SHAP optional and diagnostic; it should not be the first evidence path.",
            "",
            "## Claims To Avoid",
            "",
            "- Do not claim all selected/compiled rules have contribution.",
            "- Do not claim causal chemical mechanism from LORO deltas.",
            "- Do not use test delta to select rules.",
            "- Do not generalize these diagnostics beyond the selected rule population.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

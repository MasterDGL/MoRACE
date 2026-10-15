"""Quality-gate decisions and failure diagnostics for evaluation."""

from typing import Callable, Dict, Tuple


def literature_underdiverse(projection_metrics: Dict) -> bool:
    if not projection_metrics:
        return False
    if "underdiverse" in projection_metrics:
        return bool(projection_metrics.get("underdiverse"))
    canonical_rule_count = int(projection_metrics.get("canonical_rule_count", 0) or 0)
    compile_safe_count = int(projection_metrics.get("compile_safe_count", canonical_rule_count) or 0)
    root_feature_count = int(projection_metrics.get("root_feature_count", canonical_rule_count) or 0)
    family_diversity = int(
        projection_metrics.get(
            "family_diversity",
            projection_metrics.get("feature_family_diversity", 0),
        )
        or 0
    )
    dominant_root_share = float(projection_metrics.get("dominant_root_share", 0.0) or 0.0)
    descriptor_only = bool(projection_metrics.get("descriptor_only", False))
    return bool(
        canonical_rule_count < 2
        or compile_safe_count < 2
        or root_feature_count < 2
        or family_diversity < 2
        or (descriptor_only and dominant_root_share >= 0.67)
    )


def synthesize_soft_pass_eligible(projection_metrics: Dict, *, max_raw_rules: int) -> bool:
    if not projection_metrics:
        return False
    if float(projection_metrics.get("schema_valid_rate", 0.0)) < 1.0:
        return False
    if float(projection_metrics.get("repair_success_rate", 0.0)) < 1.0:
        return False
    if int(projection_metrics.get("raw_rule_count", 0)) <= 0:
        return False
    if int(projection_metrics.get("raw_rule_count", 0)) > max_raw_rules:
        return False
    if float(projection_metrics.get("mollogp_collapse_rate", 1.0)) > 0.0:
        return False
    family_diversity = int(
        projection_metrics.get(
            "family_diversity",
            projection_metrics.get("feature_family_diversity", 0),
        )
        or 0
    )
    compile_safe_count = int(
        projection_metrics.get(
            "compile_safe_count",
            projection_metrics.get("canonical_rule_count", 0),
        )
        or 0
    )
    root_feature_count = int(
        projection_metrics.get(
            "root_feature_count",
            projection_metrics.get("canonical_rule_count", 0),
        )
        or 0
    )
    if compile_safe_count < 2 or root_feature_count < 2 or family_diversity < 2:
        return False
    return True


def merge_count_maps(*maps: Dict) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for mapping in maps:
        for key, value in (mapping or {}).items():
            merged[str(key)] = int(merged.get(str(key), 0) or 0) + int(value or 0)
    return merged


def projection_failure_diagnostics(metrics: Dict) -> Dict:
    diagnostics = (metrics or {}).get("strict_failure_diagnostics", {}) or {}
    return {
        "total_failures": int(diagnostics.get("total_failures", 0) or 0),
        "stage_counts": dict(diagnostics.get("stage_counts", {}) or {}),
        "reason_counts": dict(diagnostics.get("reason_counts", {}) or {}),
        "label_counts": dict(diagnostics.get("label_counts", {}) or {}),
        "by_stage": dict(diagnostics.get("by_stage", {}) or {}),
    }


def infer_limiting_branch(branch_metrics: Dict, gate_reason: str, *, knowledge_type: str) -> str:
    reason = str(gate_reason or "")
    if reason == "literature_branch_empty" or "strict_valid_but_underdiverse" in reason:
        return "synthesize"
    if "heuristic_domination" in reason:
        return "inference"
    if "low_projector_valid_rate" in reason:
        synth_projection = branch_metrics.get("synthesize_projection", {}) or {}
        inference_projection = branch_metrics.get("inference_projection", {}) or {}
        synth_valid = float(
            synth_projection.get(
                "strict_projector_valid_rate",
                synth_projection.get("projector_valid_rate", 0.0),
            )
        )
        inference_valid = float(
            inference_projection.get(
                "strict_projector_valid_rate",
                inference_projection.get("projector_valid_rate", 0.0),
            )
        )
        if knowledge_type == "synthesize":
            return "synthesize"
        if knowledge_type == "inference":
            return "inference"
        return "synthesize" if synth_valid <= inference_valid else "inference"
    return "combined"


def build_branch_failure_diagnostics(
    branch_metrics: Dict,
    gate_mode: str,
    gate_reason: str,
    *,
    projection_failure_diagnostics_fn: Callable[[Dict], Dict],
    merge_count_maps_fn: Callable[..., Dict[str, int]],
    infer_limiting_branch_fn: Callable[[Dict, str], str],
) -> Dict:
    synth_diag = projection_failure_diagnostics_fn(branch_metrics.get("synthesize_projection", {}))
    inference_diag = projection_failure_diagnostics_fn(branch_metrics.get("inference_projection", {}))
    combined_metrics = branch_metrics.get("combined", {}) or {}
    return {
        "overall": {
            "total_failures": int(synth_diag.get("total_failures", 0) or 0)
            + int(inference_diag.get("total_failures", 0) or 0),
            "stage_counts": merge_count_maps_fn(
                synth_diag.get("stage_counts", {}),
                inference_diag.get("stage_counts", {}),
            ),
            "reason_counts": merge_count_maps_fn(
                synth_diag.get("reason_counts", {}),
                inference_diag.get("reason_counts", {}),
            ),
            "label_counts": merge_count_maps_fn(
                synth_diag.get("label_counts", {}),
                inference_diag.get("label_counts", {}),
            ),
        },
        "synthesize": synth_diag,
        "inference": inference_diag,
        "all": {
            "quality_gate_mode": str(gate_mode or ""),
            "quality_gate_reason": str(gate_reason or ""),
            "limiting_branch": infer_limiting_branch_fn(branch_metrics, gate_reason),
            "selected_rule_count": int(combined_metrics.get("selected_rule_count", 0) or 0),
            "compile_valid_rate": float(combined_metrics.get("compile_valid_rate", 0.0) or 0.0),
        },
    }


def evaluate_quality_gate(
    branch_metrics: Dict,
    *,
    knowledge_type: str,
    min_projector_valid_rate: float,
    max_heuristic_rule_rate: float,
    synthesize_soft_pass_eligible_fn: Callable[[Dict], bool],
    literature_underdiverse_fn: Callable[[Dict], bool],
) -> Tuple[str, str, str]:
    synth_projection = branch_metrics.get("synthesize_projection", {})
    inference_projection = branch_metrics.get("inference_projection", {})
    synth_valid = float(
        synth_projection.get(
            "strict_projector_valid_rate",
            synth_projection.get("projector_valid_rate", 0.0),
        )
    )
    inference_valid = float(
        inference_projection.get(
            "strict_projector_valid_rate",
            inference_projection.get("projector_valid_rate", 0.0),
        )
    )
    inference_heuristic = float(inference_projection.get("heuristic_rule_rate", 0.0))
    synth_soft_pass = synthesize_soft_pass_eligible_fn(synth_projection)
    synth_underdiverse = literature_underdiverse_fn(synth_projection)

    if knowledge_type == "synthesize":
        if synth_valid >= min_projector_valid_rate and synth_underdiverse:
            return "degraded", "strict_valid_but_underdiverse", "semantic_gate"
        if synth_valid < min_projector_valid_rate:
            if synth_soft_pass:
                return "success", f"literature_soft_pass(strict={synth_valid:.3f})", "literature_soft_pass"
            return "degraded", f"low_projector_valid_rate(strict={synth_valid:.3f})", "strict_gate"
        return "success", "", "strict_pass"

    if knowledge_type == "inference":
        if inference_valid < min_projector_valid_rate:
            return "degraded", f"low_projector_valid_rate(strict={inference_valid:.3f})", "strict_gate"
        if inference_heuristic > max_heuristic_rule_rate:
            return "degraded", f"heuristic_domination(rate={inference_heuristic:.3f})", "strict_gate"
        return "success", "", "strict_pass"

    if float(synth_projection.get("raw_rule_count", 0)) == 0 or synth_underdiverse:
        if inference_valid >= min_projector_valid_rate and inference_heuristic <= max_heuristic_rule_rate:
            return "success", "inference_only_fallback", "inference_only_fallback"
        return "skipped", "literature_branch_empty", "strict_gate"
    if synth_valid < min_projector_valid_rate and not synth_soft_pass:
        return "degraded", f"low_projector_valid_rate(strict={synth_valid:.3f})", "strict_gate"
    if inference_valid < min_projector_valid_rate:
        return "degraded", f"low_projector_valid_rate(strict={inference_valid:.3f})", "strict_gate"
    if inference_heuristic > max_heuristic_rule_rate:
        return "degraded", f"heuristic_domination(rate={inference_heuristic:.3f})", "strict_gate"
    if synth_soft_pass and synth_valid < min_projector_valid_rate:
        return "success", f"inference_dominant_all(strict={synth_valid:.3f})", "inference_dominant_all"
    return "success", "", "full_all"

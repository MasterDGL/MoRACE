"""Pure normalization and deduplication helpers for rule preparation."""

import copy
from typing import Callable, Dict, List, Tuple


SemanticRuleSignature = Tuple[str, str, str, str, str]


def filter_compile_safe_rules(
    rules: List[Dict],
    *,
    compile_rule_fn: Callable[[Dict], object | None],
) -> List[Dict]:
    kept = []
    for rule in rules:
        if compile_rule_fn(rule) is not None:
            kept.append(rule)
    return kept


def semantic_rule_signature(rule: Dict) -> SemanticRuleSignature:
    threshold = rule.get("threshold") or {}
    value = threshold.get("value")
    snapped_value = ""
    if value is not None:
        try:
            snapped_value = f"{float(value):.4f}"
        except (TypeError, ValueError):
            snapped_value = str(value)
    return (
        str(rule.get("feature_family", "")),
        str(rule.get("feature_name", "")),
        str(rule.get("relation_to_target", "")),
        str(threshold.get("op", "")),
        snapped_value,
    )


def dedupe_prepared_rules(
    rules: List[Dict],
    *,
    semantic_rule_signature_fn: Callable[[Dict], SemanticRuleSignature],
) -> List[Dict]:
    kept: Dict[SemanticRuleSignature, Dict] = {}
    for rule in rules:
        signature = semantic_rule_signature_fn(rule)
        existing = kept.get(signature)
        if existing is None:
            kept[signature] = rule
            continue
        existing_support = int(existing.get("support_count", 0))
        current_support = int(rule.get("support_count", 0))
        if current_support > existing_support:
            kept[signature] = rule
    return list(kept.values())


def audit_rule_id(
    rule: Dict,
    *,
    semantic_rule_signature_fn: Callable[[Dict], SemanticRuleSignature],
) -> str:
    canonical_key = str(rule.get("canonical_key", "")).strip()
    if canonical_key:
        return canonical_key
    return "|".join(semantic_rule_signature_fn(rule))


def audit_rule_summary(
    rule: Dict,
    *,
    audit_rule_id_fn: Callable[[Dict], str],
    alias_group_key_fn: Callable[[Dict], str],
    root_feature_key_fn: Callable[[Dict], str],
    semantic_rule_signature_fn: Callable[[Dict], SemanticRuleSignature],
) -> Dict:
    threshold = rule.get("threshold") or {}
    return {
        "rule_id": audit_rule_id_fn(rule),
        "canonical_key": rule.get("canonical_key", ""),
        "alias_group_key": alias_group_key_fn(rule),
        "root_feature_key": root_feature_key_fn(rule),
        "feature_family": str(rule.get("feature_family", "")),
        "feature_name": str(rule.get("feature_name", "")),
        "source": str(rule.get("source", "")),
        "primary_source": str(rule.get("primary_source", "")),
        "support_count": int(rule.get("support_count", 0) or 0),
        "p_value": float(rule.get("p_value", 1.0) or 1.0),
        "effect_size": float(rule.get("effect_size", 0.0) or 0.0),
        "threshold": threshold if threshold else None,
        "semantic_rule_signature": list(semantic_rule_signature_fn(rule)),
        "is_second_variant_participant": bool(rule.get("_audit_is_second_variant_participant", False)),
        "compile_valid": bool(rule.get("compile_valid", False)),
    }


def annotate_second_variant_rules(
    rules: List[Dict],
    second_variant_rule_ids: set[str],
    *,
    audit_rule_id_fn: Callable[[Dict], str],
) -> List[Dict]:
    annotated: List[Dict] = []
    for rule in rules:
        updated = copy.deepcopy(rule)
        updated["_audit_is_second_variant_participant"] = audit_rule_id_fn(updated) in second_variant_rule_ids
        annotated.append(updated)
    return annotated


def prepared_retention_metrics(
    selected_rules: List[Dict],
    prepared_rules: List[Dict],
    second_variant_rule_ids: set[str],
    *,
    audit_rule_id_fn: Callable[[Dict], str],
) -> Dict:
    selected_count = len(selected_rules)
    prepared_count = len(prepared_rules)
    selected_second_variant_count = sum(
        1 for rule in selected_rules if audit_rule_id_fn(rule) in second_variant_rule_ids
    )
    prepared_second_variant_count = sum(
        1 for rule in prepared_rules if audit_rule_id_fn(rule) in second_variant_rule_ids
    )
    return {
        "prepared_retention_score": float(prepared_count / max(1, selected_count)) if selected_count else 0.0,
        "selected_second_variant_rule_count": int(selected_second_variant_count),
        "prepared_second_variant_rule_count": int(prepared_second_variant_count),
        "prepared_second_variant_retention_score": (
            float(prepared_second_variant_count / max(1, selected_second_variant_count))
            if selected_second_variant_count
            else 0.0
        ),
    }


def prepare_rules_for_compilation_with_audit(
    rules: List[Dict],
    dataset: str,
    subtask: str,
    second_variant_rule_ids: set[str] | None,
    *,
    load_split_xy_fn: Callable,
    learn_threshold_if_needed_fn: Callable,
    annotate_second_variant_rules_fn: Callable[[List[Dict], set[str]], List[Dict]],
    semantic_rule_signature_fn: Callable[[Dict], SemanticRuleSignature],
    audit_rule_summary_fn: Callable[[Dict], Dict],
    audit_rule_id_fn: Callable[[Dict], str],
) -> Tuple[List[Dict], Dict]:
    second_variant_rule_ids = second_variant_rule_ids or set()
    annotated_rules = annotate_second_variant_rules_fn(rules, second_variant_rule_ids)
    smiles_train, y_train, spec = load_split_xy_fn(dataset, subtask=subtask, split="train")
    smiles_valid, y_valid, _ = load_split_xy_fn(dataset, subtask=subtask, split="valid")
    threshold_learned = []
    dropped = []
    for rule in annotated_rules:
        try:
            prepared_rule = learn_threshold_if_needed_fn(
                rule,
                smiles_train,
                y_train,
                smiles_valid,
                y_valid,
                spec.task_type,
            )
        except Exception as exc:
            dropped.append(
                {
                    "rule": audit_rule_summary_fn(rule),
                    "reason": "other_preparation_failure",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        prepared_rule["audit_origin_rule_id"] = audit_rule_id_fn(rule)
        prepared_rule["audit_origin_canonical_key"] = str(rule.get("canonical_key", ""))
        prepared_rule["_audit_is_second_variant_participant"] = bool(
            rule.get("_audit_is_second_variant_participant", False)
        )
        threshold_learned.append(prepared_rule)

    kept: Dict[SemanticRuleSignature, Dict] = {}
    for rule in threshold_learned:
        signature = semantic_rule_signature_fn(rule)
        existing = kept.get(signature)
        if existing is None:
            kept[signature] = rule
            continue
        existing_support = int(existing.get("support_count", 0))
        current_support = int(rule.get("support_count", 0))
        if current_support > existing_support:
            dropped.append(
                {
                    "rule": audit_rule_summary_fn(existing),
                    "reason": "preparation_duplicate_feature",
                    "replaced_by_rule_id": audit_rule_id_fn(rule),
                }
            )
            kept[signature] = rule
        else:
            dropped.append(
                {
                    "rule": audit_rule_summary_fn(rule),
                    "reason": "preparation_duplicate_feature",
                    "replaced_by_rule_id": audit_rule_id_fn(existing),
                }
            )

    prepared = list(kept.values())
    return prepared, {
        "threshold_learned_rules": threshold_learned,
        "dropped": dropped,
    }


def direction_relation_from_text(text: str) -> str:
    lowered = f" {str(text or '').lower()} "
    negative_markers = (
        " lower ",
        " reduced ",
        " minimize ",
        " minimized ",
        " inverse ",
        " inversely ",
        " decrease ",
        " decreased ",
        " decreasing ",
        " smaller ",
        " less ",
        " below ",
    )
    positive_markers = (
        " higher ",
        " elevated ",
        " increase ",
        " increased ",
        " increasing ",
        " greater ",
        " more ",
        " above ",
        " high ",
    )
    if any(marker in lowered for marker in negative_markers):
        return "monotonic_down"
    if any(marker in lowered for marker in positive_markers):
        return "monotonic_up"
    return ""


def normalize_directional_relation(
    rule: Dict,
    *,
    direction_relation_from_text_fn: Callable[[str], str],
) -> Dict:
    updated = copy.deepcopy(rule)
    relation = str(updated.get("relation_to_target", "")).strip().lower()
    if relation in {"positive", "monotonic_up", "negative", "monotonic_down"}:
        return updated
    inferred = direction_relation_from_text_fn(
        updated.get("rationale", "")
    ) or direction_relation_from_text_fn(updated.get("direction", ""))
    if inferred:
        updated["relation_to_target"] = inferred
        updated["direction_consistency_relation_inferred"] = True
    return updated


def enforce_direction_consistency(
    rule: Dict,
    *,
    normalize_directional_relation_fn: Callable[[Dict], Dict],
    direction_relation_from_text_fn: Callable[[str], str],
) -> Dict:
    updated = normalize_directional_relation_fn(rule)
    threshold = updated.get("threshold")
    if updated.get("feature_family") != "descriptor" or not threshold:
        return updated

    relation = str(updated.get("relation_to_target", "")).strip().lower()
    expected_op = ""
    if relation in {"negative", "monotonic_down"}:
        expected_op = "<="
    elif relation in {"positive", "monotonic_up"}:
        expected_op = ">="
    else:
        inferred = direction_relation_from_text_fn(
            updated.get("rationale", "")
        ) or direction_relation_from_text_fn(updated.get("direction", ""))
        if inferred == "monotonic_down":
            expected_op = "<="
            updated["relation_to_target"] = "monotonic_down"
        elif inferred == "monotonic_up":
            expected_op = ">="
            updated["relation_to_target"] = "monotonic_up"

    if expected_op and threshold.get("op") != expected_op:
        corrected = dict(threshold)
        corrected["op"] = expected_op
        updated["threshold"] = corrected
        updated["direction_consistency_adjusted"] = True
        updated["direction_consistency_original_op"] = threshold.get("op")
        updated.setdefault("threshold_origin", "direction_consistency_corrected")
    return updated


def snap_threshold(
    rule: Dict,
    threshold: Dict | None,
    *,
    snapped_descriptor_thresholds: Dict[str, List[float]],
) -> Dict | None:
    if threshold is None:
        return None
    feature_name = str(rule.get("feature_name", ""))
    candidates = snapped_descriptor_thresholds.get(feature_name)
    if not candidates:
        return threshold
    value = float(threshold["value"])
    snapped = min(candidates, key=lambda item: abs(item - value))
    updated = dict(threshold)
    updated["value"] = float(snapped)
    return updated


def single_feature_validation_score(
    rule: Dict,
    smiles: List[str],
    y: List[float],
    task_type: str,
    *,
    numpy_module,
    compile_rule_fn: Callable[[Dict], object | None],
    build_feature_matrix_fn: Callable,
    roc_auc_score_fn: Callable,
) -> float | None:
    compiled = compile_rule_fn(rule)
    if compiled is None:
        return None
    values = numpy_module.asarray(
        [row[0] for row in build_feature_matrix_fn([compiled], smiles)],
        dtype=float,
    )
    y = numpy_module.asarray(y, dtype=float)
    valid_mask = numpy_module.isfinite(values)
    values = values[valid_mask]
    y = y[valid_mask]
    if len(values) < 8 or len(numpy_module.unique(y)) < 2:
        return None
    if task_type == "classification":
        if len(numpy_module.unique(values)) < 2:
            return None
        try:
            score = float(roc_auc_score_fn(y, values))
        except ValueError:
            return None
        return max(score, 1.0 - score)
    return None


def threshold_hint_from_text(rule: Dict) -> bool:
    text = " ".join(
        str(rule.get(field, "")).lower()
        for field in ("direction", "rationale", "threshold_origin", "feature_mode")
    )
    markers = (
        "threshold",
        "above",
        "below",
        "greater than",
        "less than",
        "at least",
        "at most",
        "cutoff",
    )
    return any(marker in text for marker in markers)


def learn_threshold_if_needed(
    rule: Dict,
    smiles_train: List[str],
    y_train: List[float],
    smiles_valid: List[str],
    y_valid: List[float],
    task_type: str,
    *,
    numpy_module,
    threshold_keep_min_gain: float,
    compile_rule_fn: Callable[[Dict], object | None],
    build_feature_matrix_fn: Callable,
    roc_auc_score_fn: Callable,
    normalize_directional_relation_fn: Callable[[Dict], Dict],
    enforce_direction_consistency_fn: Callable[[Dict], Dict],
    snap_threshold_fn: Callable[[Dict, Dict | None], Dict | None],
    single_feature_validation_score_fn: Callable[[Dict, List[str], List[float], str], float | None],
    threshold_hint_from_text_fn: Callable[[Dict], bool],
) -> Dict:
    updated = normalize_directional_relation_fn(rule)
    if task_type != "classification":
        return enforce_direction_consistency_fn(updated)
    if updated["feature_family"] != "descriptor":
        return enforce_direction_consistency_fn(updated)
    if updated.get("threshold") is not None:
        return enforce_direction_consistency_fn(updated)

    compiled = compile_rule_fn(updated)
    if compiled is None:
        return updated
    values = numpy_module.asarray(
        [row[0] for row in build_feature_matrix_fn([compiled], smiles_train)],
        dtype=float,
    )
    y = numpy_module.asarray(y_train, dtype=float)
    valid_mask = numpy_module.isfinite(values)
    values = values[valid_mask]
    y = y[valid_mask]
    if len(values) < 8 or len(numpy_module.unique(values)) < 4 or len(numpy_module.unique(y)) < 2:
        return enforce_direction_consistency_fn(updated)

    quantiles = numpy_module.unique(numpy_module.quantile(values, [0.2, 0.35, 0.5, 0.65, 0.8]))
    best_score = -1.0
    best_threshold = None
    relation = str(updated.get("relation_to_target", "")).strip().lower()
    candidate_ops = ("<=", ">=")
    if relation in {"negative", "monotonic_down"}:
        candidate_ops = ("<=",)
    elif relation in {"positive", "monotonic_up"}:
        candidate_ops = (">=",)
    for threshold in quantiles:
        for op in candidate_ops:
            preds = (values <= threshold).astype(int) if op == "<=" else (values >= threshold).astype(int)
            if len(numpy_module.unique(preds)) < 2:
                continue
            try:
                score = roc_auc_score_fn(y, preds)
            except ValueError:
                continue
            score = max(score, 1.0 - score)
            if score > best_score:
                best_score = score
                best_threshold = {"op": op, "value": float(threshold), "unit": None}

    numeric_candidate = enforce_direction_consistency_fn(updated)
    numeric_score = single_feature_validation_score_fn(
        numeric_candidate,
        smiles_valid,
        y_valid,
        task_type,
    )
    if best_threshold is None:
        return numeric_candidate

    threshold_candidate = copy.deepcopy(updated)
    threshold_candidate["threshold"] = snap_threshold_fn(threshold_candidate, best_threshold)
    threshold_candidate["feature_mode"] = "threshold_binary"
    threshold_candidate.setdefault("threshold_origin", "train_learned_snapped")
    threshold_candidate = enforce_direction_consistency_fn(threshold_candidate)
    threshold_score = single_feature_validation_score_fn(
        threshold_candidate,
        smiles_valid,
        y_valid,
        task_type,
    )

    if threshold_score is None:
        return numeric_candidate
    if numeric_score is None:
        threshold_candidate["threshold_kept_reason"] = "numeric_score_unavailable"
        return threshold_candidate

    if threshold_score >= numeric_score + threshold_keep_min_gain:
        threshold_candidate["threshold_kept_reason"] = "validation_gain"
        threshold_candidate["threshold_score_delta"] = float(threshold_score - numeric_score)
        return threshold_candidate

    if threshold_hint_from_text_fn(rule) and threshold_score >= numeric_score - 0.005:
        threshold_candidate["threshold_kept_reason"] = "textual_threshold_hint"
        threshold_candidate["threshold_score_delta"] = float(threshold_score - numeric_score)
        return threshold_candidate

    numeric_candidate["threshold_kept_reason"] = "prefer_continuous_numeric"
    numeric_candidate["threshold_score_delta"] = float(threshold_score - numeric_score)
    return numeric_candidate

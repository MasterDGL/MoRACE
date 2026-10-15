"""Feature-matrix preparation helpers for evaluation."""

from typing import Dict, List, Tuple


class NoValidFeaturesError(RuntimeError):
    """Raised when feature filtering leaves no scientifically usable inputs."""


def rule_source_bucket(rule: Dict) -> str:
    support_sources = set(rule.get("support_sources", []))
    if support_sources == {"literature"}:
        return "literature"
    if support_sources == {"data"}:
        return "data"
    if "literature" in support_sources and "data" in support_sources:
        return "multi"
    return str(rule.get("source", "") or rule.get("primary_source", "") or "literature")


def rule_root_feature_key(rule: Dict) -> str:
    return f"{str(rule.get('feature_family', ''))}::{str(rule.get('feature_name', ''))}"


def effective_source_coverage(source_breakdown: Dict[str, int]) -> Dict[str, int]:
    multi_count = int(source_breakdown.get("multi", 0))
    return {
        "literature": int(source_breakdown.get("literature", 0)) + multi_count,
        "data": int(source_breakdown.get("data", 0)) + multi_count,
    }


def compute_rule_diversity_metrics(
    rules: List[Dict],
    *,
    rule_source_bucket_fn=rule_source_bucket,
    rule_root_feature_key_fn=rule_root_feature_key,
    effective_source_coverage_fn=effective_source_coverage,
) -> Dict:
    if not rules:
        return {
            "family_breakdown": {},
            "family_diversity": 0,
            "source_breakdown": {},
            "effective_literature_coverage": 0,
            "effective_data_coverage": 0,
            "root_feature_count": 0,
            "dominant_feature_share": 0.0,
        }
    family_breakdown = {
        family: sum(1 for rule in rules if rule.get("feature_family") == family)
        for family in ("descriptor", "pattern_presence", "count")
    }
    source_breakdown = {
        source: sum(1 for rule in rules if rule_source_bucket_fn(rule) == source)
        for source in ("literature", "data", "multi")
    }
    feature_counts = {}
    for rule in rules:
        feature_name = str(rule.get("feature_name", ""))
        feature_counts[feature_name] = feature_counts.get(feature_name, 0) + 1
    dominant_feature_share = max(feature_counts.values()) / max(1, len(rules))
    root_feature_count = len(
        {
            rule_root_feature_key_fn(rule)
            for rule in rules
            if rule_root_feature_key_fn(rule)
        }
    )
    family_diversity = sum(1 for value in family_breakdown.values() if value > 0)
    source_coverage = effective_source_coverage_fn(source_breakdown)
    return {
        "family_breakdown": family_breakdown,
        "family_diversity": family_diversity,
        "source_breakdown": source_breakdown,
        "effective_literature_coverage": source_coverage["literature"],
        "effective_data_coverage": source_coverage["data"],
        "root_feature_count": root_feature_count,
        "dominant_feature_share": float(dominant_feature_share),
    }


def promoted_multi_rule_metrics(
    rules: List[Dict],
    *,
    rule_source_bucket_fn=rule_source_bucket,
    rule_root_feature_key_fn=rule_root_feature_key,
) -> Dict:
    promoted_rules = [rule for rule in rules if bool(rule.get("consensus_promoted", False))]
    promoted_root_keys = {
        str(rule.get("promoted_root_feature_key", "") or rule_root_feature_key_fn(rule) or "").strip()
        for rule in promoted_rules
        if str(rule.get("promoted_root_feature_key", "") or rule_root_feature_key_fn(rule) or "").strip()
    }
    surviving_multi_root_keys = {
        rule_root_feature_key_fn(rule)
        for rule in rules
        if rule_source_bucket_fn(rule) == "multi" and rule_root_feature_key_fn(rule)
    }
    return {
        "promoted_multi_surviving_rule_count": len(promoted_rules),
        "promoted_multi_surviving_root_count": len(promoted_root_keys),
        "promoted_multi_surviving_root_keys": sorted(promoted_root_keys),
        "surviving_multi_root_count": len(surviving_multi_root_keys),
        "surviving_multi_root_keys": sorted(surviving_multi_root_keys),
    }


def sanitize_feature_matrices(
    X_train: List[List[float]],
    X_valid: List[List[float]],
    X_test: List[List[float]],
    feature_names: List[str],
    *,
    numpy_module,
    pandas_module,
) -> Tuple[object, object, object, List[str], Dict]:
    if len(feature_names) != len(set(feature_names)):
        duplicates = [name for name in sorted(set(feature_names)) if feature_names.count(name) > 1]
        raise RuntimeError(f"Duplicate feature names before sanitization: {duplicates[:10]}")
    train_df = pandas_module.DataFrame(X_train, columns=feature_names)
    valid_df = pandas_module.DataFrame(X_valid, columns=feature_names)
    test_df = pandas_module.DataFrame(X_test, columns=feature_names)

    keep_cols = []
    fill_values = {}
    dropped_features = []
    for idx, col in enumerate(train_df.columns):
        series = train_df.iloc[:, idx]
        non_nan = series.dropna()
        if non_nan.empty:
            dropped_features.append({"feature_name": col, "reason": "all_nan"})
            continue
        if int(non_nan.nunique()) <= 1:
            dropped_features.append({"feature_name": col, "reason": "constant_or_single_value"})
            continue
        keep_cols.append(col)
        fill_values[col] = float(non_nan.median())

    if not keep_cols:
        raise NoValidFeaturesError(
            "No valid compiled features remained after filtering. Check rule quality and compiler coverage."
        )

    train_df = train_df[keep_cols].fillna(fill_values)
    valid_df = valid_df[keep_cols].fillna(fill_values)
    test_df = test_df[keep_cols].fillna(fill_values)

    entropy_list = []
    for col in keep_cols:
        values, counts = numpy_module.unique(train_df[col].round(6), return_counts=True)
        probs = counts / counts.sum()
        entropy_list.append(float(-(probs * numpy_module.log2(numpy_module.maximum(probs, 1e-12))).sum()))

    corr = train_df.corr(numeric_only=True).abs()
    mean_corr = 0.0
    if len(keep_cols) > 1:
        mask = numpy_module.triu(numpy_module.ones(corr.shape, dtype=bool), 1)
        upper = corr.where(mask)
        vals = upper.stack().tolist()
        if vals:
            mean_corr = float(numpy_module.mean(vals))

    metrics = {
        "feature_count_before_filter": len(feature_names),
        "feature_count_after_filter": len(keep_cols),
        "non_nan_coverage": (
            float(pandas_module.DataFrame(X_train).notna().mean().mean()) if X_train else 0.0
        ),
        "feature_entropy": float(numpy_module.mean(entropy_list)) if entropy_list else 0.0,
        "pairwise_correlation": mean_corr,
        "dropped_features": dropped_features,
    }
    return train_df.values, valid_df.values, test_df.values, keep_cols, metrics

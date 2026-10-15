"""Metric interpretation helpers for rule contribution analysis."""

import math
from typing import List, Tuple


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    if len(values) <= 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def fmt_float(value: object) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    return repr(number)


def truth(value: object) -> str:
    return "yes" if bool(value) else "no"


def contribution_signal(metric: str, valid_delta: float) -> str:
    if valid_delta is None or math.isnan(valid_delta):
        return "unknown"
    if abs(valid_delta) < 0.001:
        return "near_zero"
    if metric == "roc_auc":
        return "positive_signal" if valid_delta < 0 else "negative_or_harmful_signal"
    if metric == "rmse":
        return "positive_signal" if valid_delta > 0 else "negative_or_harmful_signal"
    return "unknown"


def feature_alignment_label(
    loro_signal: str,
    family_signal: str,
    rank_mean: object,
    kept_feature_count: int,
    filtered: bool,
) -> str:
    if filtered:
        return "low_priority" if loro_signal == "near_zero" else "unknown"
    try:
        rank = float(rank_mean)
    except (TypeError, ValueError):
        return "unknown"
    if kept_feature_count <= 0:
        return "unknown"
    top_cutoff = max(3, math.ceil(kept_feature_count / 3))
    low_cutoff = max(1, math.floor(kept_feature_count * 2 / 3))
    if (
        family_signal == "positive_signal"
        and loro_signal == "positive_signal"
        and rank <= top_cutoff
    ):
        return "aligned_strong_diagnostic"
    if loro_signal == "near_zero" and rank >= low_cutoff:
        return "low_priority"
    if loro_signal or family_signal:
        return "mixed_diagnostic"
    return "unknown"


def rank_vectors(importances: List[List[float]]) -> List[List[int]]:
    ranks: List[List[int]] = []
    for row in importances:
        order = sorted(range(len(row)), key=lambda idx: row[idx], reverse=True)
        rank = [0] * len(row)
        for pos, idx in enumerate(order, start=1):
            rank[idx] = pos
        ranks.append(rank)
    return ranks

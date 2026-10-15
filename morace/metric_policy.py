"""Metric direction and comparison policy used by evaluation decisions."""

from typing import Dict


def normalize_metric_name(metric_name: str | None) -> str:
    return str(metric_name or "").strip().lower().replace("-", "_")


def metric_direction(metric_name: str | None) -> Dict:
    metric = normalize_metric_name(metric_name)
    higher_is_better = {"roc_auc", "auc", "accuracy", "acc"}
    lower_is_better = {"rmse", "mae", "mse", "loss", "log_loss", "cross_entropy", "nll"}
    if metric in higher_is_better:
        return {"higher_is_better": True, "inferred": True, "defaulted": False}
    if metric in lower_is_better:
        return {"higher_is_better": False, "inferred": True, "defaulted": False}
    return {"higher_is_better": True, "inferred": False, "defaulted": True}


def score_improvement(all_score: float, inference_score: float, metric_name: str | None) -> float:
    direction = metric_direction(metric_name)
    if direction["higher_is_better"]:
        return float(all_score - inference_score)
    return float(inference_score - all_score)


def is_better_by(all_score: float, inference_score: float, delta: float, metric_name: str | None) -> bool:
    return score_improvement(all_score, inference_score, metric_name) >= float(delta)


def is_within_delta(all_score: float, inference_score: float, delta: float, metric_name: str | None) -> bool:
    return score_improvement(all_score, inference_score, metric_name) >= -float(delta)

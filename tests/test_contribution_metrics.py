import math

from morace import contribution_analysis


def test_numeric_formatting_contract():
    empty_mean, empty_std = contribution_analysis.mean_std([])
    assert math.isnan(empty_mean)
    assert math.isnan(empty_std)
    assert contribution_analysis.mean_std([2.5]) == (2.5, 0.0)
    assert contribution_analysis.mean_std([1.0, 2.0, 3.0]) == (2.0, 1.0)

    assert contribution_analysis.fmt_float(None) == ""
    assert contribution_analysis.fmt_float(math.nan) == ""
    assert contribution_analysis.fmt_float("2.50") == "2.5"
    assert contribution_analysis.fmt_float("not-a-number") == "not-a-number"
    assert contribution_analysis.truth(1) == "yes"
    assert contribution_analysis.truth([]) == "no"


def test_contribution_signal_metric_direction_and_threshold_contract():
    signal = contribution_analysis.contribution_signal
    assert signal("roc_auc", None) == "unknown"
    assert signal("roc_auc", math.nan) == "unknown"
    assert signal("roc_auc", 0.000999) == "near_zero"
    assert signal("roc_auc", -0.000999) == "near_zero"
    assert signal("roc_auc", -0.001) == "positive_signal"
    assert signal("roc_auc", 0.001) == "negative_or_harmful_signal"
    assert signal("rmse", 0.001) == "positive_signal"
    assert signal("rmse", -0.001) == "negative_or_harmful_signal"
    assert signal("mae", 1.0) == "unknown"


def test_feature_alignment_contract():
    label = contribution_analysis.feature_alignment_label
    assert label("near_zero", "", 1, 9, True) == "low_priority"
    assert label("positive_signal", "positive_signal", 1, 9, True) == "unknown"
    assert label("positive_signal", "positive_signal", "bad", 9, False) == "unknown"
    assert label("positive_signal", "positive_signal", 1, 0, False) == "unknown"
    assert (
        label("positive_signal", "positive_signal", 3, 9, False)
        == "aligned_strong_diagnostic"
    )
    assert label("near_zero", "", 6, 9, False) == "low_priority"
    assert label("unknown", "", 1, 9, False) == "mixed_diagnostic"
    assert label("", "", 1, 9, False) == "unknown"


def test_importance_rank_vectors_preserve_stable_tie_order():
    assert contribution_analysis.rank_vectors([]) == []
    assert contribution_analysis.rank_vectors([[0.2, 0.9, 0.2], []]) == [
        [2, 1, 3],
        [],
    ]

import unittest

from morace.metric_policy import is_better_by, is_within_delta, metric_direction, score_improvement


class MetricPolicyTests(unittest.TestCase):
    def test_classification_metrics_are_maximized(self):
        self.assertTrue(metric_direction("ROC-AUC")["higher_is_better"])
        self.assertAlmostEqual(score_improvement(0.81, 0.78, "roc_auc"), 0.03)
        self.assertTrue(is_better_by(0.81, 0.78, 0.02, "roc_auc"))

    def test_error_metrics_are_minimized(self):
        self.assertFalse(metric_direction("RMSE")["higher_is_better"])
        self.assertAlmostEqual(score_improvement(0.50, 0.54, "rmse"), 0.04)
        self.assertTrue(is_within_delta(0.55, 0.54, 0.02, "rmse"))

    def test_unknown_metric_preserves_higher_is_better_default(self):
        direction = metric_direction("custom_score")
        self.assertTrue(direction["higher_is_better"])
        self.assertFalse(direction["inferred"])
        self.assertTrue(direction["defaulted"])


if __name__ == "__main__":
    unittest.main()

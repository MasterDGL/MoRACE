import json
from pathlib import Path
from types import SimpleNamespace


from morace import evaluation_pipeline as code_gen_and_eval
from morace.evaluation_pipeline import EvaluationConfig, EvaluationPipeline


def test_metric_normalization_and_direction():
    assert code_gen_and_eval.normalize_metric_name("RMSE") == "rmse"
    assert code_gen_and_eval.normalize_metric_name("rmse ") == "rmse"
    assert code_gen_and_eval.normalize_metric_name("roc-auc") == "roc_auc"
    assert code_gen_and_eval.normalize_metric_name("ROC_AUC") == "roc_auc"

    assert code_gen_and_eval.metric_direction("roc_auc") == {
        "higher_is_better": True,
        "inferred": True,
        "defaulted": False,
    }
    assert code_gen_and_eval.metric_direction("rmse") == {
        "higher_is_better": False,
        "inferred": True,
        "defaulted": False,
    }
    assert code_gen_and_eval.metric_direction("unknown_metric") == {
        "higher_is_better": True,
        "inferred": False,
        "defaulted": True,
    }


def test_score_improvement_and_delta_semantics():
    assert code_gen_and_eval.score_improvement(0.75, 0.70, "roc_auc") > 0
    assert code_gen_and_eval.score_improvement(0.65, 0.70, "roc_auc") < 0
    assert code_gen_and_eval.score_improvement(3.89, 4.68, "rmse") > 0
    assert code_gen_and_eval.score_improvement(5.10, 4.68, "rmse") < 0

    assert code_gen_and_eval.is_within_delta(0.695, 0.700, 0.010, "roc_auc")
    assert not code_gen_and_eval.is_within_delta(0.685, 0.700, 0.010, "roc_auc")
    assert code_gen_and_eval.is_within_delta(4.685, 4.680, 0.010, "rmse")
    assert not code_gen_and_eval.is_within_delta(4.695, 4.680, 0.010, "rmse")


def _configure_args(monkeypatch, tmp_path):
    return EvaluationPipeline(EvaluationConfig(
        knowledge_type="all", code_gen_folder=str(tmp_path / "code"),
        output_dir=str(tmp_path / "eval"), run_tag="", list_num=30,
        model="unit-model", target_rule_count=30,
    ))


def _write_inference_reference(pipeline, dataset, valid_score, test_score):
    code_path, metadata_path = pipeline.code_output_paths_for(dataset, "", "inference")
    Path(code_path).write_text("# inference code\n", encoding="utf-8")
    Path(metadata_path).write_text(
        json.dumps(
            {
                "dataset": dataset,
                "knowledge_type": "inference",
                "status": "success",
                "valid_scores": [valid_score],
                "test_scores": [test_score],
                "selected_rule_count": 1,
                "compiled_rule_count": 1,
                "compiled_feature_names": ["inference_feature"],
                "kept_feature_names": ["inference_feature"],
                "rules": [{"source": "data"}],
            }
        ),
        encoding="utf-8",
    )


def _all_metadata(valid_score, test_score):
    return {
        "knowledge_type": "all",
        "status": "success",
        "valid_scores": [valid_score],
        "test_scores": [test_score],
        "selected_rule_count": 3,
        "compiled_rule_count": 3,
        "compiled_feature_names": ["data_feature", "lit_feature", "count_feature"],
        "kept_feature_names": ["data_feature", "lit_feature", "count_feature"],
        "rules": [{"source": "data"}, {"source": "literature"}, {"source": "data"}],
        "quality_gate": {"mode": "full_all", "reason": ""},
        "feature_metrics": {
            "feature_count_after_filter": 3,
            "root_feature_count": 3,
            "family_diversity": 2,
            "family_breakdown": {"descriptor": 1, "count": 1, "pattern_presence": 1},
            "source_breakdown": {"data": 2, "literature": 1},
            "effective_literature_coverage": 1,
            "effective_data_coverage": 2,
        },
        "branch_metrics": {
            "combined": {
                "selected_source_breakdown": {"data": 2, "literature": 1},
                "effective_literature_coverage": 1,
                "effective_data_coverage": 2,
                "selected_family_breakdown": {"descriptor": 1, "count": 1, "pattern_presence": 1},
                "candidate_sets": [],
            }
        },
    }


def test_validation_guard_promotes_higher_is_better_auc(monkeypatch, tmp_path):
    pipeline = _configure_args(monkeypatch, tmp_path)
    _write_inference_reference(pipeline, "bace", valid_score=0.70, test_score=0.99)

    updated, _, _ = pipeline.maybe_apply_validation_promotion_guard(
        "bace",
        "",
        _all_metadata(valid_score=0.75, test_score=0.10),
        "all code",
    )

    assert updated["final_result_source"] == "all_promoted"
    assert updated["promotion"]["metric_name"] == "roc_auc"
    assert updated["promotion"]["metric_higher_is_better"] is True
    assert updated["promotion"]["valid_improvement"] > 0
    assert updated["promotion"]["test_mean_all"] == 0.10


def test_validation_guard_fallbacks_lower_auc(monkeypatch, tmp_path):
    pipeline = _configure_args(monkeypatch, tmp_path)
    _write_inference_reference(pipeline, "bace", valid_score=0.70, test_score=0.99)

    updated, _, _ = pipeline.maybe_apply_validation_promotion_guard(
        "bace",
        "",
        _all_metadata(valid_score=0.65, test_score=1.00),
        "all code",
    )

    assert updated["final_result_source"] == "inference_only"
    assert updated["quality_gate"]["mode"] == "validation_guard_reverted_to_inference"
    assert updated["promotion"]["valid_improvement"] < 0
    assert updated["candidate_all_result"]["valid_mean"] == 0.65


def test_validation_guard_promotes_lower_is_better_rmse(monkeypatch, tmp_path):
    pipeline = _configure_args(monkeypatch, tmp_path)
    _write_inference_reference(pipeline, "freesolv", valid_score=4.68, test_score=0.10)

    updated, _, _ = pipeline.maybe_apply_validation_promotion_guard(
        "freesolv",
        "",
        _all_metadata(valid_score=3.89, test_score=99.0),
        "all code",
    )

    assert updated["final_result_source"] == "all_promoted"
    assert updated["promotion"]["metric_name"] == "rmse"
    assert updated["promotion"]["metric_higher_is_better"] is False
    assert updated["promotion"]["metric_direction_inferred"] is True
    assert updated["promotion"]["metric_direction_defaulted"] is False
    assert updated["promotion"]["valid_improvement"] > 0
    assert updated["promotion"]["valid_gap_raw"] < 0
    assert updated["promotion"]["test_mean_all"] == 99.0


def test_validation_guard_fallbacks_higher_rmse(monkeypatch, tmp_path):
    pipeline = _configure_args(monkeypatch, tmp_path)
    _write_inference_reference(pipeline, "freesolv", valid_score=4.68, test_score=0.10)

    updated, _, message = pipeline.maybe_apply_validation_promotion_guard(
        "freesolv",
        "",
        _all_metadata(valid_score=5.10, test_score=0.01),
        "all code",
    )

    assert updated["final_result_source"] == "inference_only"
    assert updated["quality_gate"]["mode"] == "validation_guard_reverted_to_inference"
    assert updated["promotion"]["metric_name"] == "rmse"
    assert updated["promotion"]["metric_higher_is_better"] is False
    assert updated["promotion"]["valid_improvement"] < 0
    assert updated["promotion"]["valid_gap_raw"] > 0
    assert updated["candidate_all_result"]["valid_mean"] == 5.10
    assert "improvement=" in message

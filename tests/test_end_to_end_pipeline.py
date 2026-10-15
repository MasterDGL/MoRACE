import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_TAG = "contract_bace_seed42"
MODEL = "qwen3.5-9b"
LIST_NUM = 3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _write_contract_dataset(root: Path) -> None:
    split_root = root / "data" / "splits" / "bace"
    manifest_root = root / "data" / "manifests"
    split_root.mkdir(parents=True)
    manifest_root.mkdir(parents=True)

    molecules = [
        ("C", "0"),
        ("CC", "0"),
        ("CCC", "0"),
        ("CCO", "0"),
        ("CCN", "1"),
        ("CCCC", "1"),
        ("CCCl", "1"),
        ("c1ccccc1", "1"),
        ("CC(=O)O", "0"),
        ("C1CCCCC1", "1"),
        ("CCOC", "0"),
        ("CCS", "1"),
    ]
    split_sizes = {"train": 48, "valid": 24, "test": 24}
    manifest_rows = []
    for split_name, size in split_sizes.items():
        rows = [molecules[index % len(molecules)] for index in range(size)]
        path = split_root / f"bace_{split_name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["mol", "Class"])
            writer.writerows(rows)
        smiles = [row[0] for row in rows]
        targets = [row[1] for row in rows]
        manifest_rows.append(
            {
                "dataset": "bace",
                "split_name": split_name,
                "split_file_path": path.relative_to(root).as_posix(),
                "row_count": str(len(rows)),
                "columns": "mol|Class",
                "smiles_column": "mol",
                "target_column": "Class",
                "file_sha256": _sha256(path),
                "ordered_smiles_sha256": _line_hash(smiles),
                "ordered_smiles_target_sha256": _line_hash(
                    [f"{smiles_value}\t{target_value}" for smiles_value, target_value in rows]
                ),
            }
        )

    fields = [
        "dataset",
        "split_name",
        "split_file_path",
        "row_count",
        "columns",
        "smiles_column",
        "target_column",
        "file_sha256",
        "ordered_smiles_sha256",
        "ordered_smiles_target_sha256",
    ]
    with (manifest_root / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)


class _FakeLocalTextGenerator:
    rules = [
        {
            "feature_family": "descriptor",
            "feature_name": "MolWt",
            "feature_mode": "numeric_value",
            "relation_to_target": "positive",
        },
        {
            "feature_family": "count",
            "feature_name": "NumHBA",
            "feature_mode": "count_value",
            "relation_to_target": "negative",
        },
        {
            "feature_family": "descriptor",
            "feature_name": "TPSA",
            "feature_mode": "numeric_value",
            "relation_to_target": "negative",
        },
    ]

    def __init__(self, *_args, **_kwargs):
        self.model_family = "qwen"
        self._projection_index = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def generate_batch(self, prompts, **_kwargs):
        return [
            "Determinant A: molecular size and polarity influence BACE inhibition. "
            "Direction: context dependent. Rationale: controlled contract response."
            for _ in prompts
        ]

    def generate_structured(self, prompts, **_kwargs):
        outputs = []
        for _ in prompts:
            rule = self.rules[self._projection_index % len(self.rules)]
            self._projection_index += 1
            outputs.append(
                {
                    "text": json.dumps({"rules": [rule]}, separators=(",", ":")),
                    "generated_token_count": 24,
                    "avg_logprob": -0.1,
                    "logprob_norm": -0.1,
                    "backend": "contract_fake",
                }
            )
        return outputs


def _module_args(**values):
    return SimpleNamespace(**values)


def _run_contract() -> dict:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    with tempfile.TemporaryDirectory(prefix="morace-e2e-") as temporary:
        root = Path(temporary)
        _write_contract_dataset(root)
        (root / "configs").mkdir()
        shutil.copy2(PROJECT_ROOT / "configs" / "model_registry.json", root / "configs" / "model_registry.json")

        old_cwd = Path.cwd()
        old_dataset_root = os.environ.get("MORACE_DATASET_ROOT")
        old_profile = os.environ.get("STAGE2_PROTOCOL_PROFILE")
        try:
            os.chdir(root)
            os.environ["MORACE_DATASET_ROOT"] = str(root / "data" / "splits")
            os.environ["STAGE2_PROTOCOL_PROFILE"] = "qwen_legacy_d2"
            os.environ["LATE_SELECTION_KEEP_TOPK"] = "2"
            os.environ["LATE_SELECTION_MAX_SCORE_DELTA"] = "0.30"
            os.environ["LATE_SELECTION_MAX_SURVIVORS_PER_PROMPT"] = "2"
            os.environ["LATE_SELECTION_REQUIRE_NOVELTY"] = "1"

            from morace import dataset_provenance
            from morace import evaluation_pipeline
            from morace import inference
            from morace import pipeline_runner
            from morace import prompt_builder
            from morace import rule_summary
            from morace import synthesis
            from morace.generation_runtime import GenerationRuntime
            from morace.run_audit import audit_run
            from morace.run_summary import append_task_summary

            pipeline_runner.LocalTextGenerator = _FakeLocalTextGenerator

            for task in ("synthesize", "inference"):
                prompt_builder.args = _module_args(task=task, output_folder="prompt_file")
                prompt_builder.main()

            artifact_root = root / "artifacts" / RUN_TAG
            artifact_root.mkdir(parents=True)
            runtime_prompt_hashes = {
                name: _sha256(root / "prompt_file" / name)
                for name in ("synthesize_prompt.json", "inference_prompt.json")
            }
            prompt_semantic_hashes = {
                name: json.loads(
                    (root / "prompt_file" / name.replace(".json", ".meta.json")).read_text(
                        encoding="utf-8"
                    )
                )["sha256"]
                for name in ("synthesize_prompt.json", "inference_prompt.json")
            }
            run_config = {
                "run_tag": RUN_TAG,
                "global_seed": "42",
                "prompt_hashes": runtime_prompt_hashes,
                "resolved_config": {"projector_model": MODEL},
            }
            run_config_path = artifact_root / "run_config.json"
            run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
            provenance = dataset_provenance.record_dataset_provenance(
                run_config_path,
                "bace",
                project_root=root,
                manifest_path=root / "data" / "manifests" / "split_manifest.csv",
            )
            assert provenance["manifest"]["verification_status"] == "verified"
            assert provenance["target_column"] == "Class"

            synthesis_config = synthesis.SynthesisConfig(
                runtime=GenerationRuntime.from_environment(),
                dataset="bace",
                subtask="",
                model=MODEL,
                model_path="contract-model",
                projector_model=MODEL,
                projector_model_path="contract-model",
                extra_draft_models="",
                extra_draft_model_paths="",
                drafts_per_prompt=3,
                seed=42,
                input_folder="prompt_file",
                input_file="synthesize_prompt.json",
                output_folder="synthesize_model_response",
                max_new_tokens=64,
                projector_max_new_tokens=64,
                max_prompt_tokens=512,
                projector_max_prompt_tokens=512,
                stage1_temperature=0.45,
                stage2_temperature=0.0,
                stage1_top_p=0.92,
                stage2_top_p=1.0,
                stage1_num_beams=1,
                stage2_num_beams=1,
                run_tag=RUN_TAG,
            )
            synthesis.main(synthesis_config)

            inference_config = inference.InferenceConfig(
                runtime=GenerationRuntime.from_environment(),
                prompt_folder="prompt_file",
                prompt_file="inference_prompt.json",
                input_folder="data/splits",
                output_folder="inference_model_response",
                dataset="bace",
                subtask="",
                list_num=LIST_NUM,
                max_samples_per_chunk=8,
                seed=42,
                model=MODEL,
                model_path="contract-model",
                projector_model=MODEL,
                projector_model_path="contract-model",
                extra_draft_models="",
                extra_draft_model_paths="",
                drafts_per_prompt=2,
                max_new_tokens=64,
                projector_max_new_tokens=64,
                max_prompt_tokens=512,
                projector_max_prompt_tokens=512,
                stage1_temperature=0.45,
                stage2_temperature=0.0,
                stage1_top_p=0.92,
                stage2_top_p=1.0,
                stage1_num_beams=1,
                stage2_num_beams=1,
                run_tag=RUN_TAG,
            )
            inference.main(inference_config)

            rule_summary_config = rule_summary.RuleSummaryConfig(
                input_folder="inference_model_response",
                input_model_folder=MODEL,
                dataset="bace",
                subtask="",
                list_num=LIST_NUM,
                output_folder="summarized_inference_rules",
                split_input_folder="data/splits",
                target_rule_count=8,
                run_tag=RUN_TAG,
            )
            rule_summary.main(rule_summary_config)

            outcomes = {}
            for knowledge_type in ("synthesize", "inference", "all"):
                evaluator = evaluation_pipeline.EvaluationPipeline(evaluation_pipeline.EvaluationConfig(
                    dataset="bace",
                    subtask="",
                    model=MODEL,
                    knowledge_type=knowledge_type,
                    list_num=LIST_NUM,
                    output_dir="eval_result",
                    code_gen_folder="eval_code_generation_repo",
                    target_rule_count=8,
                    run_tag=RUN_TAG,
                    generation_seed=42,
                    max_ablation_features=3,
                )
                )
                evaluator.main()
                _, metadata_path = evaluator.code_output_paths("bace", "")
                outcome_path = Path(
                    evaluator.outcome_output_path_for(
                        "bace", "", knowledge_type, metadata_path
                    )
                )
                outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
                assert outcome["schema_version"] == "outcome_v1_1"
                assert outcome["generation_seed"] == 42
                assert outcome["compiled_rule_count"] > 0
                assert outcome["invalid_rule_count"] == 0
                assert 0.0 <= outcome["actual_test_mean"] <= 1.0
                outcomes[knowledge_type] = outcome

            log_path = artifact_root / "logs" / "bace.log"
            log_path.parent.mkdir()
            log_path.write_text(
                "\n".join(
                    [
                        "[Pipeline] dataset_provenance started",
                        "[Pipeline] dataset_provenance finished",
                        "[Pipeline] step=synthesize started",
                        "[Pipeline] step=synthesize finished",
                        "[Pipeline] step=inference started",
                        "[Pipeline] step=inference finished",
                        "[Pipeline] step=summarize started",
                        "[Pipeline] step=summarize finished",
                        "[Pipeline] step=train_eval_synthesize started",
                        "[Pipeline] step=train_eval_synthesize finished",
                        "[Pipeline] step=train_eval_inference started",
                        "[Pipeline] step=train_eval_inference finished",
                        "[Pipeline] step=train_eval_all started",
                        "[Pipeline] step=train_eval_all finished",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            task_status = append_task_summary(
                summary_path=f"artifacts/{RUN_TAG}/summary.json",
                run_tag=RUN_TAG,
                model=MODEL,
                dataset="bace",
                subtask="",
                batch_num=LIST_NUM,
                exec_status="success",
                exit_code=0,
                log_path=f"artifacts/{RUN_TAG}/logs/bace.log",
                project_root=root,
            )
            assert task_status == "success"
            summary = json.loads((artifact_root / "summary.json").read_text(encoding="utf-8"))
            audit = audit_run(artifact_root, project_root=root, require_success=True)
            assert audit["status"] == "pass", audit
            assert audit["errors"] == []
            assert audit["warnings"] == []

            return {
                "audit_status": audit["status"],
                "prompt_semantic_hashes": prompt_semantic_hashes,
                "selected_rule_counts": {
                    name: outcome["selected_rule_count"] for name, outcome in outcomes.items()
                },
                "compiled_rule_counts": {
                    name: outcome["compiled_rule_count"] for name, outcome in outcomes.items()
                },
                "branch_statuses": summary["tasks"][0]["branch_statuses"],
                "final_result_source": outcomes["all"]["final_result_source"],
            }
        finally:
            os.chdir(old_cwd)
            if old_dataset_root is None:
                os.environ.pop("MORACE_DATASET_ROOT", None)
            else:
                os.environ["MORACE_DATASET_ROOT"] = old_dataset_root
            if old_profile is None:
                os.environ.pop("STAGE2_PROTOCOL_PROFILE", None)
            else:
                os.environ["STAGE2_PROTOCOL_PROFILE"] = old_profile


def test_deterministic_six_stage_pipeline_contract():
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--harness"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, (
        "End-to-end pipeline contract failed in its isolated interpreter.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["audit_status"] == "pass"
    assert payload["prompt_semantic_hashes"] == {
        "inference_prompt.json": "4ef42221c387c279140ed20e0b485e1ff0f1eaefc6b13d9954155c328a639fde",
        "synthesize_prompt.json": "f033e7f78b5d494ee46de33c677b6e5bb3d2b050c50ddad1e94f2aeed289f52a",
    }
    assert payload["selected_rule_counts"] == {"synthesize": 1, "inference": 2, "all": 2}
    assert payload["compiled_rule_counts"] == {"synthesize": 1, "inference": 2, "all": 2}
    assert payload["branch_statuses"] == {
        "synthesize": "success",
        "inference": "success",
        "all": "success",
    }
    assert payload["final_result_source"] == "inference_only"


if __name__ == "__main__":
    print(json.dumps(_run_contract(), sort_keys=True))

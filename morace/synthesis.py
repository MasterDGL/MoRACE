import argparse
import json
import os
import time
from typing import List

from morace.generation_config import SynthesisConfig
from morace.datasets import canonical_dataset_and_subtask
from morace.pipeline_runner import (
    build_stage1_user_prompt,
    flatten_projected_rules,
    generate_stage1_drafts,
    project_drafts_to_rules,
    summarize_projection_metrics,
)
from morace.model_backend import ModelSpec, normalize_model_name
from morace.paths import with_run_tag
from morace.experiment_io import build_budget_report_payload, write_budget_report_json
from morace.rule_canonicalizer import merge_rules
from morace.diagnostics.synthesis_feature_presence import (
    build_bbbp_literature_stability_cue,
    extract_root_presence_from_synth_artifact,
)


SMALL_PROMPT_MODELS = {
    "galactica-6.7b",
    "chemdfm",
    "chemdfm-v1.5-8b",
    "glm-4-9b",
    "gemma-4-e4b-it",
    "gemma-4-e2b-it",
    "llama-3.1-8b-instruct",
    "qwen2.5-0.5b-instruct",
    "qwen2.5-3b-instruct",
    "qwen3.5-9b",
}


def build_draft_specs(config: SynthesisConfig) -> List[ModelSpec]:
    specs = [ModelSpec(name=config.model, path=(config.model_path or config.model))]
    extra_names = [item.strip() for item in config.extra_draft_models.split(",") if item.strip()]
    extra_paths = [item.strip() for item in config.extra_draft_model_paths.split(",")] if config.extra_draft_model_paths else []
    for idx, name in enumerate(extra_names):
        path = extra_paths[idx] if idx < len(extra_paths) and extra_paths[idx] else None
        specs.append(ModelSpec(name=name, path=(path or name)))
    return specs


def get_synthesize_prompt(config: SynthesisConfig):
    prompt_file = os.path.join(config.input_folder, config.input_file)
    with open(prompt_file, "r", encoding="utf-8") as handle:
        prompt_dict = json.load(handle)

    dataset, subtask = canonical_dataset_and_subtask(config.dataset, config.subtask)
    model_tag = normalize_model_name(config.model)
    dataset_key = f"{dataset}_{'small' if model_tag in SMALL_PROMPT_MODELS else 'big'}"

    if dataset == "qm9" and subtask:
        return dataset, subtask, prompt_dict[f"{subtask}_{'small' if model_tag in SMALL_PROMPT_MODELS else 'big'}"]
    if not subtask:
        return dataset, subtask, prompt_dict[dataset_key]
    return dataset, subtask, prompt_dict[dataset_key][subtask]


def output_path(dataset: str, subtask: str, config: SynthesisConfig) -> str:
    model_tag = normalize_model_name(config.model)
    output_dir = with_run_tag(config.output_folder, config.run_tag, inherit_environment=False)
    output_dir = os.path.join(output_dir, model_tag, dataset)
    os.makedirs(output_dir, exist_ok=True)
    if dataset == "qm9" and subtask:
        filename = f"{model_tag}_qm9_{subtask}_pk_response.json"
    elif subtask:
        filename = f"{model_tag}_{dataset}_{subtask}_pk_response.json"
    else:
        filename = f"{model_tag}_{dataset}_pk_response.json"
    return os.path.join(output_dir, filename)


def main(config: SynthesisConfig = None):
    """Run this stage with explicit settings, or parse the command line once."""
    if config is None:
        config = SynthesisConfig.from_namespace(build_parser().parse_args())
    dataset, subtask, base_prompt = get_synthesize_prompt(config)
    task_label = subtask if dataset == "qm9" and subtask else f"{dataset}_{subtask}" if subtask else dataset
    print(f"[Synthesize] dataset={dataset} subtask={subtask or '-'} task_label={task_label}")
    literature_stability_cue = build_bbbp_literature_stability_cue(
        dataset=dataset,
        source="literature",
        enabled=config.literature_stability_cue_enabled,
    )

    prompt_record = {
        "prompt_id": task_label,
        "source": "literature",
        "base_prompt": base_prompt,
        "task_brief": base_prompt,
        "stage1_user_prompt": build_stage1_user_prompt(base_prompt, extra_context=literature_stability_cue),
        "batch_id": None,
    }
    prompt_records = [prompt_record]
    prompt_records_by_id = {item["prompt_id"]: item for item in prompt_records}

    draft_specs = build_draft_specs(config)
    projector_spec = ModelSpec(
        name=config.projector_model or config.model,
        path=(config.projector_model_path or config.model_path or config.projector_model or config.model),
    )

    draft_outputs = generate_stage1_drafts(
        runtime=config.runtime,
        prompt_records=prompt_records,
        draft_specs=draft_specs,
        drafts_per_prompt=config.drafts_per_prompt,
        max_new_tokens=config.max_new_tokens,
        temperature=config.stage1_temperature,
        top_p=config.stage1_top_p,
        num_beams=config.stage1_num_beams,
        max_input_tokens=config.max_prompt_tokens,
        seed=config.seed,
    )
    projected_records = project_drafts_to_rules(
        runtime=config.runtime,
        prompt_records_by_id=prompt_records_by_id,
        draft_outputs=draft_outputs,
        projector_spec=projector_spec,
        max_new_tokens=config.projector_max_new_tokens,
        temperature=config.stage2_temperature,
        top_p=config.stage2_top_p,
        num_beams=config.stage2_num_beams,
        max_input_tokens=config.projector_max_prompt_tokens,
        allow_heuristic_fallback=True,
        heuristic_fallback_sources=("literature",),
        heuristic_max_rules=2,
        seed=config.seed + 1_000_000,
    )

    raw_rules = flatten_projected_rules(projected_records)
    canonical_rules = merge_rules(raw_rules, multi_consensus_promotion_enable=config.multi_consensus_promotion_enable)
    metrics = summarize_projection_metrics(projected_records, branch_name="synthesize", runtime=config.runtime)
    metrics.update(
        {
            "raw_rule_count": len(raw_rules),
            "canonical_rule_count": len(canonical_rules),
            "draft_model_count": len(draft_specs),
            "drafts_per_prompt": config.drafts_per_prompt,
        }
    )

    artifact = {
        "pipeline_mode": "dccd",
        "dataset": dataset,
        "subtask": subtask,
        "task_label": task_label,
        "source": "literature",
        "base_prompt": base_prompt,
        "run_tag": config.run_tag,
        "draft_models": [{"name": spec.name, "path": spec.path} for spec in draft_specs],
        "projector_model": {"name": projector_spec.name, "path": projector_spec.path},
        "stage1_drafts": draft_outputs,
        "stage2_projection": projected_records,
        "rules": raw_rules,
        "canonical_rules": canonical_rules,
        "metrics": metrics,
    }
    root_presence = extract_root_presence_from_synth_artifact(artifact)
    artifact.update(root_presence)
    artifact["metrics"].update(
        {
            "synth_root_observability_enabled": True,
            "synth_literature_stability_cue_enabled": config.literature_stability_cue_enabled,
            "raw_root_key_count": len(root_presence["raw_root_keys"]),
            "canonical_root_key_count": len(root_presence["canonical_root_keys"]),
            "compile_safe_root_key_count": len(root_presence["compile_safe_root_keys"]),
        }
    )

    path = output_path(dataset, subtask, config)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False)

    budget_payload = build_budget_report_payload(
        system_id="D2",
        branch_name="synthesize",
        dataset=dataset,
        run_tag=config.run_tag,
        experiment_id=config.run_tag or None,
        model_id=config.model,
        model_path=draft_specs[0].path if draft_specs else config.model_path,
        projector_model_id=projector_spec.name,
        projector_model_path=projector_spec.path,
        total_tokens=None,
        token_count_source="partial_stage2_only",
        api_call_count=int(len(draft_outputs) + int(metrics.get("dirty_retry_count", 0) or 0) + int(metrics.get("stage2_total_call_count", 0) or 0)),
        stage1_total_calls=int(len(draft_outputs) + int(metrics.get("dirty_retry_count", 0) or 0)),
        stage2_total_calls=int(metrics.get("stage2_total_call_count", 0) or 0),
        stage1_retry_calls=int(metrics.get("dirty_retry_count", 0) or 0),
        stage2_retry_calls=int(metrics.get("stage2_retry_call_count", 0) or 0),
        call_counts={
            "stage1_draft_count": len(draft_outputs),
            "stage2_projection_count": len(projected_records),
        },
        token_counts={
            "stage2_generated_token_total": int(metrics.get("stage2_generated_token_total", 0) or 0),
            "stage2_generated_token_mean": float(metrics.get("stage2_generated_token_mean", 0.0) or 0.0),
            "stage2_base_generated_token_total": int(metrics.get("stage2_base_generated_token_total", 0) or 0),
            "stage2_retry_generated_token_total": int(metrics.get("stage2_retry_generated_token_total", 0) or 0),
        },
        prompt_lengths={
            "base_prompt_char_len": len(base_prompt),
        },
        generation_caps={
            "stage1_max_new_tokens": int(config.max_new_tokens),
            "stage2_max_new_tokens": int(config.projector_max_new_tokens),
            "stage1_max_prompt_tokens": int(config.max_prompt_tokens),
            "stage2_max_prompt_tokens": int(config.projector_max_prompt_tokens),
        },
        extra={
            "draft_model_count": len(draft_specs),
            "raw_rule_count": len(raw_rules),
            "canonical_rule_count": len(canonical_rules),
            "token_counts_available": True,
        },
    )
    write_budget_report_json(os.path.join(os.path.dirname(path), "budget_report.json"), budget_payload)
    print(
        f"[Synthesize] schema_valid_rate={metrics['schema_valid_rate']:.3f} "
        f"strict_projector_valid_rate={metrics.get('strict_projector_valid_rate', 0.0):.3f} "
        f"projector_valid_rate={metrics.get('projector_valid_rate', 0.0):.3f} "
        f"heuristic_rescue_rate={metrics.get('heuristic_rescue_rate', 0.0):.3f} "
        f"repair_success_rate={metrics.get('repair_success_rate', 0.0):.3f} "
        f"raw_rules={len(raw_rules)} canonical_rules={len(canonical_rules)} "
        f"heuristic_rule_rate={metrics.get('heuristic_rule_rate', 0.0):.3f} "
        f"mollogp_collapse_rate={metrics.get('mollogp_collapse_rate', 0.0):.3f}"
    )
    print(f"[Synthesize] projection_status_counts={metrics.get('projection_status_counts', {})}")
    print(f"[Synthesize] stage2_protocol_profile_counts={metrics.get('stage2_protocol_profile_counts', {})}")
    print(f"[Synthesize] stage2_protocol_profile_source_counts={metrics.get('stage2_protocol_profile_source_counts', {})}")
    print(f"[Synthesize] stage2_protocol_profile_version_counts={metrics.get('stage2_protocol_profile_version_counts', {})}")
    print(f"[Synthesize] strict_failure_diagnostics={metrics.get('strict_failure_diagnostics', {})}")
    print(f"[Synthesize] strict_failure_diagnostics_by_source={metrics.get('strict_failure_diagnostics_by_source', {})}")
    print(f"[Synthesize] llama_native_output_discipline_summary={metrics.get('llama_native_output_discipline_summary', {})}")
    print(f"[Synthesize] llama_native_strict_fail_sample_counts={ {k: len(v) for k, v in (metrics.get('llama_native_strict_fail_samples', {}) or {}).items()} }")
    if canonical_rules:
        print("[Synthesize] top rules:")
        for idx, rule in enumerate(canonical_rules[:5], 1):
            threshold = rule.get("threshold")
            threshold_text = ""
            if threshold:
                threshold_text = f" {threshold['op']} {threshold['value']}"
            print(
                f"  - {idx}. {rule['feature_family']}::{rule['feature_name']}{threshold_text} "
                f"| support={rule.get('support_count', 1)} | rationale={rule.get('rationale', '')[:120]}"
            )
    print(f"Wrote synthesis artifact to {path}")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bbbp", help="dataset/task name")
    parser.add_argument("--subtask", type=str, default="", help="subtask of tox21/sider/qm9 dataset")
    parser.add_argument("--model", type=str, default="qwen3.5-9b", help="primary draft model label")
    parser.add_argument("--model_path", type=str, default="", help="primary draft model local path")
    parser.add_argument("--projector_model", type=str, default="", help="stage2 projector model label")
    parser.add_argument("--projector_model_path", type=str, default="", help="stage2 projector model local path")
    parser.add_argument("--extra_draft_models", type=str, default="", help="comma-separated extra draft model labels")
    parser.add_argument("--extra_draft_model_paths", type=str, default="", help="comma-separated extra draft model paths")
    parser.add_argument("--drafts_per_prompt", type=int, default=1, help="number of drafts per model and prompt")
    parser.add_argument("--seed", type=int, default=42, help="generation seed")
    parser.add_argument("--input_folder", type=str, default="prompt_file", help="synthesis prompt folder")
    parser.add_argument("--input_file", type=str, default="synthesize_prompt.json", help="synthesis prompt file")
    parser.add_argument("--output_folder", type=str, default="synthesize_model_response", help="artifact output folder")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="stage1 max new tokens")
    parser.add_argument("--projector_max_new_tokens", type=int, default=192, help="stage2 max new tokens")
    parser.add_argument("--max_prompt_tokens", type=int, default=1536, help="stage1 max input tokens after truncation")
    parser.add_argument("--projector_max_prompt_tokens", type=int, default=1792, help="stage2 max input tokens after truncation")
    parser.add_argument("--stage1_temperature", type=float, default=0.35, help="stage1 draft temperature")
    parser.add_argument("--stage2_temperature", type=float, default=0.0, help="stage2 projector temperature")
    parser.add_argument("--stage1_top_p", type=float, default=0.9, help="stage1 top-p")
    parser.add_argument("--stage2_top_p", type=float, default=0.9, help="stage2 top-p")
    parser.add_argument("--stage1_num_beams", type=int, default=1, help="stage1 beam size")
    parser.add_argument("--stage2_num_beams", type=int, default=1, help="stage2 beam size")
    parser.add_argument("--run_tag", type=str, default="", help="optional run tag for segregated artifacts")
    return parser


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Synthesize/Time elapsed: {end - start} seconds")

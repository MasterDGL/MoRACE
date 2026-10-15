import argparse
import json
import os
import random
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from scipy.stats import spearmanr

from morace.generation_config import InferenceConfig
from morace.datasets import canonical_dataset_and_subtask, format_row_for_prompt, get_dataset_spec, load_split_dataframe
from morace.pipeline_runner import (
    build_stage1_user_prompt,
    flatten_projected_rules,
    generate_stage1_drafts,
    project_drafts_to_rules,
    summarize_projection_metrics,
)
from morace.diagnostics.inference_feature_presence import (
    build_bbbp_inference_mollogp_stability_cue,
    extract_root_presence_from_inference_artifact,
)
from morace.model_backend import ModelSpec, normalize_model_name
from morace.paths import with_run_tag
from morace.experiment_io import build_budget_report_payload, write_budget_report_json
from morace.rule_canonicalizer import merge_rules


SUMMARY_DESCRIPTOR_FNS = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Crippen.MolLogP,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "NumHBD": rdMolDescriptors.CalcNumHBD,
    "NumHBA": rdMolDescriptors.CalcNumHBA,
    "NumRotatableBonds": Lipinski.NumRotatableBonds,
}

MOTIF_SMARTS = {
    "halogen": "[F,Cl,Br,I]",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
    "amide": "[NX3][CX3](=[OX1])",
    "carbonyl": "[CX3]=[OX1]",
    "aromatic_ring": "a1aaaaa1",
    "fused_ring": "a1aaa2aaaa2a1",
    "heteroaromatic_ring": "[a;r6]1[a;r6][a;r6][n,o,s;r6][a;r6][a;r6]1",
    "nitrile": "[CX2]#N",
    "carboxylate": "[CX3](=O)[O-]",
    "sulfonamide": "S(=O)(=O)N",
}



def build_draft_specs(config: InferenceConfig) -> List[ModelSpec]:
    specs = [ModelSpec(name=config.model, path=(config.model_path or config.model))]
    extra_names = [item.strip() for item in config.extra_draft_models.split(",") if item.strip()]
    extra_paths = [item.strip() for item in config.extra_draft_model_paths.split(",")] if config.extra_draft_model_paths else []
    for idx, name in enumerate(extra_names):
        path = extra_paths[idx] if idx < len(extra_paths) and extra_paths[idx] else None
        specs.append(ModelSpec(name=name, path=(path or name)))
    return specs


def get_inference_prompt(dataset: str, subtask: str, config: InferenceConfig):
    prompt_file = os.path.join(config.prompt_folder, config.prompt_file)
    with open(prompt_file, "r", encoding="utf-8") as handle:
        prompt_dict = json.load(handle)
    if dataset == "qm9" and subtask:
        return prompt_dict[subtask]
    if dataset in prompt_dict and not subtask:
        return prompt_dict[dataset]
    return prompt_dict[dataset][subtask]


def resolve_batch_size(task_type: str, config: InferenceConfig) -> int:
    if config.max_samples_per_chunk > 0:
        return config.max_samples_per_chunk
    return 10 if task_type == "classification" else 12


def stratified_batches(df: pd.DataFrame, target_column: str, task_type: str, list_num: int, batch_size: int, seed: int):
    rng = random.Random(seed)
    records = []
    if task_type == "classification":
        groups = {label: subset.to_dict("records") for label, subset in df.groupby(target_column)}
        labels = list(groups.keys())
        for _ in range(list_num):
            batch = []
            base = max(1, batch_size // max(len(labels), 1))
            for label in labels:
                source = groups[label]
                take = min(len(source), base)
                if take == 0:
                    continue
                batch.extend(rng.sample(source, take) if len(source) >= take else [rng.choice(source) for _ in range(take)])
            if len(batch) < batch_size:
                all_rows = df.to_dict("records")
                deficit = batch_size - len(batch)
                batch.extend(rng.sample(all_rows, min(deficit, len(all_rows))))
            rng.shuffle(batch)
            records.append(batch[:batch_size])
        return records

    quantiles = min(4, max(2, len(df) // max(batch_size, 1)))
    work_df = df.copy()
    work_df["_bin"] = pd.qcut(work_df[target_column], q=quantiles, labels=False, duplicates="drop")
    groups = {bin_id: subset.to_dict("records") for bin_id, subset in work_df.groupby("_bin")}
    bin_ids = list(groups.keys())
    for _ in range(list_num):
        batch = []
        base = max(1, batch_size // max(len(bin_ids), 1))
        for bin_id in bin_ids:
            source = groups[bin_id]
            take = min(len(source), base)
            if take == 0:
                continue
            batch.extend(rng.sample(source, take) if len(source) >= take else [rng.choice(source) for _ in range(take)])
        if len(batch) < batch_size:
            all_rows = df.to_dict("records")
            deficit = batch_size - len(batch)
            batch.extend(rng.sample(all_rows, min(deficit, len(all_rows))))
        rng.shuffle(batch)
        records.append(batch[:batch_size])
    return records


def select_representative_examples(batch: List[dict], spec, max_examples: int = 6) -> List[dict]:
    if len(batch) <= max_examples:
        return batch
    if spec.task_type == "classification":
        positives = [row for row in batch if int(row[spec.target_column]) == 1]
        negatives = [row for row in batch if int(row[spec.target_column]) == 0]
        selected = positives[: max_examples // 2] + negatives[: max_examples // 2]
        if len(selected) < max_examples:
            seen = {id(row) for row in selected}
            for row in batch:
                if id(row) not in seen:
                    selected.append(row)
                if len(selected) >= max_examples:
                    break
        return selected[:max_examples]

    sorted_batch = sorted(batch, key=lambda item: float(item[spec.target_column]))
    indices = {0, len(sorted_batch) // 2, len(sorted_batch) - 1}
    if len(sorted_batch) >= 6:
        indices |= {len(sorted_batch) // 3, (2 * len(sorted_batch)) // 3}
    return [sorted_batch[idx] for idx in sorted(indices)][:max_examples]


def _descriptor_vector(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    values = {}
    for name, fn in SUMMARY_DESCRIPTOR_FNS.items():
        try:
            values[name] = float(fn(mol))
        except Exception:
            continue
    return values


def _motif_vector(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    values = {}
    for name, smarts in MOTIF_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        try:
            values[name] = int(mol.HasSubstructMatch(patt))
        except Exception:
            continue
    return values


def build_descriptor_summary(batch: List[dict], spec) -> str:
    rows = []
    targets = []
    for row in batch:
        vector = _descriptor_vector(str(row[spec.smiles_column]))
        if not vector:
            continue
        rows.append(vector)
        targets.append(float(row[spec.target_column]))
    if not rows:
        return "Descriptor summary unavailable."
    frame = pd.DataFrame(rows)
    summary_lines = []
    if spec.task_type == "classification" and len(set(targets)) >= 2:
        target_arr = np.asarray(targets)
        pos = frame[target_arr == 1]
        neg = frame[target_arr == 0]
        deltas = []
        for col in frame.columns:
            if pos.empty or neg.empty:
                continue
            deltas.append((col, float(pos[col].mean() - neg[col].mean())))
        deltas.sort(key=lambda item: abs(item[1]), reverse=True)
        for name, delta in deltas[:3]:
            if delta > 0:
                summary_lines.append(f"label=1 tends to have higher {name} than label=0")
            elif delta < 0:
                summary_lines.append(f"label=1 tends to have lower {name} than label=0")
    else:
        target_arr = np.asarray(targets, dtype=float)
        corrs = []
        for col in frame.columns:
            corr, _ = spearmanr(frame[col].to_numpy(dtype=float), target_arr)
            if np.isnan(corr):
                continue
            corrs.append((col, float(corr)))
        corrs.sort(key=lambda item: abs(item[1]), reverse=True)
        for name, corr in corrs[:3]:
            direction = "increases with" if corr > 0 else "decreases with"
            summary_lines.append(f"{name} roughly {direction} the target in this batch")
    if not summary_lines:
        return "Descriptor summary was weak; rely on recurring motifs and polarity/lipophilicity contrasts."
    return "\n".join(f"- {line}" for line in summary_lines)


def build_motif_summary(batch: List[dict], spec) -> str:
    rows = []
    targets = []
    for row in batch:
        vector = _motif_vector(str(row[spec.smiles_column]))
        if not vector:
            continue
        rows.append(vector)
        targets.append(float(row[spec.target_column]))
    if not rows:
        return "Motif/ring summary unavailable."
    frame = pd.DataFrame(rows).fillna(0)
    summary_lines = []
    if spec.task_type == "classification" and len(set(targets)) >= 2:
        target_arr = np.asarray(targets)
        pos = frame[target_arr == 1]
        neg = frame[target_arr == 0]
        deltas = []
        for col in frame.columns:
            if pos.empty or neg.empty:
                continue
            deltas.append((col, float(pos[col].mean() - neg[col].mean())))
        deltas.sort(key=lambda item: abs(item[1]), reverse=True)
        for name, delta in deltas[:3]:
            if abs(delta) < 0.1:
                continue
            direction = "more common in label=1" if delta > 0 else "more common in label=0"
            summary_lines.append(f"{name} is {direction}")
    else:
        target_arr = np.asarray(targets, dtype=float)
        corrs = []
        for col in frame.columns:
            corr, _ = spearmanr(frame[col].to_numpy(dtype=float), target_arr)
            if np.isnan(corr):
                continue
            corrs.append((col, float(corr)))
        corrs.sort(key=lambda item: abs(item[1]), reverse=True)
        for name, corr in corrs[:3]:
            if abs(corr) < 0.1:
                continue
            direction = "increases with" if corr > 0 else "decreases with"
            summary_lines.append(f"{name} roughly {direction} the target")
    if not summary_lines:
        return "Motif/ring summary was weak; rely on recurring substructures only when clearly supported."
    return "\n".join(f"- {line}" for line in summary_lines)


def build_prompt_records(dataset: str, subtask: str, base_prompt: str, config: InferenceConfig):
    spec = get_dataset_spec(dataset, subtask, input_folder=config.input_folder, inherit_environment=False)
    train_df = load_split_dataframe(dataset, subtask, split="train", input_folder=config.input_folder,
                                    inherit_environment=False)
    batch_size = resolve_batch_size(spec.task_type, config)
    batches = stratified_batches(
        train_df,
        target_column=spec.target_column,
        task_type=spec.task_type,
        list_num=config.list_num,
        batch_size=batch_size,
        seed=config.seed,
    )
    prompt_records = []
    for batch_id, batch in enumerate(batches):
        representatives = select_representative_examples(batch, spec)
        examples = "\n".join(
            format_row_for_prompt(spec.dataset, str(row[spec.smiles_column]), row[spec.target_column])
            for row in representatives
        )
        descriptor_summary = build_descriptor_summary(batch, spec)
        motif_summary = build_motif_summary(batch, spec)
        extra_cue = build_bbbp_inference_mollogp_stability_cue(
            dataset=dataset,
            enabled=config.mollogp_stability_cue_enabled,
        )
        stage1_user_prompt = build_stage1_user_prompt(
            base_prompt,
            extra_context=(
                "Representative training examples:\n"
                f"{examples}\n\n"
                "Descriptor summary from this batch:\n"
                f"{descriptor_summary}\n\n"
                "Motif and ring summary from this batch:\n"
                f"{motif_summary}\n\n"
                "Instructions for this draft:\n"
                "- Infer 2-3 candidate determinants only.\n"
                "- Do not copy SMILES, labels, or training examples.\n"
                "- Explain chemistry in abstract terms.\n"
                "- Use motif/ring/count determinants when the batch evidence supports them.\n"
                "- Use the format Determinant / Direction / Rationale."
                + (f"\n- Additional stability cue:\n{extra_cue}" if extra_cue else "")
            ),
        )
        prompt_records.append(
            {
                "prompt_id": f"batch_{batch_id}",
                "batch_id": batch_id,
                "source": "data",
                "base_prompt": base_prompt,
                "task_brief": base_prompt,
                "stage1_user_prompt": stage1_user_prompt,
                "sample_size": len(batch),
                "representative_examples": representatives,
                "descriptor_summary": descriptor_summary,
                "motif_summary": motif_summary,
            }
        )
    return prompt_records, spec, batch_size


def output_path(dataset: str, subtask: str, config: InferenceConfig) -> str:
    model_tag = normalize_model_name(config.model)
    output_dir = with_run_tag(os.path.join(config.output_folder), config.run_tag, inherit_environment=False)
    output_dir = os.path.join(output_dir, model_tag, dataset)
    os.makedirs(output_dir, exist_ok=True)
    if dataset == "qm9" and subtask:
        filename = f"{model_tag}_qm9_{subtask}_dk_response_sample_{config.list_num}.json"
    elif subtask:
        filename = f"{model_tag}_{dataset}_{subtask}_dk_response_sample_{config.list_num}.json"
    else:
        filename = f"{model_tag}_{dataset}_dk_response_sample_{config.list_num}.json"
    return os.path.join(output_dir, filename)


def main(config: InferenceConfig = None):
    """Run this stage with explicit settings, or parse the command line once."""
    if config is None:
        config = InferenceConfig.from_namespace(build_parser().parse_args())
    dataset, subtask = canonical_dataset_and_subtask(config.dataset, config.subtask)
    base_prompt = get_inference_prompt(dataset, subtask, config)
    prompt_records, spec, batch_size = build_prompt_records(dataset, subtask, base_prompt, config)
    print(
        f"[Inference] dataset={dataset} subtask={subtask or '-'} task_type={spec.task_type} "
        f"batches={len(prompt_records)} batch_size={batch_size}"
    )
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
        dirty_retry_tokens=(
            "### Instruction",
            "### Response",
            "Example:",
            "Citation:",
            "Question:",
        ),
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
        heuristic_max_rules=2,
        seed=config.seed + 1_000_000,
    )

    raw_rules = flatten_projected_rules(projected_records)
    canonical_rules = merge_rules(raw_rules, multi_consensus_promotion_enable=config.multi_consensus_promotion_enable)
    metrics = summarize_projection_metrics(projected_records, branch_name="inference", runtime=config.runtime)
    metrics.update(
        {
            "raw_rule_count": len(raw_rules),
            "canonical_rule_count": len(canonical_rules),
            "batch_count": len(prompt_records),
            "task_type": spec.task_type,
            "sample_size_per_batch": batch_size,
            "inference_mollogp_stability_cue_enabled": bool(config.mollogp_stability_cue_enabled),
        }
    )

    artifact = {
        "pipeline_mode": "dccd",
        "dataset": dataset,
        "subtask": subtask,
        "task_type": spec.task_type,
        "source": "data",
        "base_prompt": base_prompt,
        "list_num": config.list_num,
        "run_tag": config.run_tag,
        "draft_models": [{"name": spec_item.name, "path": spec_item.path} for spec_item in draft_specs],
        "projector_model": {"name": projector_spec.name, "path": projector_spec.path},
        "batches": prompt_records,
        "stage1_drafts": draft_outputs,
        "stage2_projection": projected_records,
        "rules": raw_rules,
        "canonical_rules": canonical_rules,
        "metrics": metrics,
    }
    root_presence = extract_root_presence_from_inference_artifact(artifact)
    artifact.update(root_presence)
    metrics.update(
        {
            "raw_root_count": len(root_presence.get("raw_root_keys", [])),
            "canonical_root_count_distinct": len(root_presence.get("canonical_root_keys", [])),
            "compile_safe_root_count_distinct": len(root_presence.get("compile_safe_root_keys", [])),
            "selected_root_count_distinct": len(root_presence.get("selected_root_keys", [])),
        }
    )

    path = output_path(dataset, subtask, config)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False)

    budget_payload = build_budget_report_payload(
        system_id="D2",
        branch_name="inference",
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
            "batch_count": len(prompt_records),
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
            "raw_rule_count": len(raw_rules),
            "canonical_rule_count": len(canonical_rules),
            "list_num": int(config.list_num),
            "token_counts_available": True,
        },
    )
    write_budget_report_json(os.path.join(os.path.dirname(path), "budget_report.json"), budget_payload)
    print(
        f"[Inference] schema_valid_rate={metrics['schema_valid_rate']:.3f} "
        f"strict_projector_valid_rate={metrics.get('strict_projector_valid_rate', 0.0):.3f} "
        f"projector_valid_rate={metrics.get('projector_valid_rate', 0.0):.3f} "
        f"heuristic_rescue_rate={metrics.get('heuristic_rescue_rate', 0.0):.3f} "
        f"repair_success_rate={metrics.get('repair_success_rate', 0.0):.3f} "
        f"raw_rules={len(raw_rules)} canonical_rules={len(canonical_rules)} "
        f"heuristic_rule_rate={metrics.get('heuristic_rule_rate', 0.0):.3f} "
        f"mollogp_collapse_rate={metrics.get('mollogp_collapse_rate', 0.0):.3f}"
    )
    print(f"[Inference] projection_status_counts={metrics.get('projection_status_counts', {})}")
    print(f"[Inference] stage2_protocol_profile_counts={metrics.get('stage2_protocol_profile_counts', {})}")
    print(f"[Inference] stage2_protocol_profile_source_counts={metrics.get('stage2_protocol_profile_source_counts', {})}")
    print(f"[Inference] stage2_protocol_profile_version_counts={metrics.get('stage2_protocol_profile_version_counts', {})}")
    print(f"[Inference] strict_failure_diagnostics={metrics.get('strict_failure_diagnostics', {})}")
    print(f"[Inference] strict_failure_diagnostics_by_source={metrics.get('strict_failure_diagnostics_by_source', {})}")
    print(f"[Inference] llama_native_output_discipline_summary={metrics.get('llama_native_output_discipline_summary', {})}")
    print(f"[Inference] llama_native_strict_fail_sample_counts={ {k: len(v) for k, v in (metrics.get('llama_native_strict_fail_samples', {}) or {}).items()} }")
    if canonical_rules:
        print("[Inference] top canonical rules:")
        for idx, rule in enumerate(canonical_rules[:8], 1):
            threshold = rule.get("threshold")
            threshold_text = ""
            if threshold:
                threshold_text = f" {threshold['op']} {threshold['value']}"
            print(
                f"  - {idx}. {rule['feature_family']}::{rule['feature_name']}{threshold_text} "
                f"| source={rule['source']} | support={rule.get('support_count', 1)}"
            )
    print(f"Wrote inference artifact to {path}")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_folder", type=str, default="prompt_file", help="prompt file folder")
    parser.add_argument("--prompt_file", type=str, default="inference_prompt.json", help="inference prompt json file")
    parser.add_argument("--input_folder", type=str, default="data/splits", help="dataset folder")
    parser.add_argument("--output_folder", type=str, default="inference_model_response", help="artifact output folder")
    parser.add_argument("--dataset", type=str, default="bbbp", help="dataset name")
    parser.add_argument("--subtask", type=str, default="", help="subtask of tox21/sider/qm9 dataset")
    parser.add_argument("--list_num", type=int, default=30, help="number of stratified batches")
    parser.add_argument("--max_samples_per_chunk", type=int, default=0, help="max examples per batch prompt; 0 uses task-aware defaults")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed")
    parser.add_argument("--model", type=str, default="qwen3.5-9b", help="primary draft model label")
    parser.add_argument("--model_path", type=str, default="", help="primary draft model local path")
    parser.add_argument("--projector_model", type=str, default="", help="stage2 projector model label")
    parser.add_argument("--projector_model_path", type=str, default="", help="stage2 projector model local path")
    parser.add_argument("--extra_draft_models", type=str, default="", help="comma-separated extra draft model labels")
    parser.add_argument("--extra_draft_model_paths", type=str, default="", help="comma-separated extra draft model paths")
    parser.add_argument("--drafts_per_prompt", type=int, default=1, help="number of drafts per model and batch")
    parser.add_argument("--max_new_tokens", type=int, default=768, help="stage1 max new tokens")
    parser.add_argument("--projector_max_new_tokens", type=int, default=768, help="stage2 max new tokens")
    parser.add_argument("--max_prompt_tokens", type=int, default=1200, help="stage1 max input tokens after truncation")
    parser.add_argument("--projector_max_prompt_tokens", type=int, default=1600, help="stage2 max input tokens after truncation")
    parser.add_argument("--stage1_temperature", type=float, default=0.45, help="stage1 draft temperature")
    parser.add_argument("--stage2_temperature", type=float, default=0.0, help="stage2 projector temperature")
    parser.add_argument("--stage1_top_p", type=float, default=0.92, help="stage1 top-p")
    parser.add_argument("--stage2_top_p", type=float, default=1.0, help="stage2 top-p")
    parser.add_argument("--stage1_num_beams", type=int, default=1, help="stage1 beam size")
    parser.add_argument("--stage2_num_beams", type=int, default=1, help="stage2 beam size")
    parser.add_argument("--run_tag", type=str, default="", help="optional run tag for segregated artifacts")
    return parser


if __name__ == "__main__":
    main()

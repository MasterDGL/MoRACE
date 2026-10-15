import argparse
import os

from morace.rule_aggregation import aggregate_rule_candidates, load_rule_artifact, save_rule_artifact
from morace.generation_config import RuleSummaryConfig
from morace.datasets import canonical_dataset_and_subtask
from morace.model_backend import normalize_model_name
from morace.paths import with_run_tag, resolve_artifact_path


def resolve_existing_path(primary_path: str, legacy_path: str, config: RuleSummaryConfig) -> str:
    return resolve_artifact_path(primary_path, legacy_path, run_tag=config.run_tag, inherit_environment=False)


def get_input_path(dataset: str, subtask: str, config: RuleSummaryConfig) -> str:
    model_tag = normalize_model_name(config.input_model_folder)
    input_root = with_run_tag(config.input_folder, config.run_tag, inherit_environment=False)
    legacy_root = config.input_folder
    if dataset == "qm9" and subtask:
        filename = f"{model_tag}_qm9_{subtask}_dk_response_sample_{config.list_num}.json"
    elif subtask:
        filename = f"{model_tag}_{dataset}_{subtask}_dk_response_sample_{config.list_num}.json"
    else:
        filename = f"{model_tag}_{dataset}_dk_response_sample_{config.list_num}.json"
    primary = os.path.join(input_root, model_tag, dataset, filename)
    legacy = os.path.join(legacy_root, model_tag, dataset, filename)
    return resolve_existing_path(primary, legacy, config)


def get_output_path(dataset: str, subtask: str, config: RuleSummaryConfig) -> str:
    model_tag = normalize_model_name(config.input_model_folder)
    output_dir = with_run_tag(config.output_folder, config.run_tag, inherit_environment=False)
    output_dir = os.path.join(output_dir, model_tag, dataset)
    os.makedirs(output_dir, exist_ok=True)
    if dataset == "qm9" and subtask:
        filename = f"{model_tag}_qm9_{subtask}_dk_response_sample_{config.list_num}_summarized_rules.json"
    elif subtask:
        filename = f"{model_tag}_{dataset}_{subtask}_dk_response_sample_{config.list_num}_summarized_rules.json"
    else:
        filename = f"{model_tag}_{dataset}_dk_response_sample_{config.list_num}_summarized_rules.json"
    return os.path.join(output_dir, filename)


def write_text_summary(path: str, selected_rules):
    lines = []
    for idx, rule in enumerate(selected_rules, 1):
        threshold = rule.get("threshold")
        threshold_text = ""
        if threshold:
            threshold_text = f" [{threshold['op']} {threshold['value']}]"
        lines.append(
            f"Rule {idx}: {rule['feature_family']}::{rule['feature_name']}{threshold_text} "
            f"| source={rule['source']} | support={rule.get('support_count', 1)} "
            f"| p={rule.get('p_value', 1.0):.4g} | effect={rule.get('effect_size', 0.0):.4g}"
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + ("\n" if lines else ""))


def main(config: RuleSummaryConfig = None):
    """Run this stage with explicit settings, or parse the command line once."""
    if config is None:
        config = RuleSummaryConfig.from_namespace(build_parser().parse_args())
    dataset, subtask = canonical_dataset_and_subtask(config.dataset, config.subtask)
    input_path = get_input_path(dataset, subtask, config)
    print(f"[Aggregate] dataset={dataset} subtask={subtask or '-'} input={input_path}")
    artifact = load_rule_artifact(input_path)
    print(f"[Aggregate] incoming_rules={len(artifact.get('rules', []))}")
    selected_rules, metrics = aggregate_rule_candidates(
        artifact.get("rules", []),
        dataset=dataset,
        subtask=subtask,
        input_folder=os.path.abspath(config.split_input_folder),
        target_rule_count=config.target_rule_count,
        source_balance=False,
        multi_consensus_promotion_enable=config.multi_consensus_promotion_enable,
    )
    payload = {
        "dataset": dataset,
        "subtask": subtask,
        "input_path": input_path,
        "run_tag": config.run_tag,
        "selected_rules": selected_rules,
        "metrics": metrics,
        "projection_metrics": artifact.get("metrics", {}),
    }
    output_path = get_output_path(dataset, subtask, config)
    save_rule_artifact(output_path, payload)
    write_text_summary(output_path.replace(".json", ".txt"), selected_rules)
    print(
        f"[Aggregate] canonical_rule_count={metrics.get('canonical_rule_count', 0)} "
        f"selected_rule_count={metrics.get('selected_rule_count', 0)} "
        f"compile_valid_rate={metrics.get('compile_valid_rate', 0.0):.3f} "
        f"strict_projector_valid_rate={artifact.get('metrics', {}).get('strict_projector_valid_rate', 0.0):.3f} "
        f"projector_valid_rate={artifact.get('metrics', {}).get('projector_valid_rate', 0.0):.3f} "
        f"heuristic_rule_rate={artifact.get('metrics', {}).get('heuristic_rule_rate', 0.0):.3f} "
        f"mollogp_collapse_rate={metrics.get('mollogp_collapse_rate', 0.0):.3f}"
    )
    if selected_rules:
        print("[Aggregate] selected rules preview:")
        for idx, rule in enumerate(selected_rules[:8], 1):
            threshold = rule.get("threshold")
            threshold_text = ""
            if threshold:
                threshold_text = f" {threshold['op']} {threshold['value']}"
            print(
                f"  - {idx}. {rule['feature_family']}::{rule['feature_name']}{threshold_text} "
                f"| support={rule.get('support_count', 1)} | p={rule.get('p_value', 1.0):.4g}"
            )
    print(f"Wrote aggregated rules to {output_path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="inference_model_response")
    parser.add_argument("--input_model_folder", type=str, default="qwen3.5-9b")
    parser.add_argument("--dataset", type=str, default="bbbp")
    parser.add_argument("--subtask", type=str, default="", help="subtask for sider/tox21/qm9")
    parser.add_argument("--list_num", type=int, default=30, help="number of lists/batches for inference")
    parser.add_argument("--output_folder", type=str, default="summarized_inference_rules", help="aggregated rule folder")
    parser.add_argument("--split_input_folder", type=str, default="data/splits", help="dataset folder for statistics")
    parser.add_argument("--scaffold_input_folder", dest="split_input_folder", help=argparse.SUPPRESS)
    parser.add_argument("--target_rule_count", type=int, default=30, help="max selected rules")
    parser.add_argument("--run_tag", type=str, default="", help="optional run tag for segregated artifacts")
    return parser


if __name__ == "__main__":
    main()

"""Explicit settings for the three generation/aggregation entry points.

Constructing a config is independent of the process environment. CLI adapters
capture supported environment defaults once in ``from_namespace``.
"""
from dataclasses import dataclass, field, fields
import os
from typing import ClassVar, Mapping

from morace.model_backend import normalize_model_name
from morace.paths import sanitize_run_tag
from morace.generation_runtime import GenerationRuntime


class StageConfiguration:
    _dataset_folder_field: ClassVar[str | None] = None
    _stability_cue: ClassVar[tuple[str, str] | None] = None

    def __post_init__(self):
        sanitize_run_tag(self.run_tag)

    @classmethod
    def from_namespace(cls, namespace, *, environment: Mapping[str, str] = None):
        environment = dict(os.environ if environment is None else environment)
        values = {field.name: getattr(namespace, field.name) for field in fields(cls)
                  if hasattr(namespace, field.name)}
        values["run_tag"] = (values.get("run_tag") or environment.get("MORACE_RUN_TAG")
                             or environment.get("LLM4SD_RUN_TAG", ""))
        values["multi_consensus_promotion_enable"] = bool(int(environment.get(
            "MULTI_CONSENSUS_PROMOTION_ENABLE", "0")))
        if cls._stability_cue:
            field_name, environment_key = cls._stability_cue
            values[field_name] = bool(int(environment.get(environment_key, "0")))
        folder_key = cls._dataset_folder_field
        folder_value = (values.get(folder_key, "data/splits") or "").strip() or "data/splits"
        if folder_key and folder_value == "data/splits":
            values[folder_key] = (environment.get("MORACE_DATASET_ROOT")
                                  or environment.get("LLM4SD_DATASET_ROOT")
                                  or environment.get("DATASET_ROOT") or "data/splits")
        if "model" in values:
            values["runtime"] = GenerationRuntime.from_environment(environment)
            model = values["model"]
            original_path = values.get("model_path", "")
            values["model_path"] = _model_path(model, original_path, environment)
            projector = values.get("projector_model") or model
            values["projector_model_path"] = _model_path(
                projector, values.get("projector_model_path") or original_path, environment)
            names = [item.strip() for item in values.get("extra_draft_models", "").split(",") if item.strip()]
            paths = values.get("extra_draft_model_paths", "").split(",")
            values["extra_draft_model_paths"] = ",".join(
                _model_path(name, paths[i].strip() if i < len(paths) else "", environment)
                for i, name in enumerate(names))
        return cls(**values)


def _model_path(model, explicit_path, environment):
    if explicit_path:
        return explicit_path
    suffix = normalize_model_name(model).upper().replace(".", "").replace("-", "_")
    return (environment.get(f"MORACE_MODEL_PATH_{suffix}")
            or environment.get(f"LLM4SD_MODEL_PATH_{suffix}")
            or environment.get("MORACE_MODEL_PATH")
            or environment.get("LLM4SD_MODEL_PATH") or model)


@dataclass(frozen=True)
class SynthesisConfig(StageConfiguration):
    _stability_cue: ClassVar[tuple[str, str]] = (
        "literature_stability_cue_enabled", "SYNTH_BBBP_LITERATURE_STABILITY_CUE_ENABLE")
    runtime: GenerationRuntime = field(default_factory=GenerationRuntime)
    dataset: str = 'bbbp'
    subtask: str = ''
    model: str = 'qwen3.5-9b'
    model_path: str = ''
    projector_model: str = ''
    projector_model_path: str = ''
    extra_draft_models: str = ''
    extra_draft_model_paths: str = ''
    drafts_per_prompt: int = 1
    seed: int = 42
    input_folder: str = 'prompt_file'
    input_file: str = 'synthesize_prompt.json'
    output_folder: str = 'synthesize_model_response'
    max_new_tokens: int = 512
    projector_max_new_tokens: int = 192
    max_prompt_tokens: int = 1536
    projector_max_prompt_tokens: int = 1792
    stage1_temperature: float = 0.35
    stage2_temperature: float = 0.0
    stage1_top_p: float = 0.9
    stage2_top_p: float = 0.9
    stage1_num_beams: int = 1
    stage2_num_beams: int = 1
    run_tag: str = ''
    literature_stability_cue_enabled: bool = False
    multi_consensus_promotion_enable: bool = False


@dataclass(frozen=True)
class InferenceConfig(StageConfiguration):
    _dataset_folder_field: ClassVar[str] = "input_folder"
    _stability_cue: ClassVar[tuple[str, str]] = (
        "mollogp_stability_cue_enabled", "INFERENCE_BBBP_MOLLOGP_STABILITY_CUE_ENABLE")
    runtime: GenerationRuntime = field(default_factory=GenerationRuntime)
    prompt_folder: str = 'prompt_file'
    prompt_file: str = 'inference_prompt.json'
    input_folder: str = 'data/splits'
    output_folder: str = 'inference_model_response'
    dataset: str = 'bbbp'
    subtask: str = ''
    list_num: int = 30
    max_samples_per_chunk: int = 0
    seed: int = 42
    model: str = 'qwen3.5-9b'
    model_path: str = ''
    projector_model: str = ''
    projector_model_path: str = ''
    extra_draft_models: str = ''
    extra_draft_model_paths: str = ''
    drafts_per_prompt: int = 1
    max_new_tokens: int = 768
    projector_max_new_tokens: int = 768
    max_prompt_tokens: int = 1200
    projector_max_prompt_tokens: int = 1600
    stage1_temperature: float = 0.45
    stage2_temperature: float = 0.0
    stage1_top_p: float = 0.92
    stage2_top_p: float = 1.0
    stage1_num_beams: int = 1
    stage2_num_beams: int = 1
    run_tag: str = ''
    mollogp_stability_cue_enabled: bool = False
    multi_consensus_promotion_enable: bool = False


@dataclass(frozen=True)
class RuleSummaryConfig(StageConfiguration):
    _dataset_folder_field: ClassVar[str] = "split_input_folder"
    input_folder: str = 'inference_model_response'
    input_model_folder: str = 'qwen3.5-9b'
    dataset: str = 'bbbp'
    subtask: str = ''
    list_num: int = 30
    output_folder: str = 'summarized_inference_rules'
    split_input_folder: str = 'data/splits'
    target_rule_count: int = 30
    run_tag: str = ''
    multi_consensus_promotion_enable: bool = False

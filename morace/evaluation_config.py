"""Immutable arguments and validation-selection settings for one evaluation run."""

from dataclasses import dataclass, field, fields
import math
from pathlib import Path
from typing import Mapping

from morace.run_configuration import validate_run_tag


@dataclass(frozen=True)
class EvaluationConfig:
    dataset: str = "bbbp"
    dataset_root: str = field(default_factory=lambda: str(Path("data/splits").resolve()))
    subtask: str = ""
    model: str = "qwen3.5-9b"
    knowledge_type: str = "all"
    list_num: int = 30
    output_dir: str = "eval_result"
    code_gen_folder: str = "eval_code_generation_repo"
    target_rule_count: int = 30
    run_tag: str = ""
    generation_seed: int | None = None
    max_ablation_features: int = 10
    validation_promotion_epsilon: float = 0.005
    keep_all_valid_delta: float = 0.01
    revert_all_valid_delta: float = 0.015
    min_all_compiled_features: int = 3
    capacity_keep_valid_delta: float = 0.01
    threshold_keep_min_gain: float = 0.01
    all_finalist_enable: bool = False
    all_finalist_max_candidates: int = 3
    all_finalist_keep_lit_delta: float = 0.004
    all_finalist_min_effective_lit: int = 2
    all_finalist_min_effective_data: int = 5
    all_finalist_min_roots: int = 6
    all_plus_one_alias_enable: bool = False
    all_plus_one_alias_max_valid_drop: float = 0.002
    all_plus_one_alias_min_promoted_multi_roots: int = 2
    lit_only_slot_control_enable: bool = False
    lit_only_slot_max_valid_drop: float = 0.001
    lit_only_slot_min_promoted_multi_roots: int = 2
    multi_consensus_promotion_enable: bool = False
    m7_lit_preserving_swap_enable: bool = False

    def __post_init__(self) -> None:
        if self.knowledge_type not in {"all", "synthesize", "inference"}:
            raise ValueError("knowledge_type must be all, synthesize, or inference")
        for name in ("dataset", "dataset_root", "subtask", "model", "output_dir", "code_gen_folder", "run_tag"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        if self.run_tag:
            validate_run_tag(self.run_tag)
        object.__setattr__(self, "dataset_root", str(Path(self.dataset_root or "data/splits").resolve()))
        if not self.dataset or not self.model or not self.output_dir or not self.code_gen_folder:
            raise ValueError("dataset, model, and output paths must be nonempty")
        for name in ("list_num", "target_rule_count", "max_ablation_features",
                     "min_all_compiled_features", "all_finalist_max_candidates",
                     "all_finalist_min_effective_lit", "all_finalist_min_effective_data",
                     "all_finalist_min_roots", "all_plus_one_alias_min_promoted_multi_roots",
                     "lit_only_slot_min_promoted_multi_roots"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.list_num < 1 or self.target_rule_count < 1 or self.all_finalist_max_candidates < 1:
            raise ValueError("list_num, target_rule_count, and all_finalist_max_candidates must be positive")
        if self.generation_seed is not None and (type(self.generation_seed) is not int or self.generation_seed < 0):
            raise ValueError("generation_seed must be a nonnegative integer or None")
        for field, (_, kind) in _POLICY_ENVIRONMENT.items():
            value = getattr(self, field)
            if kind is bool and type(value) is not bool:
                raise ValueError(f"{field} must be a boolean")
            if kind is float and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value < 0):
                raise ValueError(f"{field} must be finite and nonnegative")

    @classmethod
    def from_namespace(cls, namespace, *, environment: Mapping[str, str] | None = None):
        """Copy CLI/Namespace fields; ambient os.environ is never read implicitly."""
        allowed = {item.name for item in fields(cls)}
        values = {name: value for name, value in vars(namespace).items() if name in allowed}
        if values.get("dataset_root") is None:
            values.pop("dataset_root", None)
        if environment is not None:
            if not values.get("run_tag"):
                values["run_tag"] = (environment.get("MORACE_RUN_TAG")
                                     or environment.get("LLM4SD_RUN_TAG") or "")
            if not values.get("dataset_root"):
                values["dataset_root"] = (environment.get("MORACE_DATASET_ROOT")
                                          or environment.get("LLM4SD_DATASET_ROOT")
                                          or environment.get("DATASET_ROOT") or "data/splits")
            for field, (env_name, kind) in _POLICY_ENVIRONMENT.items():
                if env_name not in environment:
                    continue
                raw = str(environment[env_name]).strip()
                try:
                    if kind is bool:
                        if raw not in {"0", "1"}:
                            raise ValueError("expected 0 or 1")
                        values[field] = raw == "1"
                    else:
                        values[field] = kind(raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid evaluation setting {env_name}={raw!r}") from exc
        return cls(**values)


_POLICY_ENVIRONMENT = {
    "multi_consensus_promotion_enable": ("MULTI_CONSENSUS_PROMOTION_ENABLE", bool),
    "m7_lit_preserving_swap_enable": ("M7_LIT_PRESERVING_SWAP_ENABLE", bool),
    "validation_promotion_epsilon": ("ALL_PROMOTION_VALID_EPSILON", float),
    "keep_all_valid_delta": ("KEEP_ALL_VALID_DELTA", float),
    "revert_all_valid_delta": ("REVERT_ALL_VALID_DELTA", float),
    "min_all_compiled_features": ("MIN_ALL_COMPILED_FEATURES", int),
    "capacity_keep_valid_delta": ("CAPACITY_KEEP_VALID_DELTA", float),
    "threshold_keep_min_gain": ("THRESHOLD_KEEP_MIN_GAIN", float),
    "all_finalist_enable": ("ALL_FINALIST_ENABLE", bool),
    "all_finalist_max_candidates": ("ALL_FINALIST_MAX_CANDIDATES", int),
    "all_finalist_keep_lit_delta": ("ALL_FINALIST_KEEP_LIT_DELTA", float),
    "all_finalist_min_effective_lit": ("ALL_FINALIST_MIN_EFFECTIVE_LIT", int),
    "all_finalist_min_effective_data": ("ALL_FINALIST_MIN_EFFECTIVE_DATA", int),
    "all_finalist_min_roots": ("ALL_FINALIST_MIN_ROOTS", int),
    "all_plus_one_alias_enable": ("ALL_PLUS_ONE_ALIAS_ENABLE", bool),
    "all_plus_one_alias_max_valid_drop": ("ALL_PLUS_ONE_ALIAS_MAX_VALID_DROP", float),
    "all_plus_one_alias_min_promoted_multi_roots": ("ALL_PLUS_ONE_ALIAS_MIN_PROMOTED_MULTI_ROOTS", int),
    "lit_only_slot_control_enable": ("LIT_ONLY_SLOT_CONTROL_ENABLE", bool),
    "lit_only_slot_max_valid_drop": ("LIT_ONLY_SLOT_MAX_VALID_DROP", float),
    "lit_only_slot_min_promoted_multi_roots": ("LIT_ONLY_SLOT_MIN_PROMOTED_MULTI_ROOTS", int),
}

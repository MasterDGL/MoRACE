"""Validate checkpoint/runtime compatibility without loading model weights."""

from __future__ import annotations

import argparse
from morace.generation_runtime import science_env_get
import re
import sys
from pathlib import Path


def _version_tuple(value: str) -> tuple[int, int, int]:
    numbers = [int(item) for item in re.findall(r"\d+", value)[:3]]
    return tuple((numbers + [0, 0, 0])[:3])


def _trust_remote_code_enabled() -> bool:
    value = science_env_get("MORACE_TRUST_REMOTE_CODE") or science_env_get(
        "LLM4SD_TRUST_REMOTE_CODE", "0"
    )
    return value.strip().lower() in {"1", "true", "yes", "on"}


def check_model_runtime(model_name: str, model_path: str) -> dict[str, str]:
    checkpoint = Path(model_path)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint}")

    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    pickle_weights = sorted(checkpoint.glob("*.bin"))
    if pickle_weights and _version_tuple(torch.__version__) < (2, 6, 0):
        raise RuntimeError(
            f"{model_name} uses PyTorch .bin weights, but torch {torch.__version__} is below 2.6. "
            "Current Transformers releases reject this combination because of CVE-2025-32434. "
            "Use torch>=2.6 or an equivalent safetensors checkpoint; do not bypass the safety check."
        )

    trust_remote_code = _trust_remote_code_enabled()
    try:
        config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=trust_remote_code)
    except Exception as exc:
        raise RuntimeError(
            f"{model_name} configuration is incompatible with transformers {transformers.__version__}: {exc}"
        ) from exc

    try:
        model_class = AutoModelForCausalLM._model_mapping[type(config)]
    except Exception as exc:
        raise RuntimeError(
            f"{model_name} ({getattr(config, 'model_type', 'unknown')}) has no AutoModelForCausalLM "
            f"mapping in transformers {transformers.__version__}."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
    except Exception as exc:
        raise RuntimeError(f"{model_name} tokenizer could not be loaded: {exc}") from exc

    return {
        "model": model_name,
        "model_type": str(getattr(config, "model_type", "unknown")),
        "model_class": model_class.__name__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trust_remote_code": str(trust_remote_code).lower(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    result = check_model_runtime(args.model, args.model_path)
    print("Model runtime preflight passed.")
    for key, value in result.items():
        print(f"- {key}: {value}")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print(f"Model runtime preflight failed: {exc}", file=sys.stderr)
        exit_code = 1
    raise SystemExit(exit_code)

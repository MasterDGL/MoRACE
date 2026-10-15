"""Load a local checkpoint and run a deterministic generation canary."""

from __future__ import annotations

import argparse
import sys

from morace.model_backend import LocalTextGenerator
from morace.model_preflight import check_model_runtime


def run_generation_canary(
    model_name: str,
    model_path: str,
    *,
    max_new_tokens: int = 4,
) -> dict[str, str]:
    runtime = check_model_runtime(model_name, model_path)
    generator = LocalTextGenerator(model_name, model_path, seed=42)
    outputs = generator.generate_batch(
        [("Answer briefly.", "Reply with OK.")],
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        max_input_tokens=64,
    )
    if not outputs or not outputs[0].strip():
        raise RuntimeError(f"{model_name} produced no text during the generation canary")
    return {**runtime, "output": outputs[0].strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    result = run_generation_canary(
        args.model,
        args.model_path,
        max_new_tokens=args.max_new_tokens,
    )
    print("Model generation canary passed.")
    for key, value in result.items():
        print(f"- {key}: {value}")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print(f"Model generation canary failed: {exc}", file=sys.stderr)
        exit_code = 1
    raise SystemExit(exit_code)

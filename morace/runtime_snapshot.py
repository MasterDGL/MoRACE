import argparse
import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, Iterable


LEGACY_SIGNATURES = (
    "Calculate",
    "Assumed you are",
    "Assume you are a very experienced Chemist",
)


def sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_prompt_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from iter_prompt_strings(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_prompt_strings(item)


def prompt_preview(path: str, limit: int = 160) -> str:
    payload = load_json(path)
    for text in iter_prompt_strings(payload):
        cleaned = " ".join(text.split())
        if cleaned:
            return cleaned[:limit]
    return ""


def launch_script_mentions(path: str, expected: Dict[str, str]) -> Dict[str, bool]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()
    result = {}
    for key, value in expected.items():
        result[key] = bool(value) and key in content
    return result


def parse_launcher_defaults(path: str) -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    defaults: Dict[str, str] = {}
    pattern = re.compile(r'export\s+([A-Z0-9_]+)=.*:-([^}"\']+)')
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                defaults[match.group(1)] = match.group(2).strip()
    return defaults


def is_literal_launcher_default(value: str) -> bool:
    if not value:
        return False
    return "${" not in value and "$(" not in value


def launcher_expected_config(run_config: Dict[str, Any]) -> Dict[str, str]:
    resolved = run_config.get("resolved_config", {})
    expected = {
        "PROJECTOR_MODEL": str(
            resolved.get("launcher_default_projector_model")
            or run_config.get("launcher_default_projector_model", "")
        ),
        "PROJECTOR_MODEL_PATH": str(
            resolved.get("launcher_default_projector_path")
            or run_config.get("launcher_default_projector_path", "")
        ),
    }
    return {key: value for key, value in expected.items() if value}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--synthesize-prompt", required=True)
    parser.add_argument("--inference-prompt", required=True)
    parser.add_argument("--launch-script", default="")
    args = parser.parse_args()

    issues = []
    run_config = load_json(args.run_config)
    prompt_paths = {
        "synthesize_prompt.json": args.synthesize_prompt,
        "inference_prompt.json": args.inference_prompt,
    }

    stored_prompt_hashes = run_config.get("prompt_hashes", {})
    for name, path in prompt_paths.items():
        actual_hash = sha256_file(path)
        expected_hash = stored_prompt_hashes.get(name, "")
        if expected_hash and expected_hash != actual_hash:
            issues.append(f"prompt hash mismatch for {name}: {expected_hash} != {actual_hash}")
        preview = prompt_preview(path)
        print(f"[Runtime] prompt_preview {name} {preview}")
        text = open(path, "r", encoding="utf-8").read()
        for signature in LEGACY_SIGNATURES:
            if signature in text:
                issues.append(f"stale prompt detected in {name}: {signature}")

    stored_file_hashes = run_config.get("file_hashes", {})
    for path, expected_hash in stored_file_hashes.items():
        if not os.path.exists(path):
            issues.append(f"hashed file missing: {path}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            issues.append(f"file hash mismatch for {path}: {expected_hash} != {actual_hash}")
    if stored_file_hashes:
        print(f"[Runtime] file_hashes={stored_file_hashes}")

    resolved = run_config.get("resolved_config", {})
    env_expected = {
        "MODEL": os.environ.get("MODEL", ""),
        "MODEL_PATH": os.environ.get("MODEL_PATH", ""),
        "PROJECTOR_MODEL": os.environ.get("PROJECTOR_MODEL", ""),
        "PROJECTOR_MODEL_PATH": os.environ.get("PROJECTOR_MODEL_PATH", ""),
    }
    mapping = {
        "MODEL": "model",
        "MODEL_PATH": "model_path",
        "PROJECTOR_MODEL": "projector_model",
        "PROJECTOR_MODEL_PATH": "projector_model_path",
    }
    for env_key, cfg_key in mapping.items():
        env_value = env_expected[env_key]
        cfg_value = str(resolved.get(cfg_key, ""))
        if env_value and cfg_value and env_value != cfg_value:
            issues.append(f"resolved config mismatch for {env_key}: env={env_value} config={cfg_value}")

    launch_mentions = launch_script_mentions(args.launch_script, env_expected)
    launcher_defaults = parse_launcher_defaults(args.launch_script)
    expected_launcher = launcher_expected_config(run_config)
    if args.launch_script:
        is_generic_launcher = os.path.basename(args.launch_script) == "run_pipeline.sh"
        print(f"[Runtime] launch_script={args.launch_script}")
        if launch_mentions:
            print(f"[Runtime] launch_mentions={launch_mentions}")
            missing = [key for key, present in launch_mentions.items() if env_expected.get(key) and not present]
            if missing:
                issues.append(
                    "launch script does not mention current values for: " + ", ".join(sorted(missing))
                )
        else:
            issues.append(f"launch script not found: {args.launch_script}")
        if launcher_defaults and not is_generic_launcher:
            print(f"[Runtime] launcher_defaults={launcher_defaults}")
            launcher_mapping = {
                "MODEL": "model",
                "MODEL_PATH": "model_path",
                "PROJECTOR_MODEL": "projector_model",
                "PROJECTOR_MODEL_PATH": "projector_model_path",
            }
            for env_key, cfg_key in launcher_mapping.items():
                default_value = launcher_defaults.get(env_key, "")
                resolved_value = str(resolved.get(cfg_key, ""))
                if not is_literal_launcher_default(default_value):
                    continue
                if default_value and resolved_value and default_value != resolved_value:
                    enforce_qwen = str(os.environ.get("PROJECTOR_MODEL_ENFORCE", "0")) == "1"
                    if env_key.startswith("PROJECTOR") or enforce_qwen:
                        issues.append(
                            f"launcher consistency mismatch for {env_key}: launcher={default_value} resolved={resolved_value}"
                        )
    if expected_launcher:
        print(f"[Runtime] launcher_expected={expected_launcher}")
        for env_key, expected_value in expected_launcher.items():
            cfg_key = "projector_model" if env_key == "PROJECTOR_MODEL" else "projector_model_path"
            resolved_value = str(resolved.get(cfg_key, ""))
            if expected_value and resolved_value and expected_value != resolved_value:
                issues.append(
                    f"launcher expected mismatch for {env_key}: expected={expected_value} resolved={resolved_value}"
                )

    if issues:
        for issue in issues:
            print(f"[Runtime] ERROR {issue}")
        return 1

    print("[Runtime] snapshot validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

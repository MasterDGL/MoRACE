"""Create or verify path-independent SHA256 manifests for local checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


TRACKED_SUFFIXES = {".bin", ".jinja", ".json", ".model", ".safetensors", ".txt"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in TRACKED_SUFFIXES
    )


def build_checkpoint_record(root: Path) -> dict:
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {root}")
    files = checkpoint_files(root)
    if not files:
        raise ValueError(f"no checkpoint files found in {root}")
    return {
        "files": [
            {
                "path": path.name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ]
    }


def verify_checkpoint_record(model: str, root: Path, record: dict) -> None:
    expected = {str(item["path"]): item for item in record.get("files", [])}
    actual = {path.name: path for path in checkpoint_files(root)}
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(f"checkpoint file set mismatch for {model}: missing={missing} extra={extra}")
    for name, item in expected.items():
        path = actual[name]
        if path.stat().st_size != int(item["size"]):
            raise RuntimeError(f"checkpoint size mismatch for {model}/{name}")
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"checkpoint SHA256 mismatch for {model}/{name}")


def parse_models(values: list[str]) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for value in values:
        model, separator, path = value.partition("=")
        if not separator or not model or not path:
            raise ValueError(f"invalid --model {value!r}; expected MODEL=/path/to/checkpoint")
        models[model] = Path(path).expanduser().resolve()
    return models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[], metavar="MODEL=PATH", required=True)
    parser.add_argument("--write", action="store_true", help="create the manifest instead of verifying it")
    args = parser.parse_args()
    models = parse_models(args.model)

    if args.write:
        payload = {
            "schema_version": 1,
            "hash_algorithm": "sha256",
            "models": {model: build_checkpoint_record(path) for model, path in models.items()},
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Checkpoint manifest written: {args.manifest}")
        return 0

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("hash_algorithm") != "sha256":
        raise RuntimeError("unsupported checkpoint manifest schema")
    records = payload.get("models", {})
    for model, root in models.items():
        if model not in records:
            raise RuntimeError(f"checkpoint manifest has no entry for {model}")
        verify_checkpoint_record(model, root, records[model])
        print(f"Checkpoint verified: {model}")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print(f"Checkpoint verification failed: {exc}", file=sys.stderr)
        exit_code = 1
    raise SystemExit(exit_code)

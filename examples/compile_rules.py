"""CPU-only demonstration of the rule-to-feature interface; no LLM or training."""
from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from morace.rule_compiler import build_feature_matrix, compile_rules


def main():
    artifact = json.loads((Path(__file__).with_name("rules.json")).read_text(encoding="utf-8"))
    compiled, invalid = compile_rules(artifact["rules"])
    if invalid:
        raise RuntimeError(f"The example contains unsupported rules: {invalid}")
    smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CCOc1ccccc1"]
    matrix = build_feature_matrix(compiled, smiles)
    names = [rule.name for rule in compiled]
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["smiles", *names])
    for smi, row in zip(smiles, matrix):
        writer.writerow([smi, *[f"{float(value):.4f}" for value in row]])


if __name__ == "__main__":
    main()

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, MolSurf, rdMolDescriptors

from morace.rule_registry import (
    COUNT_ALIAS,
    DESCRIPTOR_CODE_REGISTRY,
    DESCRIPTOR_REGISTRY,
    get_compiled_pattern,
    get_pattern_callable,
    get_smarts,
    is_custom_pattern,
)


def compare_values(lhs: float, op: str, rhs: float) -> bool:
    if op == "<":
        return lhs < rhs
    if op == "<=":
        return lhs <= rhs
    if op == ">":
        return lhs > rhs
    if op == ">=":
        return lhs >= rhs
    if op == "==":
        return lhs == rhs
    if op == "!=":
        return lhs != rhs
    raise ValueError(f"Unsupported operator: {op}")


def sanitize_identifier(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "feature"


def _threshold_tag(rule: Dict) -> str:
    threshold = rule.get("threshold")
    if not threshold:
        return "null"
    return f"{threshold['op']}_{round(float(threshold['value']), 4)}"


def build_compiled_feature_name(rule: Dict) -> str:
    canonical = str(rule.get("canonical_key") or "")
    canonical_hash = hashlib.md5(canonical.encode("utf-8")).hexdigest()[:8] if canonical else "nocanon"
    return sanitize_identifier(
        "_".join(
            [
                str(rule.get("source", "unknown")),
                str(rule.get("feature_family", "feature")),
                str(rule.get("feature_name", "feature")),
                str(rule.get("feature_mode", "mode")),
                str(rule.get("relation_to_target", "relation")),
                _threshold_tag(rule),
                canonical_hash,
            ]
        )
    )


@dataclass
class CompiledRule:
    name: str
    rule: Dict
    fn: Callable
    output_kind: str


def compile_rule(rule: Dict) -> Optional[CompiledRule]:
    family = rule["feature_family"]
    feature_name = rule["feature_name"]
    mode = rule["feature_mode"]
    safe_name = build_compiled_feature_name(rule)

    if family == "descriptor":
        if feature_name not in DESCRIPTOR_REGISTRY:
            return None
        base_fn = DESCRIPTOR_REGISTRY[feature_name]
        if mode == "threshold_binary" and rule.get("threshold"):
            threshold = rule["threshold"]
            return CompiledRule(
                name=safe_name,
                rule=rule,
                fn=lambda mol, fn=base_fn, threshold=threshold: int(
                    compare_values(float(fn(mol)), threshold["op"], float(threshold["value"]))
                ),
                output_kind="binary",
            )
        return CompiledRule(
            name=safe_name,
            rule=rule,
            fn=lambda mol, fn=base_fn: float(fn(mol)),
            output_kind="numeric",
        )

    if family == "pattern_presence":
        pattern_fn = get_pattern_callable(feature_name)
        if pattern_fn is None:
            return None
        return CompiledRule(
            name=safe_name,
            rule=rule,
            fn=lambda mol, fn=pattern_fn: int(bool(fn(mol))),
            output_kind="binary",
        )

    if family == "count":
        if feature_name in COUNT_ALIAS:
            base_fn = COUNT_ALIAS[feature_name]
            return CompiledRule(
                name=safe_name,
                rule=rule,
                fn=lambda mol, fn=base_fn: float(fn(mol)),
                output_kind="count",
            )
        patt = get_compiled_pattern(feature_name)
        if patt is None:
            return None
        return CompiledRule(
            name=safe_name,
            rule=rule,
            fn=lambda mol, patt=patt: float(len(mol.GetSubstructMatches(patt))),
            output_kind="count",
        )
    return None


def compile_rules(rule_list: List[Dict]) -> Tuple[List[CompiledRule], List[Dict]]:
    compiled: List[CompiledRule] = []
    invalid: List[Dict] = []
    for rule in rule_list:
        item = compile_rule(rule)
        if item is None:
            invalid.append(rule)
        else:
            compiled.append(item)
    return compiled, invalid


def feature_value(compiled_rule: CompiledRule, smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return math.nan
    try:
        value = compiled_rule.fn(mol)
    except Exception:
        return math.nan
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return float(int(value))
    return float(value)


def build_feature_matrix(compiled_rules: List[CompiledRule], smiles_list: List[str]) -> List[List[float]]:
    return [[feature_value(rule, smiles) for rule in compiled_rules] for smiles in smiles_list]


def render_rule_module(rule_list: List[Dict], compiled_rules: List[CompiledRule]) -> Tuple[str, List[str]]:
    imports = [
        "from rdkit import Chem",
        "from rdkit.Chem import Crippen, Descriptors, Lipinski, MolSurf, rdMolDescriptors",
        "",
        "",
        "def _compare(lhs, op, rhs):",
        "    if op == '<':",
        "        return lhs < rhs",
        "    if op == '<=':",
        "        return lhs <= rhs",
        "    if op == '>':",
        "        return lhs > rhs",
        "    if op == '>=':",
        "        return lhs >= rhs",
        "    if op == '==':",
        "        return lhs == rhs",
        "    if op == '!=':",
        "        return lhs != rhs",
        "    raise ValueError(f'Unsupported operator: {op}')",
        "",
    ]

    if any(rule["feature_name"] == "fused_ring" for rule in rule_list):
        imports.extend(
            [
                "def _has_fused_ring(mol):",
                "    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]",
                "    for i, left in enumerate(rings):",
                "        for right in rings[i + 1:]:",
                "            if len(left & right) >= 2:",
                "                return True",
                "    return False",
                "",
            ]
        )

    smarts_defs = []
    for rule in rule_list:
        if (
            rule["feature_family"] in {"pattern_presence", "count"}
            and rule["feature_name"] not in COUNT_ALIAS
            and not is_custom_pattern(rule["feature_name"])
        ):
            smarts = get_smarts(rule["feature_name"])
            if smarts:
                const_name = f"SMARTS_{sanitize_identifier(rule['feature_name']).upper()}"
                line = f"{const_name} = Chem.MolFromSmarts({smarts!r})"
                if line not in smarts_defs:
                    smarts_defs.append(line)
    if smarts_defs:
        imports.extend(smarts_defs)
        imports.append("")

    function_names: List[str] = []
    body: List[str] = []
    for compiled in compiled_rules:
        rule = compiled.rule
        fn_name = sanitize_identifier(compiled.name)
        function_names.append(fn_name)
        rationale = rule.get("rationale", "").replace("\n", " ").strip()
        body.append(f"# {rationale}")
        if rule["feature_family"] == "descriptor":
            expr = DESCRIPTOR_CODE_REGISTRY[rule["feature_name"]]
            if rule["feature_mode"] == "threshold_binary" and rule.get("threshold"):
                threshold = rule["threshold"]
                body.extend(
                    [
                        f"def {fn_name}(mol):",
                        f"    return int(_compare(float({expr}), {threshold['op']!r}, {float(threshold['value'])!r}))",
                        "",
                    ]
                )
            else:
                body.extend(
                    [
                        f"def {fn_name}(mol):",
                        f"    return float({expr})",
                        "",
                    ]
                )
        elif rule["feature_family"] == "pattern_presence":
            if rule["feature_name"] == "aromatic_ring":
                body.extend(
                    [
                        f"def {fn_name}(mol):",
                        "    return int(rdMolDescriptors.CalcNumAromaticRings(mol) > 0)",
                        "",
                    ]
                )
            elif rule["feature_name"] == "fused_ring":
                body.extend(
                    [
                        f"def {fn_name}(mol):",
                        "    return int(_has_fused_ring(mol))",
                        "",
                    ]
                )
            else:
                const_name = f"SMARTS_{sanitize_identifier(rule['feature_name']).upper()}"
                body.extend(
                    [
                        f"def {fn_name}(mol):",
                        f"    return int(bool(mol.HasSubstructMatch({const_name})))",
                        "",
                    ]
                )
        else:
            if rule["feature_name"] in COUNT_ALIAS:
                expr = DESCRIPTOR_CODE_REGISTRY.get(rule["feature_name"])
                if expr:
                    body.extend(
                        [
                            f"def {fn_name}(mol):",
                            f"    return float({expr})",
                            "",
                        ]
                    )
                else:
                    call_name = rule["feature_name"]
                    body.extend(
                        [
                            f"def {fn_name}(mol):",
                            f"    return float(rdMolDescriptors.CalcNumRings(mol)) if {call_name!r} == 'RingCount' else 0.0",
                            "",
                        ]
                    )
            else:
                const_name = f"SMARTS_{sanitize_identifier(rule['feature_name']).upper()}"
                body.extend(
                    [
                        f"def {fn_name}(mol):",
                        f"    return float(len(mol.GetSubstructMatches({const_name})))",
                        "",
                    ]
                )
    return "\n".join(imports + body).strip() + "\n", function_names

import json
import re
from json import JSONDecodeError
from typing import Dict, List, Optional

from morace.rule_registry import COUNT_ALIAS, DESCRIPTOR_REGISTRY, PATTERN_ALIAS, canonicalize_feature_name

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
except ImportError:  # pragma: no cover - optional at runtime
    BaseModel = None
    ConfigDict = None
    Field = None
    ValidationError = Exception


VALID_FEATURE_FAMILIES = {"descriptor", "pattern_presence", "count"}
VALID_RELATIONS = {"positive", "negative", "monotonic_up", "monotonic_down"}
VALID_THRESHOLD_OPS = {"<", "<=", ">", ">=", "==", "!="}
VALID_SOURCES = {"literature", "data"}
RELATION_ALIASES = {
    "increase": "monotonic_up",
    "increases": "monotonic_up",
    "higher": "monotonic_up",
    "greater": "monotonic_up",
    "greater than": "monotonic_up",
    "more": "monotonic_up",
    "enriched": "monotonic_up",
    "associated with active": "positive",
    "active": "positive",
    "present": "positive",
    "presence": "positive",
    "more likely": "positive",
    "lower": "monotonic_down",
    "less": "monotonic_down",
    "less than": "monotonic_down",
    "decrease": "monotonic_down",
    "decreases": "monotonic_down",
    "reduced": "monotonic_down",
    "up": "monotonic_up",
    "down": "monotonic_down",
    "positive": "positive",
    "negative": "negative",
    "pos": "positive",
    "neg": "negative",
    "yes": "positive",
    "no": "negative",
    "monotonic up": "monotonic_up",
    "monotonic down": "monotonic_down",
    "positive association": "positive",
    "negative association": "negative",
}
FAMILY_ALIASES = {
    "fragment_alias": "pattern_presence",
    "fragment alias": "pattern_presence",
    "fragment": "pattern_presence",
    "fragments": "pattern_presence",
    "pattern": "pattern_presence",
    "patterns": "pattern_presence",
    "pattern_presence": "pattern_presence",
    "pattern presence": "pattern_presence",
    "pattern-presence": "pattern_presence",
    "motif": "pattern_presence",
    "motifs": "pattern_presence",
    "substructure": "pattern_presence",
    "substructures": "pattern_presence",
    "descriptor": "descriptor",
    "descriptors": "descriptor",
    "physicochemical": "descriptor",
    "physicochemical_descriptor": "descriptor",
    "physicochemical descriptor": "descriptor",
    "property": "descriptor",
    "molecular_property": "descriptor",
    "molecular property": "descriptor",
    "count": "count",
    "counts": "count",
    "ring_count": "count",
    "ring count": "count",
    "counts or motifs": "count",
}
SOURCE_ALIASES = {
    "literature": "literature",
    "lit": "literature",
    "paper": "literature",
    "papers": "literature",
    "publication": "literature",
    "publications": "literature",
    "prior chemistry draft": "literature",
    "draft": "literature",
    "data": "data",
    "dataset": "data",
    "batch": "data",
    "training data": "data",
    "examples": "data",
}
MODE_ALIASES = {
    "": "",
    "none": "",
    "null": "",
    "value": "numeric_value",
    "numeric": "numeric_value",
    "numeric value": "numeric_value",
    "numeric_value": "numeric_value",
    "absolute": "numeric_value",
    "continuous": "numeric_value",
    "continuous value": "numeric_value",
    "range": "threshold_binary",
    "less_than": "threshold_binary",
    "greater_than": "threshold_binary",
    "threshold": "threshold_binary",
    "threshold binary": "threshold_binary",
    "threshold_binary": "threshold_binary",
    "presence": "binary_presence",
    "present": "binary_presence",
    "binary": "binary_presence",
    "binary presence": "binary_presence",
    "binary_presence": "binary_presence",
    "count": "count_value",
    "count value": "count_value",
    "count_value": "count_value",
}
MODE_DEFAULTS = {
    "descriptor": "numeric_value",
    "pattern_presence": "binary_presence",
    "count": "count_value",
}
END_JSON_MARKER = "END_JSON"


if BaseModel is not None:
    class ThresholdModel(BaseModel):
        op: str
        value: float
        unit: Optional[str] = None

        model_config = ConfigDict(extra="ignore")


    class ProvenanceModel(BaseModel):
        draft_model: Optional[str] = None
        draft_id: Optional[int] = None
        batch_id: Optional[int] = None

        model_config = ConfigDict(extra="ignore")


    class RuleModel(BaseModel):
        source: str
        feature_family: str
        feature_name: str
        feature_mode: Optional[str] = None
        relation_to_target: Optional[str] = None
        threshold: Optional[ThresholdModel] = None
        rationale: Optional[str] = None
        provenance: Optional[ProvenanceModel] = None

        model_config = ConfigDict(extra="ignore")


    class RuleEnvelopeModel(BaseModel):
        rules: List[RuleModel] = Field(default_factory=list)

        model_config = ConfigDict(extra="ignore")


def build_rule_json_schema() -> Dict:
    if BaseModel is not None:
        return RuleEnvelopeModel.model_json_schema()
    return {
        "type": "object",
        "properties": {
            "rules": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "feature_family": {"type": "string"},
                        "feature_name": {"type": "string"},
                        "feature_mode": {"type": ["string", "null"]},
                        "relation_to_target": {"type": ["string", "null"]},
                        "threshold": {"type": ["object", "null"]},
                        "rationale": {"type": ["string", "null"]},
                    },
                    "required": ["source", "feature_family", "feature_name"],
                },
            }
        },
        "required": ["rules"],
    }


def build_rule_schema_text(short_form: bool = False) -> str:
    descriptor_names = ", ".join(sorted(DESCRIPTOR_REGISTRY.keys()))
    pattern_names = ", ".join(sorted(set(PATTERN_ALIAS.keys()) | {"aromatic_ring", "fused_ring"}))
    count_names = ", ".join(sorted(set(COUNT_ALIAS.keys()) | set(PATTERN_ALIAS.keys())))
    if short_form:
        return (
            "Structured rule object requirements:\n"
            "- Output JSON only.\n"
            "- Use exactly one top-level object: {\"rules\": [...]}.\n"
            "- Do not include nested 'rules' fields inside a rule.\n"
            "- Each rule may contain only: source, feature_family, feature_name, feature_mode, relation_to_target, threshold, rationale.\n"
            "- feature_family must be one of descriptor, pattern_presence, count.\n"
            f"- descriptor feature_name must be chosen from: {descriptor_names}.\n"
            f"- pattern_presence feature_name must be chosen from: {pattern_names}.\n"
            f"- count feature_name must be chosen from: {count_names}.\n"
            "- relation_to_target must be one of positive, negative, monotonic_up, monotonic_down, or null when unclear.\n"
            "- threshold must be null unless the draft explicitly supports a threshold.\n"
            "- End the response with END_JSON.\n"
            "- Return 2 to 4 rules when supported; otherwise return the supported subset.\n"
        )
    schema_json = json.dumps(build_rule_json_schema(), ensure_ascii=False, indent=2)
    return (
        "Structured rule object requirements:\n"
        "- Return JSON only.\n"
        "- Top-level format must be {\"rules\": [...]}.\n"
        "- Use 2-6 rules only when supported by the draft.\n"
        "- feature_family must be one of descriptor, pattern_presence, count.\n"
        f"- descriptor feature_name must be chosen from: {descriptor_names}.\n"
        f"- pattern_presence feature_name must be chosen from: {pattern_names}.\n"
        f"- count feature_name must be chosen from: {count_names}.\n"
        "- relation_to_target must be one of positive, negative, monotonic_up, monotonic_down, or null when unclear.\n"
        "- threshold should be null unless the draft explicitly supports a threshold.\n"
        "- Never invent a threshold, feature, or source that does not appear in the draft.\n"
        f"- End the response with {END_JSON_MARKER}.\n"
        "JSON schema:\n"
        f"{schema_json}"
    )


def strip_code_fences(text: str) -> str:
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    return text.strip()


def truncate_at_end_json(text: str) -> str:
    marker_index = text.find(END_JSON_MARKER)
    if marker_index == -1:
        return text
    return text[:marker_index].strip()


def extract_first_json_block(text: str):
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            obj, end = decoder.raw_decode(text[idx:])
            if end > 0:
                return obj
        except JSONDecodeError:
            continue
    return None


def extract_rules_array_prefix(text: str):
    """Recover complete rule objects from a truncated {"rules": [...]} response."""
    cleaned = truncate_at_end_json(strip_code_fences(text)).strip()
    rules_key = re.search(r'"rules"\s*:', cleaned)
    if not rules_key:
        return None
    array_start = cleaned.find("[", rules_key.end())
    if array_start == -1:
        return None
    decoder = json.JSONDecoder()
    rules = []
    pos = array_start + 1
    while pos < len(cleaned):
        while pos < len(cleaned) and cleaned[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(cleaned) or cleaned[pos] == "]":
            break
        if cleaned[pos] != "{":
            pos += 1
            continue
        try:
            obj, rel_end = decoder.raw_decode(cleaned[pos:])
        except JSONDecodeError:
            break
        if isinstance(obj, dict):
            rules.append(obj)
        pos += rel_end
    if not rules:
        return None
    return {"rules": rules}


def extract_only_json_block(text: str):
    cleaned = truncate_at_end_json(strip_code_fences(text)).strip()
    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            obj, end = decoder.raw_decode(cleaned[idx:])
        except JSONDecodeError:
            continue
        prefix = cleaned[:idx].strip()
        suffix = cleaned[idx + end :].strip()
        if not prefix and not suffix:
            return obj
    return None


def repair_json_text(text: str) -> str:
    text = truncate_at_end_json(strip_code_fences(text))
    text = re.split(r"\n(?:references|notes|explanation)\b", text, flags=re.IGNORECASE)[0].strip()
    if text.startswith("rules:"):
        text = "{" + f"\"{text}" + "}"
    first_curly = text.find("{")
    last_curly = text.rfind("}")
    if first_curly != -1 and last_curly > first_curly:
        return text[first_curly : last_curly + 1]
    first_square = text.find("[")
    last_square = text.rfind("]")
    if first_square != -1 and last_square > first_square:
        return text[first_square : last_square + 1]
    return text


def _strict_profile_from_provenance(provenance: Dict) -> str:
    return str((provenance or {}).get("stage2_protocol_profile", "") or "").strip()


def _strip_llama_native_wrappers(text: str) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace("<|start_header_id|>assistant<|end_header_id|>", "")
    cleaned = cleaned.replace("<|eot_id|>", "")
    cleaned = cleaned.replace("<|eom_id|>", "")
    cleaned = cleaned.replace("<|end_of_text|>", "")
    cleaned = re.sub(r"^\s*(assistant|Assistant)\s*:?\s*", "", cleaned)
    cleaned = re.sub(r"^\s*Here is (?:the )?JSON\s*:?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*JSON\s*:?\s*", "", cleaned, flags=re.IGNORECASE)
    for marker in ("\nExplanation:", "\nNote:", "\nNotes:", "\nReasoning:"):
        idx = cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx]
    return cleaned.strip()


def normalize_threshold(threshold) -> Optional[Dict]:
    if threshold in (None, "", "null"):
        return None
    if not isinstance(threshold, dict):
        return None
    op = str(threshold.get("op", "")).strip()
    value = threshold.get("value")
    if op not in VALID_THRESHOLD_OPS:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    unit = threshold.get("unit")
    if unit is not None:
        unit = str(unit).strip() or None
    return {"op": op, "value": value, "unit": unit}


def _clean_free_text(value) -> str:
    return " ".join(str(value or "").replace("\n", " ").replace("\r", " ").split()).strip()


def _normalize_source_like_value(value, fallback: str) -> tuple[str, bool]:
    cleaned = _clean_free_text(value)
    normalized_clean = cleaned.lower().replace("-", " ").replace("_", " ")
    normalized_clean = " ".join(normalized_clean.split())
    if normalized_clean in SOURCE_ALIASES:
        return SOURCE_ALIASES[normalized_clean], False
    if any(token in normalized_clean for token in ("prior chemistry draft", "determinant", "rationale", "task brief")):
        return fallback, True
    if not normalized_clean:
        return fallback, False
    return fallback, normalized_clean not in VALID_SOURCES


def normalize_rule_with_meta(rule: Dict, source: str, provenance: Dict) -> tuple[Optional[Dict], Dict]:
    meta = {"failure_stage": "", "failure_reason": "", "source_polluted": False}
    if not isinstance(rule, dict):
        meta["failure_stage"] = "envelope_validation"
        meta["failure_reason"] = "rules_missing"
        return None, meta
    family_key = " ".join(str(rule.get("feature_family", "")).replace("-", " ").replace("_", " ").split()).lower()
    feature_family = FAMILY_ALIASES.get(family_key, None)
    if feature_family not in VALID_FEATURE_FAMILIES:
        meta["failure_stage"] = "strict_semantic_validation"
        meta["failure_reason"] = "unknown_or_unmapped_field_value"
        return None, meta
    feature_name = canonicalize_feature_name(feature_family, str(rule.get("feature_name", "")).strip())
    if not feature_name:
        meta["failure_stage"] = "strict_semantic_validation"
        meta["failure_reason"] = "unknown_or_unmapped_field_value"
        return None, meta

    relation = " ".join(str(rule.get("relation_to_target", "")).replace("-", " ").replace("_", " ").split()).lower()
    relation = RELATION_ALIASES.get(relation, relation)
    if relation not in VALID_RELATIONS:
        relation = None

    threshold = normalize_threshold(rule.get("threshold"))
    feature_mode = " ".join(str(rule.get("feature_mode", "")).replace("-", "_").split()).strip().lower()
    feature_mode = MODE_ALIASES.get(feature_mode, feature_mode or MODE_DEFAULTS[feature_family])
    if feature_family == "descriptor":
        if threshold and feature_mode not in {"threshold_binary", "numeric_value"}:
            feature_mode = "threshold_binary"
        if not threshold and feature_mode == "threshold_binary":
            feature_mode = "numeric_value"
    elif feature_family == "pattern_presence":
        feature_mode = "binary_presence"
        threshold = None
    else:
        feature_mode = "count_value"
        threshold = None

    rationale = str(rule.get("rationale", "")).strip()
    if not rationale:
        rationale = f"{feature_name} was identified as potentially relevant."

    source_value, source_polluted = _normalize_source_like_value(
        rule.get("source", rule.get("primary_source", source)),
        fallback=source,
    )
    meta["source_polluted"] = bool(source_polluted)
    if source_polluted:
        meta["failure_stage"] = "rule_normalization"
        meta["failure_reason"] = "source_polluted"

    normalized = {
        "source": source,
        "primary_source": source_value,
        "feature_family": feature_family,
        "feature_name": feature_name,
        "feature_mode": feature_mode,
        "relation_to_target": relation,
        "threshold": threshold,
        "rationale": rationale,
        "extraction_source": str(rule.get("extraction_source", "projected")).strip() or "projected",
        "provenance": {
            "draft_model": provenance.get("draft_model"),
            "draft_id": provenance.get("draft_id"),
            "batch_id": provenance.get("batch_id"),
        },
    }
    return normalized, meta


def normalize_rule(rule: Dict, source: str, provenance: Dict) -> Optional[Dict]:
    normalized, _ = normalize_rule_with_meta(rule, source=source, provenance=provenance)
    return normalized


def _validate_envelope(obj):
    if BaseModel is None:
        return obj
    if isinstance(obj, list):
        obj = {"rules": obj}
    return RuleEnvelopeModel.model_validate(obj).model_dump()


def collect_rule_dicts(obj, depth: int = 0, max_depth: int = 8):
    if depth > max_depth:
        return
    if isinstance(obj, dict):
        if "feature_family" in obj and "feature_name" in obj:
            yield obj
        if depth == 0 and "rules" in obj:
            yield from collect_rule_dicts(obj.get("rules"), depth=depth + 1, max_depth=max_depth)
            return
        for value in obj.values():
            yield from collect_rule_dicts(value, depth=depth + 1, max_depth=max_depth)
    elif isinstance(obj, list):
        for item in obj:
            yield from collect_rule_dicts(item, depth=depth + 1, max_depth=max_depth)


def analyze_rule_structure(obj, depth: int = 0) -> Dict:
    max_nesting_depth = depth
    rules_field_inside_rule_count = 0
    if isinstance(obj, dict):
        if depth > 0 and "rules" in obj:
            rules_field_inside_rule_count += 1
        for value in obj.values():
            stats = analyze_rule_structure(value, depth + 1)
            max_nesting_depth = max(max_nesting_depth, stats["max_nesting_depth"])
            rules_field_inside_rule_count += stats["rules_field_inside_rule_count"]
    elif isinstance(obj, list):
        for item in obj:
            stats = analyze_rule_structure(item, depth + 1)
            max_nesting_depth = max(max_nesting_depth, stats["max_nesting_depth"])
            rules_field_inside_rule_count += stats["rules_field_inside_rule_count"]
    return {
        "max_nesting_depth": max_nesting_depth,
        "rules_field_inside_rule_count": rules_field_inside_rule_count,
    }


def _parse_projected_rules_from_candidates(
    raw_output: str,
    source: str,
    provenance: Dict,
    *,
    strict: bool,
) -> tuple[List[Dict], Dict]:
    raw_output = truncate_at_end_json(raw_output)
    if strict and _strict_profile_from_provenance(provenance) == "llama_native":
        raw_output = _strip_llama_native_wrappers(raw_output)
    strict_failure = {"failure_stage": "", "failure_reason": "", "failure_label": ""}
    if strict:
        candidates = [extract_only_json_block(raw_output)]
    else:
        candidates = [
            extract_first_json_block(raw_output),
            extract_first_json_block(repair_json_text(raw_output)),
            extract_rules_array_prefix(raw_output),
            extract_rules_array_prefix(repair_json_text(raw_output)),
        ]
    normalized_rules: List[Dict] = []
    parse_meta = {
        "max_nesting_depth": 0,
        "rules_field_inside_rule_count": 0,
        "failure_stage": "",
        "failure_reason": "",
        "failure_label": "",
    }
    if strict and candidates[0] is None:
        if extract_first_json_block(raw_output) is not None or extract_rules_array_prefix(raw_output) is not None:
            strict_failure["failure_stage"] = "json_extraction"
            strict_failure["failure_reason"] = "trailing_text_or_prefix"
        elif any(char in raw_output for char in "{["):
            strict_failure["failure_stage"] = "json_decode"
            strict_failure["failure_reason"] = "json_decode_failed"
        else:
            strict_failure["failure_stage"] = "json_extraction"
            strict_failure["failure_reason"] = "no_pure_json_block"
    for obj in candidates:
        if obj is None:
            continue
        stats = analyze_rule_structure(obj)
        parse_meta["max_nesting_depth"] = max(parse_meta["max_nesting_depth"], stats["max_nesting_depth"])
        parse_meta["rules_field_inside_rule_count"] += stats["rules_field_inside_rule_count"]
        try:
            candidate_obj = _validate_envelope(obj)
        except ValidationError:
            if strict and not strict_failure["failure_reason"]:
                strict_failure["failure_stage"] = "envelope_validation"
                strict_failure["failure_reason"] = "envelope_invalid"
            candidate_obj = obj
        found_any_rule_dict = False
        had_source_pollution = False
        for item in collect_rule_dicts(candidate_obj):
            found_any_rule_dict = True
            normalized, normalize_meta = normalize_rule_with_meta(item, source=source, provenance=provenance)
            had_source_pollution = had_source_pollution or bool(normalize_meta.get("source_polluted"))
            if normalized is not None:
                normalized_rules.append(normalized)
            elif strict and not strict_failure["failure_reason"]:
                strict_failure["failure_stage"] = str(normalize_meta.get("failure_stage", "") or "strict_semantic_validation")
                strict_failure["failure_reason"] = str(
                    normalize_meta.get("failure_reason", "") or "unknown_or_unmapped_field_value"
                )
        if normalized_rules:
            if strict and had_source_pollution and not strict_failure["failure_reason"]:
                strict_failure["failure_stage"] = "rule_normalization"
                strict_failure["failure_reason"] = "source_polluted"
            parse_meta["failure_stage"] = strict_failure["failure_stage"]
            parse_meta["failure_reason"] = strict_failure["failure_reason"]
            parse_meta["failure_label"] = (
                f"{strict_failure['failure_stage']}::{strict_failure['failure_reason']}"
                if strict_failure["failure_stage"] and strict_failure["failure_reason"]
                else ""
            )
            return normalized_rules, parse_meta
        if strict and not found_any_rule_dict and not strict_failure["failure_reason"]:
            strict_failure["failure_stage"] = "envelope_validation"
            strict_failure["failure_reason"] = "rules_missing"
    if strict and not strict_failure["failure_reason"]:
        strict_failure["failure_stage"] = "rule_normalization"
        strict_failure["failure_reason"] = "rules_empty_after_normalization"
    parse_meta["failure_stage"] = strict_failure["failure_stage"]
    parse_meta["failure_reason"] = strict_failure["failure_reason"]
    parse_meta["failure_label"] = (
        f"{strict_failure['failure_stage']}::{strict_failure['failure_reason']}"
        if strict_failure["failure_stage"] and strict_failure["failure_reason"]
        else ""
    )
    return [], parse_meta


def strict_parse_projected_rules(raw_output: str, source: str, provenance: Dict) -> tuple[List[Dict], Dict]:
    return _parse_projected_rules_from_candidates(raw_output, source, provenance, strict=True)


def lenient_parse_projected_rules(raw_output: str, source: str, provenance: Dict) -> tuple[List[Dict], Dict]:
    return _parse_projected_rules_from_candidates(raw_output, source, provenance, strict=False)


def parse_projected_rules(raw_output: str, source: str, provenance: Dict) -> List[Dict]:
    rules, _ = lenient_parse_projected_rules(raw_output, source, provenance)
    return rules


DESCRIPTOR_PATTERNS = {
    "MolWt": ["molecular weight", "molwt", "mw"],
    "MolLogP": ["logp", "clogp", "lipophilicity"],
    "TPSA": ["tpsa", "polar surface area", "topological polar surface area"],
    "NumHBD": ["hydrogen bond donor", "hydrogen-bond donor", "h bond donor"],
    "NumHBA": ["hydrogen bond acceptor", "hydrogen-bond acceptor", "h bond acceptor"],
    "NumRotatableBonds": ["rotatable bond"],
    "NumAromaticRings": ["aromatic ring count", "number of aromatic rings"],
    "FractionCSP3": ["fraction csp3", "csp3 fraction"],
}

PATTERN_PATTERNS = {
    "halogen": ["halogen", "halogenated"],
    "nitro_group": ["nitro", "nitro group"],
    "alcohol": ["alcohol", "hydroxyl"],
    "amine": ["amine", "amino"],
    "amide": ["amide"],
    "carbonyl": ["carbonyl"],
    "carboxyl": ["carboxyl", "carboxylic acid"],
    "ester": ["ester"],
    "ether": ["ether"],
    "aromatic_ring": ["aromatic ring"],
    "fused_ring": ["fused ring", "fused bicyclic"],
}


def infer_relation(snippet: str) -> str:
    lowered = snippet.lower()
    if any(token in lowered for token in ["lower", "decrease", "reduced", "smaller", "under", "below"]):
        return "monotonic_down"
    if any(token in lowered for token in ["higher", "increase", "elevated", "larger", "above", "greater"]):
        return "monotonic_up"
    return "positive"


def split_candidate_sentences(text: str) -> List[str]:
    raw_parts = re.split(r"[\n\r]+|(?<=[.!?;])\s+|(?=Determinant\s+[A-Z0-9]+:)", text)
    cleaned = []
    for part in raw_parts:
        part = " ".join(part.split()).strip(" -\t")
        if len(part) >= 12:
            cleaned.append(part)
    return cleaned


def extract_sentence_threshold(sentence: str) -> Optional[Dict]:
    match = re.search(
        r"(<=|>=|<|>|under|below|less than|above|greater than)\s*(\d+(?:\.\d+)?)",
        sentence.lower(),
    )
    if not match:
        return None
    op_raw = match.group(1)
    value = float(match.group(2))
    if op_raw in {"under", "below", "less than"}:
        op = "<="
    elif op_raw in {"above", "greater than"}:
        op = ">="
    else:
        op = op_raw
    return {"op": op, "value": value, "unit": None}


def heuristic_extract_rules_from_draft(
    draft_text: str,
    source: str,
    provenance: Dict,
    max_rules: int = 6,
) -> List[Dict]:
    rules: List[Dict] = []
    seen = set()
    for sentence in split_candidate_sentences(draft_text):
        lowered = sentence.lower()
        local_hits: List[Dict] = []
        for feature_name, tokens in DESCRIPTOR_PATTERNS.items():
            if any(token in lowered for token in tokens):
                local_hits.append({"feature_family": "descriptor", "feature_name": feature_name})
        for feature_name, tokens in PATTERN_PATTERNS.items():
            if any(token in lowered for token in tokens):
                local_hits.append({"feature_family": "pattern_presence", "feature_name": feature_name})
        if not local_hits:
            continue

        threshold = extract_sentence_threshold(sentence) if len(local_hits) == 1 and local_hits[0]["feature_family"] == "descriptor" else None
        for hit in local_hits:
            key = (hit["feature_family"], hit["feature_name"])
            if key in seen:
                continue
            seen.add(key)
            rules.append(
                {
                    "source": source,
                    "feature_family": hit["feature_family"],
                    "feature_name": hit["feature_name"],
                    "feature_mode": "threshold_binary" if threshold and hit["feature_family"] == "descriptor" else MODE_DEFAULTS[hit["feature_family"]],
                    "relation_to_target": infer_relation(sentence),
                    "threshold": threshold if hit["feature_family"] == "descriptor" else None,
                    "rationale": sentence[:240],
                    "extraction_source": "heuristic",
                    "provenance": provenance,
                }
            )
            if len(rules) >= max_rules:
                return rules
    return rules

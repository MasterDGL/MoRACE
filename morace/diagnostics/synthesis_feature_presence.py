import re
from typing import Dict, Iterable, List


FOCUSED_ROOTS = [
    "descriptor::TPSA",
    "descriptor::MolLogP",
    "count::NumHBD",
]

ROOT_MENTION_PATTERNS = {
    "descriptor::TPSA": [
        r"\btpsa\b",
        r"\btopological polar surface area\b",
        r"\bpolar surface area\b",
    ],
    "descriptor::MolLogP": [
        r"\bmollogp\b",
        r"\blogp\b",
        r"\bclogp\b",
        r"\blipophilicity\b",
    ],
    "count::NumHBD": [
        r"\bnumhbd\b",
        r"\bhydrogen bond donor",
        r"\bh-bond donor",
        r"\bh bond donor",
    ],
}


def root_feature_key(rule: Dict) -> str:
    return f"{str(rule.get('feature_family', '')).strip()}::{str(rule.get('feature_name', '')).strip()}"


def build_bbbp_literature_stability_cue(*, dataset: str, source: str, enabled: bool) -> str:
    if not enabled:
        return ""
    if str(dataset).strip().lower() != "bbbp":
        return ""
    if str(source).strip().lower() != "literature":
        return ""
    return (
        "Focus on classic BBBP literature determinants when they are genuinely supported by the literature reasoning. "
        "Prioritize stable coverage of roots such as TPSA / polar surface area, MolLogP / lipophilicity, and "
        "hydrogen-bond donor related features. Treat these as high-value BBBP literature roots, but do not force "
        "unsupported rules."
    )


def _extract_root_mentions_from_drafts(stage1_drafts: Iterable[Dict], roots: List[str]) -> Dict[str, Dict]:
    mentions: Dict[str, Dict] = {}
    draft_texts = []
    for item in stage1_drafts:
        text = item.get("draft_text", "")
        if not text:
            text = item.get("response_text", "")
        if not text:
            text = item.get("text", "")
        draft_texts.append(str(text or ""))
    for root in roots:
        matched_terms = set()
        mention_count = 0
        patterns = ROOT_MENTION_PATTERNS.get(root, [])
        for text in draft_texts:
            lowered = text.lower()
            found_in_text = False
            for pattern in patterns:
                if re.search(pattern, lowered):
                    found_in_text = True
                    matched_terms.add(pattern)
            if found_in_text:
                mention_count += 1
        mentions[root] = {
            "mentioned": bool(mention_count),
            "mention_count": int(mention_count),
            "matched_terms": sorted(matched_terms),
        }
    return mentions


def extract_root_presence_from_synth_artifact(artifact: Dict, roots: List[str] | None = None) -> Dict:
    roots = list(roots or FOCUSED_ROOTS)
    raw_rules = list(artifact.get("rules", []) or [])
    canonical_rules = list(artifact.get("canonical_rules", []) or [])
    compile_safe_rules = [rule for rule in canonical_rules if bool(rule.get("compile_valid", False))]
    raw_root_keys = sorted({root_feature_key(rule) for rule in raw_rules if root_feature_key(rule)})
    canonical_root_keys = sorted({root_feature_key(rule) for rule in canonical_rules if root_feature_key(rule)})
    compile_safe_root_keys = sorted({root_feature_key(rule) for rule in compile_safe_rules if root_feature_key(rule)})
    raw_root_mentions = _extract_root_mentions_from_drafts(artifact.get("stage1_drafts", []) or [], roots)
    root_presence_summary: Dict[str, Dict] = {}
    for root in roots:
        raw_subset = [rule for rule in raw_rules if root_feature_key(rule) == root]
        canonical_subset = [rule for rule in canonical_rules if root_feature_key(rule) == root]
        compile_safe_subset = [rule for rule in compile_safe_rules if root_feature_key(rule) == root]
        root_presence_summary[root] = {
            "present_in_raw_rules": bool(raw_subset),
            "present_in_canonical_rules": bool(canonical_subset),
            "present_in_compile_safe_rules": bool(compile_safe_subset),
            "raw_rule_count": len(raw_subset),
            "canonical_rule_count": len(canonical_subset),
            "compile_safe_rule_count": len(compile_safe_subset),
            "sources": sorted({str(rule.get("source", "")).strip() for rule in canonical_subset if str(rule.get("source", "")).strip()}),
            "feature_modes": sorted({str(rule.get("feature_mode", "")).strip() for rule in canonical_subset if str(rule.get("feature_mode", "")).strip()}),
            "relations": sorted({str(rule.get("relation_to_target", "")).strip() for rule in canonical_subset if str(rule.get("relation_to_target", "")).strip()}),
            "raw_text_mentioned": bool(raw_root_mentions[root]["mentioned"]),
            "raw_text_mention_count": int(raw_root_mentions[root]["mention_count"]),
            "matched_terms": list(raw_root_mentions[root]["matched_terms"]),
        }
    return {
        "raw_root_keys": raw_root_keys,
        "canonical_root_keys": canonical_root_keys,
        "compile_safe_root_keys": compile_safe_root_keys,
        "root_presence_summary": root_presence_summary,
        "raw_root_mentions": raw_root_mentions,
    }


def build_root_presence_row(run_tag: str, presence: Dict) -> Dict:
    summary = presence.get("root_presence_summary", {})
    return {
        "run_tag": run_tag,
        "TPSA_present_in_raw": bool((summary.get("descriptor::TPSA") or {}).get("present_in_raw_rules", False)),
        "TPSA_present_in_canonical": bool((summary.get("descriptor::TPSA") or {}).get("present_in_canonical_rules", False)),
        "MolLogP_present_in_raw": bool((summary.get("descriptor::MolLogP") or {}).get("present_in_raw_rules", False)),
        "MolLogP_present_in_canonical": bool((summary.get("descriptor::MolLogP") or {}).get("present_in_canonical_rules", False)),
        "NumHBD_present_in_raw": bool((summary.get("count::NumHBD") or {}).get("present_in_raw_rules", False)),
        "NumHBD_present_in_canonical": bool((summary.get("count::NumHBD") or {}).get("present_in_canonical_rules", False)),
    }


def summarize_root_presence_frequency(rows: List[Dict]) -> Dict:
    total = max(1, len(rows))
    return {
        "TPSA_raw_presence_rate": sum(1 for row in rows if row.get("TPSA_present_in_raw")) / total,
        "TPSA_canonical_presence_rate": sum(1 for row in rows if row.get("TPSA_present_in_canonical")) / total,
        "MolLogP_raw_presence_rate": sum(1 for row in rows if row.get("MolLogP_present_in_raw")) / total,
        "MolLogP_canonical_presence_rate": sum(1 for row in rows if row.get("MolLogP_present_in_canonical")) / total,
        "NumHBD_raw_presence_rate": sum(1 for row in rows if row.get("NumHBD_present_in_raw")) / total,
        "NumHBD_canonical_presence_rate": sum(1 for row in rows if row.get("NumHBD_present_in_canonical")) / total,
    }

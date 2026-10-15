import re
from typing import Dict, Iterable, List, Sequence


FOCUSED_ROOTS = [
    "descriptor::MolLogP",
    "descriptor::TPSA",
    "descriptor::MolWt",
    "count::NumHBA",
    "count::NumHBD",
    "count::NumRotatableBonds",
]

ROOT_MENTION_PATTERNS = {
    "descriptor::MolLogP": [
        r"\bmollogp\b",
        r"\blogp\b",
        r"\bclogp\b",
        r"\blipophilicity\b",
        r"\blipophilic\b",
    ],
    "descriptor::TPSA": [
        r"\btpsa\b",
        r"\btopological polar surface area\b",
        r"\bpolar surface area\b",
    ],
    "descriptor::MolWt": [
        r"\bmolwt\b",
        r"\bmolecular weight\b",
    ],
    "count::NumHBA": [
        r"\bnumhba\b",
        r"\bhydrogen bond acceptor",
        r"\bh-bond acceptor",
        r"\bh bond acceptor",
    ],
    "count::NumHBD": [
        r"\bnumhbd\b",
        r"\bhydrogen bond donor",
        r"\bh-bond donor",
        r"\bh bond donor",
    ],
    "count::NumRotatableBonds": [
        r"\bnumrotatablebonds\b",
        r"\brotatable bond",
        r"\bconformational flexibility\b",
    ],
}


def root_feature_key(rule: Dict) -> str:
    return f"{str(rule.get('feature_family', '')).strip()}::{str(rule.get('feature_name', '')).strip()}"


def build_bbbp_inference_mollogp_stability_cue(*, dataset: str, enabled: bool) -> str:
    if not enabled:
        return ""
    if str(dataset).strip().lower() != "bbbp":
        return ""
    return (
        "When the batch evidence supports lipophilicity-related determinants, prefer expressing them as a canonical "
        "descriptor rule on MolLogP rather than leaving them as narrative discussion only. Keep the rule chemistry-grounded "
        "and do not force unsupported MolLogP claims."
    )


def _extract_root_mentions(stage1_drafts: Iterable[Dict], roots: Sequence[str]) -> Dict[str, Dict]:
    mentions: Dict[str, Dict] = {}
    texts: List[str] = []
    for item in stage1_drafts:
        text = item.get("draft_text", "")
        if not text:
            text = item.get("response_text", "")
        if not text:
            text = item.get("text", "")
        texts.append(str(text or ""))
    for root in roots:
        patterns = ROOT_MENTION_PATTERNS.get(root, [])
        matched_terms = set()
        mention_count = 0
        for text in texts:
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


def _selected_rules_from_projection(projected_records: Iterable[Dict]) -> List[Dict]:
    selected_rules: List[Dict] = []
    for item in projected_records or []:
        if not bool(item.get("selected_for_merge", False)):
            continue
        selected_rules.extend(list(item.get("parsed_rules", []) or []))
    return selected_rules


def _root_summary_for_subset(rules: Sequence[Dict]) -> Dict[str, Dict]:
    summary: Dict[str, Dict] = {}
    for root in FOCUSED_ROOTS:
        subset = [rule for rule in rules if root_feature_key(rule) == root]
        summary[root] = {
            "count": len(subset),
            "sources": sorted(
                {str(rule.get("source", "")).strip() for rule in subset if str(rule.get("source", "")).strip()}
            ),
            "feature_modes": sorted(
                {str(rule.get("feature_mode", "")).strip() for rule in subset if str(rule.get("feature_mode", "")).strip()}
            ),
            "relations": sorted(
                {
                    str(rule.get("relation_to_target", "")).strip()
                    for rule in subset
                    if str(rule.get("relation_to_target", "")).strip()
                }
            ),
        }
    return summary


def extract_root_presence_from_inference_artifact(artifact: Dict, roots: Sequence[str] | None = None) -> Dict:
    roots = list(roots or FOCUSED_ROOTS)
    raw_rules = list(artifact.get("rules", []) or [])
    canonical_rules = list(artifact.get("canonical_rules", []) or [])
    compile_safe_rules = [rule for rule in canonical_rules if bool(rule.get("compile_valid", False))]
    selected_rules = _selected_rules_from_projection(artifact.get("stage2_projection", []) or [])
    raw_root_keys = sorted({root_feature_key(rule) for rule in raw_rules if root_feature_key(rule)})
    canonical_root_keys = sorted({root_feature_key(rule) for rule in canonical_rules if root_feature_key(rule)})
    compile_safe_root_keys = sorted({root_feature_key(rule) for rule in compile_safe_rules if root_feature_key(rule)})
    selected_root_keys = sorted({root_feature_key(rule) for rule in selected_rules if root_feature_key(rule)})
    raw_root_mentions = _extract_root_mentions(artifact.get("stage1_drafts", []) or [], roots)
    raw_summary = _root_summary_for_subset(raw_rules)
    canonical_summary = _root_summary_for_subset(canonical_rules)
    compile_safe_summary = _root_summary_for_subset(compile_safe_rules)
    selected_summary = _root_summary_for_subset(selected_rules)
    root_presence_summary: Dict[str, Dict] = {}
    for root in roots:
        root_presence_summary[root] = {
            "present_in_raw_rules": bool(raw_summary[root]["count"]),
            "present_in_canonical_rules": bool(canonical_summary[root]["count"]),
            "present_in_compile_safe_rules": bool(compile_safe_summary[root]["count"]),
            "present_in_selected_rules": bool(selected_summary[root]["count"]),
            "raw_rule_count": int(raw_summary[root]["count"]),
            "canonical_rule_count": int(canonical_summary[root]["count"]),
            "compile_safe_rule_count": int(compile_safe_summary[root]["count"]),
            "selected_rule_count": int(selected_summary[root]["count"]),
            "sources": sorted(
                set(
                    raw_summary[root]["sources"]
                    + canonical_summary[root]["sources"]
                    + compile_safe_summary[root]["sources"]
                    + selected_summary[root]["sources"]
                )
            ),
            "feature_modes": sorted(
                set(
                    raw_summary[root]["feature_modes"]
                    + canonical_summary[root]["feature_modes"]
                    + compile_safe_summary[root]["feature_modes"]
                    + selected_summary[root]["feature_modes"]
                )
            ),
            "relations": sorted(
                set(
                    raw_summary[root]["relations"]
                    + canonical_summary[root]["relations"]
                    + compile_safe_summary[root]["relations"]
                    + selected_summary[root]["relations"]
                )
            ),
            "raw_text_mentioned": bool(raw_root_mentions[root]["mentioned"]),
            "raw_text_mention_count": int(raw_root_mentions[root]["mention_count"]),
            "matched_terms": list(raw_root_mentions[root]["matched_terms"]),
        }
    return {
        "raw_root_keys": raw_root_keys,
        "canonical_root_keys": canonical_root_keys,
        "compile_safe_root_keys": compile_safe_root_keys,
        "selected_root_keys": selected_root_keys,
        "root_presence_summary": root_presence_summary,
        "raw_root_mentions": raw_root_mentions,
    }


def build_root_presence_row(run_tag: str, presence: Dict) -> Dict:
    summary = presence.get("root_presence_summary", {})
    return {
        "run_tag": run_tag,
        "MolLogP_present_in_raw": bool((summary.get("descriptor::MolLogP") or {}).get("present_in_raw_rules", False)),
        "MolLogP_present_in_canonical": bool(
            (summary.get("descriptor::MolLogP") or {}).get("present_in_canonical_rules", False)
        ),
        "MolLogP_present_in_compile_safe": bool(
            (summary.get("descriptor::MolLogP") or {}).get("present_in_compile_safe_rules", False)
        ),
        "MolLogP_present_in_selected": bool(
            (summary.get("descriptor::MolLogP") or {}).get("present_in_selected_rules", False)
        ),
        "TPSA_present_in_raw": bool((summary.get("descriptor::TPSA") or {}).get("present_in_raw_rules", False)),
        "TPSA_present_in_canonical": bool(
            (summary.get("descriptor::TPSA") or {}).get("present_in_canonical_rules", False)
        ),
        "TPSA_present_in_compile_safe": bool(
            (summary.get("descriptor::TPSA") or {}).get("present_in_compile_safe_rules", False)
        ),
        "TPSA_present_in_selected": bool(
            (summary.get("descriptor::TPSA") or {}).get("present_in_selected_rules", False)
        ),
    }


def summarize_root_presence_frequency(rows: List[Dict]) -> Dict:
    total = max(1, len(rows))
    return {
        "MolLogP_raw_presence_rate": sum(1 for row in rows if row.get("MolLogP_present_in_raw")) / total,
        "MolLogP_canonical_presence_rate": sum(1 for row in rows if row.get("MolLogP_present_in_canonical")) / total,
        "MolLogP_compile_safe_presence_rate": sum(1 for row in rows if row.get("MolLogP_present_in_compile_safe")) / total,
        "MolLogP_selected_presence_rate": sum(1 for row in rows if row.get("MolLogP_present_in_selected")) / total,
        "TPSA_raw_presence_rate": sum(1 for row in rows if row.get("TPSA_present_in_raw")) / total,
        "TPSA_canonical_presence_rate": sum(1 for row in rows if row.get("TPSA_present_in_canonical")) / total,
        "TPSA_compile_safe_presence_rate": sum(1 for row in rows if row.get("TPSA_present_in_compile_safe")) / total,
        "TPSA_selected_presence_rate": sum(1 for row in rows if row.get("TPSA_present_in_selected")) / total,
    }

"""Projection failure accounting and output-discipline diagnostics."""

from collections import defaultdict
import json
from typing import Dict, List, Sequence

from morace.schemas import END_JSON_MARKER
from morace.stage2_protocol import STAGE2_PROTOCOL_PROFILE_VERSIONS


def normalize_failure_counter(counter: Dict[str, int]) -> Dict[str, int]:
    return {str(key): int(value) for key, value in counter.items() if int(value) > 0}


def new_strict_failure_diagnostics() -> Dict:
    return {
        "total_failures": 0,
        "stage_counts": {},
        "reason_counts": {},
        "label_counts": {},
        "by_stage": {},
    }


def truncate_sample_preview(text: str, limit: int = 1200) -> str:
    compact = str(text or "").replace("\r", "")
    if len(compact) <= limit:
        return compact
    return compact[:limit]


def audit_first_json_output(text: str) -> Dict[str, object]:
    cleaned = str(text or "").strip()
    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        suffix = cleaned[idx + end :].strip()
        suffix_after_marker = suffix
        if suffix_after_marker.startswith(END_JSON_MARKER):
            suffix_after_marker = suffix_after_marker[len(END_JSON_MARKER) :].strip()
        return {
            "first_json_parseable": True,
            "json_closes_before_suffix": bool(suffix),
            "suffix_after_first_json": suffix_after_marker,
        }
    return {
        "first_json_parseable": False,
        "json_closes_before_suffix": False,
        "suffix_after_first_json": "",
    }


def profile_matches(item: Dict, profile: str) -> bool:
    return str(item.get("stage2_protocol_profile", "") or "") == str(profile or "")


def _sample_signature(item: Dict, raw_preview: str, repair_preview: str) -> str:
    return "||".join(
        [
            str(item.get("source", "") or ""),
            str((item.get("strict_failure", {}) or {}).get("label", "") or ""),
            raw_preview,
            repair_preview,
        ]
    )


def collect_llama_native_strict_failure_samples(projected_records: Sequence[Dict], branch_name: str) -> Dict:
    by_source: Dict[str, List[Dict]] = {}
    seen_by_source: Dict[str, set[str]] = defaultdict(set)
    source_counts: Dict[str, int] = defaultdict(int)
    for item in reversed(list(projected_records)):
        if str(item.get("stage2_protocol_profile", "") or "") != "llama_native":
            continue
        if bool(item.get("strict_projector_valid")):
            continue
        failure = item.get("strict_failure", {}) or {}
        label = str(failure.get("label", "") or "")
        if not label:
            continue
        source = str(item.get("source", "") or "unknown")
        if source_counts[source] >= 10:
            continue
        raw_preview = truncate_sample_preview(item.get("projector_raw_output", ""), limit=1200)
        repair_preview = truncate_sample_preview(item.get("projector_repair_output", ""), limit=1200)
        signature = _sample_signature(item, raw_preview, repair_preview)
        if signature in seen_by_source[source]:
            continue
        seen_by_source[source].add(signature)
        source_counts[source] += 1
        by_source.setdefault(source, []).append(
            {
                "branch": str(branch_name or ""),
                "source": source,
                "stage": str(failure.get("stage", "") or ""),
                "reason": str(failure.get("reason", "") or ""),
                "label": label,
                "prompt_id": str(item.get("prompt_id", "") or ""),
                "draft_id": int(item.get("draft_id", 0) or 0),
                "raw_output_preview": raw_preview,
                "repair_output_preview": repair_preview,
            }
        )
    return {key: list(reversed(value)) for key, value in by_source.items()}


def summarize_llama_native_output_discipline(projected_records: Sequence[Dict], branch_name: str) -> Dict:
    llama_records = [item for item in projected_records if str(item.get("stage2_protocol_profile", "") or "") == "llama_native"]
    if not llama_records:
        return {}
    raw_outputs = [str(item.get("projector_raw_output", "") or "") for item in llama_records]
    return {
        "branch": str(branch_name or ""),
        "profile": "llama_native",
        "profile_version": STAGE2_PROTOCOL_PROFILE_VERSIONS["llama_native"],
        "record_count": len(llama_records),
        "strict_valid_count": sum(1 for item in llama_records if bool(item.get("strict_projector_valid"))),
        "strict_fail_count": sum(1 for item in llama_records if not bool(item.get("strict_projector_valid"))),
        "starts_with_json_object_count": sum(1 for text in raw_outputs if text.lstrip().startswith("{")),
        "ends_with_closing_brace_count": sum(1 for text in raw_outputs if text.rstrip().endswith("}")),
        "contains_code_fence_count": sum(1 for text in raw_outputs if "```" in text),
        "contains_end_json_count": sum(1 for text in raw_outputs if END_JSON_MARKER in text),
    }


def record_strict_failure(counter: Dict, failure: Dict | None) -> None:
    if not failure:
        return
    stage = str(failure.get("stage", "") or "").strip()
    reason = str(failure.get("reason", "") or "").strip()
    label = str(failure.get("label", "") or "").strip()
    if not (stage or reason or label):
        return
    counter["total_failures"] = int(counter.get("total_failures", 0) or 0) + 1
    stage_counts = counter.setdefault("stage_counts", {})
    reason_counts = counter.setdefault("reason_counts", {})
    label_counts = counter.setdefault("label_counts", {})
    by_stage = counter.setdefault("by_stage", {})
    if stage:
        stage_counts[stage] = int(stage_counts.get(stage, 0) or 0) + 1
    if reason:
        reason_counts[reason] = int(reason_counts.get(reason, 0) or 0) + 1
    if label:
        label_counts[label] = int(label_counts.get(label, 0) or 0) + 1
    if stage and reason:
        stage_bucket = by_stage.setdefault(stage, {})
        stage_bucket[reason] = int(stage_bucket.get(reason, 0) or 0) + 1

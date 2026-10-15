from typing import Dict, List, Sequence

from morace.diagnostics.logp_ingestion import (
    extract_all_gated_mollogp_objects,
    extract_consensus_group_state,
    extract_inference_selected_mollogp_objects,
    extract_synth_canonical_mollogp_objects,
)
from morace.diagnostics.logp_promotion import TARGET_ROOT, load_run_artifacts


def _has_source_evidence(item: Dict, source_name: str) -> bool:
    bucket = str(item.get("source_bucket", "")).strip()
    support_sources = {str(value).strip() for value in (item.get("support_sources") or []) if str(value).strip()}
    if bucket == "multi":
        return True
    if bucket == source_name:
        return True
    return source_name in support_sources


def _gated_row_with_match_counts(
    row: Dict,
    inference_objects: Sequence[Dict],
    synth_objects: Sequence[Dict],
    consensus_state: Dict,
) -> Dict:
    inference_exact = [
        item for item in inference_objects if item.get("canonical_key", "") == row.get("canonical_key", "")
    ]
    inference_alias = [
        item for item in inference_objects if item.get("alias_group_key", "") == row.get("alias_group_key", "")
    ]
    synth_exact = [
        item for item in synth_objects if item.get("canonical_key", "") == row.get("canonical_key", "")
    ]
    synth_alias = [
        item for item in synth_objects if item.get("alias_group_key", "") == row.get("alias_group_key", "")
    ]
    has_data = _has_source_evidence(row, "data")
    has_literature = _has_source_evidence(row, "literature")
    looks_dual = has_data and has_literature
    looks_lit_overwrite = (
        (bool(inference_exact) or bool(inference_alias))
        and (bool(synth_exact) or bool(synth_alias))
        and has_literature
        and not has_data
    )
    enriched = dict(row)
    enriched.update(
        {
            "matched_inference_objects_by_exact_key_count": len(inference_exact),
            "matched_inference_objects_by_alias_group_count": len(inference_alias),
            "matched_synth_objects_by_exact_key_count": len(synth_exact),
            "matched_synth_objects_by_alias_group_count": len(synth_alias),
            "looks_like_merged_dual_source_object": bool(looks_dual),
            "looks_like_literature_overwrite_of_data_object": bool(looks_lit_overwrite),
            "looks_like_data_object_missing_after_merge": False,
            "consensus_group_has_cross_source_overlap": bool(
                consensus_state.get("consensus_group_has_cross_source_overlap", False)
            ),
        }
    )
    return enriched


def _premerge_row(item: Dict, layer: str, run_tag: str, *, data_missing_after_merge: bool) -> Dict:
    row = dict(item)
    row["run_tag"] = run_tag
    row["layer"] = layer
    row["matched_inference_objects_by_exact_key_count"] = 0
    row["matched_inference_objects_by_alias_group_count"] = 0
    row["matched_synth_objects_by_exact_key_count"] = 0
    row["matched_synth_objects_by_alias_group_count"] = 0
    row["looks_like_merged_dual_source_object"] = False
    row["looks_like_literature_overwrite_of_data_object"] = False
    row["looks_like_data_object_missing_after_merge"] = bool(data_missing_after_merge)
    return row


def _data_side_object_retained(gated_rows: Sequence[Dict]) -> bool:
    return any(
        (row.get("matched_inference_objects_by_exact_key_count", 0) > 0 or row.get("matched_inference_objects_by_alias_group_count", 0) > 0)
        and _has_source_evidence(row, "data")
        for row in gated_rows
    )


def _dual_source_object_retained(gated_rows: Sequence[Dict]) -> bool:
    return any(bool(row.get("looks_like_merged_dual_source_object", False)) for row in gated_rows)


def _bookkeeping_issue_detected(gated_rows: Sequence[Dict]) -> bool:
    return any(bool(row.get("looks_like_literature_overwrite_of_data_object", False)) for row in gated_rows)


def _gated_has_source(gated_rows: Sequence[Dict], source_name: str) -> bool:
    return any(_has_source_evidence(row, source_name) for row in gated_rows)


def _merge_bookkeeping_summary(
    inference_objects: Sequence[Dict],
    synth_objects: Sequence[Dict],
    gated_rows: Sequence[Dict],
    consensus_state: Dict,
) -> Dict:
    gated_has_data = _gated_has_source(gated_rows, "data")
    gated_has_literature = _gated_has_source(gated_rows, "literature")
    gated_has_multi = any(str(row.get("source_bucket", "")).strip() == "multi" for row in gated_rows)
    data_side_retained = _data_side_object_retained(gated_rows)
    dual_source_retained = _dual_source_object_retained(gated_rows)
    bookkeeping_issue = _bookkeeping_issue_detected(gated_rows)

    if not inference_objects:
        primary = "no_premerge_inference_object"
        route = "upstream_inference_presence_audit"
    elif not synth_objects:
        primary = "no_premerge_synth_object"
        route = "upstream_synth_presence_audit"
    elif not gated_rows:
        primary = "all_gated_missing_mollogp"
        route = "all_input_candidate_selection_audit"
    elif not data_side_retained:
        primary = "merge_retention_gap"
        route = "merge_retention_refinement"
    elif bookkeeping_issue:
        primary = "source_bookkeeping_gap"
        route = "source_bookkeeping_refinement"
    elif gated_has_data and gated_has_literature and not dual_source_retained:
        primary = "dual_source_consolidation_gap"
        route = "dual_source_consolidation_refinement"
    elif not bool(consensus_state.get("consensus_group_has_cross_source_overlap", False)):
        primary = "consensus_ingestion_gap"
        route = "consensus_ingestion_refinement"
    else:
        primary = "present"
        route = "present_or_upstream_gap"

    return {
        "premerge_inference_object_count": len(inference_objects),
        "premerge_synth_object_count": len(synth_objects),
        "gated_object_count": len(gated_rows),
        "gated_has_data": bool(gated_has_data),
        "gated_has_literature": bool(gated_has_literature),
        "gated_has_multi": bool(gated_has_multi),
        "data_side_object_retained": bool(data_side_retained),
        "dual_source_object_retained": bool(dual_source_retained),
        "bookkeeping_issue_detected": bool(bookkeeping_issue),
        "primary_gap_reason": primary,
        "suggested_next_route": route,
    }


def audit_merge_bookkeeping(
    project_root: str,
    run_tags: Sequence[str],
    *,
    model_name: str,
    dataset: str,
    sample_dir: str = "sample_30",
) -> Dict:
    rows: List[Dict] = []
    per_run: Dict[str, Dict] = {}
    for run_tag in run_tags:
        artifacts = load_run_artifacts(
            project_root,
            run_tag,
            model_name=model_name,
            dataset=dataset,
            sample_dir=sample_dir,
        )
        inference_objects = extract_inference_selected_mollogp_objects(artifacts["inference_artifact"], run_tag)
        synth_objects = extract_synth_canonical_mollogp_objects(artifacts["synth_artifact"], run_tag)
        consensus_state = extract_consensus_group_state(artifacts["all_artifact"])
        gated_base_rows = extract_all_gated_mollogp_objects(
            artifacts["all_artifact"],
            inference_objects,
            synth_objects,
            run_tag,
        )
        gated_rows = [
            _gated_row_with_match_counts(item, inference_objects, synth_objects, consensus_state)
            for item in gated_base_rows
        ]
        for item in inference_objects:
            rows.append(
                _premerge_row(
                    item,
                    "premerge_inference_selected",
                    run_tag,
                    data_missing_after_merge=not any(
                        _has_source_evidence(row, "data")
                        and (
                            row.get("matched_inference_objects_by_exact_key_count", 0) > 0
                            or row.get("matched_inference_objects_by_alias_group_count", 0) > 0
                        )
                        for row in gated_rows
                    ),
                )
            )
        for item in synth_objects:
            rows.append(
                _premerge_row(
                    item,
                    "premerge_synth_canonical",
                    run_tag,
                    data_missing_after_merge=False,
                )
            )
        rows.extend(gated_rows)
        per_run[run_tag] = {
            "inference_objects": inference_objects,
            "synth_objects": synth_objects,
            "gated_objects": gated_rows,
            "consensus_state": consensus_state,
            "summary": _merge_bookkeeping_summary(
                inference_objects,
                synth_objects,
                gated_rows,
                consensus_state,
            ),
        }
    return {
        "project_root": project_root,
        "model_name": model_name,
        "dataset": dataset,
        "sample_dir": sample_dir,
        "run_tags": list(run_tags),
        "root_feature_key": TARGET_ROOT,
        "rows": rows,
        "per_run": per_run,
    }

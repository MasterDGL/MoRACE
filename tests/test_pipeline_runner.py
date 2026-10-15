import importlib
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest

from contextlib import ExitStack
from unittest.mock import patch

from morace import pipeline_runner as dccd_runner
from morace import projection_prompts, projection_selection


class PipelineUnitCase(unittest.TestCase):
    """Scope the original dependency fakes to each unit test, in any import order.

    These tests inspect prompt policy blocks and synthetic rule control flow.
    The isolated projection contract harness separately checks real schema text,
    parsing and compilation with fixed response digests.
    """

    def run(self, result=None):
        empty_parse = lambda *args, **kwargs: ([], {"max_nesting_depth": 0, "rules_field_inside_rule_count": 0})
        fake_compile = lambda rule: object() if rule.get("compile_ok", True) else None
        runner_fakes = {
            "LocalTextGenerator": object,
            "ModelSpec": object,
            "normalize_model_name": lambda model: str(model),
            "infer_model_family": lambda model, path="": "qwen" if "qwen" in str(model).lower() else "generic",
            "is_non_qwen_family": lambda family: family not in {"qwen"},
            "compile_rule": fake_compile,
            "heuristic_extract_rules_from_draft": lambda *args, **kwargs: [],
            "strict_parse_projected_rules": empty_parse,
            "lenient_parse_projected_rules": empty_parse,
        }
        with ExitStack() as scope:
            for name, value in runner_fakes.items():
                scope.enter_context(patch.object(dccd_runner, name, value))
            scope.enter_context(patch.object(projection_prompts, "build_rule_schema_text", lambda short_form=True: ""))
            scope.enter_context(patch.object(projection_selection, "compile_rule", fake_compile))
            return super().run(result)


def _record(prompt_id: str, draft_id: int, score_bias: float, feature_name: str, compile_ok: bool = True):
    return {
        "prompt_id": prompt_id,
        "draft_model": "qwen3.5-9b",
        "draft_id": draft_id,
        "batch_id": None,
        "draft_text": feature_name,
        "schema_valid": True,
        "strict_projector_valid": True,
        "projector_valid": True,
        "repair_used": False,
        "heuristic_rescued": False,
        "stage1_leakage_score": 0.0,
        "stage2_leakage_score": 0.0,
        "stage2_logprob_norm": score_bias,
        "parsed_rules": [
            {
                "feature_family": "descriptor",
                "feature_name": feature_name,
                "feature_mode": "continuous",
                "relation_to_target": "negative",
                "compile_ok": compile_ok,
            }
        ],
    }


class LateSelectionSurvivorTests(PipelineUnitCase):
    def setUp(self):
        self._old_keep_topk = dccd_runner.LATE_SELECTION_KEEP_TOPK
        self._old_delta = dccd_runner.LATE_SELECTION_MAX_SCORE_DELTA
        self._old_survivors = dccd_runner.LATE_SELECTION_MAX_SURVIVORS_PER_PROMPT
        self._old_novelty = dccd_runner.LATE_SELECTION_REQUIRE_NOVELTY

    def tearDown(self):
        dccd_runner.LATE_SELECTION_KEEP_TOPK = self._old_keep_topk
        dccd_runner.LATE_SELECTION_MAX_SCORE_DELTA = self._old_delta
        dccd_runner.LATE_SELECTION_MAX_SURVIVORS_PER_PROMPT = self._old_survivors
        dccd_runner.LATE_SELECTION_REQUIRE_NOVELTY = self._old_novelty

    def test_rank1_is_always_selected_for_merge(self):
        dccd_runner.LATE_SELECTION_KEEP_TOPK = 1
        dccd_runner.LATE_SELECTION_MAX_SCORE_DELTA = 0.0
        dccd_runner.LATE_SELECTION_MAX_SURVIVORS_PER_PROMPT = 1
        dccd_runner.LATE_SELECTION_REQUIRE_NOVELTY = True
        records = [
            _record("p1", 0, 1.0, "TPSA"),
            _record("p1", 1, 0.8, "MolWt"),
        ]
        selected = dccd_runner.select_best_projected_records(records)
        rank1 = next(item for item in selected if item["draft_rank"] == 1)
        self.assertTrue(rank1["selected_for_merge"])
        self.assertEqual(rank1["selected_for_merge_reason"], "rank1_primary")

    def test_selected_for_merge_count_never_exceeds_survivor_cap_per_prompt(self):
        dccd_runner.LATE_SELECTION_KEEP_TOPK = 2
        dccd_runner.LATE_SELECTION_MAX_SCORE_DELTA = 0.30
        dccd_runner.LATE_SELECTION_MAX_SURVIVORS_PER_PROMPT = 2
        dccd_runner.LATE_SELECTION_REQUIRE_NOVELTY = True
        records = [
            _record("p1", 0, 1.0, "TPSA"),
            _record("p1", 1, 0.8, "MolWt"),
            _record("p1", 2, 0.79, "NumHBA"),
            _record("p2", 0, 1.0, "TPSA"),
            _record("p2", 1, 0.8, "MolWt"),
            _record("p2", 2, 0.79, "NumHBA"),
        ]
        selected = dccd_runner.select_best_projected_records(records)
        by_prompt = {}
        for item in selected:
            by_prompt.setdefault(item["prompt_id"], []).append(item)
        for prompt_items in by_prompt.values():
            kept = sum(1 for item in prompt_items if item["selected_for_merge"])
            self.assertLessEqual(kept, dccd_runner.LATE_SELECTION_MAX_SURVIVORS_PER_PROMPT)

    def test_c3_independent_kept_variant_survives_late_selection(self):
        dccd_runner.LATE_SELECTION_KEEP_TOPK = 1
        dccd_runner.LATE_SELECTION_MAX_SCORE_DELTA = 0.0
        dccd_runner.LATE_SELECTION_MAX_SURVIVORS_PER_PROMPT = 1
        dccd_runner.LATE_SELECTION_REQUIRE_NOVELTY = True
        base = {
            **_record("p1", 0, 1.0, "MolLogP"),
            "stage2_protocol_profile": "chat_native_core",
            "rule_count_policy": "two_independent_single_rule_projections",
            "projection_variant_id": 0,
            "projection_variant_role": "base_single_rule",
        }
        independent = {
            **_record("p1", 0, 0.1, "aromatic_ring"),
            "stage2_protocol_profile": "chat_native_core",
            "rule_count_policy": "two_independent_single_rule_projections",
            "projection_variant_id": 1,
            "projection_variant_role": "independent_single_rule_2",
            "independent_single_projection_kept": True,
        }
        selected = dccd_runner.select_best_projected_records([base, independent])
        independent_out = next(item for item in selected if item["projection_variant_id"] == 1)
        self.assertTrue(independent_out["selected_for_merge"])
        self.assertEqual(
            independent_out["selected_for_merge_reason"],
            "independent_single_rule_forced_survivor",
        )

    def test_c3_independent_unkept_variant_is_not_forced(self):
        dccd_runner.LATE_SELECTION_KEEP_TOPK = 1
        dccd_runner.LATE_SELECTION_MAX_SCORE_DELTA = 0.0
        dccd_runner.LATE_SELECTION_MAX_SURVIVORS_PER_PROMPT = 1
        dccd_runner.LATE_SELECTION_REQUIRE_NOVELTY = True
        base = {
            **_record("p1", 0, 1.0, "MolLogP"),
            "stage2_protocol_profile": "chat_native_core",
            "rule_count_policy": "two_independent_single_rule_projections",
            "projection_variant_id": 0,
            "projection_variant_role": "base_single_rule",
        }
        independent = {
            **_record("p1", 0, 0.1, "aromatic_ring"),
            "stage2_protocol_profile": "chat_native_core",
            "rule_count_policy": "two_independent_single_rule_projections",
            "projection_variant_id": 1,
            "projection_variant_role": "independent_single_rule_2",
            "independent_single_projection_kept": False,
        }
        selected = dccd_runner.select_best_projected_records([base, independent])
        independent_out = next(item for item in selected if item["projection_variant_id"] == 1)
        self.assertFalse(independent_out["selected_for_merge"])
        self.assertEqual(independent_out["selected_for_merge_reason"], "rank_cutoff")

    def test_d5_family_target_kept_variant_survives_late_selection(self):
        dccd_runner.LATE_SELECTION_KEEP_TOPK = 1
        dccd_runner.LATE_SELECTION_MAX_SCORE_DELTA = 0.0
        dccd_runner.LATE_SELECTION_MAX_SURVIVORS_PER_PROMPT = 1
        dccd_runner.LATE_SELECTION_REQUIRE_NOVELTY = True
        base = {
            **_record("p1", 0, 1.0, "MolLogP"),
            "stage2_protocol_profile": "chat_native_core",
            "family_target_policy": "family_targeted_single_rule_variants",
            "projection_variant_id": 0,
            "projection_variant_role": "base_single_rule",
        }
        target = {
            **_record("p1", 0, 0.1, "NumHBA"),
            "stage2_protocol_profile": "chat_native_core",
            "family_target_policy": "family_targeted_single_rule_variants",
            "projection_variant_id": 1,
            "projection_variant_role": "family_target_count",
            "family_target_variant_kept": True,
            "parsed_rules": [
                {
                    "feature_family": "count",
                    "feature_name": "NumHBA",
                    "feature_mode": "count_value",
                    "relation_to_target": "negative",
                }
            ],
        }
        selected = dccd_runner.select_best_projected_records([base, target])
        target_out = next(item for item in selected if item["projection_variant_id"] == 1)
        self.assertTrue(target_out["selected_for_merge"])
        self.assertEqual(target_out["selected_for_merge_reason"], "family_target_forced_survivor")

    def test_d5_family_target_unkept_variant_is_not_forced(self):
        dccd_runner.LATE_SELECTION_KEEP_TOPK = 1
        dccd_runner.LATE_SELECTION_MAX_SCORE_DELTA = 0.0
        dccd_runner.LATE_SELECTION_MAX_SURVIVORS_PER_PROMPT = 1
        dccd_runner.LATE_SELECTION_REQUIRE_NOVELTY = True
        base = {
            **_record("p1", 0, 1.0, "MolLogP"),
            "stage2_protocol_profile": "chat_native_core",
            "family_target_policy": "family_targeted_single_rule_variants",
            "projection_variant_id": 0,
            "projection_variant_role": "base_single_rule",
        }
        target = {
            **_record("p1", 0, 0.1, "NumHBA"),
            "stage2_protocol_profile": "chat_native_core",
            "family_target_policy": "family_targeted_single_rule_variants",
            "projection_variant_id": 1,
            "projection_variant_role": "family_target_count",
            "family_target_variant_kept": False,
        }
        selected = dccd_runner.select_best_projected_records([base, target])
        target_out = next(item for item in selected if item["projection_variant_id"] == 1)
        self.assertFalse(target_out["selected_for_merge"])
        self.assertEqual(target_out["selected_for_merge_reason"], "rank_cutoff")

    def test_literature_projection_profile_triggers_on_family_collapse(self):
        rules = [
            {
                "feature_family": "descriptor",
                "feature_name": "TPSA",
                "feature_mode": "continuous",
                "relation_to_target": "negative",
            },
            {
                "feature_family": "descriptor",
                "feature_name": "MolWt",
                "feature_mode": "continuous",
                "relation_to_target": "negative",
            },
        ]
        profile = dccd_runner.literature_projection_profile(rules)
        self.assertTrue(profile["underdiverse"])
        self.assertIn("family_diversity_lt_2", profile["underdiverse_trigger_reasons"])

    def test_retry_without_novel_root_is_not_kept(self):
        base = _record("p1", 0, 1.0, "TPSA")
        original_rules = [
            {
                "feature_family": "descriptor",
                "feature_name": "TPSA",
                "feature_mode": "continuous",
                "relation_to_target": "negative",
            },
            {
                "feature_family": "descriptor",
                "feature_name": "MolWt",
                "feature_mode": "continuous",
                "relation_to_target": "negative",
            },
        ]
        retry_rules = [
            {
                "feature_family": "descriptor",
                "feature_name": "TPSA",
                "feature_mode": "threshold_binary",
                "relation_to_target": "negative",
                "threshold": {"op": "<=", "value": 90.0},
            },
            {
                "feature_family": "descriptor",
                "feature_name": "MolWt",
                "feature_mode": "continuous",
                "relation_to_target": "negative",
            },
        ]
        assessment = dccd_runner._literature_retry_assessment(
            base,
            original_rules,
            retry_rules,
            original_strict_valid=True,
            retry_strict_valid=True,
            original_repair_used=False,
            retry_repair_used=False,
            original_stage2_logprob_norm=1.0,
            retry_stage2_logprob_norm=1.0,
        )
        self.assertFalse(assessment["accepted"])
        self.assertEqual(assessment["novel_root_count"], 0)

    def test_retry_with_novel_root_and_quality_preserved_is_kept(self):
        base = _record("p1", 0, 1.0, "TPSA")
        original_rules = [
            {
                "feature_family": "descriptor",
                "feature_name": "TPSA",
                "feature_mode": "continuous",
                "relation_to_target": "negative",
            },
            {
                "feature_family": "descriptor",
                "feature_name": "MolWt",
                "feature_mode": "continuous",
                "relation_to_target": "negative",
            },
        ]
        retry_rules = [
            {
                "feature_family": "descriptor",
                "feature_name": "TPSA",
                "feature_mode": "continuous",
                "relation_to_target": "negative",
            },
            {
                "feature_family": "count",
                "feature_name": "NumHBA",
                "feature_mode": "count_value",
                "relation_to_target": "negative",
            },
        ]
        assessment = dccd_runner._literature_retry_assessment(
            base,
            original_rules,
            retry_rules,
            original_strict_valid=True,
            retry_strict_valid=True,
            original_repair_used=False,
            retry_repair_used=False,
            original_stage2_logprob_norm=1.0,
            retry_stage2_logprob_norm=1.0,
        )
        self.assertTrue(assessment["accepted"])
        self.assertGreaterEqual(assessment["novel_root_count"], 1)


class Stage2ProtocolProfileTests(PipelineUnitCase):
    def tearDown(self):
        os.environ.pop("STAGE2_PROTOCOL_PROFILE", None)
        os.environ.pop("STAGE2_TERMINATION_POLICY", None)
        os.environ.pop("CHAT_NATIVE_CORE_TERMINATION_POLICY", None)
        os.environ.pop("STAGE2_RULE_COUNT_POLICY", None)
        os.environ.pop("CHAT_NATIVE_CORE_RULE_COUNT_POLICY", None)
        os.environ.pop("STAGE2_EXTRACTION_POLICY", None)
        os.environ.pop("CHAT_NATIVE_CORE_EXTRACTION_POLICY", None)
        os.environ.pop("STAGE2_CANONICALIZATION_POLICY", None)
        os.environ.pop("CHAT_NATIVE_CORE_CANONICALIZATION_POLICY", None)
        os.environ.pop("STAGE2_FAMILY_FEATURE_POLICY", None)
        os.environ.pop("STAGE2_FAMILY_TARGET_POLICY", None)
        os.environ.pop("CHAT_NATIVE_CORE_FAMILY_TARGET_POLICY", None)
        os.environ.pop("STAGE2_FAMILY_TARGETS", None)
        os.environ.pop("CHAT_NATIVE_CORE_FAMILY_TARGETS", None)
        os.environ.pop("STAGE2_TARGET_VOCAB_POLICY", None)
        os.environ.pop("CHAT_NATIVE_CORE_TARGET_VOCAB_POLICY", None)
        os.environ.pop("STAGE2_TARGET_THINNING_POLICY", None)
        os.environ.pop("CHAT_NATIVE_CORE_TARGET_THINNING_POLICY", None)
        os.environ.pop("_MORACE_STAGE2_PROTOCOL_PROFILE", None)
        os.environ.pop("_LLM4SD_STAGE2_PROTOCOL_PROFILE", None)

    def test_auto_infers_qwen_legacy_d2(self):
        profile, source = dccd_runner.infer_stage2_protocol_profile("qwen", "qwen", False)
        self.assertEqual(profile, "qwen_legacy_d2")
        self.assertEqual(source, "auto_inferred")

    def test_qwen_legacy_d2_profile_version_documents_legacy_repair_semantics(self):
        self.assertEqual(
            dccd_runner.stage2_protocol_profile_version("qwen_legacy_d2"),
            "qwen_legacy_d2_v1",
        )

    def test_auto_infers_cross_qwen_for_non_qwen_draft(self):
        profile, source = dccd_runner.infer_stage2_protocol_profile("chemdfm", "qwen", True)
        self.assertEqual(profile, "cross_qwen_for_non_qwen_draft")
        self.assertEqual(source, "auto_inferred")

    def test_auto_infers_llama_native(self):
        profile, source = dccd_runner.infer_stage2_protocol_profile("llama", "llama", False)
        self.assertEqual(profile, "llama_native")
        self.assertEqual(source, "auto_inferred")

    def test_auto_infers_gemma_and_chemdfm_native(self):
        profile, source = dccd_runner.infer_stage2_protocol_profile("gemma", "gemma", False)
        self.assertEqual(profile, "gemma_native")
        self.assertEqual(source, "auto_inferred")
        profile, source = dccd_runner.infer_stage2_protocol_profile("chemdfm", "chemdfm", False)
        self.assertEqual(profile, "chemdfm_native")
        self.assertEqual(source, "auto_inferred")

    def test_env_override_can_select_chat_native_core(self):
        os.environ["STAGE2_PROTOCOL_PROFILE"] = "chat_native_core"
        profile, source = dccd_runner.infer_stage2_protocol_profile("llama", "llama", False)
        self.assertEqual(profile, "chat_native_core")
        self.assertEqual(source, "env_override")

    def test_generic_non_qwen_native_not_routed_to_llama_native(self):
        profile, source = dccd_runner.infer_stage2_protocol_profile("mimo", "mimo", False)
        self.assertEqual(profile, "non_qwen_native")
        self.assertEqual(source, "auto_inferred")

    def test_default_weak_policies_are_strict_off_after_env_reset(self):
        os.environ["STAGE2_PROTOCOL_PROFILE"] = "chat_native_core"
        os.environ["STAGE2_EXTRACTION_POLICY"] = "tolerant_first_json_object"
        os.environ["STAGE2_CANONICALIZATION_POLICY"] = "safe_enum_feature_canonicalization"
        os.environ["STAGE2_FAMILY_FEATURE_POLICY"] = "safe_family_feature_consistency"
        os.environ["STAGE2_FAMILY_TARGET_POLICY"] = "family_targeted_single_rule_variants"
        os.environ["STAGE2_TARGET_VOCAB_POLICY"] = "micro_canonical_options"
        os.environ["STAGE2_TARGET_THINNING_POLICY"] = "first_valid_target_rule"
        for key in [
            "STAGE2_PROTOCOL_PROFILE",
            "STAGE2_EXTRACTION_POLICY",
            "STAGE2_CANONICALIZATION_POLICY",
            "STAGE2_FAMILY_FEATURE_POLICY",
            "STAGE2_FAMILY_TARGET_POLICY",
            "STAGE2_TARGET_VOCAB_POLICY",
            "STAGE2_TARGET_THINNING_POLICY",
        ]:
            os.environ.pop(key, None)
        self.assertEqual(dccd_runner.stage2_extraction_policy(), "strict_only")
        self.assertEqual(dccd_runner.stage2_canonicalization_policy(), "strict_canonical_only")
        self.assertEqual(dccd_runner.stage2_family_feature_policy(), "strict_family")
        self.assertEqual(dccd_runner.stage2_family_target_policy(), "none")
        self.assertEqual(dccd_runner.stage2_target_vocab_policy(), "none")
        self.assertEqual(dccd_runner.stage2_target_thinning_policy(), "none")

    def test_env_override_wins(self):
        os.environ["STAGE2_PROTOCOL_PROFILE"] = "cross_qwen_for_non_qwen_draft"
        profile, source = dccd_runner.infer_stage2_protocol_profile("qwen", "qwen", False)
        self.assertEqual(profile, "cross_qwen_for_non_qwen_draft")
        self.assertEqual(source, "env_override")

    def test_invalid_env_override_raises(self):
        os.environ["STAGE2_PROTOCOL_PROFILE"] = "cross_qwen_nonqwen"
        with self.assertRaises(ValueError):
            dccd_runner.infer_stage2_protocol_profile("qwen", "qwen", False)

    def test_cross_qwen_prompt_mentions_non_qwen_noise_constraints(self):
        prompt = dccd_runner.build_stage2_user_prompt(
            task_brief="bace",
            draft_text="Determinant A: MolWt",
            source="data",
            projector_family="qwen",
            stage2_protocol_profile="cross_qwen_for_non_qwen_draft",
        )
        self.assertIn("input draft may contain envelope text", prompt)
        self.assertIn("Never copy prior draft sentences into source", prompt)

    def test_llama_native_prompt_mentions_terminal_json_constraint(self):
        prompt = dccd_runner.build_stage2_user_prompt(
            task_brief="bace",
            draft_text="Determinant A: MolWt",
            source="literature",
            projector_family="llama",
            stage2_protocol_profile="llama_native",
        )
        self.assertIn("final two characters are ]}", prompt)
        self.assertIn("Only return 2 rules when the draft clearly supports 2 clean rules", prompt)
        self.assertIn("Prefer 2 different feature families", prompt)
        self.assertIn("Prefer 1 descriptor rule and 1 non-descriptor rule", prompt)
        self.assertIn("Do not return two descriptor rules", prompt)
        self.assertIn("at most 8 words", prompt)
        self.assertIn("If no clean rule is supported, return exactly {\"rules\":[]}.", prompt)
        self.assertIn("\"feature_family\":\"<family>\"", prompt)
        self.assertNotIn("\"feature_name\":\"MolWt\"", prompt)

    def test_chat_native_core_pure_json_policy_prompt_omits_end_json(self):
        prompt = dccd_runner.build_stage2_user_prompt(
            task_brief="bace",
            draft_text="Determinant A: aromatic ring count",
            source="literature",
            projector_family="generic",
            stage2_protocol_profile="chat_native_core",
        )
        self.assertIn("Termination policy: pure_json_single_rule.", prompt)
        self.assertIn("Return exactly one JSON object only.", prompt)
        self.assertIn("If a clean rule is supported, return exactly 1 rule.", prompt)
        self.assertIn("If no clean rule is supported, return {\"rules\":[]}.", prompt)
        self.assertIn("Do not return 2 rules.", prompt)
        self.assertIn("Each rationale must be 1 to 3 words.", prompt)
        self.assertNotIn("END_JSON", prompt)
        self.assertNotIn("DeepSeek", prompt)

    def test_chat_native_core_marker_policy_prompt_mentions_marker_constraints(self):
        os.environ["STAGE2_TERMINATION_POLICY"] = "end_json_marker_single_rule"
        prompt = dccd_runner.build_stage2_user_prompt(
            task_brief="bace",
            draft_text="Determinant A: aromatic ring count",
            source="literature",
            projector_family="generic",
            stage2_protocol_profile="chat_native_core",
        )
        self.assertIn("Termination policy: end_json_marker_single_rule.", prompt)
        self.assertIn("Return exactly one JSON object followed by END_JSON.", prompt)
        self.assertIn("Output must start with {.", prompt)
        self.assertIn("Output must end with }.", prompt)
        self.assertIn("Do not write anything before {.", prompt)
        self.assertIn("After the final }, write END_JSON and stop.", prompt)
        self.assertIn("Do not write anything after END_JSON.", prompt)
        self.assertIn("If a clean rule is supported, return exactly 1 rule.", prompt)
        self.assertIn("If no clean rule is supported, return {\"rules\":[]} followed by END_JSON.", prompt)
        self.assertIn("Do not return 2 rules.", prompt)
        self.assertIn("}]}END_JSON", prompt)
        self.assertIn("Allowed feature_family: descriptor, pattern_presence, count.", prompt)
        self.assertIn("Allowed relation_to_target: positive, negative, monotonic_up, monotonic_down, null.", prompt)
        self.assertIn("Allowed feature_mode: numeric_value, threshold_binary, binary_presence, count_value.", prompt)
        self.assertIn("Compact feature_name examples: MolLogP, aromatic_ring, ring_count.", prompt)
        self.assertNotIn("NumRotatableBonds, NumAromaticRings, RingCount", prompt)
        self.assertNotIn("heteroaromatic_ring, fused_ring, nitro_group", prompt)
        self.assertNotIn("DeepSeek", prompt)
        self.assertNotIn("No END_JSON", prompt)
        self.assertIn("unsupported field or value", prompt)
        self.assertIn("Each rationale must be 1 to 3 words.", prompt)

    def test_chat_native_core_short_output_policy_prompt_uses_conditional_two_rules(self):
        os.environ["STAGE2_TERMINATION_POLICY"] = "pure_json_short_output"
        prompt = dccd_runner.build_stage2_user_prompt(
            task_brief="bace",
            draft_text="Determinant A: aromatic ring count",
            source="literature",
            projector_family="generic",
            stage2_protocol_profile="chat_native_core",
        )
        self.assertIn("Termination policy: pure_json_short_output.", prompt)
        self.assertIn("Valid single rule is better than two risky rules.", prompt)
        self.assertIn("If returning 2 rules may make the JSON longer or less certain, return 1 rule.", prompt)
        self.assertIn("Return 1 clean rule by default.", prompt)
        self.assertIn("Return 2 rules only if both are short, valid, and from different feature families.", prompt)
        self.assertIn("Never return more than 2 rules.", prompt)
        self.assertIn("Each rationale must be 1 to 3 words and a short phrase, not a sentence.", prompt)
        self.assertIn("Compact feature_name examples: MolLogP, aromatic_ring, ring_count.", prompt)
        self.assertNotIn("END_JSON", prompt)
        self.assertNotIn("DeepSeek", prompt)

    def test_chat_native_core_vocab_rescue_prompt_uses_exactly_one_and_compact_vocabulary(self):
        os.environ["STAGE2_TERMINATION_POLICY"] = "pure_json_single_vocab_rescue"
        prompt = dccd_runner.build_stage2_user_prompt(
            task_brief="bace",
            draft_text="Determinant A: aromatic ring count",
            source="literature",
            projector_family="generic",
            stage2_protocol_profile="chat_native_core",
        )
        self.assertIn("Termination policy: pure_json_single_vocab_rescue.", prompt)
        self.assertIn("If a clean rule is supported, return exactly 1 rule.", prompt)
        self.assertIn("Do not return 2 rules.", prompt)
        self.assertIn("Each rationale must be 1 to 3 words and a short noun phrase, not a sentence.", prompt)
        self.assertIn("Allowed feature_family: descriptor, pattern_presence, count.", prompt)
        self.assertIn("Allowed feature_mode: numeric_value, threshold_binary, binary_presence, count_value.", prompt)
        self.assertIn('"count" is allowed only as feature_family; for feature_mode use "count_value", never "count".', prompt)
        self.assertIn("Never use feature_mode values: presence, binary, continuous, count, numeric, boolean.", prompt)
        self.assertIn("Allowed relation_to_target: positive, negative, monotonic_up, monotonic_down, null.", prompt)
        self.assertIn('Use JSON null, not the string "null".', prompt)
        self.assertIn("Allowed descriptor feature_name:", prompt)
        self.assertIn("MolLogP", prompt)
        self.assertIn("Allowed pattern_presence feature_name:", prompt)
        self.assertIn("aromatic_ring", prompt)
        self.assertIn("Allowed count feature_name:", prompt)
        self.assertIn("Choose feature_name only from the allowed vocabulary above.", prompt)
        self.assertIn("If no allowed feature_name fits, return {\"rules\":[]}.", prompt)
        self.assertNotIn("END_JSON", prompt)
        self.assertNotIn("DeepSeek", prompt)

    def test_chat_native_core_vocab_rescue_allowed_blocks_exclude_bad_values(self):
        os.environ["STAGE2_TERMINATION_POLICY"] = "pure_json_single_vocab_rescue"
        prompt = dccd_runner.build_stage2_user_prompt(
            task_brief="bace",
            draft_text="Determinant A: aromatic ring count",
            source="literature",
            projector_family="generic",
            stage2_protocol_profile="chat_native_core",
        )
        allowed_lines = "\n".join(
            line for line in prompt.splitlines() if line.startswith("Allowed ") or line.startswith("Choose feature_name")
        )
        feature_mode_line = next(line for line in prompt.splitlines() if line.startswith("Allowed feature_mode:"))
        feature_name_lines = "\n".join(line for line in prompt.splitlines() if "feature_name:" in line)
        self.assertNotIn("phenyl_group", allowed_lines)
        self.assertNotIn("thiophene_ring", allowed_lines)
        self.assertNotIn("continuous", feature_mode_line)
        self.assertNotIn("numeric,", feature_mode_line)
        self.assertNotIn("phenyl_group", feature_name_lines)
        self.assertNotIn("thiophene_ring", feature_name_lines)

    def test_repair_prompt_uses_cross_profile_constraints(self):
        os.environ["_MORACE_STAGE2_PROTOCOL_PROFILE"] = "cross_qwen_for_non_qwen_draft"
        prompt = dccd_runner.build_stage2_repair_prompt("bad output", "literature", projector_family="qwen")
        self.assertIn("If no clean rule survives, return exactly {\"rules\":[]}.", prompt)
        self.assertIn("No markdown. No prose. No code fence. No END_JSON. No prefix. No suffix.", prompt)
        self.assertIn("Top-level object must be exactly", prompt)
        self.assertIn("Normalize polluted source text", prompt)

    def test_chat_native_core_repair_prompt_mentions_supported_vocabulary(self):
        os.environ["_MORACE_STAGE2_PROTOCOL_PROFILE"] = "chat_native_core"
        os.environ["STAGE2_TERMINATION_POLICY"] = "end_json_marker_single_rule"
        prompt = dccd_runner.build_stage2_repair_prompt("bad output", "literature", projector_family="generic")
        self.assertIn("Allowed feature_family: descriptor, pattern_presence, count.", prompt)
        self.assertIn("Allowed relation_to_target: positive, negative, monotonic_up, monotonic_down, null.", prompt)
        self.assertIn("Allowed feature_mode: numeric_value, threshold_binary, binary_presence, count_value.", prompt)
        self.assertIn("Compact feature_name examples: MolLogP, aromatic_ring, ring_count.", prompt)
        self.assertIn("After the final }, write END_JSON and stop.", prompt)
        self.assertIn("Do not write anything after END_JSON.", prompt)
        self.assertIn("If a clean rule survives, return exactly 1 rule.", prompt)
        self.assertIn("If no clean rule survives, return {\"rules\":[]} followed by END_JSON.", prompt)
        self.assertIn("Do not return 2 rules.", prompt)
        self.assertIn("Each rationale must be 1 to 3 words.", prompt)
        self.assertIn("Return only repaired JSON followed by END_JSON.", prompt)
        self.assertNotIn("No END_JSON", prompt)

    def test_chat_native_core_pure_json_repair_prompt_omits_marker(self):
        os.environ["_MORACE_STAGE2_PROTOCOL_PROFILE"] = "chat_native_core"
        prompt = dccd_runner.build_stage2_repair_prompt("bad output", "literature", projector_family="generic")
        self.assertIn("Termination policy: pure_json_single_rule.", prompt)
        self.assertIn("If a clean rule survives, return exactly 1 rule.", prompt)
        self.assertIn("If no clean rule survives, return {\"rules\":[]}.", prompt)
        self.assertIn("Each rationale must be 1 to 3 words.", prompt)
        self.assertNotIn("END_JSON", prompt)

    def test_chat_native_core_short_output_repair_prompt_does_not_force_second_rule(self):
        os.environ["_MORACE_STAGE2_PROTOCOL_PROFILE"] = "chat_native_core"
        os.environ["STAGE2_TERMINATION_POLICY"] = "pure_json_short_output"
        prompt = dccd_runner.build_stage2_repair_prompt("bad output", "literature", projector_family="generic")
        self.assertIn("Termination policy: pure_json_short_output.", prompt)
        self.assertIn("Repair the existing JSON shape only. Do not add a second rule unless it was already clearly present.", prompt)
        self.assertIn("Valid single rule is better than two risky rules.", prompt)
        self.assertIn("Never return more than 2 rules.", prompt)
        self.assertIn("Each rationale must be 1 to 3 words and a short phrase, not a sentence.", prompt)
        self.assertNotIn("END_JSON", prompt)

    def test_chat_native_core_vocab_rescue_repair_prompt_drops_unsupported_feature_names(self):
        os.environ["_MORACE_STAGE2_PROTOCOL_PROFILE"] = "chat_native_core"
        os.environ["STAGE2_TERMINATION_POLICY"] = "pure_json_single_vocab_rescue"
        prompt = dccd_runner.build_stage2_repair_prompt("bad output", "literature", projector_family="generic")
        self.assertIn("Termination policy: pure_json_single_vocab_rescue.", prompt)
        self.assertIn("Repair the existing JSON shape, enum values, and vocabulary only.", prompt)
        self.assertIn("Do not add a new rule during repair.", prompt)
        self.assertIn("If feature_name is unsupported, drop the rule.", prompt)
        self.assertIn("Do not replace unsupported feature_name with a guessed chemistry term.", prompt)
        self.assertIn("If a clean rule survives, return exactly 1 rule.", prompt)
        self.assertIn("Do not return 2 rules.", prompt)
        self.assertIn("Allowed feature_mode: numeric_value, threshold_binary, binary_presence, count_value.", prompt)
        self.assertNotIn("END_JSON", prompt)

    def test_literature_retry_prompt_uses_empty_rules_fallback(self):
        prompt = dccd_runner.build_stage2_literature_retry_prompt(
            task_brief="bace",
            draft_text="Determinant A: MolWt",
            previous_output="Here is JSON {\"rules\": []} explanation",
            current_roots=["descriptor::MolWt"],
            current_families=["descriptor"],
            projector_family="qwen",
            stage2_protocol_profile="cross_qwen_for_non_qwen_draft",
        )
        self.assertIn("If no clean new rule is supported, return exactly {\"rules\":[]}.", prompt)
        self.assertIn("No markdown. No prose. No code fence. No END_JSON. No prefix. No suffix.", prompt)
        self.assertIn("Previous malformed output to improve:", prompt)

    def test_chat_native_core_retry_prompt_mentions_marker_and_vocab_constraints(self):
        os.environ["STAGE2_TERMINATION_POLICY"] = "end_json_marker_single_rule"
        prompt = dccd_runner.build_stage2_literature_retry_prompt(
            task_brief="bace",
            draft_text="Determinant A: aromatic ring count",
            previous_output="{\"rules\":[]}",
            current_roots=["descriptor::MolWt"],
            current_families=["descriptor"],
            projector_family="generic",
            stage2_protocol_profile="chat_native_core",
        )
        self.assertIn("Output must start with {.", prompt)
        self.assertIn("After the final }, write END_JSON and stop.", prompt)
        self.assertIn("Do not write anything after END_JSON.", prompt)
        self.assertIn("If a clean new literature rule is supported, return exactly 1 rule.", prompt)
        self.assertIn("If no clean new rule is supported, return {\"rules\":[]} followed by END_JSON.", prompt)
        self.assertIn("Do not return 2 rules.", prompt)
        self.assertIn("supported schema vocabulary", prompt)
        self.assertIn("Do not invent aliases", prompt)
        self.assertIn("Each rationale must be 1 to 3 words.", prompt)
        self.assertIn("}]}END_JSON", prompt)
        self.assertNotIn("No END_JSON", prompt)

    def test_chat_native_core_pure_json_retry_prompt_omits_marker(self):
        prompt = dccd_runner.build_stage2_literature_retry_prompt(
            task_brief="bace",
            draft_text="Determinant A: aromatic ring count",
            previous_output="{\"rules\":[]}",
            current_roots=["descriptor::MolWt"],
            current_families=["descriptor"],
            projector_family="generic",
            stage2_protocol_profile="chat_native_core",
        )
        self.assertIn("Termination policy: pure_json_single_rule.", prompt)
        self.assertIn("If a clean new literature rule is supported, return exactly 1 rule.", prompt)
        self.assertIn("If no clean new rule is supported, return {\"rules\":[]}.", prompt)
        self.assertIn("Each rationale must be 1 to 3 words.", prompt)
        self.assertNotIn("END_JSON", prompt)

    def test_chat_native_core_short_output_retry_prompt_uses_conditional_two_rules(self):
        os.environ["STAGE2_TERMINATION_POLICY"] = "pure_json_short_output"
        prompt = dccd_runner.build_stage2_literature_retry_prompt(
            task_brief="bace",
            draft_text="Determinant A: aromatic ring count",
            previous_output="{\"rules\":[]}",
            current_roots=["descriptor::MolWt"],
            current_families=["descriptor"],
            projector_family="generic",
            stage2_protocol_profile="chat_native_core",
        )
        self.assertIn("Termination policy: pure_json_short_output.", prompt)
        self.assertIn("Valid single rule is better than two risky rules.", prompt)
        self.assertIn("If returning 2 rules may make the JSON longer or less certain, return 1 rule.", prompt)
        self.assertIn("Return 2 rules only if both are short, valid, and from different feature families.", prompt)
        self.assertIn("Never return more than 2 rules.", prompt)
        self.assertNotIn("END_JSON", prompt)

    def test_chat_native_core_vocab_rescue_retry_prompt_uses_exactly_one_and_vocab_constraints(self):
        os.environ["STAGE2_TERMINATION_POLICY"] = "pure_json_single_vocab_rescue"
        prompt = dccd_runner.build_stage2_literature_retry_prompt(
            task_brief="bace",
            draft_text="Determinant A: aromatic ring count",
            previous_output="{\"rules\":[]}",
            current_roots=["descriptor::MolWt"],
            current_families=["descriptor"],
            projector_family="generic",
            stage2_protocol_profile="chat_native_core",
        )
        self.assertIn("Termination policy: pure_json_single_vocab_rescue.", prompt)
        self.assertIn("If a clean new literature rule is supported, return exactly 1 rule.", prompt)
        self.assertIn("Do not return 2 rules.", prompt)
        self.assertIn("Allowed feature_family: descriptor, pattern_presence, count.", prompt)
        self.assertIn('"count" is allowed only as feature_family; for feature_mode use "count_value", never "count".', prompt)
        self.assertIn('Use JSON null, not the string "null".', prompt)
        self.assertIn("Choose feature_name only from the allowed vocabulary above.", prompt)
        self.assertNotIn("END_JSON", prompt)

    def test_chat_native_core_prompts_do_not_contain_invalid_rules_placeholder(self):
        for policy in [
            "pure_json_single_rule",
            "pure_json_single_vocab_rescue",
            "pure_json_short_output",
            "end_json_marker_single_rule",
        ]:
            os.environ["STAGE2_TERMINATION_POLICY"] = policy
            os.environ["_MORACE_STAGE2_PROTOCOL_PROFILE"] = "chat_native_core"
            user_prompt = dccd_runner.build_stage2_user_prompt(
                task_brief="bace",
                draft_text="Determinant A: MolLogP",
                source="literature",
                projector_family="generic",
                stage2_protocol_profile="chat_native_core",
            )
            repair_prompt = dccd_runner.build_stage2_repair_prompt(
                "bad output",
                "literature",
                projector_family="generic",
            )
            retry_prompt = dccd_runner.build_stage2_literature_retry_prompt(
                task_brief="bace",
                draft_text="Determinant A: MolLogP",
                previous_output='{"rules":[]}',
                current_roots=[],
                current_families=[],
                projector_family="generic",
                stage2_protocol_profile="chat_native_core",
            )
            joined = "\n".join([user_prompt, repair_prompt, retry_prompt])
            self.assertNotIn('{"rules":[.]}', joined)
            self.assertNotIn('{"rules":[...]}', joined)
            self.assertIn('Top-level object must be a JSON object with exactly one key: "rules"', joined)

    def test_chat_native_core_profile_version_is_d5_2_target_thinning_v1(self):
        self.assertEqual(
            dccd_runner.stage2_protocol_profile_version("chat_native_core"),
            "chat_native_core_d5_2_target_thinning_v1",
        )

    def test_chat_native_core_termination_policy_can_select_marker(self):
        os.environ["STAGE2_TERMINATION_POLICY"] = "end_json_marker_single_rule"
        self.assertEqual(dccd_runner.chat_native_core_termination_policy(), "end_json_marker_single_rule")
        self.assertEqual(dccd_runner._stage2_stop_strings("chat_native_core"), ["END_JSON"])

    def test_chat_native_core_termination_policy_defaults_to_pure_json(self):
        self.assertEqual(dccd_runner.chat_native_core_termination_policy(), "pure_json_single_rule")
        self.assertEqual(dccd_runner._stage2_stop_strings("chat_native_core"), [])

    def test_chat_native_core_termination_policy_can_select_short_output(self):
        os.environ["STAGE2_TERMINATION_POLICY"] = "pure_json_short_output"
        self.assertEqual(dccd_runner.chat_native_core_termination_policy(), "pure_json_short_output")
        self.assertEqual(dccd_runner._stage2_stop_strings("chat_native_core"), [])

    def test_chat_native_core_termination_policy_can_select_vocab_rescue(self):
        os.environ["STAGE2_TERMINATION_POLICY"] = "pure_json_single_vocab_rescue"
        self.assertEqual(dccd_runner.chat_native_core_termination_policy(), "pure_json_single_vocab_rescue")
        self.assertEqual(dccd_runner._stage2_stop_strings("chat_native_core"), [])

    def test_chat_native_core_rule_count_policy_defaults_to_single_rule(self):
        self.assertEqual(dccd_runner.chat_native_core_rule_count_policy(), "single_rule")

    def test_chat_native_core_rule_count_policy_can_select_two_independent(self):
        os.environ["STAGE2_RULE_COUNT_POLICY"] = "two_independent_single_rule_projections"
        self.assertEqual(
            dccd_runner.chat_native_core_rule_count_policy(),
            "two_independent_single_rule_projections",
        )

    def test_chat_native_core_extraction_policy_defaults_to_strict_only(self):
        self.assertEqual(dccd_runner.chat_native_core_extraction_policy(), "strict_only")

    def test_chat_native_core_extraction_policy_can_select_tolerant_first_json_object(self):
        os.environ["STAGE2_EXTRACTION_POLICY"] = "tolerant_first_json_object"
        self.assertEqual(dccd_runner.chat_native_core_extraction_policy(), "tolerant_first_json_object")

    def test_chat_native_core_canonicalization_policy_defaults_to_strict_only(self):
        self.assertEqual(dccd_runner.chat_native_core_canonicalization_policy(), "strict_canonical_only")

    def test_chat_native_core_canonicalization_policy_can_select_safe_policy(self):
        os.environ["STAGE2_CANONICALIZATION_POLICY"] = "safe_enum_feature_canonicalization"
        self.assertEqual(
            dccd_runner.chat_native_core_canonicalization_policy(),
            "safe_enum_feature_canonicalization",
        )

    def test_chat_native_core_canonicalization_policy_rejects_invalid_value(self):
        os.environ["STAGE2_CANONICALIZATION_POLICY"] = "alias_mapping"
        with self.assertRaises(ValueError):
            dccd_runner.chat_native_core_canonicalization_policy()

    def test_chat_native_core_family_target_policy_defaults_to_none(self):
        self.assertEqual(dccd_runner.chat_native_core_family_target_policy(), "none")

    def test_chat_native_core_family_target_policy_can_select_family_variants(self):
        os.environ["STAGE2_FAMILY_TARGET_POLICY"] = "family_targeted_single_rule_variants"
        self.assertEqual(
            dccd_runner.chat_native_core_family_target_policy(),
            "family_targeted_single_rule_variants",
        )

    def test_chat_native_core_family_target_policy_rejects_invalid_value(self):
        os.environ["STAGE2_FAMILY_TARGET_POLICY"] = "all_the_families"
        with self.assertRaises(ValueError):
            dccd_runner.chat_native_core_family_target_policy()

    def test_chat_native_core_family_targets_default_and_dedupe_order(self):
        self.assertEqual(dccd_runner.chat_native_core_family_targets(), ["pattern_presence", "count"])
        os.environ["STAGE2_FAMILY_TARGETS"] = " count, pattern_presence, count "
        self.assertEqual(dccd_runner.chat_native_core_family_targets(), ["count", "pattern_presence"])

    def test_chat_native_core_family_targets_reject_empty_or_invalid(self):
        os.environ["STAGE2_FAMILY_TARGETS"] = " , "
        with self.assertRaises(ValueError):
            dccd_runner.chat_native_core_family_targets()
        os.environ["STAGE2_FAMILY_TARGETS"] = "motif,count"
        with self.assertRaises(ValueError):
            dccd_runner.chat_native_core_family_targets()

    def test_chat_native_core_target_vocab_policy_defaults_and_validates(self):
        self.assertEqual(dccd_runner.chat_native_core_target_vocab_policy(), "none")
        os.environ["STAGE2_TARGET_VOCAB_POLICY"] = "micro_canonical_options"
        self.assertEqual(dccd_runner.chat_native_core_target_vocab_policy(), "micro_canonical_options")
        os.environ["STAGE2_TARGET_VOCAB_POLICY"] = "semantic_aliases"
        with self.assertRaises(ValueError):
            dccd_runner.chat_native_core_target_vocab_policy()

    def test_chat_native_core_target_thinning_policy_defaults_and_validates(self):
        self.assertEqual(dccd_runner.chat_native_core_target_thinning_policy(), "none")
        os.environ["STAGE2_TARGET_THINNING_POLICY"] = "first_valid_target_rule"
        self.assertEqual(dccd_runner.chat_native_core_target_thinning_policy(), "first_valid_target_rule")
        os.environ["STAGE2_TARGET_THINNING_POLICY"] = "keep_all_targets"
        with self.assertRaises(ValueError):
            dccd_runner.chat_native_core_target_thinning_policy()

    def test_chat_native_core_target_micro_vocab_options_are_registry_backed(self):
        pattern_options = dccd_runner.chat_native_core_target_micro_vocab_options("pattern_presence")
        count_options = dccd_runner.chat_native_core_target_micro_vocab_options("count")
        self.assertIn("aromatic_ring", pattern_options)
        self.assertIn("NumHBA", count_options)
        with self.assertRaises(ValueError):
            dccd_runner.chat_native_core_target_micro_vocab_options("descriptor")

    def test_chat_native_core_independent_single_prompt_is_short_and_pure_json(self):
        prompt = dccd_runner.build_stage2_independent_single_rule_prompt(
            task_brief="bace",
            draft_text="Determinant A: MolLogP",
            previous_roots=["descriptor::MolLogP"],
            previous_families=["descriptor"],
            source="literature",
        )
        self.assertIn("Projection variant role: independent_single_rule_2.", prompt)
        self.assertIn("Return exactly 1 clean JSON rule", prompt)
        self.assertIn("If no clean novel rule is supported, return {\"rules\":[]}.", prompt)
        self.assertIn("Do not return 2 rules.", prompt)
        self.assertIn("Prefer a different root.", prompt)
        self.assertIn("Prefer a different feature_family only when clearly supported.", prompt)
        self.assertIn("Validity is more important than diversity.", prompt)
        self.assertIn("Previous roots: descriptor::MolLogP", prompt)
        self.assertNotIn("END_JSON", prompt)
        self.assertNotIn("pure_json_single_vocab_rescue", prompt)
        self.assertNotIn("DeepSeek", prompt)

    def test_chat_native_core_family_target_prompt_is_short_and_family_specific(self):
        pattern_options = dccd_runner.chat_native_core_target_micro_vocab_options("pattern_presence")
        pattern_prompt = dccd_runner.build_stage2_family_target_single_rule_prompt(
            task_brief="bace",
            draft_text="Aromatic motif may matter.",
            target_family="pattern_presence",
            source="literature",
            target_vocab_policy="micro_canonical_options",
            target_vocab_options=pattern_options,
        )
        self.assertIn("Projection variant role: family_target_pattern_presence.", pattern_prompt)
        self.assertIn("feature_family must be exactly pattern_presence", pattern_prompt)
        self.assertIn("feature_name must be exactly one of: aromatic_ring", pattern_prompt)
        self.assertIn("Do not invent feature_name.", pattern_prompt)
        self.assertIn("aromaticity", pattern_prompt)
        self.assertIn("phenyl_group", pattern_prompt)
        self.assertIn("If only another family is supported, return {\"rules\":[]} instead.", pattern_prompt)
        self.assertIn("Use feature_mode exactly binary_presence.", pattern_prompt)
        self.assertIn("Return at most 1 rule.", pattern_prompt)
        self.assertNotIn("END_JSON", pattern_prompt)
        self.assertNotIn("DeepSeek", pattern_prompt)
        self.assertNotIn("2 rules", pattern_prompt)
        count_options = dccd_runner.chat_native_core_target_micro_vocab_options("count")
        count_prompt = dccd_runner.build_stage2_family_target_single_rule_prompt(
            task_brief="bace",
            draft_text="HBA count may matter.",
            target_family="count",
            source="literature",
            target_vocab_policy="micro_canonical_options",
            target_vocab_options=count_options,
        )
        self.assertIn("feature_family must be exactly count", count_prompt)
        self.assertIn("feature_name must be exactly one of: NumHBA", count_prompt)
        self.assertIn("polar_groups", count_prompt)
        self.assertIn("hbond_groups", count_prompt)
        self.assertIn("Use feature_mode exactly count_value.", count_prompt)
        self.assertNotIn("phenyl, aromatic_ring, amide", count_prompt)

    def test_family_target_candidate_assessment_keeps_valid_target_rule(self):
        rules = [
            {
                "feature_family": "count",
                "feature_name": "NumHBA",
                "feature_mode": "count_value",
                "relation_to_target": "negative",
            }
        ]
        assessment = dccd_runner._family_target_candidate_assessment(
            rules,
            "count",
            [],
            effective_strict_valid=True,
            target_vocab_policy="micro_canonical_options",
            target_vocab_options=["NumHBA", "NumHBD"],
        )
        self.assertTrue(assessment["family_target_variant_kept"])
        self.assertEqual(assessment["family_target_drop_reason"], "")
        self.assertTrue(assessment["target_family_match"])
        self.assertTrue(assessment["target_vocab_match"])
        self.assertEqual(assessment["target_feature_name"], "NumHBA")

    def test_family_target_candidate_assessment_drops_off_target_and_multi_rule(self):
        descriptor_rules = [
            {
                "feature_family": "descriptor",
                "feature_name": "MolLogP",
                "feature_mode": "numeric_value",
                "relation_to_target": "positive",
            }
        ]
        off_target = dccd_runner._family_target_candidate_assessment(
            descriptor_rules,
            "count",
            [],
            effective_strict_valid=True,
        )
        self.assertFalse(off_target["family_target_variant_kept"])
        self.assertTrue(off_target["off_target_family_drop"])
        self.assertEqual(off_target["family_target_drop_reason"], "off_target_family")
        multi_rule = dccd_runner._family_target_candidate_assessment(
            descriptor_rules + descriptor_rules,
            "descriptor",
            [],
            effective_strict_valid=True,
        )
        self.assertFalse(multi_rule["family_target_variant_kept"])
        self.assertTrue(multi_rule["multi_rule_target_violation"])
        self.assertEqual(multi_rule["family_target_drop_reason"], "multi_rule_target_violation")

    def test_family_target_micro_vocab_drops_semantic_alias(self):
        rules = [
            {
                "feature_family": "pattern_presence",
                "feature_name": "aromaticity",
                "feature_mode": "binary_presence",
                "relation_to_target": "positive",
            }
        ]
        assessment = dccd_runner._family_target_candidate_assessment(
            rules,
            "pattern_presence",
            [],
            effective_strict_valid=True,
            target_vocab_policy="micro_canonical_options",
            target_vocab_options=["aromatic_ring", "amide"],
        )
        self.assertFalse(assessment["family_target_variant_kept"])
        self.assertTrue(assessment["target_family_match"])
        self.assertFalse(assessment["target_vocab_match"])
        self.assertEqual(assessment["family_target_drop_reason"], "feature_name_not_in_target_micro_vocab")
        self.assertEqual(assessment["target_vocab_drop_reason"], "feature_name_not_in_target_micro_vocab")

    def test_stage2_family_feature_policy_defaults_and_validates(self):
        self.assertEqual(dccd_runner.stage2_family_feature_policy(), "strict_family")
        os.environ["STAGE2_FAMILY_FEATURE_POLICY"] = "safe_family_feature_consistency"
        self.assertEqual(dccd_runner.stage2_family_feature_policy(), "safe_family_feature_consistency")
        os.environ["STAGE2_FAMILY_FEATURE_POLICY"] = "semantic_alias_mapping"
        with self.assertRaises(ValueError):
            dccd_runner.stage2_family_feature_policy()

    def test_family_feature_consistency_corrects_count_like_descriptor(self):
        diagnostics = dccd_runner._blank_d4_extraction_diagnostics(
            "tolerant_first_json_object",
            "safe_enum_feature_canonicalization",
            "safe_family_feature_consistency",
        )
        cleaned = dccd_runner._d4_clean_rule(
            {
                "feature_family": "descriptor",
                "feature_name": "NumHBA",
                "feature_mode": "numeric_value",
                "relation_to_target": "positive",
                "threshold": {"op": ">", "value": 2},
                "rationale": "HBA count",
                "source": "literature",
            },
            "literature",
            diagnostics,
        )
        self.assertIsNotNone(cleaned)
        self.assertEqual(cleaned["feature_family"], "count")
        self.assertEqual(cleaned["feature_mode"], "count_value")
        self.assertIsNone(cleaned["threshold"])
        self.assertEqual(diagnostics["descriptor_to_count_corrected_count"], 1)
        self.assertEqual(diagnostics["family_feature_threshold_cleared_count"], 1)
        self.assertEqual(diagnostics["non_descriptor_rescued_by_family_consistency_count"], 1)

    def test_family_feature_consistency_does_not_guess_chemistry_alias(self):
        diagnostics = dccd_runner._blank_d4_extraction_diagnostics(
            "tolerant_first_json_object",
            "safe_enum_feature_canonicalization",
            "safe_family_feature_consistency",
        )
        cleaned = dccd_runner._d4_clean_rule(
            {
                "feature_family": "descriptor",
                "feature_name": "hydroxyl",
                "feature_mode": "numeric_value",
                "relation_to_target": "positive",
                "threshold": None,
                "rationale": "OH group",
                "source": "literature",
            },
            "literature",
            diagnostics,
        )
        self.assertIsNone(cleaned)
        self.assertEqual(diagnostics["family_feature_corrected_count"], 0)
        self.assertEqual(diagnostics["non_descriptor_rescued_by_family_consistency_count"], 0)

    def test_chemdfm_family_target_enabled_only_for_explicit_literature_route(self):
        self.assertFalse(dccd_runner._stage2_family_target_enabled("chemdfm_native", "literature"))
        os.environ["STAGE2_FAMILY_TARGET_POLICY"] = "family_targeted_single_rule_variants"
        self.assertTrue(dccd_runner._stage2_family_target_enabled("chemdfm_native", "literature"))
        self.assertFalse(dccd_runner._stage2_family_target_enabled("chemdfm_native", "data"))
        self.assertFalse(dccd_runner._stage2_family_target_enabled("gemma_native", "literature"))

    def test_target_thinning_keeps_first_valid_target_rule(self):
        old_parser = dccd_runner.strict_parse_projected_rules

        def fake_strict_parser(text, source="literature", provenance=None):
            return (
                [
                    {
                        "source": "literature",
                        "feature_family": "count",
                        "feature_name": "NumHBA",
                        "feature_mode": "count_value",
                        "relation_to_target": "negative",
                        "threshold": None,
                        "rationale": "hba count",
                    }
                ],
                {"max_nesting_depth": 1, "rules_field_inside_rule_count": 0},
            )

        dccd_runner.strict_parse_projected_rules = fake_strict_parser
        try:
            result = dccd_runner._thin_first_valid_target_rule(
                [
                    {
                        "feature_family": "count",
                        "feature_name": "NumHBA",
                        "feature_mode": "count_value",
                        "relation_to_target": "negative",
                        "threshold": None,
                        "rationale": "hba count",
                        "compile_ok": True,
                    },
                    {
                        "feature_family": "count",
                        "feature_name": "NumHBD",
                        "feature_mode": "count_value",
                        "relation_to_target": "negative",
                        "threshold": None,
                        "rationale": "hbd count",
                        "compile_ok": True,
                    },
                ],
                "count",
                ["NumHBA", "NumHBD"],
                [],
                source="literature",
                provenance={},
                thinning_policy="first_valid_target_rule",
                trigger_reason="multi_rule_target_violation",
            )
        finally:
            dccd_runner.strict_parse_projected_rules = old_parser
        self.assertTrue(result["target_thinning_attempted"])
        self.assertTrue(result["target_thinning_success"])
        self.assertEqual(result["target_thinning_selected_root"], "count::NumHBA")
        self.assertEqual(result["target_thinning_valid_candidate_count"], 1)

    def test_target_thinning_skips_off_target_then_keeps_second_valid_rule(self):
        old_parser = dccd_runner.strict_parse_projected_rules

        def fake_strict_parser(text, source="literature", provenance=None):
            return (
                [
                    {
                        "source": "literature",
                        "feature_family": "pattern_presence",
                        "feature_name": "aromatic_ring",
                        "feature_mode": "binary_presence",
                        "relation_to_target": "positive",
                        "threshold": None,
                        "rationale": "aromatic",
                    }
                ],
                {"max_nesting_depth": 1, "rules_field_inside_rule_count": 0},
            )

        dccd_runner.strict_parse_projected_rules = fake_strict_parser
        try:
            result = dccd_runner._thin_first_valid_target_rule(
                [
                    {
                        "feature_family": "descriptor",
                        "feature_name": "MolLogP",
                        "feature_mode": "numeric_value",
                        "relation_to_target": "positive",
                        "compile_ok": True,
                    },
                    {
                        "feature_family": "pattern_presence",
                        "feature_name": "aromatic_ring",
                        "feature_mode": "binary_presence",
                        "relation_to_target": "positive",
                        "threshold": None,
                        "rationale": "aromatic",
                        "compile_ok": True,
                    },
                ],
                "pattern_presence",
                ["aromatic_ring"],
                [],
                source="literature",
                provenance={},
                thinning_policy="first_valid_target_rule",
                trigger_reason="multi_rule_target_violation",
            )
        finally:
            dccd_runner.strict_parse_projected_rules = old_parser
        self.assertTrue(result["target_thinning_success"])
        self.assertEqual(result["target_thinning_selected_root"], "pattern_presence::aromatic_ring")

    def test_target_thinning_rejects_alias_compile_invalid_and_duplicate(self):
        old_parser = dccd_runner.strict_parse_projected_rules
        dccd_runner.strict_parse_projected_rules = lambda *args, **kwargs: ([], {})
        try:
            alias_result = dccd_runner._thin_first_valid_target_rule(
                [
                    {
                        "feature_family": "pattern_presence",
                        "feature_name": "aromaticity",
                        "feature_mode": "binary_presence",
                        "compile_ok": True,
                    }
                ],
                "pattern_presence",
                ["aromatic_ring"],
                [],
                source="literature",
                provenance={},
                thinning_policy="first_valid_target_rule",
                trigger_reason="feature_name_not_in_target_micro_vocab",
            )
            compile_result = dccd_runner._thin_first_valid_target_rule(
                [
                    {
                        "feature_family": "count",
                        "feature_name": "NumHBA",
                        "feature_mode": "count_value",
                        "compile_ok": False,
                    }
                ],
                "count",
                ["NumHBA"],
                [],
                source="literature",
                provenance={},
                thinning_policy="first_valid_target_rule",
                trigger_reason="multi_rule_target_violation",
            )
            duplicate_result = dccd_runner._thin_first_valid_target_rule(
                [
                    {
                        "feature_family": "count",
                        "feature_name": "NumHBA",
                        "feature_mode": "count_value",
                        "compile_ok": True,
                    }
                ],
                "count",
                ["NumHBA"],
                ["count::NumHBA"],
                source="literature",
                provenance={},
                thinning_policy="first_valid_target_rule",
                trigger_reason="multi_rule_target_violation",
            )
            no_candidate_result = dccd_runner._thin_first_valid_target_rule(
                [],
                "count",
                ["NumHBA"],
                [],
                source="literature",
                provenance={},
                thinning_policy="first_valid_target_rule",
                trigger_reason="strict_invalid",
            )
        finally:
            dccd_runner.strict_parse_projected_rules = old_parser
        self.assertFalse(alias_result["target_thinning_success"])
        self.assertEqual(alias_result["target_thinning_drop_reason"], "feature_name_not_in_target_micro_vocab")
        self.assertFalse(compile_result["target_thinning_success"])
        self.assertEqual(compile_result["target_thinning_drop_reason"], "compile_invalid")
        self.assertFalse(duplicate_result["target_thinning_success"])
        self.assertEqual(duplicate_result["target_thinning_drop_reason"], "duplicate_root")
        self.assertFalse(no_candidate_result["target_thinning_success"])
        self.assertEqual(no_candidate_result["target_thinning_drop_reason"], "strict_invalid_no_candidates")

    def test_d4_extracts_first_balanced_json_object_from_prefix_suffix(self):
        json_text, audit = dccd_runner._extract_first_balanced_json_object(
            'Here is JSON: {"rules":[{"feature_family":"descriptor"}]} trailing text'
        )
        self.assertEqual(json_text, '{"rules":[{"feature_family":"descriptor"}]}')
        self.assertTrue(audit["balanced_object_found"])
        self.assertTrue(audit["prefix_removed"])
        self.assertTrue(audit["suffix_removed"])

    def test_d4_extracts_object_inside_markdown_fence(self):
        json_text, audit = dccd_runner._extract_first_balanced_json_object(
            '```json\n{"rules":[]}\n```\nexplanation'
        )
        self.assertEqual(json_text, '{"rules":[]}')
        self.assertTrue(audit["prefix_removed"])
        self.assertTrue(audit["suffix_removed"])

    def test_d4_multiple_json_objects_takes_first_only(self):
        json_text, audit = dccd_runner._extract_first_balanced_json_object('{"rules":[]} {"rules":[1]}')
        self.assertEqual(json_text, '{"rules":[]}')
        self.assertTrue(audit["multiple_json_objects_detected"])

    def test_d4_no_balanced_object_returns_none(self):
        json_text, audit = dccd_runner._extract_first_balanced_json_object('prefix {"rules":[')
        self.assertIsNone(json_text)
        self.assertFalse(audit["balanced_object_found"])

    def test_d4_1_finds_only_json_like_rules_array_prefix(self):
        start_index, audit = dccd_runner._find_rules_array_start('prefix {"rules": [')
        self.assertGreaterEqual(start_index, 0)
        self.assertTrue(audit["rules_array_prefix_found"])
        start_index, audit = dccd_runner._find_rules_array_start('The following rules are: [{"feature_family":"descriptor"}]')
        self.assertEqual(start_index, -1)
        self.assertFalse(audit["rules_array_prefix_found"])

    def test_d4_1_extracts_complete_objects_from_unclosed_rules_array(self):
        text = (
            '{"rules":['
            '{"feature_family":"descriptor","feature_name":"MolLogP","feature_mode":"numeric_value"},'
            '{"feature_family":"pattern_presence","feature_name":"fused_ring","feature_mode":"binary_presence"}'
        )
        start_index, _ = dccd_runner._find_rules_array_start(text)
        objects, audit = dccd_runner._extract_balanced_objects_from_array_prefix(text, start_index)
        self.assertEqual(len(objects), 2)
        self.assertEqual(audit["candidate_object_count"], 2)
        self.assertFalse(audit["stopped_on_unclosed_object"])

    def test_d4_1_extracts_threshold_nested_object(self):
        text = (
            '{"rules":['
            '{"feature_family":"descriptor","feature_name":"MolLogP","feature_mode":"threshold_binary",'
            '"threshold":{"op":">","value":3.5},"rationale":"short"}'
        )
        start_index, _ = dccd_runner._find_rules_array_start(text)
        objects, audit = dccd_runner._extract_balanced_objects_from_array_prefix(text, start_index)
        self.assertEqual(len(objects), 1)
        self.assertIn('"threshold":{"op":">","value":3.5}', objects[0])
        self.assertFalse(audit["stopped_on_unclosed_object"])

    def test_d4_1_does_not_salvage_half_rule_object(self):
        text = '{"rules":[{"feature_family":"descriptor","feature_name":"MolLogP","feature_mode":"numeric_value"'
        start_index, _ = dccd_runner._find_rules_array_start(text)
        objects, audit = dccd_runner._extract_balanced_objects_from_array_prefix(text, start_index)
        self.assertEqual(objects, [])
        self.assertTrue(audit["stopped_on_unclosed_object"])

    def _with_d4_registry_stub(self):
        old_rule_registry = sys.modules.get("morace.rule_registry")
        registry_stub = types.ModuleType("morace.rule_registry")
        registry_stub.DESCRIPTOR_REGISTRY = {"MolLogP": object(), "TPSA": object()}
        registry_stub.PATTERN_ALIAS = {"benzene_ring": "c1ccccc1", "phenyl": "c1ccccc1"}
        registry_stub.CUSTOM_PATTERN_FUNCTIONS = {"fused_ring": object()}
        registry_stub.COUNT_ALIAS = {"RingCount": object(), "NumHBD": object()}
        sys.modules["morace.rule_registry"] = registry_stub
        return old_rule_registry

    def _restore_rule_registry_stub(self, old_rule_registry):
        if old_rule_registry is None:
            sys.modules.pop("morace.rule_registry", None)
        else:
            sys.modules["morace.rule_registry"] = old_rule_registry

    def test_d4_cleaning_drops_extra_fields_defaults_relation_and_overwrites_source(self):
        old = self._with_d4_registry_stub()
        try:
            cleaned_json, diagnostics = dccd_runner._d4_extract_cleaned_json(
                '{"rules":[{"feature_family":"descriptor","feature_name":"MolLogP",'
                '"feature_mode":"numeric_value","source":"model","confidence":0.9}]}',
                "literature",
                "tolerant_first_json_object",
            )
        finally:
            self._restore_rule_registry_stub(old)
        self.assertIn('"relation_to_target":null', cleaned_json)
        self.assertIn('"source":"literature"', cleaned_json)
        self.assertEqual(diagnostics["invalid_field_dropped_count"], 1)
        self.assertEqual(diagnostics["relation_defaulted_to_null_count"], 1)
        self.assertEqual(diagnostics["source_overwritten_count"], 1)
        self.assertEqual(diagnostics["rules_after_extraction"], 1)

    def test_d4_cleaning_drops_bad_enum_unsupported_feature_missing_field_and_bad_threshold(self):
        old = self._with_d4_registry_stub()
        try:
            cleaned_json, diagnostics = dccd_runner._d4_extract_cleaned_json(
                '{"rules":['
                '{"feature_family":"descriptor","feature_name":"MolLogP","feature_mode":"presence"},'
                '{"feature_family":"pattern_presence","feature_name":"phenyl_group","feature_mode":"binary_presence"},'
                '{"feature_family":"descriptor","feature_mode":"numeric_value"},'
                '{"feature_family":"descriptor","feature_name":"TPSA","feature_mode":"threshold_binary"}'
                ']}',
                "literature",
                "tolerant_first_json_object",
            )
        finally:
            self._restore_rule_registry_stub(old)
        self.assertEqual(cleaned_json, '{"rules":[]}')
        self.assertEqual(diagnostics["invalid_enum_dropped_count"], 1)
        self.assertEqual(diagnostics["unsupported_feature_name_dropped_count"], 1)
        self.assertEqual(diagnostics["missing_required_field_dropped_count"], 1)
        self.assertEqual(diagnostics["threshold_invalid_dropped_count"], 1)

    def test_d4_pattern_allowlist_does_not_accept_alias_keys(self):
        old = self._with_d4_registry_stub()
        try:
            cleaned_json, diagnostics = dccd_runner._d4_extract_cleaned_json(
                '{"rules":[{"feature_family":"pattern_presence","feature_name":"benzene_ring",'
                '"feature_mode":"binary_presence"}]}',
                "literature",
                "tolerant_first_json_object",
            )
        finally:
            self._restore_rule_registry_stub(old)
        self.assertEqual(cleaned_json, '{"rules":[]}')
        self.assertEqual(diagnostics["unsupported_feature_name_dropped_count"], 1)

    def test_d4_safe_canonicalization_rescues_mechanical_non_descriptor_variant(self):
        old = self._with_d4_registry_stub()
        try:
            cleaned_json, diagnostics = dccd_runner._d4_extract_cleaned_json(
                '{"rules":[{"feature_family":"pattern Presence","feature_name":"aromatic Ring",'
                '"feature_mode":"binary Presence","relation_to_target":"monotonic Up"}]}',
                "literature",
                "tolerant_first_json_object",
                "safe_enum_feature_canonicalization",
            )
        finally:
            self._restore_rule_registry_stub(old)
        self.assertIn('"feature_family":"pattern_presence"', cleaned_json)
        self.assertIn('"feature_name":"aromatic_ring"', cleaned_json)
        self.assertEqual(diagnostics["enum_canonicalized_count"], 3)
        self.assertEqual(diagnostics["feature_name_canonicalized_count"], 1)
        self.assertEqual(diagnostics["non_descriptor_canonicalized_count"], 1)
        self.assertEqual(diagnostics["non_descriptor_rescued_by_canonicalization_count"], 1)
        self.assertEqual(diagnostics["feature_name_canonicalized_samples"][0]["family"], "pattern_presence")

    def test_d4_safe_canonicalization_does_not_guess_semantic_aliases(self):
        for feature_name in ["phenolic OH group", "hydrogen-Bonding groups", "hydroxyl", "thiophene_ring"]:
            old = self._with_d4_registry_stub()
            try:
                cleaned_json, diagnostics = dccd_runner._d4_extract_cleaned_json(
                    '{"rules":[{"feature_family":"pattern_presence","feature_name":"'
                    + feature_name
                    + '","feature_mode":"binary_presence"}]}',
                    "literature",
                    "tolerant_first_json_object",
                    "safe_enum_feature_canonicalization",
                )
            finally:
                self._restore_rule_registry_stub(old)
            self.assertEqual(cleaned_json, '{"rules":[]}')
            self.assertEqual(diagnostics["feature_name_canonicalized_count"], 0)
            self.assertEqual(diagnostics["unsupported_feature_name_dropped_count"], 1)

    def test_d4_1_salvages_unclosed_rules_array_prefix(self):
        old = self._with_d4_registry_stub()
        try:
            cleaned_json, diagnostics = dccd_runner._d4_salvage_rules_array_prefix(
                '{"rules":[{"feature_family":"descriptor","feature_name":"MolLogP","feature_mode":"numeric_value"}',
                "literature",
                "tolerant_first_json_object",
            )
        finally:
            self._restore_rule_registry_stub(old)
        self.assertIn('"feature_name":"MolLogP"', cleaned_json)
        self.assertEqual(diagnostics["d4_extraction_method"], "rules_array_prefix_salvage")
        self.assertTrue(diagnostics["d4_rules_array_salvage_attempted"])
        self.assertTrue(diagnostics["d4_rules_array_prefix_found"])
        self.assertTrue(diagnostics["d4_any_json_fragment_parseable"])
        self.assertEqual(diagnostics["d4_rules_array_candidate_object_count"], 1)
        self.assertEqual(diagnostics["d4_rules_array_parseable_object_count"], 1)
        self.assertEqual(diagnostics["d4_rules_array_cleaned_rule_count"], 1)

    def test_d4_1_drops_invalid_rules_during_salvage(self):
        old = self._with_d4_registry_stub()
        try:
            cleaned_json, diagnostics = dccd_runner._d4_salvage_rules_array_prefix(
                '{"rules":['
                '{"feature_family":"descriptor","feature_name":"MolLogP","feature_mode":"presence"},'
                '{"feature_family":"pattern_presence","feature_name":"phenyl_group","feature_mode":"binary_presence"},'
                '{"feature_family":"descriptor","feature_name":"TPSA","feature_mode":"numeric_value"}',
                "literature",
                "tolerant_first_json_object",
            )
        finally:
            self._restore_rule_registry_stub(old)
        self.assertIn('"feature_name":"TPSA"', cleaned_json)
        self.assertNotIn("MolLogP", cleaned_json)
        self.assertNotIn("phenyl_group", cleaned_json)
        self.assertEqual(diagnostics["invalid_enum_dropped_count"], 1)
        self.assertEqual(diagnostics["unsupported_feature_name_dropped_count"], 1)
        self.assertEqual(diagnostics["d4_rules_array_cleaned_rule_count"], 1)

    def test_d4_run_projection_attempt_sets_effective_strict_after_extraction(self):
        class FakeBackend:
            def generate_structured(self, *args, **kwargs):
                return [{"text": 'prefix {"rules":[{"feature_family":"descriptor","feature_name":"MolLogP","feature_mode":"numeric_value"}]} suffix', "generated_token_count": 10}]

        old = self._with_d4_registry_stub()
        old_strict = dccd_runner.strict_parse_projected_rules
        old_lenient = dccd_runner.lenient_parse_projected_rules
        os.environ["STAGE2_EXTRACTION_POLICY"] = "tolerant_first_json_object"
        try:
            def fake_strict(raw_output, source, provenance):
                if str(raw_output).startswith('{"rules"') and str(raw_output).endswith("}"):
                    return (
                        [{
                            "feature_family": "descriptor",
                            "feature_name": "MolLogP",
                            "feature_mode": "numeric_value",
                            "relation_to_target": None,
                            "compile_ok": True,
                        }],
                        {"max_nesting_depth": 1, "rules_field_inside_rule_count": 0, "failure_stage": "", "failure_reason": "", "failure_label": ""},
                    )
                return (
                    [],
                    {
                        "max_nesting_depth": 0,
                        "rules_field_inside_rule_count": 0,
                        "failure_stage": "json_extraction",
                        "failure_reason": "trailing_text_or_prefix",
                        "failure_label": "json_extraction::trailing_text_or_prefix",
                    },
                )

            dccd_runner.strict_parse_projected_rules = fake_strict
            dccd_runner.lenient_parse_projected_rules = lambda *args, **kwargs: (
                [],
                {"max_nesting_depth": 0, "rules_field_inside_rule_count": 0, "failure_stage": "", "failure_reason": "", "failure_label": ""},
            )
            attempt = dccd_runner._run_projection_attempt(
                FakeBackend(),
                system_prompt="system",
                user_prompt="user",
                source="literature",
                provenance={"stage2_protocol_profile": "chat_native_core"},
                max_new_tokens=64,
                temperature=0.0,
                top_p=1.0,
                num_beams=1,
                max_input_tokens=512,
                allow_repair_pass=False,
                stage2_protocol_profile="chat_native_core",
            )
        finally:
            dccd_runner.strict_parse_projected_rules = old_strict
            dccd_runner.lenient_parse_projected_rules = old_lenient
            self._restore_rule_registry_stub(old)
        diagnostics = attempt["chat_native_extraction_diagnostics"]
        self.assertEqual(len(attempt["strict_rules"]), 0)
        self.assertEqual(len(attempt["d4_strict_rules"]), 1)
        self.assertTrue(diagnostics["d4_extraction_used"])
        self.assertFalse(diagnostics["original_strict_valid"])
        self.assertTrue(diagnostics["strict_after_extraction_valid"])

    def test_d4_1_run_projection_attempt_salvages_rules_array_prefix(self):
        class FakeBackend:
            def generate_structured(self, *args, **kwargs):
                return [{"text": '{"rules":[{"feature_family":"descriptor","feature_name":"MolLogP","feature_mode":"numeric_value"}', "generated_token_count": 10}]

        old = self._with_d4_registry_stub()
        old_strict = dccd_runner.strict_parse_projected_rules
        old_lenient = dccd_runner.lenient_parse_projected_rules
        os.environ["STAGE2_EXTRACTION_POLICY"] = "tolerant_first_json_object"
        try:
            def fake_strict(raw_output, source, provenance):
                text = str(raw_output)
                if text.startswith('{"rules"') and text.endswith("}]}"):
                    return (
                        [{
                            "feature_family": "descriptor",
                            "feature_name": "MolLogP",
                            "feature_mode": "numeric_value",
                            "relation_to_target": None,
                            "compile_ok": True,
                        }],
                        {"max_nesting_depth": 1, "rules_field_inside_rule_count": 0, "failure_stage": "", "failure_reason": "", "failure_label": ""},
                    )
                return (
                    [],
                    {
                        "max_nesting_depth": 0,
                        "rules_field_inside_rule_count": 0,
                        "failure_stage": "json_extraction",
                        "failure_reason": "trailing_text_or_prefix",
                        "failure_label": "json_extraction::trailing_text_or_prefix",
                    },
                )

            dccd_runner.strict_parse_projected_rules = fake_strict
            dccd_runner.lenient_parse_projected_rules = lambda *args, **kwargs: (
                [],
                {"max_nesting_depth": 0, "rules_field_inside_rule_count": 0, "failure_stage": "", "failure_reason": "", "failure_label": ""},
            )
            attempt = dccd_runner._run_projection_attempt(
                FakeBackend(),
                system_prompt="system",
                user_prompt="user",
                source="literature",
                provenance={"stage2_protocol_profile": "chat_native_core"},
                max_new_tokens=64,
                temperature=0.0,
                top_p=1.0,
                num_beams=1,
                max_input_tokens=512,
                allow_repair_pass=False,
                stage2_protocol_profile="chat_native_core",
            )
        finally:
            dccd_runner.strict_parse_projected_rules = old_strict
            dccd_runner.lenient_parse_projected_rules = old_lenient
            self._restore_rule_registry_stub(old)
        diagnostics = attempt["chat_native_extraction_diagnostics"]
        self.assertEqual(len(attempt["strict_rules"]), 0)
        self.assertEqual(len(attempt["d4_strict_rules"]), 1)
        self.assertEqual(diagnostics["d4_extraction_method"], "rules_array_prefix_salvage")
        self.assertTrue(diagnostics["d4_rules_array_salvage_attempted"])
        self.assertTrue(diagnostics["d4_rules_array_salvage_strict_valid"])

    def test_d4_does_not_run_when_original_strict_succeeds(self):
        class FakeBackend:
            def generate_structured(self, *args, **kwargs):
                return [{"text": '{"rules":[{"feature_family":"descriptor","feature_name":"MolLogP","feature_mode":"numeric_value"}]}', "generated_token_count": 10}]

        old_strict = dccd_runner.strict_parse_projected_rules
        old_lenient = dccd_runner.lenient_parse_projected_rules
        os.environ["STAGE2_EXTRACTION_POLICY"] = "tolerant_first_json_object"
        try:
            dccd_runner.strict_parse_projected_rules = lambda *args, **kwargs: (
                [{
                    "feature_family": "descriptor",
                    "feature_name": "MolLogP",
                    "feature_mode": "numeric_value",
                    "relation_to_target": None,
                    "compile_ok": True,
                }],
                {"max_nesting_depth": 1, "rules_field_inside_rule_count": 0, "failure_stage": "", "failure_reason": "", "failure_label": ""},
            )
            dccd_runner.lenient_parse_projected_rules = lambda *args, **kwargs: (
                [],
                {"max_nesting_depth": 0, "rules_field_inside_rule_count": 0, "failure_stage": "", "failure_reason": "", "failure_label": ""},
            )
            attempt = dccd_runner._run_projection_attempt(
                FakeBackend(),
                system_prompt="system",
                user_prompt="user",
                source="literature",
                provenance={"stage2_protocol_profile": "chat_native_core"},
                max_new_tokens=64,
                temperature=0.0,
                top_p=1.0,
                num_beams=1,
                max_input_tokens=512,
                allow_repair_pass=False,
                stage2_protocol_profile="chat_native_core",
            )
        finally:
            dccd_runner.strict_parse_projected_rules = old_strict
            dccd_runner.lenient_parse_projected_rules = old_lenient
        diagnostics = attempt["chat_native_extraction_diagnostics"]
        self.assertEqual(len(attempt["strict_rules"]), 1)
        self.assertEqual(len(attempt["d4_strict_rules"]), 0)
        self.assertFalse(diagnostics["d4_extraction_used"])
        self.assertTrue(diagnostics["original_strict_valid"])
        self.assertFalse(diagnostics["strict_after_extraction_valid"])

    def test_d4_does_not_run_for_non_chat_native_core(self):
        class FakeBackend:
            def generate_structured(self, *args, **kwargs):
                return [{"text": 'prefix {"rules":[]} suffix', "generated_token_count": 10}]

        os.environ["STAGE2_EXTRACTION_POLICY"] = "tolerant_first_json_object"
        attempt = dccd_runner._run_projection_attempt(
            FakeBackend(),
            system_prompt="system",
            user_prompt="user",
            source="literature",
            provenance={"stage2_protocol_profile": "non_qwen_native"},
            max_new_tokens=64,
            temperature=0.0,
            top_p=1.0,
            num_beams=1,
            max_input_tokens=512,
            allow_repair_pass=False,
            stage2_protocol_profile="non_qwen_native",
        )
        diagnostics = attempt["chat_native_extraction_diagnostics"]
        self.assertEqual(diagnostics["extraction_policy"], "strict_only")
        self.assertFalse(diagnostics["d4_extraction_used"])

    def test_qwen_legacy_d2_counts_lenient_projection_as_effective_strict_only_for_qwen(self):
        class FakeBackend:
            def generate_structured(self, *args, **kwargs):
                return [{"text": 'prefix {"rules":[{"feature_family":"descriptor","feature_name":"MolLogP","feature_mode":"numeric_value"}]} suffix', "generated_token_count": 10}]

        old_strict = dccd_runner.strict_parse_projected_rules
        old_lenient = dccd_runner.lenient_parse_projected_rules
        try:
            dccd_runner.strict_parse_projected_rules = lambda *args, **kwargs: (
                [],
                {
                    "max_nesting_depth": 0,
                    "rules_field_inside_rule_count": 0,
                    "failure_stage": "json_extraction",
                    "failure_reason": "trailing_text_or_prefix",
                    "failure_label": "json_extraction::trailing_text_or_prefix",
                },
            )
            dccd_runner.lenient_parse_projected_rules = lambda *args, **kwargs: (
                [{
                    "feature_family": "descriptor",
                    "feature_name": "MolLogP",
                    "feature_mode": "numeric_value",
                    "relation_to_target": None,
                    "compile_ok": True,
                }],
                {"max_nesting_depth": 1, "rules_field_inside_rule_count": 0, "failure_stage": "", "failure_reason": "", "failure_label": ""},
            )
            qwen_attempt = dccd_runner._run_projection_attempt(
                FakeBackend(),
                system_prompt="system",
                user_prompt="user",
                source="literature",
                provenance={"stage2_protocol_profile": "qwen_legacy_d2"},
                max_new_tokens=64,
                temperature=0.0,
                top_p=1.0,
                num_beams=1,
                max_input_tokens=512,
                allow_repair_pass=False,
                stage2_protocol_profile="qwen_legacy_d2",
            )
            non_qwen_attempt = dccd_runner._run_projection_attempt(
                FakeBackend(),
                system_prompt="system",
                user_prompt="user",
                source="literature",
                provenance={"stage2_protocol_profile": "non_qwen_native"},
                max_new_tokens=64,
                temperature=0.0,
                top_p=1.0,
                num_beams=1,
                max_input_tokens=512,
                allow_repair_pass=False,
                stage2_protocol_profile="non_qwen_native",
            )
        finally:
            dccd_runner.strict_parse_projected_rules = old_strict
            dccd_runner.lenient_parse_projected_rules = old_lenient

        self.assertFalse(qwen_attempt["strict_rules"])
        self.assertTrue(qwen_attempt["projected_rules"])
        self.assertTrue(
            dccd_runner._qwen_legacy_d2_effective_strict_valid(
                stage2_protocol_profile="qwen_legacy_d2",
                original_strict_valid=bool(qwen_attempt["strict_rules"] or qwen_attempt["repair_strict_rules"]),
                d4_strict_rules=qwen_attempt["d4_strict_rules"],
                projected_rules=qwen_attempt["projected_rules"],
            )
        )
        self.assertFalse(
            dccd_runner._qwen_legacy_d2_effective_strict_valid(
                stage2_protocol_profile="non_qwen_native",
                original_strict_valid=bool(non_qwen_attempt["strict_rules"] or non_qwen_attempt["repair_strict_rules"]),
                d4_strict_rules=non_qwen_attempt["d4_strict_rules"],
                projected_rules=non_qwen_attempt["projected_rules"],
            )
        )

    def test_summarize_projection_metrics_reports_qwen_legacy_d2_effective_strict(self):
        records = [
            {
                "stage2_protocol_profile": "qwen_legacy_d2",
                "stage2_protocol_profile_source": "auto_inferred",
                "stage2_protocol_profile_version": "qwen_legacy_d2_v1",
                "projection_status": "projected",
                "schema_valid": True,
                "projector_valid": True,
                "strict_projector_valid": True,
                "original_strict_projector_valid": False,
                "effective_strict_projector_valid": True,
                "qwen_legacy_d2_effective_strict": True,
                "projector_parsed_rules": [{
                    "feature_family": "descriptor",
                    "feature_name": "MolLogP",
                    "feature_mode": "numeric_value",
                }],
                "parsed_rules": [{
                    "feature_family": "descriptor",
                    "feature_name": "MolLogP",
                    "feature_mode": "numeric_value",
                }],
            },
            {
                "stage2_protocol_profile": "gemma_native",
                "stage2_protocol_profile_source": "auto_inferred",
                "stage2_protocol_profile_version": "gemma_native_v1",
                "projection_status": "projected",
                "schema_valid": True,
                "projector_valid": True,
                "strict_projector_valid": True,
                "original_strict_projector_valid": True,
                "effective_strict_projector_valid": True,
                "qwen_legacy_d2_effective_strict": False,
                "projector_parsed_rules": [{
                    "feature_family": "descriptor",
                    "feature_name": "MolWt",
                    "feature_mode": "numeric_value",
                }],
                "parsed_rules": [{
                    "feature_family": "descriptor",
                    "feature_name": "MolWt",
                    "feature_mode": "numeric_value",
                }],
            },
        ]
        metrics = dccd_runner.summarize_projection_metrics(records, branch_name="inference")
        self.assertEqual(metrics["qwen_legacy_d2_effective_strict_count"], 1)
        self.assertEqual(metrics["qwen_legacy_d2_effective_strict_rate"], 0.5)
        self.assertEqual(metrics["raw_strict_projector_valid_count"], 1)
        self.assertEqual(metrics["effective_projector_valid_count"], 2)
        self.assertEqual(metrics["original_strict_projector_valid_count"], 1)
        self.assertEqual(metrics["effective_strict_projector_valid_count"], 2)

    def test_llama_native_retry_prompt_uses_terminal_json_constraint(self):
        prompt = dccd_runner.build_stage2_literature_retry_prompt(
            task_brief="bace",
            draft_text="Determinant A: MolWt",
            previous_output="{\"rules\":[]}",
            current_roots=["descriptor::MolWt"],
            current_families=["descriptor"],
            projector_family="llama",
            stage2_protocol_profile="llama_native",
        )
        self.assertIn("final two characters are ]}", prompt)
        self.assertIn("Add at most 1 new literature rule", prompt)
        self.assertIn("Prefer a new feature family not already covered.", prompt)
        self.assertIn("Prefer pattern_presence or count when the current families are descriptor-only.", prompt)
        self.assertIn("at most 8 words", prompt)
        self.assertIn("If no clean new rule is supported, return exactly {\"rules\":[]}.", prompt)
        self.assertIn("\"feature_family\":\"<family>\"", prompt)

    def test_llama_native_stage2_max_new_tokens_respects_requested_budget_up_to_256(self):
        self.assertEqual(dccd_runner._stage2_max_new_tokens("llama_native", 192), 192)
        self.assertEqual(dccd_runner._stage2_max_new_tokens("llama_native", 256), 256)
        self.assertEqual(dccd_runner._stage2_max_new_tokens("llama_native", 320), 256)

    def test_llama_native_stage2_max_new_tokens_has_192_default(self):
        self.assertEqual(dccd_runner._stage2_max_new_tokens("llama_native", 0), 192)

    def test_llama_native_stage2_max_new_tokens_enforces_96_floor(self):
        self.assertEqual(dccd_runner._stage2_max_new_tokens("llama_native", 64), 96)

    def test_non_llama_stage2_max_new_tokens_behavior_is_unchanged(self):
        self.assertEqual(dccd_runner._stage2_max_new_tokens("non_qwen_native", 64), 64)

    def test_chat_native_core_stage2_max_new_tokens_caps_and_defaults(self):
        self.assertEqual(dccd_runner._stage2_max_new_tokens("chat_native_core", 0), 128)
        self.assertEqual(dccd_runner._stage2_max_new_tokens("chat_native_core", 64), 96)
        self.assertEqual(dccd_runner._stage2_max_new_tokens("chat_native_core", 192), 160)
        self.assertEqual(dccd_runner._stage2_max_new_tokens("chat_native_core", 320), 160)

    def test_real_strict_parser_truncates_end_json_marker(self):
        schema_path = Path(__file__).resolve().parents[1] / "morace/schemas.py"
        spec = importlib.util.spec_from_file_location("real_schemas_for_end_json_test", schema_path)
        real_schemas = importlib.util.module_from_spec(spec)
        old_rule_registry = sys.modules.get("morace.rule_registry")
        registry_stub = types.ModuleType("morace.rule_registry")
        registry_stub.COUNT_ALIAS = {}
        registry_stub.DESCRIPTOR_REGISTRY = {"MolWt": {}}
        registry_stub.PATTERN_ALIAS = {}
        registry_stub.canonicalize_feature_name = lambda family, name: str(name)
        sys.modules["morace.rule_registry"] = registry_stub
        try:
            spec.loader.exec_module(real_schemas)
        finally:
            if old_rule_registry is None:
                sys.modules.pop("morace.rule_registry", None)
            else:
                sys.modules["morace.rule_registry"] = old_rule_registry
        raw_output = (
            "{\"rules\":[{\"source\":\"literature\",\"feature_family\":\"descriptor\","
            "\"feature_name\":\"MolWt\",\"feature_mode\":\"numeric_value\","
            "\"relation_to_target\":\"negative\",\"threshold\":null,\"rationale\":\"short\"}]}"
            "END_JSON"
        )
        rules, meta = real_schemas.strict_parse_projected_rules(
            raw_output,
            source="literature",
            provenance={"stage2_protocol_profile": "chat_native_core"},
        )
        self.assertEqual(len(rules), 1)
        self.assertEqual(meta.get("failure_label", ""), "")

    def test_summarize_projection_metrics_includes_failure_diagnostics_by_source(self):
        records = [
            {
                **_record("p1", 0, 1.0, "TPSA"),
                "source": "literature",
                "strict_projector_valid": False,
                "strict_failure": {
                    "stage": "json_extraction",
                    "reason": "trailing_text_or_prefix",
                    "label": "json_extraction::trailing_text_or_prefix",
                },
            }
        ]
        metrics = dccd_runner.summarize_projection_metrics(records, branch_name="synthesize")
        by_source = metrics["strict_failure_diagnostics_by_source"]["literature"]
        self.assertEqual(by_source["total_failures"], 1)
        self.assertEqual(by_source["by_stage"]["json_extraction"]["trailing_text_or_prefix"], 1)

    def test_summarize_projection_metrics_collects_llama_native_fail_samples(self):
        records = [
            {
                **_record("p1", 0, 1.0, "TPSA"),
                "source": "literature",
                "stage2_protocol_profile": "llama_native",
                "stage2_protocol_profile_version": "llama_native_v1",
                "strict_projector_valid": False,
                "strict_failure": {
                    "stage": "json_extraction",
                    "reason": "trailing_text_or_prefix",
                    "label": "json_extraction::trailing_text_or_prefix",
                },
                "projector_raw_output": "{\"rules\":[]\nExplanation: extra text",
                "projector_repair_output": "",
            }
        ]
        metrics = dccd_runner.summarize_projection_metrics(records, branch_name="synthesize")
        self.assertEqual(metrics["llama_native_output_discipline_summary"]["profile"], "llama_native")
        self.assertEqual(
            metrics["llama_native_output_discipline_summary"]["strict_fail_count"],
            1,
        )
        sample = metrics["llama_native_strict_fail_samples"]["literature"][0]
        self.assertEqual(sample["branch"], "synthesize")
        self.assertEqual(sample["reason"], "trailing_text_or_prefix")
        self.assertIn("{\"rules\":[]", sample["raw_output_preview"])

    def test_summarize_projection_metrics_collects_chat_native_core_synth_observability(self):
        records = [
            {
                **_record("p1", 0, 1.0, "MolWt"),
                "source": "literature",
                "stage2_protocol_profile": "chat_native_core",
                "stage2_protocol_profile_version": "chat_native_core_d4_2_safe_canonicalization_v1",
                "projection_variant_id": 0,
                "selected_for_merge": False,
                "selected_for_merge_reason": "rank_cutoff",
                "projected_rule_count": 1,
                "projector_parsed_rules": [
                    {
                        "feature_family": "descriptor",
                        "feature_name": "MolWt",
                        "feature_mode": "continuous",
                        "relation_to_target": "negative",
                        "compile_ok": True,
                    }
                ],
                "parsed_rules": [
                    {
                        "feature_family": "descriptor",
                        "feature_name": "MolWt",
                        "feature_mode": "continuous",
                        "relation_to_target": "negative",
                        "compile_ok": True,
                    }
                ],
            }
        ]
        metrics = dccd_runner.summarize_projection_metrics(records, branch_name="synthesize")
        obs = metrics["chat_native_core_synth_observability"]
        self.assertEqual(obs["profile"], "chat_native_core")
        self.assertEqual(obs["profile_version"], "chat_native_core_d5_2_target_thinning_v1")
        self.assertEqual(obs["synth_record_count_base_variant"], 1)
        self.assertEqual(obs["synth_record_count_retry_variant"], 0)
        self.assertEqual(obs["synth_selected_rule_count"], 0)
        self.assertEqual(obs["synth_selected_feature_families"], [])
        self.assertEqual(obs["synth_failure_stage"], "selection_empty")

    def test_summarize_projection_metrics_collects_chat_native_core_output_discipline_summary(self):
        records = [
            {
                **_record("p1", 0, 1.0, "TPSA"),
                "source": "literature",
                "stage2_protocol_profile": "chat_native_core",
                "stage2_protocol_profile_version": "chat_native_core_d4_2_safe_canonicalization_v1",
                "strict_projector_valid": False,
                "strict_failure": {
                    "stage": "json_extraction",
                    "reason": "trailing_text_or_prefix",
                    "label": "json_extraction::trailing_text_or_prefix",
                },
                "projector_raw_output": "{\"rules\":[]}\nextra text",
                "projector_repair_output": "",
            },
            {
                **_record("p2", 0, 1.0, "MolWt"),
                "source": "literature",
                "stage2_protocol_profile": "chat_native_core",
                "stage2_protocol_profile_version": "chat_native_core_d4_2_safe_canonicalization_v1",
                "strict_projector_valid": False,
                "strict_failure": {
                    "stage": "json_decode",
                    "reason": "json_decode_failed",
                    "label": "json_decode::json_decode_failed",
                },
                "projector_raw_output": "{\"rules\":[",
                "projector_repair_output": "",
            },
            {
                **_record("p3", 0, 1.0, "RingCount"),
                "source": "literature",
                "stage2_protocol_profile": "chat_native_core",
                "stage2_protocol_profile_version": "chat_native_core_d4_2_safe_canonicalization_v1",
                "strict_projector_valid": True,
                "projector_raw_output": "{\"rules\":[]}END_JSON",
                "projector_repair_output": "",
            },
        ]
        metrics = dccd_runner.summarize_projection_metrics(records, branch_name="synthesize")
        summary = metrics["chat_native_core_output_discipline_summary"]
        self.assertEqual(summary["profile"], "chat_native_core")
        self.assertEqual(summary["profile_version"], "chat_native_core_d5_2_target_thinning_v1")
        self.assertEqual(summary["strict_fail_count"], 2)
        self.assertEqual(summary["strict_fail_count_base_variant"], 2)
        self.assertEqual(summary["strict_fail_count_retry_variant"], 0)
        self.assertEqual(summary["raw_first_json_parseable_count"], 2)
        self.assertEqual(summary["json_object_closes_before_suffix_count"], 2)
        self.assertEqual(summary["suffix_after_first_json_count"], 1)
        self.assertEqual(summary["suffix_after_first_json_preview"], ["extra text"])
        self.assertEqual(metrics["strict_valid_rule_count"], 1)
        self.assertEqual(metrics["strict_valid_root_keys"], ["descriptor::RingCount"])
        self.assertEqual(metrics["strict_valid_feature_families"], ["descriptor"])
        self.assertEqual(metrics["strict_valid_family_count"], 1)

    def test_summarize_projection_metrics_counts_chat_native_core_short_output_policy(self):
        records = [
            {
                **_record("p1", 0, 1.0, "MolLogP"),
                "stage2_protocol_profile": "chat_native_core",
                "stage2_protocol_profile_version": "chat_native_core_d4_2_safe_canonicalization_v1",
                "termination_policy": "pure_json_short_output",
                "rule_count_policy": "single_rule",
            }
        ]
        metrics = dccd_runner.summarize_projection_metrics(records, branch_name="synthesize")
        self.assertEqual(metrics["chat_native_core_termination_policy_counts"]["pure_json_short_output"], 1)
        self.assertEqual(metrics["chat_native_core_rule_count_policy_counts"]["single_rule"], 1)
        self.assertEqual(metrics["strict_valid_rule_count"], 1)

    def test_summarize_projection_metrics_reports_original_and_effective_strict(self):
        rule = {
            "feature_family": "descriptor",
            "feature_name": "MolLogP",
            "feature_mode": "numeric_value",
            "relation_to_target": None,
            "compile_ok": True,
        }
        records = [
            {
                **_record("p1", 0, 1.0, "MolLogP"),
                "stage2_protocol_profile": "chat_native_core",
                "stage2_protocol_profile_version": "chat_native_core_d4_2_safe_canonicalization_v1",
                "termination_policy": "pure_json_single_rule",
                "rule_count_policy": "single_rule",
                "strict_projector_valid": True,
                "original_strict_projector_valid": False,
                "effective_strict_projector_valid": True,
                "projector_parsed_rules": [rule],
                "parsed_rules": [rule],
                "chat_native_extraction_diagnostics": {
                    "extraction_policy": "tolerant_first_json_object",
                    "canonicalization_policy": "safe_enum_feature_canonicalization",
                    "d4_extraction_method": "rules_array_prefix_salvage",
                    "d4_extraction_used": True,
                    "raw_first_json_parseable": True,
                    "d4_any_json_fragment_parseable": True,
                    "strict_after_extraction_valid": True,
                    "rules_before_extraction": 1,
                    "rules_after_extraction": 1,
                    "suffix_removed": True,
                    "invalid_enum_dropped_count": 0,
                    "unsupported_feature_name_dropped_count": 0,
                    "missing_required_field_dropped_count": 0,
                    "d4_rules_array_salvage_attempted": True,
                    "d4_rules_array_prefix_found": True,
                    "d4_rules_array_candidate_object_count": 1,
                    "d4_rules_array_parseable_object_count": 1,
                    "d4_rules_array_cleaned_rule_count": 1,
                    "d4_rules_array_salvage_strict_valid": True,
                    "enum_canonicalized_count": 1,
                    "enum_canonicalized_samples": [
                        {
                            "field": "feature_mode",
                            "original_value": "numeric Value",
                            "canonical_value": "numeric_value",
                            "reason": "enum_case_space_normalized",
                            "family": "descriptor",
                        }
                    ],
                    "feature_name_canonicalized_count": 1,
                    "feature_name_canonicalized_samples": [
                        {
                            "field": "feature_name",
                            "original_value": "mol LogP",
                            "canonical_value": "MolLogP",
                            "reason": "remove_spaces_exact_match",
                            "family": "descriptor",
                        }
                    ],
                    "canonicalization_failed_count": 0,
                    "non_descriptor_canonicalized_count": 0,
                    "non_descriptor_rescued_by_canonicalization_count": 0,
                },
            }
        ]
        metrics = dccd_runner.summarize_projection_metrics(records, branch_name="synthesize")
        self.assertEqual(metrics["strict_projector_valid_rate"], 1.0)
        self.assertEqual(metrics["original_strict_projector_valid_rate"], 0.0)
        self.assertEqual(metrics["effective_strict_projector_valid_rate"], 1.0)
        self.assertEqual(metrics["strict_after_extraction_rate"], 1.0)
        self.assertEqual(metrics["chat_native_core_extraction_policy_counts"]["tolerant_first_json_object"], 1)
        self.assertEqual(
            metrics["chat_native_core_canonicalization_policy_counts"]["safe_enum_feature_canonicalization"],
            1,
        )
        self.assertEqual(metrics["d4_extraction_used_count"], 1)
        self.assertEqual(metrics["d4_extraction_success_count"], 1)
        self.assertEqual(metrics["d4_extraction_parseable_count"], 1)
        self.assertEqual(metrics["d4_extraction_strict_valid_count"], 1)
        self.assertEqual(metrics["d4_any_json_fragment_parseable_count"], 1)
        self.assertEqual(metrics["d4_extraction_parseable_rate_on_attempted"], 1.0)
        self.assertEqual(metrics["d4_extraction_strict_valid_rate_on_attempted"], 1.0)
        self.assertEqual(metrics["d4_suffix_removed_count"], 1)
        self.assertEqual(metrics["d4_compile_safe_rule_count"], 1)
        self.assertEqual(metrics["d4_extraction_method_counts"]["rules_array_prefix_salvage"], 1)
        self.assertEqual(metrics["d4_rules_array_salvage_attempted_count"], 1)
        self.assertEqual(metrics["d4_rules_array_prefix_found_count"], 1)
        self.assertEqual(metrics["d4_rules_array_candidate_object_count"], 1)
        self.assertEqual(metrics["d4_rules_array_parseable_object_count"], 1)
        self.assertEqual(metrics["d4_rules_array_cleaned_rule_count"], 1)
        self.assertEqual(metrics["d4_rules_array_salvage_strict_valid_count"], 1)
        self.assertEqual(metrics["d4_rules_array_salvage_strict_valid_rate_on_attempted"], 1.0)
        self.assertEqual(metrics["d4_enum_canonicalized_count"], 1)
        self.assertEqual(metrics["d4_feature_name_canonicalized_count"], 1)
        self.assertEqual(metrics["d4_canonicalization_failed_count"], 0)
        self.assertEqual(metrics["non_descriptor_canonicalized_count"], 0)
        self.assertEqual(metrics["non_descriptor_rescued_by_canonicalization_count"], 0)
        self.assertEqual(metrics["strict_projector_valid_rate_semantics"], "effective_when_extraction_policy_enabled")

    def test_summarize_projection_metrics_reports_chemdfm_family_feature_consistency(self):
        rule = {
            "feature_family": "count",
            "feature_name": "NumHBA",
            "feature_mode": "count_value",
            "relation_to_target": "positive",
            "threshold": None,
            "compile_ok": True,
        }
        records = [
            {
                **_record("p1", 0, 1.0, "NumHBA"),
                "source": "literature",
                "stage2_protocol_profile": "chemdfm_native",
                "stage2_protocol_profile_version": "chemdfm_native_v2_9_b3_family_target_thinning_v1",
                "family_target_policy": "none",
                "target_vocab_policy": "none",
                "target_thinning_policy": "none",
                "strict_projector_valid": True,
                "original_strict_projector_valid": False,
                "effective_strict_projector_valid": True,
                "projector_parsed_rules": [rule],
                "parsed_rules": [rule],
                "chat_native_extraction_diagnostics": {
                    "extraction_policy": "tolerant_first_json_object",
                    "canonicalization_policy": "safe_enum_feature_canonicalization",
                    "family_feature_policy": "safe_family_feature_consistency",
                    "d4_extraction_method": "first_balanced_json_object",
                    "d4_extraction_used": True,
                    "raw_first_json_parseable": True,
                    "strict_after_extraction_valid": True,
                    "rules_before_extraction": 1,
                    "rules_after_extraction": 1,
                    "family_feature_corrected_count": 1,
                    "family_feature_corrected_samples": [
                        {
                            "feature_name": "NumHBA",
                            "original_feature_family": "descriptor",
                            "corrected_feature_family": "count",
                            "reason": "feature_name_in_count_registry",
                        }
                    ],
                    "family_feature_threshold_cleared_count": 1,
                    "family_feature_threshold_cleared_samples": [
                        {
                            "feature_name": "NumHBA",
                            "original_threshold": {"op": ">", "value": 2},
                            "corrected_threshold": None,
                        }
                    ],
                    "count_feature_family_corrected_count": 1,
                    "descriptor_to_count_corrected_count": 1,
                    "non_descriptor_rescued_by_family_consistency_count": 1,
                },
            }
        ]
        metrics = dccd_runner.summarize_projection_metrics(records, branch_name="synthesize")
        self.assertEqual(
            metrics["chemdfm_native_family_feature_policy_counts"]["safe_family_feature_consistency"],
            1,
        )
        self.assertEqual(metrics["family_feature_corrected_count"], 1)
        self.assertEqual(metrics["family_feature_threshold_cleared_count"], 1)
        self.assertEqual(metrics["descriptor_to_count_corrected_count"], 1)
        self.assertEqual(metrics["non_descriptor_rescued_by_family_consistency_count"], 1)
        self.assertEqual(metrics["family_feature_corrected_samples"][0]["feature_name"], "NumHBA")

    def test_summarize_projection_metrics_splits_c3_base_and_independent_variants(self):
        base_rule = {
            "feature_family": "descriptor",
            "feature_name": "MolLogP",
            "feature_mode": "numeric_value",
            "relation_to_target": "negative",
            "compile_ok": True,
        }
        independent_rule = {
            "feature_family": "pattern_presence",
            "feature_name": "aromatic_ring",
            "feature_mode": "binary_presence",
            "relation_to_target": "negative",
            "compile_ok": True,
        }
        records = [
            {
                **_record("p1", 0, 1.0, "MolLogP"),
                "stage2_protocol_profile": "chat_native_core",
                "stage2_protocol_profile_version": "chat_native_core_d4_2_safe_canonicalization_v1",
                "termination_policy": "pure_json_single_rule",
                "rule_count_policy": "two_independent_single_rule_projections",
                "projection_variant_id": 0,
                "projection_variant_role": "base_single_rule",
                "projector_parsed_rules": [base_rule],
                "parsed_rules": [base_rule],
                "selected_for_merge": True,
            },
            {
                **_record("p1", 1, 1.0, "aromatic_ring"),
                "stage2_protocol_profile": "chat_native_core",
                "stage2_protocol_profile_version": "chat_native_core_d4_2_safe_canonicalization_v1",
                "termination_policy": "pure_json_single_rule",
                "rule_count_policy": "two_independent_single_rule_projections",
                "projection_variant_id": 1,
                "projection_variant_role": "independent_single_rule_2",
                "projector_parsed_rules": [independent_rule],
                "parsed_rules": [independent_rule],
                "selected_for_merge": True,
                "independent_single_projection_used": True,
                "independent_single_projection_kept": True,
                "independent_single_rule_novel_root_count": 1,
            },
        ]
        metrics = dccd_runner.summarize_projection_metrics(records, branch_name="synthesize")
        self.assertEqual(
            metrics["chat_native_core_rule_count_policy_counts"]["two_independent_single_rule_projections"],
            2,
        )
        self.assertEqual(metrics["base_single_rule_strict_valid_count"], 1)
        self.assertEqual(metrics["independent_single_rule_strict_valid_count"], 1)
        self.assertEqual(metrics["independent_single_rule_compile_valid_count"], 1)
        self.assertEqual(metrics["independent_single_rule_novel_root_count"], 1)
        self.assertEqual(metrics["independent_single_rule_kept_count"], 1)
        self.assertEqual(metrics["independent_single_projection_used_count"], 1)
        self.assertEqual(metrics["independent_single_projection_kept_count"], 1)
        self.assertEqual(metrics["base_single_rule_selected_root_keys"], ["descriptor::MolLogP"])
        self.assertEqual(metrics["independent_single_rule_selected_root_keys"], ["pattern_presence::aromatic_ring"])
        self.assertEqual(metrics["independent_single_rule_selected_feature_families"], ["pattern_presence"])

    def test_summarize_projection_metrics_reports_d5_family_targets(self):
        base_rule = {
            "feature_family": "descriptor",
            "feature_name": "MolLogP",
            "feature_mode": "numeric_value",
            "relation_to_target": "negative",
            "compile_ok": True,
        }
        count_rule = {
            "feature_family": "count",
            "feature_name": "NumHBA",
            "feature_mode": "count_value",
            "relation_to_target": "negative",
            "compile_ok": True,
        }
        rejected_rule = {
            "feature_family": "descriptor",
            "feature_name": "MolMR",
            "feature_mode": "numeric_value",
            "relation_to_target": "positive",
            "compile_ok": True,
        }
        records = [
            {
                **_record("p1", 0, 1.0, "MolLogP"),
                "stage2_protocol_profile": "chat_native_core",
                "stage2_protocol_profile_version": "chat_native_core_d5_2_target_thinning_v1",
                "termination_policy": "pure_json_single_rule",
                "rule_count_policy": "single_rule",
                "family_target_policy": "family_targeted_single_rule_variants",
                "target_vocab_policy": "micro_canonical_options",
                "target_thinning_policy": "first_valid_target_rule",
                "projection_variant_id": 0,
                "projection_variant_role": "base_single_rule",
                "projector_parsed_rules": [base_rule],
                "parsed_rules": [base_rule],
                "selected_for_merge": True,
            },
            {
                **_record("p1", 1, 0.8, "NumHBA"),
                "stage2_protocol_profile": "chat_native_core",
                "stage2_protocol_profile_version": "chat_native_core_d5_2_target_thinning_v1",
                "termination_policy": "pure_json_single_rule",
                "rule_count_policy": "single_rule",
                "family_target_policy": "family_targeted_single_rule_variants",
                "target_vocab_policy": "micro_canonical_options",
                "target_thinning_policy": "first_valid_target_rule",
                "projection_variant_id": 1,
                "projection_variant_role": "family_target_count",
                "family_target_family": "count",
                "family_target_variant_kept": True,
                "family_target_original_drop_reason": "multi_rule_target_violation",
                "family_target_variant_kept_reason": "target_thinning_rescue",
                "target_vocab_options": ["NumHBA", "NumHBD"],
                "target_vocab_match": True,
                "target_vocab_drop_reason": "",
                "target_feature_name": "NumHBA",
                "target_thinning_attempted": True,
                "target_thinning_success": True,
                "target_thinning_trigger_reason": "multi_rule_target_violation",
                "target_thinning_candidate_count": 2,
                "target_thinning_valid_candidate_count": 1,
                "target_thinning_selected_root": "count::NumHBA",
                "target_thinning_selected_family": "count",
                "target_thinning_selected_feature_name": "NumHBA",
                "target_thinning_drop_reason": "",
                "target_root_keys": ["count::NumHBA"],
                "target_feature_families": ["count"],
                "projector_parsed_rules": [count_rule],
                "parsed_rules": [count_rule],
                "selected_for_merge": True,
            },
            {
                **_record("p1", 2, 0.7, "MolMR"),
                "stage2_protocol_profile": "chat_native_core",
                "stage2_protocol_profile_version": "chat_native_core_d5_2_target_thinning_v1",
                "termination_policy": "pure_json_single_rule",
                "rule_count_policy": "single_rule",
                "family_target_policy": "family_targeted_single_rule_variants",
                "target_vocab_policy": "micro_canonical_options",
                "target_thinning_policy": "first_valid_target_rule",
                "projection_variant_id": 2,
                "projection_variant_role": "family_target_pattern_presence",
                "family_target_family": "pattern_presence",
                "family_target_variant_kept": False,
                "off_target_family_drop": True,
                "family_target_drop_reason": "off_target_family",
                "target_vocab_options": ["aromatic_ring", "amide"],
                "target_vocab_match": False,
                "target_vocab_drop_reason": "",
                "target_feature_name": "MolMR",
                "target_thinning_attempted": True,
                "target_thinning_success": False,
                "target_thinning_trigger_reason": "off_target_family",
                "target_thinning_candidate_count": 1,
                "target_thinning_valid_candidate_count": 0,
                "target_thinning_drop_reason": "off_target_family",
                "target_root_keys": ["descriptor::MolMR"],
                "target_feature_families": ["descriptor"],
                "projector_parsed_rules": [rejected_rule],
                "parsed_rules": [],
                "selected_for_merge": False,
            },
        ]
        metrics = dccd_runner.summarize_projection_metrics(records, branch_name="synthesize")
        self.assertEqual(
            metrics["chat_native_core_family_target_policy_counts"]["family_targeted_single_rule_variants"],
            3,
        )
        self.assertEqual(metrics["family_target_variant_used_count"], 2)
        self.assertEqual(metrics["family_target_variant_kept_count"], 1)
        self.assertEqual(metrics["family_target_variant_effective_strict_count"], 2)
        self.assertEqual(metrics["family_target_variant_compile_safe_count"], 2)
        self.assertEqual(metrics["family_target_variant_off_target_drop_count"], 1)
        self.assertEqual(metrics["family_target_non_descriptor_kept_count"], 1)
        self.assertEqual(metrics["family_target_non_descriptor_root_keys"], ["count::NumHBA"])
        self.assertEqual(metrics["family_target_by_family"]["count"]["kept_count"], 1)
        self.assertEqual(metrics["family_target_by_family"]["pattern_presence"]["used_count"], 1)
        self.assertEqual(metrics["target_vocab_policy_counts"]["micro_canonical_options"], 3)
        self.assertEqual(metrics["target_vocab_variant_used_count"], 2)
        self.assertEqual(metrics["target_vocab_match_count"], 1)
        self.assertEqual(metrics["target_vocab_drop_count"], 1)
        self.assertEqual(metrics["target_vocab_drop_reason_histogram"]["off_target_family"], 1)
        self.assertEqual(metrics["target_vocab_non_descriptor_kept_count"], 1)
        self.assertEqual(metrics["target_vocab_non_descriptor_root_keys"], ["count::NumHBA"])
        self.assertEqual(metrics["target_vocab_by_family"]["count"]["options"], ["NumHBA", "NumHBD"])
        self.assertEqual(metrics["target_vocab_by_family"]["count"]["kept_count"], 1)
        self.assertEqual(metrics["target_thinning_policy_counts"]["first_valid_target_rule"], 3)
        self.assertEqual(metrics["target_thinning_attempted_count"], 2)
        self.assertEqual(metrics["target_thinning_success_count"], 1)
        self.assertEqual(metrics["target_thinning_candidate_count"], 3)
        self.assertEqual(metrics["target_thinning_valid_candidate_count"], 1)
        self.assertEqual(metrics["target_thinning_success_rate_on_attempted"], 0.5)
        self.assertEqual(metrics["target_thinning_non_descriptor_kept_count"], 1)
        self.assertEqual(metrics["target_thinning_non_descriptor_root_keys"], ["count::NumHBA"])
        self.assertEqual(metrics["target_thinning_by_family"]["count"]["success_count"], 1)
        self.assertEqual(metrics["target_thinning_by_family"]["count"]["selected_roots"], ["count::NumHBA"])
        self.assertEqual(
            metrics["target_thinning_by_family"]["pattern_presence"]["drop_reason_histogram"]["off_target_family"],
            1,
        )
        self.assertEqual(metrics["family_target_selected_root_keys"], ["count::NumHBA"])
        self.assertEqual(
            metrics["base_vs_family_target_comparison"]["combined_feature_families"],
            ["descriptor", "count"],
        )
        self.assertEqual(
            metrics["base_vs_family_target_comparison"]["new_non_descriptor_roots_from_family_targets"],
            ["count::NumHBA"],
        )

    def test_summarize_projection_metrics_collects_chat_native_core_strict_fail_samples(self):
        records = []
        for idx in range(6):
            records.append(
                {
                    **_record("p1", idx, 1.0, f"TPSA_{idx}"),
                    "source": "literature",
                    "stage2_protocol_profile": "chat_native_core",
                    "stage2_protocol_profile_version": "chat_native_core_d4_2_safe_canonicalization_v1",
                    "projection_variant_id": idx % 2,
                    "strict_projector_valid": False,
                    "strict_failure": {
                        "stage": "json_extraction",
                        "reason": "trailing_text_or_prefix",
                        "label": "json_extraction::trailing_text_or_prefix",
                    },
                    "projector_raw_output": "{\"rules\":[]}" + (str(idx) * 800),
                    "projector_repair_output": "",
                }
            )
        metrics = dccd_runner.summarize_projection_metrics(records, branch_name="synthesize")
        fail_samples = metrics["chat_native_core_strict_fail_samples"]["trailing_text_or_prefix"]
        self.assertEqual(len(fail_samples), 5)
        self.assertTrue(all(len(item["raw_output_preview"]) <= 500 for item in fail_samples))
        self.assertEqual(
            {item["projection_variant_id"] for item in fail_samples},
            {0, 1},
        )

    def test_run_pipeline_snapshot_contains_profile_override_and_cross_model_flag(self):
        import json
        import tempfile
        from pathlib import Path
        from morace.run_configuration import prepare_run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_run(root, {
                "RUN_TAG": "snapshot-test",
                "USE_CROSS_MODEL_PROJECTOR": "1",
                "STAGE2_PROTOCOL_PROFILE": "chat_native_core",
            }, "bace")
            payload = json.loads((root / "artifacts/snapshot-test/run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["use_cross_model_projector"], "1")
            self.assertEqual(payload["stage2_protocol_profile_override"], "chat_native_core")


if __name__ == "__main__":
    unittest.main()

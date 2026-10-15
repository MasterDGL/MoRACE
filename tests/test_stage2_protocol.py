import os
import unittest
from unittest.mock import patch

from morace import feature_vocabulary, stage2_protocol


PROTOCOL_ENV_KEYS = {
    "STAGE2_PROTOCOL_PROFILE",
    "STAGE2_EXTRACTION_POLICY",
    "CHAT_NATIVE_CORE_EXTRACTION_POLICY",
    "STAGE2_CANONICALIZATION_POLICY",
    "CHAT_NATIVE_CORE_CANONICALIZATION_POLICY",
    "STAGE2_FAMILY_FEATURE_POLICY",
    "STAGE2_FAMILY_TARGET_POLICY",
    "CHAT_NATIVE_CORE_FAMILY_TARGET_POLICY",
    "STAGE2_FAMILY_TARGETS",
    "CHAT_NATIVE_CORE_FAMILY_TARGETS",
    "STAGE2_TARGET_VOCAB_POLICY",
    "CHAT_NATIVE_CORE_TARGET_VOCAB_POLICY",
    "STAGE2_TARGET_THINNING_POLICY",
    "CHAT_NATIVE_CORE_TARGET_THINNING_POLICY",
    "STAGE2_TERMINATION_POLICY",
    "CHAT_NATIVE_CORE_TERMINATION_POLICY",
    "STAGE2_RULE_COUNT_POLICY",
    "CHAT_NATIVE_CORE_RULE_COUNT_POLICY",
}


class Stage2ProtocolTests(unittest.TestCase):
    def clean_protocol_environment(self):
        return patch.dict(
            os.environ,
            {key: value for key, value in os.environ.items() if key not in PROTOCOL_ENV_KEYS},
            clear=True,
        )

    def test_default_protocol_policies(self):
        with self.clean_protocol_environment():
            self.assertEqual(stage2_protocol.stage2_extraction_policy(), "strict_only")
            self.assertEqual(stage2_protocol.stage2_canonicalization_policy(), "strict_canonical_only")
            self.assertEqual(stage2_protocol.stage2_family_feature_policy(), "strict_family")
            self.assertEqual(stage2_protocol.stage2_family_targets(), ["pattern_presence", "count"])
            self.assertEqual(stage2_protocol.chat_native_core_termination_policy(), "pure_json_single_rule")
            self.assertEqual(stage2_protocol.chat_native_core_rule_count_policy(), "single_rule")

    def test_protocol_environment_values_are_validated(self):
        with self.clean_protocol_environment(), patch.dict(
            os.environ,
            {
                "STAGE2_EXTRACTION_POLICY": "tolerant_first_json_object",
                "STAGE2_FAMILY_TARGETS": "count,pattern_presence,count",
            },
        ):
            self.assertEqual(stage2_protocol.stage2_extraction_policy(), "tolerant_first_json_object")
            self.assertEqual(stage2_protocol.stage2_family_targets(), ["count", "pattern_presence"])

        with self.clean_protocol_environment(), patch.dict(
            os.environ,
            {"STAGE2_EXTRACTION_POLICY": "unsupported"},
        ):
            with self.assertRaises(ValueError):
                stage2_protocol.stage2_extraction_policy()

    def test_chat_native_core_output_policies_are_validated(self):
        with self.clean_protocol_environment(), patch.dict(
            os.environ,
            {
                "CHAT_NATIVE_CORE_TERMINATION_POLICY": "end_json_marker_single_rule",
                "CHAT_NATIVE_CORE_RULE_COUNT_POLICY": "two_independent_single_rule_projections",
            },
        ):
            self.assertEqual(
                stage2_protocol.chat_native_core_termination_policy(),
                "end_json_marker_single_rule",
            )
            self.assertEqual(
                stage2_protocol.chat_native_core_rule_count_policy(),
                "two_independent_single_rule_projections",
            )

        for environment_key, policy_reader in (
            ("STAGE2_TERMINATION_POLICY", stage2_protocol.chat_native_core_termination_policy),
            ("STAGE2_RULE_COUNT_POLICY", stage2_protocol.chat_native_core_rule_count_policy),
        ):
            with self.subTest(environment_key=environment_key), self.clean_protocol_environment(), patch.dict(
                os.environ,
                {environment_key: "unsupported"},
            ):
                with self.assertRaises(ValueError):
                    policy_reader()

    def test_resolved_context_preserves_qwen_defaults(self):
        with self.clean_protocol_environment():
            context = stage2_protocol.resolve_stage2_protocol_context("qwen", "qwen", False)

        self.assertEqual(
            context,
            {
                "profile": "qwen_legacy_d2",
                "profile_source": "auto_inferred",
                "profile_version": "qwen_legacy_d2_v1",
                "termination_policy": "",
                "rule_count_policy": "",
                "family_target_policy": "none",
                "target_vocab_policy": "none",
                "target_thinning_policy": "none",
                "family_target_families": [],
            },
        )

    def test_resolved_context_preserves_chat_native_policy_bundle(self):
        with self.clean_protocol_environment(), patch.dict(
            os.environ,
            {
                "STAGE2_PROTOCOL_PROFILE": "chat_native_core",
                "STAGE2_TERMINATION_POLICY": "end_json_marker_single_rule",
                "STAGE2_RULE_COUNT_POLICY": "two_independent_single_rule_projections",
                "STAGE2_FAMILY_TARGET_POLICY": "family_targeted_single_rule_variants",
                "STAGE2_TARGET_VOCAB_POLICY": "micro_canonical_options",
                "STAGE2_TARGET_THINNING_POLICY": "first_valid_target_rule",
            },
        ):
            context = stage2_protocol.resolve_stage2_protocol_context("llama", "llama", False)

        self.assertEqual(context["profile"], "chat_native_core")
        self.assertEqual(context["profile_source"], "env_override")
        self.assertEqual(context["termination_policy"], "end_json_marker_single_rule")
        self.assertEqual(context["rule_count_policy"], "two_independent_single_rule_projections")
        self.assertEqual(context["family_target_policy"], "family_targeted_single_rule_variants")
        self.assertEqual(context["target_vocab_policy"], "micro_canonical_options")
        self.assertEqual(context["target_thinning_policy"], "first_valid_target_rule")
        self.assertEqual(context["family_target_families"], ["pattern_presence", "count"])

    def test_pipeline_runner_keeps_legacy_import_surface(self):
        try:
            from morace import pipeline_runner
        except ModuleNotFoundError as exc:
            if exc.name == "rdkit":
                self.skipTest("pipeline compatibility requires the core RDKit dependency")
            raise

        self.assertIs(pipeline_runner.stage2_extraction_policy, stage2_protocol.stage2_extraction_policy)
        self.assertIs(
            pipeline_runner.infer_stage2_protocol_profile,
            stage2_protocol.infer_stage2_protocol_profile,
        )
        self.assertIs(
            pipeline_runner.resolve_stage2_protocol_context,
            stage2_protocol.resolve_stage2_protocol_context,
        )
        self.assertIs(
            pipeline_runner.chat_native_core_family_targets,
            stage2_protocol.chat_native_core_family_targets,
        )
        self.assertEqual(
            pipeline_runner.STAGE2_PROTOCOL_PROFILE_VERSIONS,
            stage2_protocol.STAGE2_PROTOCOL_PROFILE_VERSIONS,
        )

    def test_feature_vocabulary_uses_canonical_family_precedence(self):
        self.assertIn("NumHBA", feature_vocabulary.d4_feature_name_allowlist("count"))
        self.assertEqual(feature_vocabulary.expected_family_for_feature("NumHBA"), "count")
        self.assertEqual(feature_vocabulary.expected_family_for_feature("aromatic_ring"), "pattern_presence")
        self.assertEqual(feature_vocabulary.expected_family_for_feature("unknown_feature"), "")

    def test_feature_vocabulary_fails_visibly_when_rule_registry_import_breaks(self):
        real_import = __import__

        def broken_registry_import(name, *args, **kwargs):
            if name == "morace.rule_registry":
                raise RuntimeError("rule registry import failed")
            return real_import(name, *args, **kwargs)

        for reader in (
            lambda: feature_vocabulary.registry_supported_names("DESCRIPTOR_REGISTRY", ("MolWt",)),
            lambda: feature_vocabulary.d4_feature_name_allowlist("descriptor"),
        ):
            with self.subTest(reader=reader), patch("builtins.__import__", new=broken_registry_import):
                with self.assertRaisesRegex(RuntimeError, "rule registry import failed"):
                    reader()

    def test_feature_vocabulary_rejects_unknown_registry_name(self):
        with self.assertRaisesRegex(AttributeError, "UNKNOWN_REGISTRY"):
            feature_vocabulary.registry_supported_names("UNKNOWN_REGISTRY", ("MolWt",))


if __name__ == "__main__":
    unittest.main()

import os
import unittest
from unittest.mock import patch


def load_projection_prompts():
    try:
        from morace import projection_prompts
    except ModuleNotFoundError as exc:
        if exc.name == "rdkit":
            raise unittest.SkipTest("projection prompts require the core RDKit dependency") from exc
        raise
    return projection_prompts


class ProjectionPromptTests(unittest.TestCase):
    def test_stage1_prompt_normalizes_input_whitespace(self):
        prompts = load_projection_prompts()

        rendered = prompts.build_stage1_user_prompt("  task text  ", "  context text  ")

        self.assertEqual(rendered, "task text\n\nContext for this draft:\ncontext text")

    def test_llama_native_prompt_preserves_pure_json_contract(self):
        prompts = load_projection_prompts()

        rendered = prompts.build_stage2_user_prompt(
            task_brief="BBBP classification",
            draft_text="Lipophilicity may increase permeability.",
            source="literature",
            stage2_protocol_profile="llama_native",
        )

        self.assertIn("native Llama projection", rendered)
        self.assertIn('Use source exactly "literature"', rendered)
        self.assertIn("Do not output markdown, code fences, END_JSON", rendered)
        self.assertTrue(rendered.endswith('"rationale":"short reason"}]}'))

    def test_chat_native_marker_policy_controls_prompt_and_stop_string(self):
        prompts = load_projection_prompts()

        with patch.dict(
            os.environ,
            {"STAGE2_TERMINATION_POLICY": "end_json_marker_single_rule"},
            clear=False,
        ):
            rendered = prompts.build_stage2_user_prompt(
                task_brief="BBBP classification",
                draft_text="Aromaticity may matter.",
                source="data",
                stage2_protocol_profile="chat_native_core",
            )
            self.assertEqual(prompts.stage2_stop_strings("chat_native_core"), ["END_JSON"])

        self.assertIn("Termination policy: end_json_marker_single_rule.", rendered)
        self.assertIn("write END_JSON and stop", rendered)

    def test_family_target_prompt_validates_family_and_micro_vocabulary(self):
        prompts = load_projection_prompts()

        rendered = prompts.build_stage2_family_target_single_rule_prompt(
            task_brief="BBBP classification",
            draft_text="Hydrogen-bond acceptor count may matter.",
            target_family="count",
            target_vocab_policy="micro_canonical_options",
            target_vocab_options=["NumHBA", "NumHBD"],
        )

        self.assertIn("feature_family must be exactly count", rendered)
        self.assertIn("feature_name must be exactly one of: NumHBA, NumHBD", rendered)
        with self.assertRaises(ValueError):
            prompts.build_stage2_family_target_single_rule_prompt(
                task_brief="task",
                draft_text="draft",
                target_family="unsupported",
            )

    def test_compact_vocabulary_fails_visibly_when_rule_registry_import_breaks(self):
        prompts = load_projection_prompts()
        real_import = __import__

        def broken_registry_import(name, *args, **kwargs):
            if name == "morace.rule_registry":
                raise RuntimeError("rule registry import failed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", new=broken_registry_import):
            with self.assertRaisesRegex(RuntimeError, "rule registry import failed"):
                prompts._chat_native_core_compact_vocabulary_block()

    def test_pipeline_runner_keeps_prompt_import_surface(self):
        prompts = load_projection_prompts()
        from morace import pipeline_runner

        self.assertIs(pipeline_runner.build_stage1_user_prompt, prompts.build_stage1_user_prompt)
        self.assertIs(pipeline_runner.build_stage2_user_prompt, prompts.build_stage2_user_prompt)
        self.assertIs(pipeline_runner._stage2_stop_strings, prompts.stage2_stop_strings)
        self.assertIs(pipeline_runner._stage2_max_new_tokens, prompts.stage2_max_new_tokens)


if __name__ == "__main__":
    unittest.main()

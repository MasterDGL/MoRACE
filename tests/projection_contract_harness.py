import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from morace.generation_runtime import GenerationRuntime


PROJECTION_ENV_KEYS = {
    "USE_CROSS_MODEL_PROJECTOR",
    "STAGE2_PROTOCOL_PROFILE",
    "STAGE2_TERMINATION_POLICY",
    "CHAT_NATIVE_CORE_TERMINATION_POLICY",
    "STAGE2_RULE_COUNT_POLICY",
    "CHAT_NATIVE_CORE_RULE_COUNT_POLICY",
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
    "_MORACE_STAGE2_PROTOCOL_PROFILE",
}


def load_pipeline_runner():
    try:
        from morace import pipeline_runner
    except ModuleNotFoundError as exc:
        if exc.name == "rdkit":
            raise unittest.SkipTest("projection contracts require the core RDKit dependency") from exc
        raise
    return pipeline_runner


def descriptor_rule(feature_name="MolLogP"):
    return {
        "source": "literature",
        "feature_family": "descriptor",
        "feature_name": feature_name,
        "feature_mode": "numeric_value",
        "relation_to_target": "positive",
        "threshold": None,
        "rationale": "signal",
    }


def pattern_rule(feature_name="aromatic_ring"):
    return {
        "source": "literature",
        "feature_family": "pattern_presence",
        "feature_name": feature_name,
        "feature_mode": "binary_presence",
        "relation_to_target": "positive",
        "threshold": None,
        "rationale": "motif",
    }


def count_rule(feature_name="NumHBA"):
    return {
        "source": "literature",
        "feature_family": "count",
        "feature_name": feature_name,
        "feature_mode": "count_value",
        "relation_to_target": "positive",
        "threshold": None,
        "rationale": "count",
    }


def projection_attempt(rule, raw_output="raw-json", generated_tokens=11):
    rules = [copy.deepcopy(rule)] if rule else []
    parse_meta = {
        "max_nesting_depth": 0,
        "rules_field_inside_rule_count": 0,
        "failure_stage": "",
        "failure_reason": "",
        "failure_label": "",
    }
    return {
        "raw_output": raw_output,
        "repair_output": "",
        "projected_rules": rules,
        "strict_rules": copy.deepcopy(rules),
        "strict_parse_meta": dict(parse_meta),
        "d4_cleaned_json": "",
        "d4_strict_rules": [],
        "d4_strict_parse_meta": dict(parse_meta),
        "chat_native_extraction_diagnostics": {},
        "lenient_rules": copy.deepcopy(rules),
        "lenient_parse_meta": dict(parse_meta),
        "repair_strict_rules": [],
        "repair_strict_parse_meta": dict(parse_meta),
        "repair_lenient_rules": [],
        "repair_lenient_parse_meta": dict(parse_meta),
        "repair_used": False,
        "parse_meta": {
            "max_nesting_depth": 0,
            "rules_field_inside_rule_count": 0,
        },
        "stage2_backend": "contract-backend",
        "stage2_avg_logprob": -0.25,
        "stage2_logprob_norm": 0.75,
        "stage2_projection_generated_token_count": generated_tokens,
        "stage2_repair_generated_token_count": 0,
        "stage2_generated_token_count": generated_tokens,
        "stage2_projection_call_count": 1,
        "stage2_repair_call_count": 0,
        "stage2_total_call_count": 1,
    }


def canonical_digest(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fake_generator_class(model_family, constructor_calls):
    class FakeLocalTextGenerator:
        def __init__(self, model_name, model_path, seed=42):
            constructor_calls.append(
                {"model_name": model_name, "model_path": model_path, "seed": seed}
            )
            self.model_family = model_family

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    return FakeLocalTextGenerator


class ProjectionContractTests(unittest.TestCase):
    def clean_projection_environment(self):
        return patch.dict(
            os.environ,
            {key: value for key, value in os.environ.items() if key not in PROJECTION_ENV_KEYS},
            clear=True,
        )

    def run_projector(
        self,
        pipeline_runner,
        *,
        model_family,
        attempt_outputs,
        environment=None,
        allow_heuristic_fallback=False,
        heuristic_rules=None,
        enable_literature_retry=False,
    ):
        constructor_calls = []
        attempt_calls = []
        remaining_outputs = iter(attempt_outputs)

        def fake_attempt(_backend, **kwargs):
            attempt_calls.append(copy.deepcopy(kwargs))
            return copy.deepcopy(next(remaining_outputs))

        prompt_records = {
            "prompt-1": {
                "prompt_id": "prompt-1",
                "source": "literature",
                "base_prompt": "Predict BBBP.",
                "task_brief": "BBBP classification",
            }
        }
        draft_outputs = [
            {
                "prompt_id": "prompt-1",
                "source": "literature",
                "draft_model": {
                    "llama": "llama-3.1-8b-instruct",
                    "chemdfm": "chemdfm-v1.5-8b",
                    "qwen": "qwen3.5-9b",
                }.get(model_family, "generic-chat-model"),
                "draft_model_path": f"/models/{model_family}",
                "draft_id": 0,
                "batch_id": "batch-1",
                "stage1_user_prompt": "stage-1 prompt",
                "draft_text": "Lipophilicity, aromatic motifs, and hydrogen-bond counts may matter.",
            }
        ]
        environment = dict(environment or {})
        with (
            self.clean_projection_environment(),
            patch.dict(os.environ, environment),
            patch.object(
                pipeline_runner,
                "LocalTextGenerator",
                fake_generator_class(model_family, constructor_calls),
            ),
            patch.object(pipeline_runner, "LITERATURE_RECALL_RETRY_ENABLE", enable_literature_retry),
            patch.object(
                pipeline_runner,
                "heuristic_extract_rules_from_draft",
                return_value=copy.deepcopy(list(heuristic_rules or [])),
            ),
            patch.object(pipeline_runner, "_run_projection_attempt", side_effect=fake_attempt),
        ):
            records = pipeline_runner.project_drafts_to_rules(
                prompt_records,
                draft_outputs,
                pipeline_runner.ModelSpec(f"{model_family}-projector", f"/models/{model_family}"),
                max_new_tokens=128,
                temperature=0.65,
                top_p=0.92,
                num_beams=3,
                max_input_tokens=1536,
                allow_heuristic_fallback=allow_heuristic_fallback,
                allow_repair_pass=True,
                seed=73,
                runtime=GenerationRuntime({
                    **environment,
                    "LITERATURE_RECALL_RETRY_ENABLE": str(int(enable_literature_retry)),
                }),
            )
        return records, constructor_calls, attempt_calls

    def test_projection_attempt_preserves_generation_and_repair_call_contract(self):
        pipeline_runner = load_pipeline_runner()
        valid_repair = (
            '{"rules":[{"source":"literature","feature_family":"descriptor",'
            '"feature_name":"MolLogP","feature_mode":"numeric_value",'
            '"relation_to_target":"positive","threshold":null,"rationale":"signal"}]}'
        )

        class RecordingBackend:
            def __init__(self):
                self.calls = []
                self.outputs = iter(
                    [
                        {"text": "not-json", "generated_token_count": 7, "backend": "fake"},
                        {"text": valid_repair, "generated_token_count": 5, "backend": "fake"},
                    ]
                )

            def generate_structured(self, messages, **kwargs):
                self.calls.append({"messages": copy.deepcopy(messages), "kwargs": copy.deepcopy(kwargs)})
                return [next(self.outputs)]

        backend = RecordingBackend()
        with self.clean_projection_environment():
            result = pipeline_runner._run_projection_attempt(
                backend,
                system_prompt="base-system",
                user_prompt="base-user",
                source="literature",
                provenance={"draft_model": "llama", "draft_id": 0},
                max_new_tokens=192,
                temperature=0.7,
                top_p=0.9,
                num_beams=2,
                max_input_tokens=1024,
                allow_repair_pass=True,
                projector_family="llama",
                stage2_protocol_profile="llama_native",
            )

        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(backend.calls[0]["messages"], [("base-system", "base-user")])
        self.assertEqual(
            backend.calls[0]["kwargs"],
            {
                "max_new_tokens": 192,
                "temperature": 0.7,
                "top_p": 0.9,
                "num_beams": 2,
                "max_input_tokens": 1024,
                "preserve_head_tokens": 256,
                "preserve_tail_tokens": 512,
                "stop_strings": ["```"],
                "stage2_protocol_profile": "llama_native",
                "return_metadata": True,
            },
        )
        self.assertEqual(
            backend.calls[1]["messages"][0][0],
            pipeline_runner._stage2_repair_system_prompt_for("llama_native"),
        )
        with patch.dict(
            os.environ,
            {"_MORACE_STAGE2_PROTOCOL_PROFILE": "llama_native"},
        ):
            expected_repair_prompt = pipeline_runner.build_stage2_repair_prompt(
                "not-json",
                "literature",
                projector_family="llama",
            )
        self.assertEqual(
            backend.calls[1]["messages"][0][1],
            expected_repair_prompt,
        )
        self.assertEqual(
            backend.calls[1]["kwargs"],
            {
                "max_new_tokens": 192,
                "temperature": 0.0,
                "top_p": 1.0,
                "num_beams": 2,
                "max_input_tokens": 1024,
                "preserve_head_tokens": 64,
                "preserve_tail_tokens": 768,
                "stop_strings": ["```"],
                "stage2_protocol_profile": "llama_native",
                "return_metadata": True,
            },
        )
        self.assertTrue(result["repair_used"])
        self.assertEqual(result["stage2_generated_token_count"], 12)
        self.assertEqual(result["stage2_total_call_count"], 2)
        self.assertEqual(result["projected_rules"][0]["feature_name"], "MolLogP")

    def test_llama_base_projection_call_and_record_contract(self):
        pipeline_runner = load_pipeline_runner()
        outputs = [projection_attempt(descriptor_rule())]

        records, constructor_calls, attempt_calls = self.run_projector(
            pipeline_runner,
            model_family="llama",
            attempt_outputs=outputs,
        )

        self.assertEqual(
            constructor_calls,
            [{"model_name": "llama-projector", "model_path": "/models/llama", "seed": 73}],
        )
        self.assertEqual(len(attempt_calls), 1)
        call = attempt_calls[0]
        self.assertEqual(call["stage2_protocol_profile"], "llama_native")
        self.assertEqual(call["max_new_tokens"], 256)
        self.assertEqual(call["temperature"], 0.65)
        self.assertEqual(call["top_p"], 0.92)
        self.assertEqual(call["num_beams"], 3)
        self.assertEqual(call["max_input_tokens"], 1536)
        self.assertTrue(call["allow_repair_pass"])
        self.assertEqual(call["system_prompt"], pipeline_runner._stage2_system_prompt_for("literature", "llama_native"))
        self.assertEqual(
            call["user_prompt"],
            pipeline_runner.build_stage2_user_prompt(
                task_brief="BBBP classification",
                draft_text="Lipophilicity, aromatic motifs, and hydrogen-bond counts may matter.",
                source="literature",
                short_schema=True,
                projector_family="llama",
                stage2_protocol_profile="llama_native",
            ),
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["projection_variant_role"], "base_single_rule")
        self.assertEqual(record["stage2_protocol_profile"], "llama_native")
        self.assertEqual(record["projector_model_path"], "/models/llama")
        self.assertEqual(record["parsed_rules"], outputs[0]["projected_rules"])
        self.assertEqual(record["stage2_total_call_count"], 1)
        self.assertTrue(record["selected_for_merge"])
        self.assertEqual(record["selected_for_merge_reason"], "rank1_primary")
        self.assertEqual(
            canonical_digest(records),
            "01922becc22b65ee3ec3a8cf58c95a493404138779069fab4f6f5c3f93694732",
        )

    def test_qwen_legacy_projection_preserves_native_token_budget_and_profile(self):
        pipeline_runner = load_pipeline_runner()
        outputs = [projection_attempt(descriptor_rule())]

        records, _, attempt_calls = self.run_projector(
            pipeline_runner,
            model_family="qwen",
            attempt_outputs=outputs,
        )

        self.assertEqual(len(attempt_calls), 1)
        call = attempt_calls[0]
        self.assertEqual(call["stage2_protocol_profile"], "qwen_legacy_d2")
        self.assertEqual(call["max_new_tokens"], 128)
        self.assertEqual(call["temperature"], 0.65)
        self.assertEqual(call["top_p"], 0.92)
        self.assertEqual(call["num_beams"], 3)
        self.assertEqual(records[0]["stage2_protocol_profile"], "qwen_legacy_d2")
        self.assertEqual(records[0]["stage2_protocol_profile_source"], "auto_inferred")
        self.assertEqual(
            canonical_digest(records),
            "4294021fcdd32d888f4b3ba3161b2279f80ba83ef6b16afa9c2a8b7c8f96606c",
        )

    def test_literature_retry_preserves_second_call_and_accounting_contract(self):
        pipeline_runner = load_pipeline_runner()
        outputs = [
            projection_attempt(descriptor_rule(), raw_output="base-output", generated_tokens=11),
            projection_attempt(count_rule(), raw_output="retry-output", generated_tokens=7),
        ]

        records, _, attempt_calls = self.run_projector(
            pipeline_runner,
            model_family="llama",
            attempt_outputs=outputs,
            enable_literature_retry=True,
        )

        self.assertEqual(len(attempt_calls), 2)
        self.assertEqual(attempt_calls[0]["temperature"], 0.65)
        self.assertEqual(attempt_calls[1]["temperature"], 0.0)
        self.assertEqual(attempt_calls[1]["top_p"], 1.0)
        self.assertEqual(attempt_calls[1]["num_beams"], 3)
        self.assertIn("native Llama literature retry", attempt_calls[1]["user_prompt"])
        self.assertIn("Already covered roots: descriptor::MolLogP", attempt_calls[1]["user_prompt"])
        self.assertTrue(records[0]["underdiverse_retry_used"])
        self.assertTrue(records[0]["underdiverse_retry_success"])
        self.assertEqual(records[0]["stage2_base_call_count"], 1)
        self.assertEqual(records[0]["stage2_retry_call_count"], 1)
        self.assertEqual(records[0]["stage2_generated_token_count"], 18)
        retry_records = [record for record in records if record["projection_variant_role"] == "literature_recall_retry"]
        self.assertEqual(len(retry_records), 1)
        self.assertTrue(retry_records[0]["literature_recall_retry_kept"])
        self.assertEqual(retry_records[0]["literature_recall_novel_root_count"], 1)
        self.assertEqual(
            canonical_digest(records),
            "f40194f0a14ff437f4185500f627197986d3bd08776667c89c231746adb5ff62",
        )

    def test_heuristic_fallback_preserves_rescue_status_and_record_fields(self):
        pipeline_runner = load_pipeline_runner()
        outputs = [projection_attempt(None, raw_output="unparseable")]
        fallback_rule = pattern_rule()

        records, _, attempt_calls = self.run_projector(
            pipeline_runner,
            model_family="llama",
            attempt_outputs=outputs,
            allow_heuristic_fallback=True,
            heuristic_rules=[fallback_rule],
        )

        self.assertEqual(len(attempt_calls), 1)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["projection_status"], "heuristic_rescued")
        self.assertTrue(record["heuristic_rescued"])
        self.assertEqual(record["heuristic_rule_count"], 1)
        self.assertEqual(record["projected_rule_count"], 0)
        self.assertEqual(record["projector_parsed_rules"], [])
        self.assertEqual(record["parsed_rules"], [fallback_rule])
        self.assertFalse(record["strict_projector_valid"])
        self.assertTrue(record["schema_valid"])
        self.assertEqual(
            canonical_digest(records),
            "81a1b8269024decd1e5d3b077c886b880f7acc2096683acbfd3046726cfe6660",
        )

    def test_chat_native_independent_variant_preserves_call_order_and_parameters(self):
        pipeline_runner = load_pipeline_runner()
        outputs = [
            projection_attempt(descriptor_rule(), raw_output="base"),
            projection_attempt(pattern_rule(), raw_output="independent"),
        ]
        environment = {
            "STAGE2_PROTOCOL_PROFILE": "chat_native_core",
            "STAGE2_TERMINATION_POLICY": "pure_json_single_rule",
            "STAGE2_RULE_COUNT_POLICY": "two_independent_single_rule_projections",
        }

        records, _, attempt_calls = self.run_projector(
            pipeline_runner,
            model_family="llama",
            attempt_outputs=outputs,
            environment=environment,
        )

        self.assertEqual(len(attempt_calls), 2)
        self.assertNotIn("projection_variant_role", attempt_calls[0]["provenance"])
        self.assertEqual(
            attempt_calls[1]["provenance"]["projection_variant_role"],
            "independent_single_rule_2",
        )
        self.assertEqual(attempt_calls[0]["temperature"], 0.65)
        self.assertEqual(attempt_calls[0]["top_p"], 0.92)
        self.assertEqual(attempt_calls[1]["temperature"], 0.0)
        self.assertEqual(attempt_calls[1]["top_p"], 1.0)
        self.assertEqual(attempt_calls[1]["num_beams"], 3)
        self.assertIn("Previous roots: descriptor::MolLogP", attempt_calls[1]["user_prompt"])
        self.assertEqual(
            [record["projection_variant_role"] for record in records],
            ["base_single_rule", "independent_single_rule_2"],
        )
        independent = records[1]
        self.assertTrue(independent["independent_single_projection_kept"])
        self.assertEqual(independent["independent_single_rule_novel_root_keys"], ["pattern_presence::aromatic_ring"])
        self.assertTrue(independent["selected_for_merge"])
        self.assertEqual(
            canonical_digest(records),
            "2903602d97c3100faf257a85c057661926940ec0e36ca07d6880be7b002c6c72",
        )

    def test_chemdfm_family_target_variants_preserve_order_and_family_contract(self):
        pipeline_runner = load_pipeline_runner()
        outputs = [
            projection_attempt(descriptor_rule(), raw_output="base"),
            projection_attempt(pattern_rule(), raw_output="pattern"),
            projection_attempt(count_rule(), raw_output="count"),
        ]
        environment = {
            "STAGE2_PROTOCOL_PROFILE": "chemdfm_native",
            "STAGE2_FAMILY_TARGET_POLICY": "family_targeted_single_rule_variants",
            "STAGE2_TARGET_VOCAB_POLICY": "micro_canonical_options",
            "STAGE2_TARGET_THINNING_POLICY": "none",
        }

        records, _, attempt_calls = self.run_projector(
            pipeline_runner,
            model_family="chemdfm",
            attempt_outputs=outputs,
            environment=environment,
        )

        self.assertEqual(len(attempt_calls), 3)
        self.assertEqual(
            [call["provenance"].get("projection_variant_role", "base") for call in attempt_calls],
            ["base", "family_target_pattern_presence", "family_target_count"],
        )
        self.assertEqual(attempt_calls[0]["temperature"], 0.65)
        self.assertEqual([call["temperature"] for call in attempt_calls[1:]], [0.0, 0.0])
        self.assertEqual([call["top_p"] for call in attempt_calls[1:]], [1.0, 1.0])
        self.assertIn("feature_family must be exactly pattern_presence", attempt_calls[1]["user_prompt"])
        self.assertIn("feature_family must be exactly count", attempt_calls[2]["user_prompt"])
        self.assertEqual(
            [record["projection_variant_role"] for record in records],
            ["base_single_rule", "family_target_pattern_presence", "family_target_count"],
        )
        self.assertEqual([record["family_target_variant_kept"] for record in records[1:]], [True, True])
        self.assertEqual(
            [record["target_feature_name"] for record in records[1:]],
            ["aromatic_ring", "NumHBA"],
        )
        self.assertEqual(
            canonical_digest(records),
            "03175a2708ceb1f03beb5bfe1f369aa57f145463819077fce8ffa6011990c9b2",
        )


if __name__ == "__main__":
    unittest.main()

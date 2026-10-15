import pytest

from morace.diagnostics import (
    logp_ingestion,
    logp_matching,
    logp_promotion,
    synthesis_feature_stability,
)


@pytest.mark.parametrize("rule,trimmed,untrimmed", [
    ({"support_sources": [" data "], "source": " data "}, "data", " data "),
    ({"support_sources": [" literature ", "data"], "source": "literature"}, "multi", "literature"),
    ({"support_sources": [None, 0, False], "source": " literature "}, "literature", " literature "),
    ({"source": "", "primary_source": " data "}, "data", " data "),
])
def test_source_whitespace_remains_specific_to_each_diagnostic(rule, trimmed, untrimmed):
    for module in (logp_ingestion, logp_matching, logp_promotion):
        assert module._source_bucket(rule) == trimmed
    assert synthesis_feature_stability._source_bucket(rule) == untrimmed


@pytest.mark.parametrize("rule,expected", [
    ({}, "unknown|unknown|null|null|null"),
    ({"canonical_key": " custom "}, "custom"),
    ({"canonical_key": None}, "None"),
    ({"feature_family": " descriptor ", "feature_name": " MolLogP ",
      "feature_mode": None, "relation_to_target": " POSITIVE ",
      "threshold": {"op": " >= ", "value": 0}}, "descriptor|MolLogP|none|positive|>=0"),
])
def test_canonical_keys_keep_explicit_values_and_threshold_tokens(rule, expected):
    for module in (logp_ingestion, logp_matching):
        assert module._effective_canonical_key(rule) == expected


@pytest.mark.parametrize("value,expected", [
    ([" data ", None, 0, False, " "], ["data", "None", "0", "False"]),
    (" data ", ["data"]),
    (("data",), []),
    (None, []),
])
def test_source_lists_keep_original_coercion(value, expected):
    for module in (logp_promotion, synthesis_feature_stability):
        assert module._as_list(value) == expected


@pytest.mark.parametrize("selected", [True, "false", 1])
def test_projection_selection_keeps_truthiness_order_and_rule_identity(selected):
    first = {"feature_name": "MolLogP"}
    second = {"feature_name": "TPSA"}
    records = [
        {"selected_for_merge": False, "parsed_rules": [second]},
        {"selected_for_merge": selected, "parsed_rules": [first, second]},
        {"selected_for_merge": True, "parsed_rules": None},
    ]
    for module in (logp_ingestion, logp_matching, logp_promotion):
        assert module._selected_rules_from_projection(None) == []
        actual = module._selected_rules_from_projection(records)
        assert actual == [first, second]
        assert actual[0] is first and actual[1] is second

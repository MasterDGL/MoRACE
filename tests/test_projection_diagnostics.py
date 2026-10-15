import unittest


class ProjectionDiagnosticsTests(unittest.TestCase):
    def load_module(self):
        try:
            from morace import projection_diagnostics
        except ModuleNotFoundError as exc:
            if exc.name == "rdkit":
                self.skipTest("projection diagnostics require the core RDKit dependency")
            raise
        return projection_diagnostics

    def test_audits_json_suffix_after_end_marker(self):
        diagnostics = self.load_module()
        audit = diagnostics.audit_first_json_output('{"rules":[]} END_JSON trailing')

        self.assertTrue(audit["first_json_parseable"])
        self.assertTrue(audit["json_closes_before_suffix"])
        self.assertEqual(audit["suffix_after_first_json"], "trailing")

    def test_records_and_normalizes_strict_failures(self):
        diagnostics = self.load_module()
        counter = diagnostics.new_strict_failure_diagnostics()
        diagnostics.record_strict_failure(
            counter,
            {"stage": "initial", "reason": "invalid_json", "label": "initial:invalid_json"},
        )

        self.assertEqual(counter["total_failures"], 1)
        self.assertEqual(diagnostics.normalize_failure_counter(counter["reason_counts"]), {"invalid_json": 1})


if __name__ == "__main__":
    unittest.main()

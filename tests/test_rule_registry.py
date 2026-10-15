import unittest

from rdkit import Chem

from morace.rule_registry import get_pattern_callable, get_pattern_code


class RuleRegistryTests(unittest.TestCase):
    def test_nitro_group_matches_charged_form(self):
        mol = Chem.MolFromSmiles("c1ccc([N+](=O)[O-])cc1")
        self.assertTrue(get_pattern_callable("nitro_group")(mol))

    def test_fused_ring_detects_naphthalene(self):
        mol = Chem.MolFromSmiles("c1cccc2ccccc12")
        self.assertTrue(get_pattern_callable("fused_ring")(mol))

    def test_smarts_pattern_code_uses_molecule_input(self):
        code = get_pattern_code("halogen")
        self.assertIsNotNone(code)
        self.assertIn("Chem.MolFromSmarts", code)
        self.assertNotIn("placeholder", code)

        positive_namespace = {"Chem": Chem, "mol": Chem.MolFromSmiles("CCCl")}
        negative_namespace = {"Chem": Chem, "mol": Chem.MolFromSmiles("CCC")}
        self.assertEqual(eval(code, positive_namespace), 1)
        self.assertEqual(eval(code, negative_namespace), 0)


if __name__ == "__main__":
    unittest.main()

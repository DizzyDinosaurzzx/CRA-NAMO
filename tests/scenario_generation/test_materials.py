"""The themed vocabularies only measure anything while the tables misread them.

These tests fail the moment an edit to ``risk.RISK_LABELS``,
``risk.RISK_KEYWORDS`` or ``llm_difficulty.MATERIAL_*`` makes a trap legible.
That would not break any run; it would quietly turn the random corpus into one
where the offline heuristic and an LLM agree, which is the one outcome the
corpus exists to rule out.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "CRA-NAMO"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import risk
from llm_difficulty import MATERIAL_MU_RHO
from scenario_generation import materials
from scenario_generation.profiles import PROFILES


class MaterialCatalogTests(unittest.TestCase):
    def test_every_theme_still_separates_the_arms(self):
        problems = materials.validate_all()
        self.assertEqual(problems, [], "\n".join(problems))

    def test_risk_traps_read_as_low_and_reveal_something_readable(self):
        for theme in materials.THEMES.values():
            for material in theme.risk_traps:
                with self.subTest(theme=theme.name, label=material.label):
                    self.assertEqual(materials.heuristic_risk(material.label),
                                     risk.LOW)
                    self.assertNotEqual(material.risk, risk.LOW)
                    self.assertNotEqual(
                        materials.heuristic_risk(material.reveals), risk.LOW)

    def test_weight_traps_are_priced_far_below_their_truth(self):
        for theme in materials.THEMES.values():
            for material in theme.weight_traps:
                with self.subTest(theme=theme.name, label=material.label):
                    true_mu_rho = material.mu * sum(material.density) / 2.0
                    guessed = materials.heuristic_mu_rho(material.label)
                    self.assertLess(guessed * 3.0, true_mu_rho)

    def test_no_trap_is_anchored_to_a_table_value(self):
        """An anchored label pins the LLM to the table, erasing the contrast."""
        for theme in materials.THEMES.values():
            for material in (*theme.risk_traps, *theme.weight_traps):
                with self.subTest(theme=theme.name, label=material.label):
                    self.assertFalse(materials.is_anchored(material.label))

    def test_plain_materials_are_read_correctly(self):
        """The honest half of the vocabulary must stay honest."""
        unknown = MATERIAL_MU_RHO["unknown"]
        for theme in materials.THEMES.values():
            for material in theme.plain:
                with self.subTest(theme=theme.name, label=material.label):
                    self.assertEqual(material.risk, risk.LOW)
                    self.assertEqual(materials.heuristic_risk(material.label),
                                     risk.LOW)
                    guessed = materials.heuristic_mu_rho(material.label)
                    true_mu_rho = material.mu * sum(material.density) / 2.0
                    if guessed != unknown:
                        self.assertLess(abs(guessed - true_mu_rho),
                                        max(true_mu_rho, guessed))

    def test_every_profile_names_a_real_theme(self):
        for profile in PROFILES.values():
            with self.subTest(profile=profile.name):
                self.assertTrue(profile.themes)
                for name in profile.themes:
                    self.assertIn(name, materials.THEMES)


if __name__ == "__main__":
    unittest.main()

"""Regression and invariant tests for generated scenarios."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "CRA-NAMO"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenario_generation import RandomScenarioRequest, ScenarioGenerator
from scenario_generation.fingerprint import fingerprint_manifest
from scenario_generation.naming import experiment_id, map_stem, run_stem
from scenario_generation.serialization import candidate_from_manifest
from dynamics import MoveTo


class SeededRandomTests(unittest.TestCase):
    def setUp(self):
        self.generator = ScenarioGenerator()

    def test_same_seed_is_byte_stable(self):
        request = RandomScenarioRequest(seed=42, profile="hospital")
        first = self.generator.generate(request)
        second = self.generator.generate(request)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(json.dumps(first.manifest, sort_keys=True),
                         json.dumps(second.manifest, sort_keys=True))

    def test_exact_obstacle_and_dynamic_counts(self):
        generated = self.generator.generate(RandomScenarioRequest(
            seed=42, profile="depot", obstacle_count=16, event_count=1))
        self.assertEqual(len(generated.scenario.movable), 16)
        dynamic_oids = {
            event.effect.oid for event in generated.scenario.events
            if isinstance(event.effect, MoveTo)
        }
        self.assertEqual(len(dynamic_oids), 1)
        self.assertEqual(
            generated.validation.metrics["dynamic_obstacle_count"], 1)

    def test_artifacts_use_experiment_numbers(self):
        generated = self.generator.generate(RandomScenarioRequest(
            seed=42, profile="hospital"))
        self.assertEqual(experiment_id(42), "experiment_0042")
        self.assertEqual(experiment_id(-3), "experiment_neg_0003")
        self.assertTrue(map_stem(generated.manifest).startswith(
            "experiment_0042_hospital_map_"))
        self.assertTrue(run_stem(generated.manifest, "no-llm").startswith(
            "experiment_0042_hospital_no-llm_"))

    def test_manifest_round_trip(self):
        generated = self.generator.generate(RandomScenarioRequest(seed=7))
        restored = candidate_from_manifest(generated.manifest)
        self.assertEqual(len(restored.static), len(generated.scenario.static))
        self.assertEqual(len(restored.movable), len(generated.scenario.movable))
        self.assertEqual(restored.start, generated.scenario.start)
        self.assertEqual(fingerprint_manifest(generated.manifest),
                         generated.fingerprint)
        self.assertTrue(generated.scenario.cfg.save_frames)
        self.assertTrue(restored.cfg.save_frames)
        self.assertEqual(restored.metadata["calibration"],
                         generated.scenario.metadata["calibration"])
        self.assertEqual(
            [p.metadata["oracle"] for p in restored.decision_points],
            [p.metadata["oracle"]
             for p in generated.scenario.decision_points])

    def test_one_hundred_seeds_are_valid_and_diverse(self):
        fingerprints = set()
        topology_signatures = set()
        families = set()
        oracle_actions = set()
        for seed in range(100):
            generated = self.generator.generate(RandomScenarioRequest(
                seed=seed, profile="benchmark"))
            self.assertTrue(generated.validation.accepted)
            metrics = generated.validation.metrics
            # Six gates, plus a dynamic decision when the map carries an event.
            self.assertGreaterEqual(metrics["decision_count"], 6)
            fingerprints.add(generated.fingerprint)
            families.add(metrics["topology_family"])
            if "room_count" in metrics:          # staged topologies have no graph
                topology_signatures.add((
                    metrics["room_count"], metrics["graph_edge_count"],
                    metrics["cycle_rank"], metrics["dead_end_count"],
                    metrics["junction_count"], metrics["shortest_hops"]))
                self.assertGreaterEqual(metrics["angled_wall_count"], 1)
                self.assertTrue(all(hops >= 0
                                    for hops in metrics["decision_detour_hops"]))
            self.assertGreaterEqual(metrics["calibrated_decision_count"], 3)
            self.assertGreaterEqual(metrics["blind_decision_count"], 1)
            self.assertGreaterEqual(metrics["oracle_margin_min"], 0.05)
            self.assertLessEqual(metrics["oracle_margin_max"], 0.30)
            for point in generated.scenario.decision_points:
                oracle = point.metadata["oracle"]
                oracle_actions.add(oracle["best_action"])
                if not oracle.get("qualitative"):
                    self.assertGreaterEqual(oracle["relative_margin"], 0.05)
                    self.assertLessEqual(oracle["relative_margin"], 0.30)
        self.assertEqual(len(fingerprints), 100)
        self.assertLessEqual({"room_graph", "junction"}, families)
        self.assertGreaterEqual(len(topology_signatures), 25)
        self.assertLessEqual({"move", "detour", "safe_move"}, oracle_actions)

    def test_hidden_decision_has_belief_and_ground_truth_labels(self):
        # Whether a given map carries a hidden-difficulty gate is a seeded
        # choice, so look across a handful of them.
        hidden = None
        for seed in range(12):
            generated = self.generator.generate(RandomScenarioRequest(
                seed=seed, profile="hospital"))
            hidden = next(
                (point for point in generated.scenario.decision_points
                 if point.kind == "hidden_difficulty"), None)
            if hidden is not None:
                break
        self.assertIsNotNone(hidden, "no seed produced a hidden-difficulty gate")
        oracle = hidden.metadata["oracle"]
        self.assertEqual(oracle["stage"], "contact")
        self.assertEqual(oracle["belief_action_before_contact"], "move")
        self.assertEqual(oracle["best_action"], "detour")

    def test_blind_gates_are_invisible_to_the_offline_tables(self):
        """The whole point of a blind gate: the keyword table must misread it."""
        import risk
        seen = set()
        for seed in range(20):
            generated = self.generator.generate(RandomScenarioRequest(
                seed=seed, profile="earthquake"))
            for point in generated.scenario.decision_points:
                if point.kind not in ("blind_risk", "blind_weight",
                                      "risk_or_risk"):
                    continue
                label = point.metadata["material"]
                seen.add(point.kind)
                self.assertEqual(
                    risk.keyword_level(label), risk.LOW,
                    f"{label} is already legible to the keyword table")
                if point.kind == "blind_risk":
                    self.assertNotEqual(point.metadata["true_risk"], risk.LOW)
                    self.assertNotEqual(
                        risk.keyword_level(point.metadata["reveals"]), risk.LOW,
                        "contact must teach a blind arm something")
        self.assertTrue(seen, "no blind gate was generated in twenty seeds")

    def test_event_and_decision_oids_exist(self):
        generated = self.generator.generate(RandomScenarioRequest(
            seed=3, profile="depot", event_count=(1, 1)))
        oids = {obs.oid for obs in generated.scenario.movable}
        for point in generated.scenario.decision_points:
            self.assertLessEqual(set(point.involved_oids), oids)
        for event in generated.scenario.events:
            self.assertIn(event.effect.oid, oids)

    def test_dynamic_profile_varies_event_templates(self):
        names = set()
        timed_actions = set()
        for seed in range(60):
            generated = self.generator.generate(RandomScenarioRequest(
                seed=seed, profile="depot", event_count=(1, 1)))
            names.update(event.name for event in generated.scenario.events)
            timed_actions.update(
                point.metadata["oracle"]["best_action"]
                for point in generated.scenario.decision_points
                if point.kind == "wait_or_replan")
        self.assertTrue(any("trolley" in name for name in names))
        self.assertTrue(any("load changes" in name for name in names))
        self.assertEqual(timed_actions, {"wait", "replan"})


if __name__ == "__main__":
    unittest.main()

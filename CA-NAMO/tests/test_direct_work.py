import unittest

from config import Config
from obstacle import MovableObstacle
from perception import Belief
from planner import OnlineNAMO
import scenarios


class DirectWorkTests(unittest.TestCase):
    def test_cost_config_has_no_work_multiplier(self):
        cfg = Config()
        self.assertEqual(cfg.work_source, "direct")
        self.assertFalse(hasattr(cfg, "lambda_w"))
        with self.assertRaises(ValueError):
            Config(lambda_d=-1.0)

    def test_work_is_revealed_with_obstacle(self):
        obs = MovableObstacle(
            x=1.0, y=2.0, l=1.0, d=1.0,
            material="box", difficulty=99.0, work=7.5, oid=3)
        belief = Belief(roadmap=None, cfg=Config())
        belief.perceived[obs.oid] = obs

        self.assertEqual(obs.observation()["work"], 7.5)
        self.assertEqual(belief.get_work(obs.oid), 7.5)
        self.assertEqual(obs.moved_copy(2.0, 3.0).work, 7.5)

    def test_direct_mode_uses_j_equals_motion_plus_work(self):
        scene = scenarios.load("two_doors")
        scene["cfg"].save_frames = False
        scene["cfg"].verbose = False
        sim = OnlineNAMO(
            scene["workspace"], scene["static"], scene["movable"],
            scene["start"], scene["goal"], scene["cfg"])

        result = sim.run()

        self.assertTrue(result.success)
        self.assertEqual(result.work_source, "direct")
        self.assertEqual(result.llm_calls, 0)
        self.assertAlmostEqual(
            result.J, result.walk_cost + result.work_cost, places=3)


if __name__ == "__main__":
    unittest.main()

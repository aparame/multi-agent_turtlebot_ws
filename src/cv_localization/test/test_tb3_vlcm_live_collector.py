import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

import numpy as np

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from cv_localization.tb3_vlcm_live_collector import (
    EpisodeStep,
    EpisodeTarWriter,
    GoalSamplingConfig,
    RewardConfig,
    adjacency_from_distances,
    compute_step_rewards,
    distance,
    distance_matrix,
    goals_are_valid,
    sample_random_goals,
)


class _DeterministicRandom:
    def __init__(self, values):
        self.values = list(values)
        self.index = 0

    def uniform(self, _low, _high):
        value = self.values[self.index % len(self.values)]
        self.index += 1
        return value


class TestTB3VLCMLiveCollector(unittest.TestCase):
    def test_random_goals_respect_spacing_and_safe_bounds(self):
        names = ["tb_1", "tb_2", "tb_3"]
        positions = {
            "tb_1": (0.0, 0.0),
            "tb_2": (0.6, 0.0),
            "tb_3": (-0.6, 0.0),
        }
        config = GoalSamplingConfig(
            workspace_width_m=3.048,
            workspace_height_m=3.048,
            wall_margin_m=0.2,
            goal_min_separation_m=0.5,
            goal_min_robot_distance_m=0.35,
        )
        rng = _DeterministicRandom([
            -1.0, -1.0,
            1.0, 1.0,
            -1.0, 1.0,
        ])

        goals = sample_random_goals(names, positions, config, rng)

        self.assertTrue(goals_are_valid(goals, positions, config))
        goal_items = list(goals.values())
        for i, first in enumerate(goal_items):
            for second in goal_items[i + 1:]:
                self.assertGreaterEqual(distance(first, second), 0.5)

    def test_rewards_include_progress_success_proximity_and_failure(self):
        config = RewardConfig(
            goal_radius_m=0.12,
            proximity_penalty_distance_m=0.2,
            success_reward=5.0,
            proximity_penalty=-1.0,
            failure_terminal_reward=-25.0,
        )
        distances = np.array(
            [
                [0.0, 0.1, 1.0],
                [0.1, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        agent_rewards, scalar = compute_step_rewards(
            previous_distances=[0.5, 0.2, 0.4],
            current_distances=[0.4, 0.1, 0.5],
            reached_before=[False, False, False],
            reached_now=[False, True, False],
            distances=distances,
            config=config,
            terminal_failure=True,
        )

        self.assertAlmostEqual(agent_rewards[0], 0.1)
        self.assertAlmostEqual(agent_rewards[1], 5.1)
        self.assertAlmostEqual(agent_rewards[2], -0.1)
        self.assertAlmostEqual(scalar, np.mean(agent_rewards) - 1.0 - 25.0)

    def test_episode_tar_writer_outputs_ma_vlcm_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = EpisodeTarWriter(Path(tmp), "episode_test")
            dist = distance_matrix([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
            writer.write_step(EpisodeStep(
                key="episode_test_step0000",
                frame_png=b"\x89PNG\r\n\x1a\nfake",
                state={
                    "episode_meta": {"done": True},
                    "agents": [],
                    "reward": 1.0,
                    "cumulative_reward": 1.0,
                },
                reward=1.0,
                episode_reward=1.0,
                distances=dist,
                adjacency=adjacency_from_distances(dist),
            ))
            tar_path = writer.close()

            with tarfile.open(tar_path, "r") as tar:
                names = set(tar.getnames())
                prefix = "episode_test_step0000"
                self.assertIn(f"{prefix}.overhead.png", names)
                self.assertIn(f"{prefix}.state.json", names)
                self.assertIn(f"{prefix}.reward.json", names)
                self.assertIn(f"{prefix}.episode_reward.json", names)
                self.assertIn(f"{prefix}.dist.npy", names)
                self.assertIn(f"{prefix}.adj.npy", names)
                state = json.load(tar.extractfile(f"{prefix}.state.json"))
                reward = json.load(tar.extractfile(f"{prefix}.reward.json"))
                dist_bytes = tar.extractfile(f"{prefix}.dist.npy").read()

            self.assertTrue(state["episode_meta"]["done"])
            self.assertEqual(reward, 1.0)
            loaded_dist = np.load(io.BytesIO(dist_bytes))
            np.testing.assert_allclose(loaded_dist, dist)

    def test_collection_launch_declares_expected_nodes(self):
        launch_path = (
            Path(__file__).resolve().parents[1]
            / "launch"
            / "cv_rl_vlcm_collect.launch.py"
        )
        text = launch_path.read_text()
        self.assertIn("cv_rl_direct_controller", text)
        self.assertIn("cv_mppi_direct_gui", text)
        self.assertIn("tb3_vlcm_live_collector", text)
        self.assertIn("goal_min_separation_m", text)
        self.assertIn("goal_min_robot_distance_m", text)
        self.assertIn("default_value='from_config'", text)
        self.assertIn("vlcm_collection", text)


if __name__ == "__main__":
    unittest.main()

import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from cv_localization.cv_rl_direct_controller import (  # noqa: E402
    Pose2D,
    RlControllerParams,
    apply_goal_termination_commands,
    build_policy_inputs,
    filter_commands_for_safety,
    obs_dim_for_agents,
    safety_stop_reason,
    scale_policy_action,
    state_dim_for_agents,
    validate_aero_marl_root,
)


class TestCvRlDirectController(unittest.TestCase):
    def test_policy_input_shapes_for_three_agents(self):
        params = RlControllerParams()
        poses = [
            Pose2D(-0.8, -0.6, 0.0),
            Pose2D(0.0, 0.0, 1.0),
            Pose2D(0.8, 0.6, -1.0),
        ]
        goals = [
            Pose2D(0.8, 0.6, 0.0),
            Pose2D(-0.8, 0.6, 0.0),
            Pose2D(-0.8, -0.6, 0.0),
        ]
        last_cmds = np.zeros((3, 2), dtype=np.float32)

        obs, share_obs, edge_index = build_policy_inputs(
            poses,
            goals,
            last_cmds,
            params,
        )

        self.assertEqual(obs.shape, (3, obs_dim_for_agents(3)))
        self.assertEqual(obs.shape, (3, 17))
        self.assertEqual(share_obs.shape, (3, state_dim_for_agents(3)))
        self.assertEqual(share_obs.shape, (3, 24))
        self.assertEqual(edge_index.shape, (2, 9))

    def test_action_scaling_and_safety_filter_respect_velocity_limits(self):
        params = RlControllerParams(max_v_mps=0.1, max_w_radps=1.0)

        high = scale_policy_action([100.0, 100.0], params)
        low = scale_policy_action([-100.0, -100.0], params)

        self.assertLessEqual(high[0], params.max_v_mps + 1e-6)
        self.assertGreaterEqual(high[0], 0.0)
        self.assertLessEqual(low[0], params.max_v_mps + 1e-6)
        self.assertGreaterEqual(low[0], 0.0)
        self.assertLessEqual(abs(high[1]), params.max_w_radps + 1e-6)
        self.assertLessEqual(abs(low[1]), params.max_w_radps + 1e-6)

        poses = [
            Pose2D(-0.8, -0.6, 0.0),
            Pose2D(0.0, 0.0, 0.0),
            Pose2D(0.8, 0.6, 0.0),
        ]
        unsafe_commands = np.array(
            [
                [10.0, 10.0],
                [10.0, -10.0],
                [10.0, 0.5],
            ],
            dtype=np.float32,
        )
        filtered = filter_commands_for_safety(poses, unsafe_commands, params)

        self.assertTrue(np.all(filtered[:, 0] >= 0.0))
        self.assertTrue(np.all(filtered[:, 0] <= params.max_v_mps + 1e-6))
        self.assertTrue(np.all(np.abs(filtered[:, 1]) <= params.max_w_radps + 1e-6))

    def test_goal_termination_zeroes_reached_agent_command(self):
        params = RlControllerParams(goal_radius_m=0.12, max_v_mps=0.1, max_w_radps=1.0)
        poses = [
            Pose2D(0.03, 0.04, 0.0),
            Pose2D(0.0, 0.0, 0.0),
            Pose2D(0.8, 0.6, 0.0),
        ]
        goals = [
            Pose2D(0.0, 0.0, 0.0),
            Pose2D(1.0, 0.0, 0.0),
            Pose2D(-1.0, 0.0, 0.0),
        ]
        commands = np.array(
            [
                [0.08, 0.7],
                [0.06, -0.4],
                [0.05, 0.3],
            ],
            dtype=np.float32,
        )

        terminated = apply_goal_termination_commands(poses, goals, commands, params)

        np.testing.assert_allclose(terminated[0], [0.0, 0.0])
        np.testing.assert_allclose(terminated[1:], commands[1:])

    def test_aero_marl_root_validation_requires_mat_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mat_dir = root / "mat"
            mat_dir.mkdir()
            (mat_dir / "config.py").write_text("# test config\n")

            self.assertEqual(validate_aero_marl_root(root), root)

    def test_aero_marl_root_validation_rejects_checkpoint_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "transformer_800.pt"
            checkpoint.write_bytes(b"checkpoint")

            with self.assertRaises(NotADirectoryError):
                validate_aero_marl_root(checkpoint)

    def test_safety_stop_reasons_cover_stale_bounds_and_spacing(self):
        params = RlControllerParams()
        goals = [
            Pose2D(0.8, 0.6, 0.0),
            Pose2D(-0.8, 0.6, 0.0),
            Pose2D(-0.8, -0.6, 0.0),
        ]
        safe_poses = [
            Pose2D(-0.8, -0.6, 0.0),
            Pose2D(0.0, 0.0, 1.0),
            Pose2D(0.8, 0.6, -1.0),
        ]

        stale = safety_stop_reason(
            safe_poses,
            goals,
            [0.0, params.pose_timeout_s + 0.1, 0.0],
            params,
        )
        self.assertEqual(stale, "stale CV pose for tb_2")

        out_of_bounds = safe_poses.copy()
        out_of_bounds[0] = Pose2D(2.0, 0.0, 0.0)
        boundary = safety_stop_reason(out_of_bounds, goals, [0.0, 0.0, 0.0], params)
        self.assertEqual(boundary, "tb_1 pose outside safe boundary")

        close = [
            Pose2D(0.0, 0.0, 0.0),
            Pose2D(0.1, 0.0, 0.0),
            Pose2D(0.8, 0.6, 0.0),
        ]
        spacing = safety_stop_reason(close, goals, [0.0, 0.0, 0.0], params)
        self.assertTrue(spacing.startswith("unsafe live spacing between tb_1 and tb_2"))


if __name__ == "__main__":
    unittest.main()

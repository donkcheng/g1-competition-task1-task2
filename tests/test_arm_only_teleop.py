from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from task2_right_arm_teleop import (  # noqa: E402
    RightArmIK,
    apply_deadzone,
    clamp_sphere,
    controller_position_in_root,
    ik_solution_is_feasible,
    rate_limit_vector,
    rotation_between_vectors,
    tracked_rotation_in_root,
    update_deadman,
)


class ArmOnlyMathTests(unittest.TestCase):
    def test_large_ik_residual_is_rejected(self) -> None:
        self.assertTrue(ik_solution_is_feasible(0.03, math.radians(10)))
        self.assertFalse(ik_solution_is_feasible(0.15, math.radians(10)))
        self.assertFalse(ik_solution_is_feasible(0.03, math.radians(70)))

    def test_unity_controller_position_maps_to_robot_root(self) -> None:
        root = np.asarray([1, 2, 3, 0, 0, 0, 1], dtype=np.float64)
        controller = np.asarray([2, 4, 6, 0, 0, 0, 1], dtype=np.float64)
        np.testing.assert_allclose(
            controller_position_in_root(controller, root), [-3, -1, 2]
        )

    def test_official_pico_root_axes(self) -> None:
        root = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float64)
        # Unity physical forward is -Z and right is +X for this PICO stream.
        forward = np.asarray([0, 0, -1, 0, 0, 0, 1], dtype=np.float64)
        right = np.asarray([1, 0, 0, 0, 0, 0, 1], dtype=np.float64)
        np.testing.assert_allclose(
            controller_position_in_root(forward, root), [1, 0, 0], atol=1e-8
        )
        np.testing.assert_allclose(
            controller_position_in_root(right, root), [0, -1, 0], atol=1e-8
        )

    def test_controller_relative_rotation_is_preserved(self) -> None:
        root = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float64)
        anchor = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float64)
        turned = anchor.copy()
        turned[3:] = Rotation.from_euler("y", 30, degrees=True).as_quat()
        anchor_rotation = tracked_rotation_in_root(anchor, root)
        turned_rotation = tracked_rotation_in_root(turned, root)
        relative = anchor_rotation.T @ turned_rotation
        self.assertAlmostEqual(
            float(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec())),
            math.radians(30),
            places=7,
        )

    def test_workspace_sphere_is_hard_bounded(self) -> None:
        result = clamp_sphere(np.asarray([2.0, 0, 0]), np.zeros(3), 0.42)
        self.assertAlmostEqual(float(np.linalg.norm(result)), 0.42)

    def test_joint_rate_limit(self) -> None:
        result = rate_limit_vector(
            np.zeros(2), np.asarray([1.0, -1.0]), max_rate=0.6, dt=0.02
        )
        np.testing.assert_allclose(result, [0.012, -0.012])

    def test_axis_deadzone_is_symmetric(self) -> None:
        self.assertEqual(apply_deadzone(0.1), 0.0)
        self.assertAlmostEqual(apply_deadzone(1.0), 1.0)
        self.assertAlmostEqual(apply_deadzone(-1.0), -1.0)

    def test_grip_deadman_has_hysteresis_but_drops_on_stale_xr(self) -> None:
        self.assertTrue(update_deadman(False, 0.60, True, False))
        self.assertTrue(update_deadman(True, 0.30, True, False))
        self.assertFalse(update_deadman(True, 0.10, True, False))
        self.assertFalse(update_deadman(True, 1.00, False, False))
        self.assertFalse(update_deadman(True, 1.00, True, True))

    def test_vector_alignment_rotation(self) -> None:
        rotation = rotation_between_vectors([1, 0, 0], [0, 1, 0])
        np.testing.assert_allclose(rotation @ [1, 0, 0], [0, 1, 0], atol=1e-8)


class RightArmIKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.solver = RightArmIK()

    def test_solver_uses_only_physical_joint_limits(self) -> None:
        elbow_index = self.solver.active_indices[3]
        wrist_pitch_index = self.solver.active_indices[5]
        self.assertAlmostEqual(
            self.solver.lower[3],
            self.solver.robot.lower_joint_limits[elbow_index],
        )
        self.assertAlmostEqual(
            self.solver.upper[5],
            self.solver.robot.upper_joint_limits[wrist_pitch_index],
        )

    def test_solver_tracks_position_and_orientation(self) -> None:
        body = self.solver.robot.get_body_actuated_joints(
            self.solver.robot.get_default_body_pose()
        )
        body[25] = 0.75
        full = self.solver.full_configuration(body)
        position, rotation = self.solver.tool_pose(full)
        target = position + np.asarray([0.04, 0.0, 0.02])
        target_rotation = rotation @ Rotation.from_euler(
            "z", 12, degrees=True
        ).as_matrix()
        solved, position_error, rotation_error = self.solver.solve(
            body, target, target_rotation
        )
        self.assertLess(position_error, 0.03)
        self.assertLess(rotation_error, np.deg2rad(5.0))
        self.assertFalse(np.allclose(solved[26:29], body[26:29]))

    def test_exact_seed_pose_does_not_move_on_takeover(self) -> None:
        body = self.solver.robot.get_body_actuated_joints(
            self.solver.robot.get_default_body_pose()
        )
        body[25] = 1.0
        full = self.solver.full_configuration(body)
        position, rotation = self.solver.tool_pose(full)
        solved, position_error, rotation_error = self.solver.solve(
            body,
            position,
            rotation,
        )
        np.testing.assert_allclose(
            solved[22:29], body[22:29], atol=1.0e-9
        )
        self.assertLess(position_error, 1.0e-9)
        self.assertLess(rotation_error, 1.0e-9)

    def test_far_target_returns_closest_physical_solution_without_freezing(self) -> None:
        body = self.solver.robot.get_body_actuated_joints(
            self.solver.robot.get_default_body_pose()
        )
        body[25] = 0.75
        full = self.solver.full_configuration(body)
        shoulder = self.solver.shoulder_position(full)
        neutral = self.solver.neutral_orientation(body)
        solved, _, _ = self.solver.solve(
            body, shoulder + np.asarray([2.0, 0.0, 0.0]), neutral
        )
        self.assertFalse(np.allclose(solved[22:29], body[22:29]))
        self.assertTrue(np.all(solved[22:29] >= self.solver.lower - 1.0e-8))
        self.assertTrue(np.all(solved[22:29] <= self.solver.upper + 1.0e-8))

    def test_logged_forward_raise_does_not_flip_to_downward_branch(self) -> None:
        """Regression for the real PICO trajectory that exposed the bad IK branch."""
        body = self.solver.robot.get_body_actuated_joints(
            self.solver.robot.get_default_body_pose()
        )
        body[25] = 1.0
        full = self.solver.full_configuration(body)
        position, rotation = self.solver.tool_pose(full)
        solved, position_error, rotation_error = self.solver.solve(
            body,
            position + np.asarray([0.31, 0.05, 0.46]),
            rotation,
        )
        self.assertLess(position_error, 0.06)
        self.assertLess(float(solved[22]), 0.0)
        self.assertLess(rotation_error, math.radians(20.0))


if __name__ == "__main__":
    unittest.main()

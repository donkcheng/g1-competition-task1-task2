from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from pico_wrist_mapping import (  # noqa: E402
    BilateralControllerPoseBridge,
    ControllerPoseFilter,
    LEFT_HAND_JOINT_INDEX,
    RIGHT_HAND_JOINT_INDEX,
    RightControllerPoseBridge,
    WristMappingConfig,
    quaternion_angle_rad,
)


def z_rotation_xyzw(degrees: float) -> np.ndarray:
    half = math.radians(degrees) / 2.0
    return np.asarray([0.0, 0.0, math.sin(half), math.cos(half)])


class ControllerPoseFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = WristMappingConfig(
            max_angular_speed_deg_s=180.0,
            max_linear_speed_m_s=1.0,
            smoothing_time_constant_s=0.0,
        )

    def test_quaternion_sign_flip_is_not_motion(self) -> None:
        pose_filter = ControllerPoseFilter(self.config)
        first = np.r_[np.zeros(3), z_rotation_xyzw(25.0)]
        second = first.copy()
        second[3:] *= -1.0
        output_first = pose_filter.update(first, 10.0)
        output_second = pose_filter.update(second, 10.02)
        self.assertLess(
            quaternion_angle_rad(output_first[3:], output_second[3:]), 1.0e-8
        )
        self.assertGreater(float(np.dot(output_first[3:], output_second[3:])), 0.0)

    def test_ninety_degree_jump_is_rate_limited(self) -> None:
        pose_filter = ControllerPoseFilter(self.config)
        first = np.r_[np.zeros(3), z_rotation_xyzw(0.0)]
        second = np.r_[np.zeros(3), z_rotation_xyzw(90.0)]
        output_first = pose_filter.update(first, 5.0)
        output_second = pose_filter.update(second, 5.02)
        delta_deg = math.degrees(
            quaternion_angle_rad(output_first[3:], output_second[3:])
        )
        self.assertLessEqual(delta_deg, 3.6001)

    def test_position_jump_is_rate_limited(self) -> None:
        pose_filter = ControllerPoseFilter(self.config)
        first = np.r_[np.zeros(3), z_rotation_xyzw(0.0)]
        second = np.r_[[1.0, 0.0, 0.0], z_rotation_xyzw(0.0)]
        output_first = pose_filter.update(first, 2.0)
        output_second = pose_filter.update(second, 2.02)
        self.assertLessEqual(
            float(np.linalg.norm(output_second[:3] - output_first[:3])), 0.020001
        )


class RightControllerPoseBridgeTests(unittest.TestCase):
    def test_only_right_hand_joint_is_replaced(self) -> None:
        bridge = RightControllerPoseBridge(WristMappingConfig())
        body = np.zeros((24, 7), dtype=np.float64)
        body[:, 6] = 1.0
        body[RIGHT_HAND_JOINT_INDEX, :3] = [0.35, -0.18, 1.05]
        controller = np.asarray([0.4, -0.2, 1.1, 0.0, 0.0, 0.0, 1.0])
        bridge.register(body, controller, 1.0)
        mapped, used = bridge.substitute(body)
        self.assertTrue(used)
        np.testing.assert_allclose(mapped[RIGHT_HAND_JOINT_INDEX], controller)
        np.testing.assert_allclose(mapped[:RIGHT_HAND_JOINT_INDEX], body[:RIGHT_HAND_JOINT_INDEX])
        self.assertFalse(np.shares_memory(mapped, body))

    def test_unpaired_body_falls_back_unchanged(self) -> None:
        bridge = RightControllerPoseBridge(WristMappingConfig())
        body = np.zeros((24, 7), dtype=np.float64)
        body[:, 6] = 1.0
        mapped, used = bridge.substitute(body)
        self.assertFalse(used)
        self.assertIs(mapped, body)

    def test_final_wxyz_quaternion_keeps_one_hemisphere(self) -> None:
        bridge = RightControllerPoseBridge(WristMappingConfig())
        first = np.asarray([0.8, 0.1, -0.2, 0.55])
        first /= np.linalg.norm(first)
        stable_first = bridge.stabilize_output_wxyz(first)
        stable_second = bridge.stabilize_output_wxyz(-first)
        np.testing.assert_allclose(stable_second, stable_first, atol=1.0e-9)


class BilateralControllerPoseBridgeTests(unittest.TestCase):
    def test_both_inferred_hand_joints_are_replaced(self) -> None:
        bridge = BilateralControllerPoseBridge(WristMappingConfig())
        body = np.zeros((24, 7), dtype=np.float64)
        body[:, 6] = 1.0
        body[LEFT_HAND_JOINT_INDEX, :3] = [0.35, 0.18, 1.05]
        body[RIGHT_HAND_JOINT_INDEX, :3] = [0.35, -0.18, 1.05]
        left = np.asarray([0.4, 0.2, 1.1, 0.0, 0.0, 0.0, 1.0])
        right = np.asarray([0.4, -0.2, 1.1, 0.0, 0.0, 0.0, 1.0])
        bridge.register(body, left, right, 1.0)
        mapped, used = bridge.substitute(body)
        self.assertTrue(used)
        np.testing.assert_allclose(mapped[LEFT_HAND_JOINT_INDEX], left)
        np.testing.assert_allclose(mapped[RIGHT_HAND_JOINT_INDEX], right)
        np.testing.assert_allclose(mapped[:LEFT_HAND_JOINT_INDEX], body[:LEFT_HAND_JOINT_INDEX])
        self.assertFalse(np.shares_memory(mapped, body))

    def test_bilateral_quaternions_are_continuous_independently(self) -> None:
        bridge = BilateralControllerPoseBridge(WristMappingConfig())
        left = np.asarray([0.8, 0.1, -0.2, 0.55])
        right = np.asarray([0.6, -0.4, 0.2, 0.65])
        left /= np.linalg.norm(left)
        right /= np.linalg.norm(right)
        stable_left = bridge.stabilize_output_wxyz(0, left)
        stable_right = bridge.stabilize_output_wxyz(1, right)
        np.testing.assert_allclose(
            bridge.stabilize_output_wxyz(0, -left), stable_left, atol=1.0e-9
        )
        np.testing.assert_allclose(
            bridge.stabilize_output_wxyz(1, -right), stable_right, atol=1.0e-9
        )


if __name__ == "__main__":
    unittest.main()

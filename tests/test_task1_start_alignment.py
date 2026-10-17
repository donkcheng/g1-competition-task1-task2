from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from task1_start_alignment import (
    AlignmentError,
    alignment_error,
    detect_marker,
    guidance,
    is_aligned,
)


class AlignmentMathTests(unittest.TestCase):
    CONFIG = {
        "forward_tolerance_m": 0.04,
        "lateral_tolerance_m": 0.04,
        "yaw_tolerance_deg": 2.0,
    }

    @staticmethod
    def camera_from_marker_for_camera_pose(right: float, forward: float, yaw_left_deg: float):
        theta = math.radians(yaw_left_deg)
        pitch = math.radians(35.0)
        marker_from_reference_camera = np.eye(4)
        marker_from_reference_camera[:3, :3] = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, -math.sin(pitch), math.cos(pitch)],
             [0.0, -math.cos(pitch), -math.sin(pitch)]]
        )
        yaw = np.array(
            [[math.cos(theta), -math.sin(theta), 0.0],
             [math.sin(theta), math.cos(theta), 0.0],
             [0.0, 0.0, 1.0]]
        )
        marker_from_current_camera = marker_from_reference_camera.copy()
        marker_from_current_camera[:3, :3] = yaw @ marker_from_reference_camera[:3, :3]
        marker_from_current_camera[:3, 3] = [right, forward, 1.0]
        return np.linalg.inv(marker_from_current_camera)

    @staticmethod
    def reference_camera_from_marker():
        return AlignmentMathTests.camera_from_marker_for_camera_pose(0.0, 0.0, 0.0)

    def test_identical_pose_is_aligned(self):
        ref = self.reference_camera_from_marker()
        error = alignment_error(ref, ref)
        self.assertTrue(is_aligned(error, self.CONFIG))

    def test_translation_signs(self):
        ref = self.reference_camera_from_marker()
        cur = self.camera_from_marker_for_camera_pose(0.10, 0.20, 0.0)
        error = alignment_error(ref, cur)
        self.assertAlmostEqual(error.forward_m, 0.20, places=5)
        self.assertAlmostEqual(error.left_m, -0.10, places=5)
        self.assertIn("向后移动", " ".join(guidance(error, self.CONFIG)))
        self.assertIn("向左横移", " ".join(guidance(error, self.CONFIG)))

    def test_yaw_guidance(self):
        error = AlignmentError(0.0, 0.0, 8.0)
        self.assertIn("向右转", guidance(error, self.CONFIG)[0])

    def test_generated_marker_is_detected(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 42, 180)
        frame = np.full((480, 640, 3), 255, np.uint8)
        frame[150:330, 230:410] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        config = {
            "dictionary": "DICT_4X4_50",
            "marker_id": 42,
            "marker_size_m": 0.18,
            "camera_width": 640,
            "camera_height": 480,
            "camera_matrix": [[607.853, 0.0, 327.088], [0.0, 608.059, 242.929], [0.0, 0.0, 1.0]],
            "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
        pose = detect_marker(frame, config)
        self.assertIsNotNone(pose)
        self.assertGreater(pose.camera_from_marker[2, 3], 0.0)


if __name__ == "__main__":
    unittest.main()

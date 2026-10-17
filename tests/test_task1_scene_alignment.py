from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from task1_scene_alignment import (
    SceneError,
    choose_motion,
    execute_odometry_bounded_motion,
    estimate_scene_error,
    guidance,
    is_aligned,
)


class SceneAlignmentTests(unittest.TestCase):
    @staticmethod
    def textured_scene() -> np.ndarray:
        rng = np.random.default_rng(42)
        image = rng.integers(35, 210, size=(480, 640), dtype=np.uint8)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        cv2.line(image, (40, 90), (590, 120), 255, 14)
        cv2.rectangle(image, (90, 190), (280, 390), 15, 12)
        cv2.putText(image, "TASK1", (330, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 245, 5)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def test_identical_scene_is_aligned(self):
        scene = self.textured_scene()
        error, _ = estimate_scene_error(scene, scene.copy())
        self.assertTrue(is_aligned(error, 8.0, 1.2))
        self.assertGreaterEqual(error.inliers, 15)

    def test_guidance_signs(self):
        right_and_back = SceneError(-20.0, -18.0, 0.0, 1.0, 30, 0.8, 20.0)
        text = " ".join(guidance(right_and_back, 8.0, 1.2))
        self.assertIn("向右", text)
        self.assertIn("向后", text)
        self.assertIn("向左转", guidance(SceneError(0, 0, 3, 1, 30, .8, 1), 8, 1.2)[0])

    def test_auto_motion_is_single_axis_and_low_direction(self):
        yaw = choose_motion(SceneError(0, 0, 3, 1, 30, .8, 1), 8, 1.2, .05, .04, .08)
        self.assertEqual(yaw[:3], (0.0, 0.0, 0.08))
        lateral = choose_motion(SceneError(20, 2, 0, 1, 30, .8, 1), 8, 1.2, .05, .04, .08)
        self.assertEqual(lateral[:3], (0.0, 0.04, 0.0))
        forward = choose_motion(SceneError(1, -20, 0, 1, 30, .8, 1), 8, 1.2, .05, .04, .08)
        self.assertEqual(forward[:3], (-0.05, 0.0, 0.0))

    def test_odometry_bounded_motion_stops_at_small_yaw(self):
        from task1_route import MockBackend

        backend = MockBackend()
        moved, mode = execute_odometry_bounded_motion(
            backend, 0.0, 0.0, 0.2,
            yaw_target_rad=0.015,
            translation_target_m=0.01,
            max_action_time=0.8,
        )
        self.assertEqual(mode, "转角")
        self.assertGreaterEqual(moved, 0.015)
        self.assertLess(moved, 0.05)


if __name__ == "__main__":
    unittest.main()

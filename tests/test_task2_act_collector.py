from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import task2_act_collector as collector


class Task2ActCollectorTests(unittest.TestCase):
    def test_act_schema_has_two_cameras_and_right_only_action(self) -> None:
        features = collector.dataset_features()
        self.assertEqual((30,), features["observation.state"]["shape"])
        self.assertEqual((8,), features["action"]["shape"])
        self.assertEqual(
            [480, 640, 3], features["observation.images.ego_view"]["shape"]
        )
        self.assertEqual(
            [480, 640, 3], features["observation.images.right_wrist"]["shape"]
        )
        self.assertNotIn("observation.images.left_wrist", features)

    def test_right_arm_indices_match_g1_tail(self) -> None:
        self.assertEqual(list(range(22, 29)), collector.RIGHT_ARM_INDICES.tolist())
        self.assertEqual(
            collector.BODY_JOINT_NAMES[22:29], collector.RIGHT_ARM_ACTION_NAMES
        )

    def test_aborted_attempt_reuses_episode_slot(self) -> None:
        pending = collector.PendingEpisode()
        pending.start(1.0)
        zero = np.zeros((480, 640, 3), dtype=np.uint8)
        jpeg = collector.encode_rgb(zero)
        pending.frames.append(
            collector.BufferedFrame(
                jpeg, jpeg, np.zeros(30, dtype=np.float32),
                np.zeros(8, dtype=np.float32),
            )
        )
        self.assertEqual(1, pending.abort())
        self.assertFalse(pending.active)
        self.assertEqual([], pending.frames)

    def test_image_buffer_round_trip(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[..., 0] = 200
        decoded = collector.decode_rgb(collector.encode_rgb(image))
        self.assertEqual(image.shape, decoded.shape)
        self.assertGreater(float(decoded[..., 0].mean()), 190.0)


if __name__ == "__main__":
    unittest.main()

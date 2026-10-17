from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import pico_collection
import task1_route
from common import load_env


class Task1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = task1_route.load_config(Path(__file__).resolve().parents[1] / "config/task1.json")

    def test_example_route_is_valid(self) -> None:
        route = task1_route.load_route(Path(__file__).resolve().parents[1] / "routes/example_route.json")
        self.assertEqual([], task1_route.validate_route(route, self.config, 3))

    def test_speed_limit_rejected(self) -> None:
        route = {
            "schema_version": 1,
            "duration_s": 1,
            "events": [{"t": 0, "vx": 99, "vy": 0, "wz": 0}],
            "poses": [],
        }
        self.assertTrue(task1_route.validate_route(route, self.config))

    def test_marker_order_rejected(self) -> None:
        route = {
            "schema_version": 1,
            "duration_s": 3,
            "events": [
                {"t": 0, "vx": 0, "vy": 0, "wz": 0, "marker": "point_2"},
                {"t": 1, "vx": 0, "vy": 0, "wz": 0, "marker": "point_1"},
                {"t": 2, "vx": 0, "vy": 0, "wz": 0, "marker": "point_3"},
            ],
            "poses": [],
        }
        errors = task1_route.validate_route(route, self.config, 3)
        self.assertTrue(any("marker order" in error for error in errors))

    def test_bad_marker_pose_index_rejected(self) -> None:
        route = {
            "schema_version": 1,
            "duration_s": 1,
            "events": [
                {"t": 0, "vx": 0, "vy": 0, "wz": 0, "marker": "point_1", "pose_index": 5}
            ],
            "poses": [{"t": 0, "x": 0, "y": 0, "yaw": 0}],
        }
        errors = task1_route.validate_route(route, self.config)
        self.assertTrue(any("pose_index" in error for error in errors))

    def test_nan_rejected(self) -> None:
        route = {
            "schema_version": 1,
            "duration_s": 1,
            "events": [{"t": 0, "vx": float("nan"), "vy": 0, "wz": 0}],
            "poses": [],
        }
        self.assertTrue(task1_route.validate_route(route, self.config))

    def test_relative_angle(self) -> None:
        self.assertAlmostEqual(task1_route.wrap_angle(3.5), -2.7831853071795862)

    def test_pose_route_yaw_compensation_rotates_positions_and_headings(self) -> None:
        route = {"poses": [{"x": 1.0, "y": 0.0, "yaw": 0.0}], "events": []}
        rotated = task1_route.rotate_pose_route(route, 3.141592653589793 / 2.0)
        self.assertAlmostEqual(rotated["poses"][0]["x"], 0.0, places=6)
        self.assertAlmostEqual(rotated["poses"][0]["y"], 1.0, places=6)
        self.assertAlmostEqual(rotated["poses"][0]["yaw"], 3.141592653589793 / 2.0)
        self.assertEqual(route["poses"][0]["x"], 1.0)

    def test_terminal_bias_only_changes_last_checkpoint_approach(self) -> None:
        path = [
            {"x": 0.0, "y": 0.0, "yaw": 0.0},
            {"x": 1.0, "y": 0.0, "yaw": 0.0, "marker": "point_1"},
            {"x": 2.0, "y": 0.0, "yaw": 0.0},
            {"x": 3.0, "y": 0.0, "yaw": 0.0, "marker": "point_3"},
            {"x": 3.3, "y": 0.0, "yaw": 0.0},
        ]
        corrected, start_index = task1_route.apply_terminal_right_bias(path, 0.06, 0.8)
        self.assertGreaterEqual(start_index, 2)
        self.assertEqual(0.0, corrected[1]["y"])
        self.assertAlmostEqual(-0.06, corrected[3]["y"], places=6)
        self.assertAlmostEqual(-0.06, corrected[4]["y"], places=6)

    def test_current_g1_service_is_default(self) -> None:
        args = task1_route.build_parser().parse_args(
            ["--config", "config/task1.json", "preflight"]
        )
        self.assertEqual("sport", args.loco_service)

    def test_running_fsm_blocks_external_motion(self) -> None:
        reason = task1_route.motion_fsm_block_reason(802)
        self.assertIsNotNone(reason)
        self.assertIn("R1+X", reason)

    def test_locked_stand_fsm_blocks_external_motion(self) -> None:
        reason = task1_route.motion_fsm_block_reason(4)
        self.assertIsNotNone(reason)
        self.assertIn("锁定站立", reason)

    def test_verified_three_waist_fsm_is_allowed(self) -> None:
        self.assertIsNone(task1_route.motion_fsm_block_reason(501))

    def test_unverified_fsm_is_blocked(self) -> None:
        self.assertIsNotNone(task1_route.motion_fsm_block_reason(0))

    def test_single_velocity_timeout_is_tolerated_but_three_are_not(self) -> None:
        class FakeClient:
            def SetVelocity(self, *_args):
                return 3104

        backend = object.__new__(task1_route.UnitreeBackend)
        backend._client = FakeClient()
        backend._motion_enabled = True
        backend._duration = 0.5
        backend._consecutive_velocity_timeouts = 0
        backend.set_velocity(0.1, 0.0, 0.0)
        backend.set_velocity(0.1, 0.0, 0.0)
        with self.assertRaises(RuntimeError):
            backend.set_velocity(0.1, 0.0, 0.0)

    def test_streaming_velocity_uses_no_reply_and_retries_send_failure(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.results = [3102, 0]
                self.calls: list[tuple[int, str]] = []

            def _CallNoReply(self, api_id, parameter):
                self.calls.append((api_id, parameter))
                return self.results.pop(0)

        backend = object.__new__(task1_route.UnitreeBackend)
        backend._client = FakeClient()
        backend._motion_enabled = True
        backend._duration = 0.8
        backend._velocity_no_reply = True
        backend._consecutive_velocity_timeouts = 0
        backend.set_velocity(0.1, -0.02, 0.03)
        self.assertEqual(2, len(backend._client.calls))
        self.assertTrue(all(api_id == 7105 for api_id, _ in backend._client.calls))
        payload = json.loads(backend._client.calls[-1][1])
        self.assertEqual([0.1, -0.02, 0.03], payload["velocity"])
        self.assertEqual(0.8, payload["duration"])

    def test_pose_replay_finishes_at_final_tolerance(self) -> None:
        route = {
            "schema_version": 1,
            "duration_s": 1,
            "events": [{"t": 0, "vx": 0, "vy": 0, "wz": 0}],
            "poses": [
                {"t": 0, "x": 0, "y": 0, "yaw": 0},
                {"t": 1, "x": 0.05, "y": 0, "yaw": 0},
            ],
        }
        config = dict(self.config)
        config["max_route_duration_s"] = 0.2
        task1_route.replay_pose(task1_route.MockBackend(), route, config)

    def test_path_progress_advances_past_lookahead_samples(self) -> None:
        path = [
            {"x": 0.0, "y": 0.0, "yaw": 0.0},
            {"x": 0.1, "y": 0.0, "yaw": 0.0},
            {"x": 0.2, "y": 0.0, "yaw": 0.0},
            {"x": 0.3, "y": 0.0, "yaw": 0.0},
        ]
        pose = task1_route.Pose2D(0.3, 0.0, 0.0, 0.0)
        self.assertEqual(3, task1_route.advance_path_index(path, 1, pose))

    def test_path_progress_does_not_skip_marker(self) -> None:
        path = [
            {"x": 0.0, "y": 0.0, "yaw": 0.0},
            {"x": 0.1, "y": 0.0, "yaw": 0.0, "marker": "point_1"},
            {"x": 0.2, "y": 0.0, "yaw": 0.0},
        ]
        pose = task1_route.Pose2D(0.2, 0.0, 0.0, 0.0)
        self.assertEqual(1, task1_route.advance_path_index(path, 0, pose))
        self.assertEqual(1, task1_route.advance_path_index(path, 1, pose))

    def test_marker_is_crossed_instead_of_stopping_short(self) -> None:
        route = {
            "schema_version": 1,
            "duration_s": 1,
            "events": [
                {"t": 0, "vx": 0, "vy": 0, "wz": 0},
                {"t": 1, "vx": 0, "vy": 0, "wz": 0, "marker": "point_1", "pose_index": 1},
            ],
            "poses": [
                {"t": 0, "x": 0, "y": 0, "yaw": 0},
                {"t": 1, "x": 0.10, "y": 0, "yaw": 0},
            ],
        }
        config = dict(self.config)
        config["marker_pause_s"] = 0.0
        config["max_route_duration_s"] = 2.0
        backend = task1_route.MockBackend()
        task1_route.replay_pose(backend, route, config)
        self.assertGreaterEqual(backend.pose().x, 0.10)

    def test_marker_gate_rejects_close_but_not_crossed_pose(self) -> None:
        path = [
            {"x": 0.0, "y": 0.0, "yaw": 0.0},
            {"x": 0.1, "y": 0.0, "yaw": 0.0, "marker": "point_1"},
        ]
        before = task1_route.Pose2D(0.09, 0.0, 0.0, 0.0)
        after = task1_route.Pose2D(0.11, 0.04, 0.0, 0.0)
        before_along, _, _, _ = task1_route.marker_gate_error(path, 1, before)
        after_along, after_cross, _, _ = task1_route.marker_gate_error(path, 1, after)
        self.assertLess(before_along, 0.0)
        self.assertGreaterEqual(after_along, 0.0)
        self.assertAlmostEqual(after_cross, 0.04)


class ConfigTests(unittest.TestCase):
    def test_env_parser(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("A=one\nB='two words'\n", encoding="utf-8")
            self.assertEqual({"A": "one", "B": "two words"}, load_env(path))

    def test_dataset_name_guard(self) -> None:
        class Fake:
            dataset_name = ""

        self.assertRegex(pico_collection.safe_dataset_name(Fake(), "button", None), r"^button_\d{8}_\d{6}$")
        with self.assertRaises(ValueError):
            pico_collection.safe_dataset_name(Fake(), "button", "../../bad")

    def test_tensorrt_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            include = Path(directory) / "include"
            include.mkdir()
            (include / "NvInferVersion.h").write_text(
                "#define NV_TENSORRT_MAJOR 10\n"
                "#define NV_TENSORRT_MINOR 13\n"
                "#define NV_TENSORRT_PATCH 3\n",
                encoding="utf-8",
            )
            self.assertEqual((10, 13, 3), pico_collection.tensorrt_version(Path(directory)))

    def test_tensorrt_version_indirect_macros(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            include = Path(directory) / "include"
            include.mkdir()
            (include / "NvInferVersion.h").write_text(
                "#define TRT_MAJOR_ENTERPRISE 10\n"
                "#define TRT_MINOR_ENTERPRISE 13\n"
                "#define TRT_PATCH_ENTERPRISE 3\n"
                "#define NV_TENSORRT_MAJOR TRT_MAJOR_ENTERPRISE\n"
                "#define NV_TENSORRT_MINOR TRT_MINOR_ENTERPRISE\n"
                "#define NV_TENSORRT_PATCH TRT_PATCH_ENTERPRISE\n",
                encoding="utf-8",
            )
            self.assertEqual((10, 13, 3), pico_collection.tensorrt_version(Path(directory)))


if __name__ == "__main__":
    unittest.main()

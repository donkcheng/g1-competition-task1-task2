#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import copy

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/task2_act_inference.py"
SPEC = importlib.util.spec_from_file_location("task2_act_inference", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MeasuredVelocityTests(unittest.TestCase):
    def sample(self, position=0.0, dq=0.2, received=1.0, tick=100):
        body = np.zeros(29)
        body[26] = position
        velocity = np.zeros(7)
        velocity[4] = dq
        return MODULE.RobotMeasurement(
            body, 5.2, np.zeros(29), velocity, received, tick, np.zeros(3)
        )

    def test_delayed_frame_does_not_use_control_loop_dt(self):
        first = self.sample()
        latest = self.sample(position=0.02, received=1.1, tick=200)
        # 100ms between state samples, while a 20ms loop would misreport 1rad/s.
        self.assertAlmostEqual(latest.arm_speed, 0.2)
        self.assertIn("q_fd=+0.200rad/s", latest.speed_detail(first, 1.11))
        self.assertIn("sample_dt=100.0ms", latest.speed_detail(first, 1.11))

    def test_repeated_frame_is_not_reported_as_zero_speed(self):
        sample = self.sample(dq=0.82)
        self.assertGreater(sample.arm_speed, 0.60)
        self.assertIn("q_fd=n/a", sample.speed_detail(sample, 1.02))
        self.assertIn("RightWristRoll", sample.speed_detail(sample, 1.02))

    def test_nonfinite_velocity_is_fail_closed(self):
        for velocity in (float("nan"), float("inf"), -float("inf")):
            sample = self.sample(dq=velocity)
            self.assertEqual(sample.arm_speed, float("inf"))
            self.assertEqual(sample.speed_index, 4)

    def test_body_and_dq_are_taken_from_same_snapshot(self):
        sources = MODULE.RobotSources.__new__(MODULE.RobotSources)
        motors = [SimpleNamespace(q=0.1, dq=0.2, temperature=[30, 31]) for _ in range(35)]
        first = SimpleNamespace(motor_state=motors, tick=100, imu_state=SimpleNamespace(rpy=[0, 0, 0]))
        sources._low_snapshot = (first, 1.0)
        sources._dex_state = (5.2, 1.0)
        measured = sources.measured()
        second = copy.deepcopy(first)
        second.tick = 200
        second.motor_state[26].q = 1.0
        second.motor_state[26].dq = 2.0
        sources._low_snapshot = (second, 2.0)
        self.assertEqual(measured.body[26], 0.1)
        self.assertEqual(measured.arm_dq[4], 0.2)
        self.assertEqual(measured.received_at, 1.0)
        self.assertEqual(measured.source_tick, 100)
        self.assertEqual(sources.measured().arm_dq[4], 2.0)


class StationaryBodyTests(unittest.TestCase):
    def make_guard(self):
        body = np.linspace(-0.15, 0.15, 29)
        return body, MODULE.StationaryBodyGuard(body, [0, 0, 3.13], [1, 2, 0])

    def test_support_is_captured_once_and_does_not_write_legs_or_right_arm(self):
        body, guard = self.make_guard()
        target = body.copy()
        body[:] = 9.0
        commands = [SimpleNamespace(q=42, dq=43, tau=44, kp=45, kd=46)
                    for _ in range(35)]
        original = copy.deepcopy(commands)
        guard.apply_support(SimpleNamespace(motor_cmd=commands))
        for index in range(35):
            if 12 <= index < 22:
                self.assertAlmostEqual(commands[index].q, target[index])
                self.assertGreater(commands[index].kp, 0)
                self.assertGreater(commands[index].kd, 0)
            else:
                self.assertEqual(commands[index], original[index])

    def test_detects_lean_waist_motion_and_steps(self):
        body, guard = self.make_guard()
        cases = [
            (body, [0.09, 0, 3.13], [1, 2, 0], [0, 0, 0], "tilt_delta"),
            (body, [0, 0, 3.13], [1.05, 2, 0], [0, 0, 0], "travel"),
            (body, [0, 0, 3.13], [1, 2, 0], [0.15, 0, 0], "speed"),
        ]
        moved = body.copy()
        moved[14] += 0.11
        cases.append((moved, [0, 0, 3.13], [1, 2, 0], [0, 0, 0], "waist"))
        for state, rpy, position, velocity, expected in cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, guard.reason(state, rpy, position, velocity))

    def test_right_arm_motion_and_yaw_wrap_are_not_body_motion(self):
        body, guard = self.make_guard()
        body[22:29] += 0.8
        self.assertIsNone(guard.reason(body, [0, 0, -3.13], [1, 2, 0], [0, 0, 0]))
        self.assertIsNotNone(guard.reason(body, [float("nan"), 0, 0], [1, 2, 0], [0, 0, 0]))

    def test_actual_publisher_contains_support_and_final_zero_weight(self):
        body, guard = self.make_guard()
        sources = MODULE.RobotSources.__new__(MODULE.RobotSources)
        sources.real = True
        sources.body_guard = guard
        sources.arm_publisher = Mock()
        sources.dex_publisher = Mock()
        sources._crc = SimpleNamespace(Crc=lambda command: 123)
        sources._MotorCmd = sources._MotorCmds = SimpleNamespace
        sources.low_command = SimpleNamespace(motor_cmd=[
            SimpleNamespace(q=0., dq=0., tau=0., kp=0., kd=0.) for _ in range(35)
        ])
        action = np.arange(8) * 0.1
        sources.publish(action, 0.9, 28., 1.8)
        for index in range(12, 22):
            self.assertEqual(sources.low_command.motor_cmd[index].q, body[index])
            self.assertGreater(sources.low_command.motor_cmd[index].kp, 0.)
        for index in range(12, 15):
            self.assertEqual(sources.low_command.motor_cmd[index].kp, 40.)
            self.assertEqual(sources.low_command.motor_cmd[index].kd, 2.)
        for index in range(15, 22):
            self.assertEqual(sources.low_command.motor_cmd[index].kp, 28.)
            self.assertEqual(sources.low_command.motor_cmd[index].kd, 1.8)
        for index in range(12):
            self.assertEqual(sources.low_command.motor_cmd[index].kp, 0.)
        for index in range(7):
            self.assertEqual(sources.low_command.motor_cmd[22 + index].q, action[index])
        sources.publish(action, 0., 16., 1.5)
        self.assertEqual(sources.arm_publisher.Write.call_args.args[0].motor_cmd[29].q, 0.)
        sources.body_guard = None
        with self.assertRaises(RuntimeError):
            sources.publish(action, 0.9, 28., 1.8)
        sources.low_command.motor_cmd[29].q = 0.9
        sources.arm_publisher.Write.reset_mock()
        sources.release_arm_ownership()
        self.assertEqual(sources.arm_publisher.Write.call_count, 3)
        self.assertEqual(sources.low_command.motor_cmd[29].q, 0.)
        sources.arm_publisher.Write.reset_mock()
        sources.dex_publisher.Write.reset_mock()
        sources.emergency_stop_requested = lambda: True
        sources.publish(action, 0.9, 28., 1.8)
        self.assertEqual(sources.low_command.motor_cmd[29].q, 0.)
        self.assertEqual(sources.arm_publisher.Write.call_count, 3)
        sources.dex_publisher.Write.assert_not_called()


class TemporalChunkEnsemblerTests(unittest.TestCase):
    def test_time_alignment_and_interpolation(self):
        ensemble = MODULE.TemporalChunkEnsembler(max_chunks=4, newer_weight_decay=0.0)
        first = np.tile(np.arange(10, dtype=np.float64)[:, None], (1, 8))
        second = first + 10.0
        ensemble.add(MODULE.ActionChunk(1, 0, first, 1.0))
        ensemble.add(MODULE.ActionChunk(1, 2, second, 1.1))
        np.testing.assert_allclose(ensemble.action_at(1, 1.5), 1.5)
        # At absolute step 2.5: first says 2.5, second says 10.5.
        np.testing.assert_allclose(ensemble.action_at(1, 2.5), 6.5)

    def test_episode_change_drops_old_chunks(self):
        ensemble = MODULE.TemporalChunkEnsembler()
        actions = np.zeros((3, 8))
        ensemble.add(MODULE.ActionChunk(1, 0, actions, 1.0))
        ensemble.add(MODULE.ActionChunk(2, 0, actions + 2.0, 2.0))
        self.assertEqual(ensemble.size, 1)
        np.testing.assert_allclose(ensemble.action_at(2, 0), 2.0)

    def test_rejects_non_finite_chunk(self):
        ensemble = MODULE.TemporalChunkEnsembler()
        actions = np.zeros((3, 8))
        actions[1, 2] = np.nan
        with self.assertRaises(ValueError):
            ensemble.add(MODULE.ActionChunk(1, 0, actions, 1.0))


class SmoothActionFilterTests(unittest.TestCase):
    def test_policy_handoff_preserves_existing_hold_command(self):
        staging = self.make_filter()
        policy = self.make_filter()
        measured = np.zeros(8)
        held = np.asarray([0.10, -0.03, 0.04, 0.12, -0.13, 0.03, 0.02, 0.0])
        staging.reset(held)
        policy.handoff_from(staging)
        np.testing.assert_array_equal(policy.command, held)
        # Even if the policy initially predicts the measured pose, the first
        # output must slew, not drop 0.13rad of holding error in one frame.
        command = policy.step(measured, measured, 0.02)
        self.assertLessEqual(float(np.max(np.abs(command - held))), 0.000401)
        self.assertGreater(abs(command[4]), 0.12)

    def make_filter(self):
        return MODULE.SmoothActionFilter(
            np.asarray([-2.0] * 7 + [0.0]),
            np.asarray([2.0] * 7 + [5.2]),
            cutoff_hz=2.0,
            arm_speed=0.5,
            arm_acceleration=1.0,
            gripper_speed=1.0,
            gripper_acceleration=4.0,
            max_follow_error=0.35,
        )

    def test_limits_first_step_speed_and_acceleration(self):
        filt = self.make_filter()
        measured = np.zeros(8)
        filt.reset(measured)
        result = filt.step(np.ones(8) * 2.0, measured, 0.02)
        self.assertTrue(np.all(np.abs(filt.velocity[:7]) <= 0.0200001))
        self.assertTrue(np.all(np.abs(result[:7]) <= 0.0004001))

    def test_clamps_to_measured_follow_window(self):
        filt = self.make_filter()
        measured = np.zeros(8)
        desired = filt.clamp_raw(np.ones(8) * 10.0, measured)
        self.assertTrue(np.all(desired[:7] <= 0.350001))
        self.assertLessEqual(desired[7], 0.400001)

    def test_demonstration_bounds_are_respected(self):
        filt = self.make_filter()
        measured = np.zeros(8)
        desired = filt.clamp_raw(np.ones(8) * -10.0, measured)
        self.assertTrue(np.all(desired[:7] >= -0.350001))
        self.assertGreaterEqual(desired[7], 0.0)


class ArgumentTests(unittest.TestCase):
    def test_arming_rejects_invalid_hold_speed_tracking_and_gains(self):
        parser = MODULE.build_parser(Path(__file__).resolve().parents[1])
        invalid = {
            "preposition_hold_seconds": [0., -1.],
            "preposition_stable_speed": [0., -1., 0.61],
            "preposition_tracking_error": [0., -1.],
            "takeover_kp": [0., -1.],
            "takeover_kd": [0., -1.],
            "waist_hold_kp": [0., -1., 61.],
            "waist_hold_kd": [0., -1., 6.],
            "left_arm_hold_kp": [0., -1., 61.],
            "left_arm_hold_kd": [0., -1., 6.],
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    args = parser.parse_args(["button", "--real"])
                    setattr(args, name, value)
                    with patch.object(parser, "error", side_effect=ValueError), self.assertRaises(ValueError):
                        MODULE.validate_args(args, parser)

    def test_all_nonfinite_float_arguments_are_rejected(self):
        parser = MODULE.build_parser(Path(__file__).resolve().parents[1])
        defaults = parser.parse_args(["button", "--real"])
        for name, value in vars(defaults).items():
            if not isinstance(value, float):
                continue
            for invalid in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(name=name, value=invalid):
                    args = copy.deepcopy(defaults)
                    setattr(args, name, invalid)
                    with patch.object(parser, "error", side_effect=ValueError) as error, self.assertRaises(ValueError):
                        MODULE.validate_args(args, parser)
                    self.assertIn("finite", error.call_args.args[0])

    def test_arming_rejects_timeout_shorter_than_ramp_and_hold(self):
        parser = MODULE.build_parser(Path(__file__).resolve().parents[1])
        args = parser.parse_args(["button", "--real", "--preposition-timeout", "7"])
        with patch.object(parser, "error", side_effect=ValueError) as error, self.assertRaises(ValueError):
            MODULE.validate_args(args, parser)
        self.assertIn("arming ramp", error.call_args.args[0])

    def test_short_automatic_timeout_run_is_valid(self):
        parser = MODULE.build_parser(Path(__file__).resolve().parents[1])
        args = parser.parse_args([
            "button", "--real", "--auto-start", "--max-run-seconds", "3",
            "--exit-after-attempt",
        ])
        MODULE.validate_args(args, parser)
        self.assertTrue(args.auto_start)
        self.assertEqual(args.completion_mode, "timeout")

    def test_short_motion_cycle_still_rejects_impossible_minimum(self):
        parser = MODULE.build_parser(Path(__file__).resolve().parents[1])
        args = parser.parse_args([
            "button", "--real", "--max-run-seconds", "3",
            "--completion-mode", "motion-cycle",
        ])
        with self.assertRaises(SystemExit):
            MODULE.validate_args(args, parser)

    def test_sequence_flags_are_accepted_and_temperature_defaults_to_warning(self):
        project = Path(__file__).resolve().parents[1]
        args = MODULE.build_parser(project).parse_args([
            "button", "--real", "--auto-start", "--exit-after-attempt"
        ])
        self.assertTrue(args.auto_start)
        self.assertTrue(args.exit_after_attempt)
        self.assertEqual(args.thermal_action, "warn")
        self.assertLessEqual(args.arm_speed, 0.35)
        self.assertLessEqual(args.max_follow_error, 0.18)
        self.assertLessEqual(args.takeover_weight, 0.25)
        self.assertGreater(args.preposition_kp, args.takeover_kp)
        self.assertLessEqual(args.preposition_kp, args.right_arm_kp)
        self.assertLess(args.max_follow_error, args.max_tracking_error)

    def test_all_skill_start_targets_are_valid(self):
        project = Path(__file__).resolve().parents[1]
        bounds = project / "config/task2_act_deploy.json"
        for skill in ("button", "knob", "switch"):
            start = MODULE.parse_start_action(bounds, skill)
            lower, upper = MODULE.parse_action_bounds(bounds, skill)
            start_obs_lower, start_obs_upper = (
                MODULE.parse_start_observation_envelope(bounds, skill)
            )
            self.assertEqual(start.shape, (8,))
            self.assertTrue(np.isfinite(start).all())
            self.assertTrue(np.all(start >= lower - 1.0e-5))
            self.assertTrue(np.all(start <= upper + 1.0e-5))
            self.assertEqual(start_obs_lower.shape, (8,))
            self.assertEqual(start_obs_upper.shape, (8,))
            self.assertTrue(np.all(start_obs_lower <= start_obs_upper))

    def test_button_reached_wrist_pose_is_inside_training_observation_envelope(self):
        project = Path(__file__).resolve().parents[1]
        bounds = project / "config/task2_act_deploy.json"
        lower, upper = MODULE.parse_start_observation_envelope(bounds, "button")
        # The second real preposition run settled near -0.689rad WristRoll.
        # That value is well within the first-observation distribution seen by
        # the trained button policy and must not be rejected by a median-only
        # tolerance.
        self.assertLessEqual(lower[4], -0.689)
        self.assertGreaterEqual(upper[4], -0.689)

    def test_real_hardware_margin_accepts_latest_settled_shoulder_roll(self):
        project = Path(__file__).resolve().parents[1]
        parser = MODULE.build_parser(project)
        args = parser.parse_args(["button", "--real", "--preposition-only"])
        lower, upper = MODULE.parse_start_observation_envelope(
            project / "config/task2_act_deploy.json", "button"
        )
        # The 12:36 real run became motionless with only ShoulderRoll 0.03rad
        # outside the central P10-P90 envelope.  That is 1.7 degrees and must
        # be accepted rather than increasing impedance or waiting forever.
        measured = np.clip((lower + upper) / 2.0, lower, upper)
        measured[1] = lower[1] - 0.03
        error = np.maximum(
            np.maximum(lower - measured, 0.0),
            np.maximum(measured - upper, 0.0),
        )
        self.assertEqual(args.preposition_envelope_margin, 0.05)
        self.assertLessEqual(float(np.max(error)), args.preposition_envelope_margin)

    def test_preposition_only_is_real_mode_only(self):
        project = Path(__file__).resolve().parents[1]
        parser = MODULE.build_parser(project)
        real_args = parser.parse_args(["button", "--real", "--preposition-only"])
        MODULE.validate_args(real_args, parser)
        self.assertTrue(real_args.preposition_only)
        dry_args = parser.parse_args(["button", "--dry-run", "--preposition-only"])
        with self.assertRaises(SystemExit):
            MODULE.validate_args(dry_args, parser)

    def test_physical_limit_staging_does_not_jump_to_demo_boundary(self):
        # A normal-locomotion pose may be valid for G1 but slightly outside a
        # skill's demonstration envelope.  Staging must still start exactly
        # from the measured pose and advance only by its configured limits.
        lower = np.concatenate((MODULE.RIGHT_ARM_COMMAND_LOWER, [0.0]))
        upper = np.concatenate((MODULE.RIGHT_ARM_COMMAND_UPPER, [5.2]))
        staging = MODULE.SmoothActionFilter(
            lower,
            upper,
            cutoff_hz=1.0,
            arm_speed=0.12,
            arm_acceleration=0.35,
            gripper_speed=0.60,
            gripper_acceleration=2.0,
            max_follow_error=0.16,
        )
        measured = np.asarray([0.2, -0.2, 0.1, 0.8, 1.5, 0.1, -0.1, 0.0])
        target = np.asarray([0.1, -0.1, 0.2, 1.3, -0.8, 0.05, -0.3, 5.2])
        staging.reset(measured)
        command = staging.step(target, measured, 0.02)
        self.assertLessEqual(float(np.max(np.abs(command[:7] - measured[:7]))), 0.00015)

    def test_accumulated_command_cannot_exceed_measured_follow_window(self):
        staging = MODULE.SmoothActionFilter(
            np.asarray([-2.0] * 7 + [0.0]),
            np.asarray([2.0] * 7 + [5.2]),
            cutoff_hz=2.0,
            arm_speed=0.5,
            arm_acceleration=2.0,
            gripper_speed=1.0,
            gripper_acceleration=4.0,
            max_follow_error=0.16,
        )
        measured = np.zeros(8)
        staging.reset(measured)
        for _ in range(100):
            # Simulate normal locomotion holding the measured arm still while
            # the staging target remains far away.
            command = staging.step(np.ones(8), measured, 0.02)
        self.assertLessEqual(float(np.max(np.abs(command[:7] - measured[:7]))), 0.160001)


class MotionCycleCompletionTests(unittest.TestCase):
    def test_requires_departure_return_and_stable_hold(self):
        detector = MODULE.MotionCycleCompletion(
            excursion_threshold=1.5,
            return_threshold=0.85,
            stable_speed=1.0,
            hold_seconds=1.5,
            minimum_seconds=2.0,
        )
        start = np.zeros(7)
        detector.reset(start)
        self.assertFalse(detector.update(np.ones(7) * 0.7, 0.1, 1.0))
        self.assertTrue(detector.departed)
        # Returning is not enough: it must remain settled for the hold time.
        for index in range(15):
            self.assertFalse(detector.update(start, 0.1, 2.0 + 0.1 * index))
        self.assertTrue(detector.update(start, 0.1, 3.6))

    def test_never_completes_without_large_excursion(self):
        detector = MODULE.MotionCycleCompletion(1.5, 0.85, 1.0, 0.5, 0.0)
        start = np.zeros(7)
        detector.reset(start)
        for index in range(20):
            self.assertFalse(
                detector.update(np.ones(7) * 0.1, 0.1, 0.1 * index)
            )


if __name__ == "__main__":
    unittest.main()

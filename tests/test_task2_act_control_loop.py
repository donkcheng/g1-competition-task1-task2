"""Exercise the real main-loop branches with fake sensors, model and clock.

No DDS, cameras, CUDA or robot connection is created by these tests.
"""
import contextlib
import importlib.util
import io
from pathlib import Path
import signal
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("act_loop_under_test", PROJECT / "src/task2_act_inference.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ControlLoopTests(unittest.TestCase):
    def exercise(self, fault=None, initial_offset=None, refuse_low_weight=False,
                 holding_lag=0., argv_extra=None):
        now = [100.0]
        events, publishes, instances, handlers = [], [], [], {}
        quit_sent = [False]
        observation_count = [0]
        start = MODULE.parse_start_action(PROJECT / "config/task2_act_deploy.json", "button")
        initial = start + (np.zeros(8) if initial_offset is None else np.asarray(initial_offset))

        def sleep(seconds):
            now[0] += seconds
            if fault == "interrupt_in_quit_fade" and quit_sent[0] and "fault:fade" not in events:
                events.append("fault:fade")
                handlers[signal.SIGINT](signal.SIGINT, None)
            if now[0] > 140:
                raise AssertionError("control loop failed to finish")

        def commands():
            if fault == "interrupt_in_quit_fade" and now[0] >= 104 and not quit_sent[0]:
                quit_sent[0] = True
                return ["q"]
            return []

        class Sources:
            def __init__(self, *args):
                self.q = np.zeros(29)
                self.q[22:29] = initial[:7]
                self.gripper = initial[7]
                self.dq = np.zeros(7)
                self._odometry = (np.zeros(3), np.zeros(3), now[0])
                self.injected = False
                self.hold = None
                self.authority_ready = False
                self.lag_applied = False
                instances.append(self)

            def poll_camera(self):
                self._odometry = (np.zeros(3), np.zeros(3), now[0])
                if holding_lag and now[0] >= 104 and not self.lag_applied:
                    self.q[26] -= holding_lag
                    self.lag_applied = True
                if fault and not self.injected and now[0] >= 104:
                    self.injected = True
                    events.append("fault:" + fault)
                    if fault == "interrupt":
                        handlers[signal.SIGINT](signal.SIGINT, None)
                    elif fault == "exception":
                        raise RuntimeError("camera read failed")
                    elif fault == "overspeed":
                        self.dq[4] = 0.82
                    elif fault == "invalid_velocity":
                        self.dq[4] = float("nan")
                    elif fault == "arming_unsettled":
                        self.dq[4] = 0.20
                    elif fault == "arming_hold_error":
                        self.q[26] += 0.20

            def readiness(self, age):
                if fault == "stale_camera" and self.injected:
                    return False, "fake camera stale"
                return True, "fresh fake sources"

            def measured(self):
                if fault == "missing_state" and self.injected:
                    return None
                return MODULE.RobotMeasurement(
                    self.q.copy(), self.gripper, np.ones(29) * 40,
                    self.dq.copy(), now[0], int(now[0] * 1000), np.zeros(3)
                )

            def capture_body_hold(self, sample):
                self.hold = sample.body[12:22].copy()

            def body_motion(self, sample):
                return None, {}

            def observation(self, episode_id, policy_step, age):
                observation_count[0] += 1
                if fault == "interrupt_in_observation" and observation_count[0] == 2:
                    events.append("fault:observation")
                    handlers[signal.SIGINT](signal.SIGINT, None)
                return SimpleNamespace(episode_id=episode_id, policy_step=policy_step)

            def publish(self, action, weight, kp, kd):
                events.append("positive" if weight > 0 else "zero")
                publishes.append((now[0], action.copy(), weight, kp, kd))
                self.authority_ready |= weight >= 0.90 - 1.e-9
                if refuse_low_weight and not self.authority_ready:
                    return
                self.q[22:29] = action[:7]
                if self.lag_applied:
                    self.q[26] -= holding_lag
                self.gripper = action[7]

            def release_arm_ownership(self):
                events.append("zero")

            def close(self):
                events.append("sources_close")

        class Worker:
            def __init__(self, engine):
                self.pending = []

            def clear_requests(self):
                self.pending.clear()

            def submit(self, request):
                events.append("policy_submit")
                self.pending.append(SimpleNamespace(
                    episode_id=request.episode_id, policy_step=request.policy_step,
                    error=None, actions=np.tile(start, (100, 1)),
                    completed_at=now[0], latency=0.01,
                ))

            def drain(self):
                values, self.pending = self.pending, []
                return values

            def close(self):
                events.append("worker_close")

        argv = ["act", "button", "--real", "--auto-start", "--max-run-seconds", "3", "--exit-after-attempt"]
        if argv_extra:
            argv.extend(argv_extra)
        output = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(MODULE.sys, "argv", argv))
            stack.enter_context(patch.object(MODULE.sys.stdin, "isatty", return_value=True))
            stack.enter_context(patch.object(MODULE, "add_runtime_dependencies"))
            stack.enter_context(patch.object(MODULE.Path, "is_file", return_value=True))
            stack.enter_context(patch.object(MODULE, "ACTEngine", return_value=SimpleNamespace(device="cuda", warmup=lambda: 0.01)))
            stack.enter_context(patch.object(MODULE, "InferenceWorker", Worker))
            stack.enter_context(patch.object(MODULE, "RobotSources", Sources))
            stack.enter_context(patch.object(MODULE, "terminal_commands", side_effect=commands))
            stack.enter_context(patch.object(MODULE.time, "monotonic", side_effect=lambda: now[0]))
            stack.enter_context(patch.object(MODULE.time, "sleep", side_effect=sleep))
            stack.enter_context(patch.object(MODULE.signal, "signal", side_effect=lambda sig, fn: handlers.update({sig: fn})))
            stack.enter_context(contextlib.redirect_stdout(output))
            if fault == "exception":
                with self.assertRaisesRegex(RuntimeError, "camera read failed"):
                    MODULE.main()
                result = None
            else:
                result = MODULE.main()
        self.assertEqual(events[-2:], ["worker_close", "sources_close"])
        self.assertEqual(events[-3], "zero")
        return result, output.getvalue(), events, publishes

    def test_automatic_takeover_policy_and_release(self):
        result, output, _, publishes = self.exercise()
        self.assertEqual(result, 0)
        self.assertIn("Attempt 1 RUNNING", output)
        self.assertIn("3.0s attempt limit reached", output)
        self.assertEqual(publishes[-1][2], 0.)

    def test_motion_cycle_timeout_has_distinct_released_exit_code(self):
        result, output, events, _ = self.exercise(argv_extra=[
            "--completion-mode", "motion-cycle", "--max-run-seconds", "7"
        ])
        self.assertEqual(result, 6)
        self.assertIn("completion was not detected before 7.0s timeout", output)
        self.assertEqual(events[-3], "zero")

    def test_real_motor_overspeed_stops_before_policy(self):
        result, output, events, _ = self.exercise("overspeed")
        self.assertEqual(result, 5)
        self.assertIn("motor_dq=+0.820", output)
        self.assertIn("RightWristRoll", output)
        self.assertIn("unexpected motion during stationary arming", output)
        self.assertNotIn("policy_submit", events)
        self.assertNotIn("Attempt 1 RUNNING", output)

    def test_arming_does_not_move_arm_or_gripper_before_gains_and_ownership_settle(self):
        offset = np.array([0., 0., 0., 0., 0.30, 0., 0., -0.8])
        result, output, _, publishes = self.exercise(
            initial_offset=offset, refuse_low_weight=True
        )
        self.assertEqual(result, 0)
        self.assertIn("ARMING stable; ownership/gains fixed", output)
        initial = publishes[0][1]
        moving_index = next(i for i, p in enumerate(publishes)
                            if not np.allclose(p[1], initial, rtol=0., atol=1.e-10))
        for p in publishes[:moving_index]:
            np.testing.assert_allclose(p[1], initial, rtol=0., atol=1.e-10)
        first_moving = publishes[moving_index]
        self.assertAlmostEqual(first_moving[2], 0.90)
        self.assertAlmostEqual(first_moving[3], 28.)
        self.assertAlmostEqual(first_moving[4], 1.8)
        full_authority_at = next(p[0] for p in publishes if p[2] >= 0.90 - 1.e-9)
        self.assertGreaterEqual(first_moving[0] - full_authority_at, 0.98)
        # Both the arm and Dex1 start moving only after the stationary stage.
        self.assertLess(first_moving[1][4], initial[4])
        self.assertGreater(first_moving[1][7], initial[7])

    def test_arming_handoff_keeps_holding_error_instead_of_resetting_to_measured(self):
        result, _, _, publishes = self.exercise(
            initial_offset=[0., 0., 0., 0., 0.30, 0., 0., -0.8],
            refuse_low_weight=True, holding_lag=0.02,
        )
        self.assertEqual(result, 0)
        initial = publishes[0][1]
        first_moving = next(p for p in publishes
                            if not np.allclose(p[1], initial, rtol=0., atol=1.e-10))
        # A reset to measured would introduce a 0.02rad step. Only the
        # acceleration-limited first 20ms command increment is permitted.
        self.assertLessEqual(np.max(np.abs(first_moving[1][:7] - initial[:7])),
                             0.35 * 0.02 ** 2 + 1.e-6)

    def test_stale_camera_during_arming_releases_without_inference(self):
        result, output, events, _ = self.exercise("stale_camera")
        self.assertEqual(result, 5)
        self.assertIn("stale source: fake camera stale", output)
        self.assertNotIn("policy_submit", events)

    def test_unsettled_arming_times_out_without_inference(self):
        result, output, events, _ = self.exercise("arming_unsettled")
        self.assertEqual(result, 5)
        self.assertIn("stationary arming timed out", output)
        self.assertNotIn("policy_submit", events)

    def test_arming_hold_error_releases_without_inference(self):
        result, output, events, _ = self.exercise("arming_hold_error")
        self.assertEqual(result, 5)
        self.assertIn("stationary arming hold error", output)
        self.assertNotIn("policy_submit", events)

    def test_invalid_velocity_releases(self):
        result, output, _, _ = self.exercise("invalid_velocity")
        self.assertEqual(result, 5)
        self.assertIn("Invalid joint feedback", output)

    def test_exception_releases_before_shutdown(self):
        self.exercise("exception")

    def test_interrupt_during_camera_poll_sends_no_more_positive_weight(self):
        result, _, events, _ = self.exercise("interrupt")
        self.assertEqual(result, 130)
        self.assertNotIn("positive", events[events.index("fault:interrupt") + 1:])

    def test_missing_feedback_does_not_hang_release(self):
        result, _, _, _ = self.exercise("missing_state")
        self.assertEqual(result, 5)

    def test_interrupt_during_observation_sends_no_more_positive_weight(self):
        result, _, events, _ = self.exercise("interrupt_in_observation")
        self.assertEqual(result, 130)
        self.assertNotIn("positive", events[events.index("fault:observation") + 1:])

    def test_interrupt_during_quit_fade_sends_no_more_positive_weight(self):
        result, _, events, _ = self.exercise("interrupt_in_quit_fade")
        self.assertEqual(result, 130)
        self.assertNotIn("positive", events[events.index("fault:fade") + 1:])


if __name__ == "__main__":
    unittest.main()

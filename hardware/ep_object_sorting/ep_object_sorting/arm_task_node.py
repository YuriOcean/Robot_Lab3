#!/usr/bin/env python3
"""RoboMaster EP arm task node for object sorting.

Command topic
-------------
/arm_task/command (std_msgs/msg/String containing JSON)

Supported commands
------------------
{"command":"observe", "id":"observe_1"}
{"command":"prepare", "id":"prepare_1"}
{"command":"pick", "id":"G1_pick"}
{"command":"place", "id":"G1_place"}
{"command":"home", "id":"home_1"}
{"command":"stop", "id":"stop_1"}

Status topic
------------
/arm_task/status (std_msgs/msg/String containing JSON)

This node controls only the arm and gripper. It never commands the chassis.
"""

import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

from robomaster_msgs.action import GripperControl, MoveArm


class ArmTaskNode(Node):
    SOFTWARE_VERSION = "2026-09-21-direct-arm-motion-v13"
    WAITING = "WAITING"
    IDLE = "IDLE"
    SENDING = "SENDING"
    RUNNING = "RUNNING"
    VERIFYING = "VERIFYING"
    COOLDOWN = "COOLDOWN"
    STOPPING = "STOPPING"

    def __init__(self) -> None:
        super().__init__("arm_task")

        # ROS interfaces.
        self.declare_parameter("command_topic", "/arm_task/command")
        self.declare_parameter("status_topic", "/arm_task/status")
        self.declare_parameter("arm_position_topic", "/arm_position")
        self.declare_parameter("move_arm_action", "/move_arm")
        self.declare_parameter("gripper_action", "/gripper")

        # Coordinates verified in Experiment 2, in metres.
        self.declare_parameter("home_x", 0.083)
        self.declare_parameter("home_z", 0.058)
        # Final fixed snapshot pose selected on the real EP. It only affects
        # the pre-snapshot camera position, not pick/place coordinates.
        self.declare_parameter("observe_x", 0.106)
        self.declare_parameter("observe_z", 0.030)
        self.declare_parameter("pick_x", 0.200)
        # Hardware feedback shows that the EP arm bottoms out at about 0.030 m.
        # Commanding 0.015 m makes pose verification time out before gripping.
        self.declare_parameter("pick_z", 0.030)
        self.declare_parameter("pick_safe_z", 0.120)
        self.declare_parameter("place_x", 0.209)
        self.declare_parameter("place_z", 0.020)
        self.declare_parameter("place_safe_z", 0.120)

        # Conservative software workspace verified in Experiment 2.
        self.declare_parameter("x_min", 0.075)
        self.declare_parameter("x_max", 0.212)
        # z_min is a software validation limit, not a hardware minimum.
        # The current observation and picking defaults both use z=0.030 m.
        self.declare_parameter("z_min", 0.005)
        self.declare_parameter("z_max", 0.125)

        self.declare_parameter("gripper_power", 0.15)
        self.declare_parameter("action_timeout", 15.0)
        self.declare_parameter("pose_timeout", 3.0)
        self.declare_parameter("arm_state_timeout", 1.0)
        self.declare_parameter("action_cooldown", 1.0)
        self.declare_parameter("gripper_cooldown", 2.0)
        self.declare_parameter("stop_hold", 1.0)
        self.declare_parameter("pose_tolerance", 0.010)
        self.declare_parameter("safe_pose_tolerance", 0.020)
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("return_home_after_place", True)
        self.declare_parameter("enforce_holding_state", True)

        self.command_topic = self._string_param("command_topic")
        self.status_topic = self._string_param("status_topic")
        self.arm_position_topic = self._string_param(
            "arm_position_topic"
        )
        self.move_arm_action = self._string_param("move_arm_action")
        self.gripper_action = self._string_param("gripper_action")

        self.home_x = self._float_param("home_x")
        self.home_z = self._float_param("home_z")
        self.observe_x = self._float_param("observe_x")
        self.observe_z = self._float_param("observe_z")
        self.pick_x = self._float_param("pick_x")
        self.pick_z = self._float_param("pick_z")
        self.pick_safe_z = self._float_param("pick_safe_z")
        self.place_x = self._float_param("place_x")
        self.place_z = self._float_param("place_z")
        self.place_safe_z = self._float_param("place_safe_z")

        self.x_min = self._float_param("x_min")
        self.x_max = self._float_param("x_max")
        self.z_min = self._float_param("z_min")
        self.z_max = self._float_param("z_max")

        self.gripper_power = self._float_param("gripper_power")
        self.action_timeout = self._float_param("action_timeout")
        self.pose_timeout = self._float_param("pose_timeout")
        self.arm_state_timeout = self._float_param(
            "arm_state_timeout"
        )
        self.action_cooldown = self._float_param("action_cooldown")
        self.gripper_cooldown = self._float_param(
            "gripper_cooldown"
        )
        self.stop_hold = self._float_param("stop_hold")
        self.pose_tolerance = self._float_param("pose_tolerance")
        self.safe_pose_tolerance = self._float_param(
            "safe_pose_tolerance"
        )
        self.control_rate = self._float_param("control_rate")
        self.return_home_after_place = self._bool_param(
            "return_home_after_place"
        )
        self.enforce_holding_state = self._bool_param(
            "enforce_holding_state"
        )

        self._validate_configuration()

        self.status_pub = self.create_publisher(
            String, self.status_topic, 10
        )
        self.command_sub = self.create_subscription(
            String, self.command_topic, self._command_callback, 10
        )
        self.arm_position_sub = self.create_subscription(
            PointStamped,
            self.arm_position_topic,
            self._arm_position_callback,
            10,
        )

        self.arm_client = ActionClient(
            self, MoveArm, self.move_arm_action
        )
        self.gripper_client = ActionClient(
            self, GripperControl, self.gripper_action
        )

        self.state = self.WAITING
        self.ready_announced = False
        self.latest_arm_position: Optional[Tuple[float, float]] = None
        self.last_arm_state_time: Optional[float] = None

        # False: known empty; True: known holding; None: uncertain.
        self.holding_object: Optional[bool] = False

        self.active_task = ""
        self.command_id = ""
        self.steps: List[Dict[str, Any]] = []
        self.step_index = -1
        self.current_step: Optional[Dict[str, Any]] = None

        self.send_future = None
        self.result_future = None
        self.goal_handle = None
        self.action_started = 0.0
        self.verify_deadline = 0.0
        self.verify_not_before = 0.0
        self.cooldown_deadline = 0.0

        self.stop_deadline = 0.0
        self.stop_event = "stopped"
        self.stop_success = True
        self.stop_message = ""

        self.timer = self.create_timer(
            1.0 / self.control_rate, self._timer_callback
        )

        self.get_logger().info(
            f"Arm task node {self.SOFTWARE_VERSION} started; "
            "no automatic motion will occur"
        )
        self.get_logger().info(
            f"Pick height={self.pick_z:.3f} m; "
            f"safe height={self.pick_safe_z:.3f} m"
        )
        self.get_logger().info(
            "Arm movement mode: one absolute MoveArm goal per task waypoint; "
            "segmented interpolation is disabled"
        )
        self.get_logger().info(
            f"Command: {self.command_topic}; status: {self.status_topic}"
        )
        self._publish_status(
            "waiting",
            False,
            "Waiting for arm state and action servers",
        )

    def _string_param(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _float_param(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _bool_param(self, name: str) -> bool:
        return bool(self.get_parameter(name).value)

    def _now(self) -> float:
        return time.monotonic()

    def _validate_point(self, name: str, x: float, z: float) -> None:
        if not math.isfinite(x) or not math.isfinite(z):
            raise ValueError(f"{name} contains a non-finite coordinate")
        if not self.x_min <= x <= self.x_max:
            raise ValueError(
                f"{name}: x={x:.3f} outside "
                f"[{self.x_min:.3f}, {self.x_max:.3f}]"
            )
        if not self.z_min <= z <= self.z_max:
            raise ValueError(
                f"{name}: z={z:.3f} outside "
                f"[{self.z_min:.3f}, {self.z_max:.3f}]"
            )

    def _validate_configuration(self) -> None:
        if self.x_min >= self.x_max or self.z_min >= self.z_max:
            raise ValueError("Invalid software workspace limits")

        points = [
            ("home", self.home_x, self.home_z),
            ("observe", self.observe_x, self.observe_z),
            ("pick_safe", self.pick_x, self.pick_safe_z),
            ("pick", self.pick_x, self.pick_z),
            ("place_safe", self.place_x, self.place_safe_z),
            ("place", self.place_x, self.place_z),
        ]
        for name, x, z in points:
            self._validate_point(name, x, z)

        if self.pick_safe_z <= self.pick_z:
            raise ValueError("pick_safe_z must be above pick_z")
        if self.place_safe_z <= self.place_z:
            raise ValueError("place_safe_z must be above place_z")
        if not 0.0 <= self.gripper_power <= 1.0:
            raise ValueError("gripper_power must be in [0, 1]")
        if self.action_timeout <= 0.0:
            raise ValueError("action_timeout must be positive")
        if self.pose_timeout <= 0.0:
            raise ValueError("pose_timeout must be positive")
        if self.arm_state_timeout <= 0.0:
            raise ValueError("arm_state_timeout must be positive")
        if self.action_cooldown < 0.0 or self.gripper_cooldown < 0.0:
            raise ValueError("Cooldown values cannot be negative")
        if self.stop_hold < 0.2:
            raise ValueError("stop_hold must be at least 0.2 s")
        if self.pose_tolerance <= 0.0:
            raise ValueError("pose_tolerance must be positive")
        if self.safe_pose_tolerance < self.pose_tolerance:
            raise ValueError(
                "safe_pose_tolerance cannot be smaller than "
                "pose_tolerance"
            )
        if not 5.0 <= self.control_rate <= 100.0:
            raise ValueError("control_rate must be between 5 and 100 Hz")

    def _arm_position_callback(self, msg: PointStamped) -> None:
        self.latest_arm_position = (
            float(msg.point.x),
            float(msg.point.z),
        )
        self.last_arm_state_time = self._now()

    def _servers_ready(self) -> bool:
        return (
            self.arm_client.server_is_ready()
            and self.gripper_client.server_is_ready()
        )

    def _system_ready(self) -> bool:
        if not self._servers_ready():
            return False
        if self.latest_arm_position is None:
            return False
        if self.last_arm_state_time is None:
            return False
        return (
            self._now() - self.last_arm_state_time
            <= self.arm_state_timeout
        )

    def _command_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self._reject(f"Invalid JSON: {exc}")
            return

        if not isinstance(data, dict):
            self._reject("Command JSON must be an object")
            return

        command = str(data.get("command", "")).strip().lower()
        requested_id = str(data.get("id", ""))

        if command in ("stop", "cancel", "emergency_stop"):
            self.command_id = requested_id
            self._request_stop("Stop command received")
            return

        if command == "set_empty":
            if self.state != self.IDLE:
                self._reject("Cannot reset holding state while busy")
                return
            self.command_id = requested_id
            self.holding_object = False
            self._publish_status(
                "state_reset",
                True,
                "Operator confirmed that the gripper is empty",
            )
            return

        if self.state != self.IDLE:
            self._reject(f"Arm is busy: state={self.state}")
            return

        if not self._system_ready():
            self._reject(
                "Arm state or action servers are not ready"
            )
            return

        if command not in ("observe", "prepare", "home", "pick", "place"):
            self._reject(
                "Unknown command. Use observe, prepare, home, pick, place, "
                "stop, or set_empty"
            )
            return

        if (
            command == "pick"
            and self.enforce_holding_state
            and self.holding_object is not False
        ):
            self._reject(
                "Pick rejected because the gripper is not known empty; "
                "inspect it and send set_empty if appropriate"
            )
            return

        if (
            command == "place"
            and self.enforce_holding_state
            and self.holding_object is not True
        ):
            self._reject(
                "Place rejected because no held object is confirmed"
            )
            return

        self.command_id = requested_id
        self._start_task(command)

    def _move_step(
        self,
        name: str,
        x: float,
        z: float,
        tolerance: float,
        cooldown: Optional[float] = None,
    ) -> Dict[str, Any]:
        return {
            "kind": "move",
            "name": name,
            "x": float(x),
            "z": float(z),
            "tolerance": float(tolerance),
            "cooldown": (
                self.action_cooldown
                if cooldown is None
                else float(cooldown)
            ),
        }

    def _gripper_step(
        self, name: str, target_state: int
    ) -> Dict[str, Any]:
        return {
            "kind": "gripper",
            "name": name,
            "target_state": int(target_state),
            "cooldown": self.gripper_cooldown,
        }

    def _build_sequence(self, task: str) -> List[Dict[str, Any]]:
        if self.latest_arm_position is None:
            raise ValueError("Cannot build arm sequence without fresh arm pose")

        open_gripper = self._gripper_step(
            "open_gripper", GripperControl.Goal.OPEN
        )

        if task == "home":
            return [
                self._move_step(
                    "move_home",
                    self.home_x,
                    self.home_z,
                    self.pose_tolerance,
                )
            ]

        if task == "observe":
            return [
                self._move_step(
                    "move_to_observe",
                    self.observe_x,
                    self.observe_z,
                    self.safe_pose_tolerance,
                )
            ]

        if task == "prepare":
            return [
                self._move_step(
                    "move_home",
                    self.home_x,
                    self.home_z,
                    self.pose_tolerance,
                ),
                open_gripper,
            ]

        if task == "pick":
            return [
                open_gripper,
                self._move_step(
                    "move_above_pick",
                    self.pick_x,
                    self.pick_safe_z,
                    self.safe_pose_tolerance,
                ),
                self._move_step(
                    "move_to_pick",
                    self.pick_x,
                    self.pick_z,
                    self.pose_tolerance,
                ),
                self._gripper_step(
                    "close_gripper", GripperControl.Goal.CLOSE
                ),
                self._move_step(
                    "lift_from_pick",
                    self.pick_x,
                    self.pick_safe_z,
                    self.safe_pose_tolerance,
                ),
            ]

        if task == "place":
            steps = [
                self._move_step(
                    "move_above_place",
                    self.place_x,
                    self.place_safe_z,
                    self.safe_pose_tolerance,
                ),
                self._move_step(
                    "move_to_place",
                    self.place_x,
                    self.place_z,
                    self.pose_tolerance,
                ),
                open_gripper,
                self._move_step(
                    "lift_from_place",
                    self.place_x,
                    self.place_safe_z,
                    self.safe_pose_tolerance,
                ),
            ]
            if self.return_home_after_place:
                steps.append(
                    self._move_step(
                        "move_home",
                        self.home_x,
                        self.home_z,
                        self.pose_tolerance,
                    )
                )
            return steps

        raise ValueError(f"Unsupported task: {task}")

    def _start_task(self, task: str) -> None:
        self.active_task = task
        self.steps = self._build_sequence(task)
        self.step_index = -1
        self.current_step = None
        self.state = self.IDLE

        self.get_logger().info(
            f"Task started: {task}, id={self.command_id}"
        )
        self._publish_status(
            "started", True, f"Task {task} started"
        )
        self._advance_sequence()

    def _advance_sequence(self) -> None:
        self.step_index += 1

        if self.step_index >= len(self.steps):
            self._complete_task()
            return

        self.current_step = self.steps[self.step_index]
        step_name = str(self.current_step["name"])
        step_kind = str(self.current_step["kind"])

        self.get_logger().info(
            f"Step {self.step_index + 1}/{len(self.steps)}: {step_name}"
        )
        self._publish_status(
            "step_started", True, f"Executing {step_name}"
        )

        if step_kind == "move":
            self._send_move_goal(self.current_step)
        elif step_kind == "gripper":
            self._send_gripper_goal(self.current_step)
        else:
            self._fail_task(f"Unknown step kind: {step_kind}")

    def _send_move_goal(self, step: Dict[str, Any]) -> None:
        x = float(step["x"])
        z = float(step["z"])
        self._validate_point(str(step["name"]), x, z)

        goal = MoveArm.Goal()
        goal.x = x
        goal.z = z
        goal.relative = False

        self.send_future = self.arm_client.send_goal_async(goal)
        self.result_future = None
        self.goal_handle = None
        self.action_started = self._now()
        self.state = self.SENDING

    def _send_gripper_goal(self, step: Dict[str, Any]) -> None:
        goal = GripperControl.Goal()
        goal.target_state = int(step["target_state"])
        goal.power = self.gripper_power

        self.send_future = self.gripper_client.send_goal_async(goal)
        self.result_future = None
        self.goal_handle = None
        self.action_started = self._now()
        self.state = self.SENDING

    def _timer_callback(self) -> None:
        now = self._now()

        if self.state == self.WAITING:
            if self._system_ready():
                self.state = self.IDLE
                self.ready_announced = True
                x, z = self.latest_arm_position or (0.0, 0.0)
                self.get_logger().info(
                    f"Arm task node ready: x={x:.3f}, z={z:.3f}"
                )
                self._publish_status(
                    "ready", True, "Arm and gripper are ready"
                )
            return

        if self.state == self.IDLE:
            return

        if self.state == self.STOPPING:
            if now >= self.stop_deadline:
                self.state = self.IDLE
                self._publish_status(
                    self.stop_event,
                    self.stop_success,
                    self.stop_message,
                )
            return

        if self.state == self.SENDING:
            self._poll_send_future(now)
            return

        if self.state == self.RUNNING:
            self._poll_result_future(now)
            return

        if self.state == self.VERIFYING:
            self._verify_pose(now)
            return

        if self.state == self.COOLDOWN:
            if now >= self.cooldown_deadline:
                self._advance_sequence()

    def _poll_send_future(self, now: float) -> None:
        if now - self.action_started > self.action_timeout:
            self._fail_task("Timed out while sending action goal")
            return
        if self.send_future is None or not self.send_future.done():
            return
        if self.send_future.exception() is not None:
            self._fail_task(
                f"Action goal error: {self.send_future.exception()}"
            )
            return

        self.goal_handle = self.send_future.result()
        if self.goal_handle is None or not self.goal_handle.accepted:
            self._fail_task("Action goal was rejected")
            return

        self.result_future = self.goal_handle.get_result_async()
        self.action_started = now
        self.state = self.RUNNING

    def _poll_result_future(self, now: float) -> None:
        if now - self.action_started > self.action_timeout:
            self._cancel_active_goal()
            self._fail_task("Action execution timeout")
            return
        if self.result_future is None or not self.result_future.done():
            return
        if self.result_future.exception() is not None:
            self._fail_task(
                f"Action result error: {self.result_future.exception()}"
            )
            return

        wrapped_result = self.result_future.result()
        if wrapped_result is None:
            self._fail_task("Action returned no result")
            return
        if wrapped_result.status != GoalStatus.STATUS_SUCCEEDED:
            self._fail_task(
                f"Action failed with status {wrapped_result.status}"
            )
            return

        self.goal_handle = None
        self.send_future = None
        self.result_future = None

        if self.current_step is None:
            self._fail_task("Current step disappeared")
            return

        if self.current_step["kind"] == "move":
            # 位姿反馈必须晚于本次动作开始，不能用动作前残留缓存判成功。
            self.verify_not_before = self.action_started
            self.verify_deadline = now + self.pose_timeout
            self.state = self.VERIFYING
        else:
            self._step_succeeded(now)

    def _verify_pose(self, now: float) -> None:
        if self.current_step is None:
            self._fail_task("Current move step disappeared")
            return
        feedback_is_fresh = (
            self.latest_arm_position is not None
            and self.last_arm_state_time is not None
            and self.last_arm_state_time >= self.verify_not_before
            and now - self.last_arm_state_time <= self.arm_state_timeout
        )
        if feedback_is_fresh:
            actual_x, actual_z = self.latest_arm_position
            target_x = float(self.current_step["x"])
            target_z = float(self.current_step["z"])
            tolerance = float(self.current_step["tolerance"])

            error_x = abs(actual_x - target_x)
            error_z = abs(actual_z - target_z)
            if error_x <= tolerance and error_z <= tolerance:
                self._step_succeeded(now)
                return

        if now >= self.verify_deadline:
            if self.latest_arm_position is None or self.last_arm_state_time is None:
                detail = "no arm position received"
            elif self.last_arm_state_time < self.verify_not_before:
                detail = "arm position feedback predates the current move"
            elif now - self.last_arm_state_time > self.arm_state_timeout:
                detail = (
                    "arm position feedback is stale: "
                    f"age={now - self.last_arm_state_time:.2f} s"
                )
            else:
                actual_x, actual_z = self.latest_arm_position
                target_x = float(self.current_step["x"])
                target_z = float(self.current_step["z"])
                detail = (
                    f"target=({target_x:.3f},{target_z:.3f}), "
                    f"actual=({actual_x:.3f},{actual_z:.3f})"
                )
            self._fail_task(f"Arm pose verification timeout: {detail}")

    def _step_succeeded(self, now: float) -> None:
        if self.current_step is None:
            self._fail_task("Current step disappeared")
            return

        step_name = str(self.current_step["name"])
        self.get_logger().info(f"Step succeeded: {step_name}")
        self._publish_status(
            "step_succeeded", True, f"Completed {step_name}"
        )
        self.cooldown_deadline = now + float(
            self.current_step.get("cooldown", 0.0)
        )
        self.state = self.COOLDOWN

    def _complete_task(self) -> None:
        completed_task = self.active_task
        if completed_task == "pick":
            self.holding_object = True
        elif completed_task in ("place", "prepare"):
            self.holding_object = False

        self.state = self.IDLE
        self.current_step = None
        self.steps = []
        self.step_index = -1
        self.get_logger().info(f"Task succeeded: {completed_task}")
        self._publish_status(
            "succeeded", True, f"Task {completed_task} succeeded"
        )

    def _cancel_active_goal(self) -> None:
        if self.goal_handle is not None:
            try:
                self.goal_handle.cancel_goal_async()
            except Exception as exc:
                self.get_logger().error(f"Goal cancellation failed: {exc}")

    def _send_gripper_pause(self) -> None:
        if not self.gripper_client.server_is_ready():
            return
        try:
            goal = GripperControl.Goal()
            goal.target_state = GripperControl.Goal.PAUSE
            goal.power = self.gripper_power
            self.gripper_client.send_goal_async(goal)
        except Exception as exc:
            self.get_logger().error(f"Gripper pause failed: {exc}")

    def _request_stop(self, message: str) -> None:
        was_active = self.state not in (self.IDLE, self.WAITING)
        self._cancel_active_goal()
        self._send_gripper_pause()

        if was_active:
            self.holding_object = None

        self.steps = []
        self.current_step = None
        self.send_future = None
        self.result_future = None
        self.goal_handle = None

        self.stop_event = "stopped"
        self.stop_success = True
        self.stop_message = message
        self.stop_deadline = self._now() + self.stop_hold
        self.state = self.STOPPING
        self.get_logger().warning(message)
        self._publish_status("stopping", True, message)

    def _fail_task(self, message: str) -> None:
        failed_task = self.active_task
        self._cancel_active_goal()
        self._send_gripper_pause()

        if failed_task in ("pick", "place"):
            self.holding_object = None

        self.steps = []
        self.current_step = None
        self.send_future = None
        self.result_future = None
        self.goal_handle = None

        self.stop_event = "failed"
        self.stop_success = False
        self.stop_message = f"Task {failed_task} failed: {message}"
        self.stop_deadline = self._now() + self.stop_hold
        self.state = self.STOPPING
        self.get_logger().error(self.stop_message)
        self._publish_status("stopping", False, self.stop_message)

    def _reject(self, message: str) -> None:
        self.get_logger().error(f"Command rejected: {message}")
        self._publish_status("rejected", False, message)

    def _publish_status(
        self, event: str, success: bool, message: str
    ) -> None:
        if self.holding_object is True:
            holding_state = "holding"
        elif self.holding_object is False:
            holding_state = "empty"
        else:
            holding_state = "unknown"

        if self.latest_arm_position is None:
            arm_x = None
            arm_z = None
        else:
            arm_x = round(self.latest_arm_position[0], 6)
            arm_z = round(self.latest_arm_position[1], 6)

        step_name = ""
        if self.current_step is not None:
            step_name = str(self.current_step.get("name", ""))

        payload = {
            "event": event,
            "success": success,
            "state": self.state,
            "id": self.command_id,
            "task": self.active_task,
            "step": step_name,
            "step_index": self.step_index,
            "step_count": len(self.steps),
            "holding_state": holding_state,
            "arm_x": arm_x,
            "arm_z": arm_z,
            "message": message,
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)

    def safe_shutdown(self) -> None:
        self._cancel_active_goal()
        self._send_gripper_pause()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[ArmTaskNode] = None

    try:
        node = ArmTaskNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().warning(
                "Ctrl+C received; cancelling the active arm task"
            )
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f"Fatal error: {exc}")
        else:
            print(f"Fatal error: {exc}")
        raise
    finally:
        if node is not None:
            node.safe_shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()


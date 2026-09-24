#!/usr/bin/env python3
"""Closed-loop translational chassis controller for RoboMaster EP.

Inputs
------
/chassis_motion/command (std_msgs/msg/String, JSON)

Examples
--------
{"command":"move", "axis":"x", "distance":0.10, "speed":0.08}
{"command":"compensate_heading", "right_degrees":4.0}
{"command":"next_column"}
{"command":"next_row"}
{"command":"stop"}

Outputs
-------
/chassis_motion/status (std_msgs/msg/String, JSON)

The controller publishes geometry_msgs/msg/Twist to /cmd_vel and closes the
position loop with nav_msgs/msg/Odometry from /odom. Translation continuously
holds a configurable heading target. A deliberate clockwise compensation can
update only that heading target without rotating the fixed field coordinates.
"""

import json
import math
from typing import Any, Dict, Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class ChassisMotionNode(Node):
    SOFTWARE_VERSION = "2026-09-21-one-way-reset-v19"
    IDLE = "IDLE"
    MOVING = "MOVING"
    STOPPING = "STOPPING"

    def __init__(self) -> None:
        super().__init__("chassis_motion")

        # ROS interfaces.
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter(
            "command_topic", "/chassis_motion/command"
        )
        self.declare_parameter(
            "status_topic", "/chassis_motion/status"
        )

        # Calibrated real-table grid spacing.
        self.declare_parameter("column_spacing", 0.20)
        self.declare_parameter("row_spacing", 0.20)

        # Mapping from logical grid movement to chassis axes.
        # On the real robot, columns vary along chassis Y and rows along X.
        self.declare_parameter("column_axis", "y")
        self.declare_parameter("column_sign", 1)
        self.declare_parameter("row_axis", "x")
        self.declare_parameter("row_sign", 1)

        # Convert logical X/Y commands into the physical chassis directions.
        # On the tested EP, physical Y is opposite to the table convention.
        self.declare_parameter("x_command_sign", 1)
        self.declare_parameter("y_command_sign", -1)

        # Closed-loop controller and safety limits.
        self.declare_parameter("control_rate", 30.0)
        self.declare_parameter("default_speed", 0.06)
        # 兼容旧launch的统一最低速度；正式控制使用下面两个分轴参数。
        self.declare_parameter("minimum_speed", 0.045)
        self.declare_parameter("minimum_x_speed", 0.045)
        # EP麦克纳姆轮横移在0.045 m/s附近存在明显静摩擦死区。
        self.declare_parameter("minimum_y_speed", 0.060)
        # Y轴进入4 cm末端区且1 s无有效进展时，用短距离高驱动力克服死区。
        self.declare_parameter("y_terminal_boost_speed", 0.080)
        self.declare_parameter("terminal_boost_error", 0.040)
        self.declare_parameter("terminal_boost_delay", 1.0)
        # 允许启动文件显式传入最高 1.0 m/s。该参数是闭环速度上限，
        # 接近目标时仍会由 position_kp 自动降速，并非全程恒速。
        self.declare_parameter("maximum_speed", 1.0)
        self.declare_parameter("position_kp", 1.2)
        self.declare_parameter("cross_track_kp", 1.0)
        self.declare_parameter("maximum_cross_speed", 0.04)
        self.declare_parameter("heading_kp", 3.0)
        self.declare_parameter("maximum_angular_speed", 0.40)
        # 每轮固定航向补偿使用更小的终止容差和最低角速度，避免小角度
        # 命令落入底盘旋转静摩擦死区。实际补偿角由调度器逐次传入。
        self.declare_parameter("compensation_heading_tolerance_deg", 0.50)
        # 小角度修正进入停止保持后，麦克纳姆轮可能回弹几十分之一度。
        # 运动阶段仍按0.5度收敛，停止后使用独立的稍宽验收容差，避免
        # 0.01度量级的边界抖动错误终止整轮分拣。
        self.declare_parameter(
            "compensation_post_stop_tolerance_deg", 1.0
        )
        self.declare_parameter("minimum_compensation_angular_speed", 0.10)
        self.declare_parameter("maximum_heading_compensation_deg", 15.0)
        self.declare_parameter("position_tolerance", 0.008)
        self.declare_parameter("x_position_tolerance", 0.008)
        # 两次实机日志显示Y轴反馈会在目标外约1 cm处稳定不再变化。
        # 1.2 cm只用于Y轴，X轴仍保持8 mm；绝对坐标控制避免误差累计。
        self.declare_parameter("y_position_tolerance", 0.012)
        self.declare_parameter("cross_track_tolerance", 0.025)
        self.declare_parameter("heading_tolerance_deg", 2.0)
        # 平移动作已经连续稳定达到运动阈值后才进入STOPPING。停车保持期间
        # 允许有限的里程计抖动/麦克纳姆轮回弹，避免刚刚合格的动作因
        # 毫米或零点几度边界变化被错误判为整轮失败。该范围只用于停车
        # 后复验，不改变运动阶段的闭环精度。
        self.declare_parameter("post_stop_position_margin", 0.005)
        self.declare_parameter("post_stop_cross_track_tolerance", 0.035)
        self.declare_parameter("post_stop_heading_tolerance_deg", 2.0)
        # Zero means the physical heading present when the node receives its
        # first odometry sample becomes the global forward heading. Translation
        # continuously holds this heading, but no automatic startup turn occurs.
        self.declare_parameter("initial_heading_correction_deg", 0.0)
        # If heading error exceeds this threshold, stop linear motion and
        # rotate in place first. This prevents approaching a cell diagonally.
        self.declare_parameter("max_translation_heading_error_deg", 5.0)
        self.declare_parameter("stable_cycles", 5)
        self.declare_parameter("stop_hold", 0.50)
        self.declare_parameter("odom_timeout", 0.60)
        self.declare_parameter("maximum_distance", 0.50)
        self.declare_parameter("maximum_motion_time", 35.0)
        self.declare_parameter("minimum_motion_timeout", 20.0)
        self.declare_parameter("timeout_scale", 4.0)
        self.declare_parameter("timeout_margin", 4.0)
        self.declare_parameter("stall_timeout", 12.0)
        self.declare_parameter("progress_epsilon", 0.0010)
        self.declare_parameter("feedback_rate", 5.0)

        self.cmd_vel_topic = self._string_param("cmd_vel_topic")
        self.odom_topic = self._string_param("odom_topic")
        self.command_topic = self._string_param("command_topic")
        self.status_topic = self._string_param("status_topic")

        self.column_spacing = self._float_param("column_spacing")
        self.row_spacing = self._float_param("row_spacing")
        self.column_axis = self._string_param("column_axis").lower()
        self.column_sign = self._int_param("column_sign")
        self.row_axis = self._string_param("row_axis").lower()
        self.row_sign = self._int_param("row_sign")
        self.x_command_sign = self._int_param("x_command_sign")
        self.y_command_sign = self._int_param("y_command_sign")

        self.control_rate = self._float_param("control_rate")
        self.default_speed = self._float_param("default_speed")
        self.minimum_speed = self._float_param("minimum_speed")
        self.minimum_x_speed = self._float_param("minimum_x_speed")
        self.minimum_y_speed = self._float_param("minimum_y_speed")
        self.y_terminal_boost_speed = self._float_param(
            "y_terminal_boost_speed"
        )
        self.terminal_boost_error = self._float_param(
            "terminal_boost_error"
        )
        self.terminal_boost_delay = self._float_param(
            "terminal_boost_delay"
        )
        self.maximum_speed = self._float_param("maximum_speed")
        self.position_kp = self._float_param("position_kp")
        self.cross_track_kp = self._float_param("cross_track_kp")
        self.maximum_cross_speed = self._float_param(
            "maximum_cross_speed"
        )
        self.heading_kp = self._float_param("heading_kp")
        self.maximum_angular_speed = self._float_param(
            "maximum_angular_speed"
        )
        self.compensation_heading_tolerance = math.radians(
            self._float_param("compensation_heading_tolerance_deg")
        )
        self.compensation_post_stop_tolerance = math.radians(
            self._float_param("compensation_post_stop_tolerance_deg")
        )
        self.minimum_compensation_angular_speed = self._float_param(
            "minimum_compensation_angular_speed"
        )
        self.maximum_heading_compensation = math.radians(
            self._float_param("maximum_heading_compensation_deg")
        )
        self.position_tolerance = self._float_param(
            "position_tolerance"
        )
        self.x_position_tolerance = self._float_param(
            "x_position_tolerance"
        )
        self.y_position_tolerance = self._float_param(
            "y_position_tolerance"
        )
        self.cross_track_tolerance = self._float_param(
            "cross_track_tolerance"
        )
        self.heading_tolerance = math.radians(
            self._float_param("heading_tolerance_deg")
        )
        self.post_stop_position_margin = self._float_param(
            "post_stop_position_margin"
        )
        self.post_stop_cross_track_tolerance = self._float_param(
            "post_stop_cross_track_tolerance"
        )
        self.post_stop_heading_tolerance = math.radians(
            self._float_param("post_stop_heading_tolerance_deg")
        )
        self.initial_heading_correction = math.radians(
            self._float_param("initial_heading_correction_deg")
        )
        self.max_translation_heading_error = math.radians(
            self._float_param("max_translation_heading_error_deg")
        )
        self.stable_cycles_required = self._int_param("stable_cycles")
        self.stop_hold = self._float_param("stop_hold")
        self.odom_timeout = self._float_param("odom_timeout")
        self.maximum_distance = self._float_param("maximum_distance")
        self.maximum_motion_time = self._float_param(
            "maximum_motion_time"
        )
        self.minimum_motion_timeout = self._float_param(
            "minimum_motion_timeout"
        )
        self.timeout_scale = self._float_param("timeout_scale")
        self.timeout_margin = self._float_param("timeout_margin")
        self.stall_timeout = self._float_param("stall_timeout")
        self.progress_epsilon = self._float_param("progress_epsilon")
        self.feedback_rate = self._float_param("feedback_rate")

        self._validate_parameters()

        self.cmd_pub = self.create_publisher(
            Twist, self.cmd_vel_topic, 10
        )
        self.status_pub = self.create_publisher(
            String, self.status_topic, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self._odom_callback, 10
        )
        self.command_sub = self.create_subscription(
            String, self.command_topic, self._command_callback, 10
        )

        self.state = self.IDLE
        self.latest_pose: Optional[Tuple[float, float, float]] = None
        self.last_odom_time: Optional[float] = None

        self.start_pose: Optional[Tuple[float, float, float]] = None
        # 场地坐标原点：首次 phase1 在 G5 抓取位显式锁定。后续 move_to
        # 始终相对此原点闭环，机械臂动作造成的底盘位移也会被补偿。
        self.origin_pose: Optional[Tuple[float, float, float]] = None
        # reference_yaw固定定义场地X/Y坐标方向，整个任务期间不改变。
        # heading_target_yaw只定义底盘应保持的航向；每轮右修正后允许更新。
        # 两者必须分离，否则修正航向会旋转整个G1~G6坐标系。
        self.reference_yaw: Optional[float] = None
        self.heading_target_yaw: Optional[float] = None
        self.compensation_target_yaw: Optional[float] = None
        self.compensation_right_deg = 0.0
        self.compensation_count = 0
        self.absolute_mode = False
        # 仅显式人工微调动作启用：成功后记录位移，平移后续路线原点。
        self.reset_forward_active = False
        self.manual_translation_world = (0.0, 0.0)
        self.target_position: Optional[Tuple[float, float]] = None
        self.active_axis = "x"
        self.target_distance = 0.0
        self.active_speed = self.default_speed
        self.command_id = ""
        self.motion_started = 0.0
        self.motion_timeout = self.maximum_motion_time
        self.best_directed_progress = 0.0
        self.best_abs_error = float("inf")
        self.terminal_boost_active = False
        self.last_progress_time = 0.0
        self.stable_count = 0

        self.stop_started = 0.0
        self.pending_success = False
        self.pending_message = ""

        self.last_progress = 0.0
        self.last_cross_track = 0.0
        self.last_heading_error = 0.0
        self.last_feedback_time = 0.0
        self.ready_announced = False

        self.control_timer = self.create_timer(
            1.0 / self.control_rate, self._control_callback
        )

        self.get_logger().info(
            f"Chassis motion node {self.SOFTWARE_VERSION} started. "
            f"column={self.column_spacing:.3f} m on "
            f"{self.column_axis}{self.column_sign:+d}, "
            f"row={self.row_spacing:.3f} m on "
            f"{self.row_axis}{self.row_sign:+d}"
        )
        self.get_logger().info(
            f"Command: {self.command_topic}; status: {self.status_topic}"
        )
        self.get_logger().info(
            "Physical command signs: "
            f"x={self.x_command_sign:+d}, y={self.y_command_sign:+d}"
        )
        self.get_logger().info(
            f"Heading hold: kp={self.heading_kp:.2f}, "
            f"tolerance={math.degrees(self.heading_tolerance):.1f} deg, "
            f"initial correction="
            f"{math.degrees(self.initial_heading_correction):+.1f} deg"
        )
        self.get_logger().info(
            "Cycle heading compensation command is available; positive "
            "right_degrees means clockwise rotation"
        )
        self._publish_status(
            "waiting",
            False,
            "Waiting for /odom and a /cmd_vel subscriber",
        )

    def _string_param(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _float_param(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _int_param(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _validate_parameters(self) -> None:
        for name, value in (
            ("column_axis", self.column_axis),
            ("row_axis", self.row_axis),
        ):
            if value not in ("x", "y"):
                raise ValueError(f"{name} must be 'x' or 'y'")

        for name, value in (
            ("column_sign", self.column_sign),
            ("row_sign", self.row_sign),
            ("x_command_sign", self.x_command_sign),
            ("y_command_sign", self.y_command_sign),
        ):
            if value not in (-1, 1):
                raise ValueError(f"{name} must be 1 or -1")

        if self.column_spacing <= 0.0 or self.row_spacing <= 0.0:
            raise ValueError("Grid spacing must be positive")
        if not 5.0 <= self.control_rate <= 100.0:
            raise ValueError("control_rate must be between 5 and 100 Hz")
        if not 0.0 < self.minimum_speed <= self.default_speed:
            raise ValueError(
                "minimum_speed must be positive and <= default_speed"
            )
        for name, speed in (
            ("minimum_x_speed", self.minimum_x_speed),
            ("minimum_y_speed", self.minimum_y_speed),
        ):
            if not 0.0 < speed <= self.maximum_speed:
                raise ValueError(f"{name} must be in (0, maximum_speed]")
        if not self.minimum_y_speed <= self.y_terminal_boost_speed <= self.maximum_speed:
            raise ValueError(
                "y_terminal_boost_speed must be between minimum_y_speed "
                "and maximum_speed"
            )
        if self.terminal_boost_error <= self.y_position_tolerance:
            raise ValueError(
                "terminal_boost_error must exceed y_position_tolerance"
            )
        if self.terminal_boost_delay < 0.0:
            raise ValueError("terminal_boost_delay cannot be negative")
        if not self.default_speed <= self.maximum_speed <= 1.0:
            raise ValueError(
                "maximum_speed must be >= default_speed and <= 1.0 m/s"
            )
        if self.position_kp <= 0.0:
            raise ValueError("position_kp must be positive")
        if self.cross_track_kp < 0.0 or self.heading_kp < 0.0:
            raise ValueError("Correction gains cannot be negative")
        if not 0.0 < self.compensation_heading_tolerance <= self.heading_tolerance:
            raise ValueError(
                "compensation_heading_tolerance_deg must be positive and "
                "not exceed heading_tolerance_deg"
            )
        if not (
            self.compensation_heading_tolerance
            <= self.compensation_post_stop_tolerance
            <= math.radians(3.0)
        ):
            raise ValueError(
                "compensation_post_stop_tolerance_deg must be no smaller "
                "than compensation_heading_tolerance_deg and no greater "
                "than 3 degrees"
            )
        if not 0.0 < self.minimum_compensation_angular_speed <= self.maximum_angular_speed:
            raise ValueError(
                "minimum_compensation_angular_speed must be positive and "
                "not exceed maximum_angular_speed"
            )
        if not math.radians(0.5) <= self.maximum_heading_compensation <= math.radians(30.0):
            raise ValueError(
                "maximum_heading_compensation_deg must be in [0.5, 30]"
            )
        if not math.isfinite(self.initial_heading_correction):
            raise ValueError("initial_heading_correction_deg must be finite")
        if not math.radians(1.0) <= self.max_translation_heading_error <= math.radians(30.0):
            raise ValueError(
                "max_translation_heading_error_deg must be in [1, 30]"
            )
        if self.position_tolerance <= 0.0:
            raise ValueError("position_tolerance must be positive")
        if self.x_position_tolerance <= 0.0 or self.y_position_tolerance <= 0.0:
            raise ValueError("Axis position tolerances must be positive")
        if not 0.0 <= self.post_stop_position_margin <= 0.010:
            raise ValueError(
                "post_stop_position_margin must be in [0, 0.010] m"
            )
        if not (
            self.cross_track_tolerance
            <= self.post_stop_cross_track_tolerance
            <= 0.050
        ):
            raise ValueError(
                "post_stop_cross_track_tolerance must be no smaller than "
                "cross_track_tolerance and no greater than 0.050 m"
            )
        if not (
            self.heading_tolerance
            <= self.post_stop_heading_tolerance
            <= math.radians(3.0)
        ):
            raise ValueError(
                "post_stop_heading_tolerance_deg must be no smaller than "
                "heading_tolerance_deg and no greater than 3 degrees"
            )
        if self.stable_cycles_required < 1:
            raise ValueError("stable_cycles must be at least 1")
        if self.stop_hold < 0.20:
            raise ValueError("stop_hold must be at least 0.20 s")
        if self.odom_timeout <= 0.0:
            raise ValueError("odom_timeout must be positive")
        if not 0.0 < self.maximum_distance <= 1.0:
            raise ValueError("maximum_distance must be in (0, 1.0] m")
        if self.maximum_motion_time <= 0.0:
            raise ValueError("maximum_motion_time must be positive")
        if not 3.0 <= self.minimum_motion_timeout <= self.maximum_motion_time:
            raise ValueError(
                "minimum_motion_timeout must be between 3 seconds and "
                "maximum_motion_time"
            )
        if self.timeout_scale <= 0.0 or self.timeout_margin < 0.0:
            raise ValueError(
                "timeout_scale must be positive and timeout_margin cannot be negative"
            )
        if not 1.0 <= self.stall_timeout <= self.maximum_motion_time:
            raise ValueError(
                "stall_timeout must be between 1 second and maximum_motion_time"
            )
        if self.progress_epsilon <= 0.0:
            raise ValueError("progress_epsilon must be positive")
        if self.feedback_rate <= 0.0:
            raise ValueError("feedback_rate must be positive")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1.0e9

    def _axis_minimum_speed(self, axis: str) -> float:
        return self.minimum_x_speed if axis == "x" else self.minimum_y_speed

    def _axis_position_tolerance(self, axis: str) -> float:
        if getattr(self, "reset_forward_active", False):
            return 0.002  # 1 cm微调不能使用普通8 mm到位阈值。
        return (
            self.x_position_tolerance
            if axis == "x"
            else self.y_position_tolerance
        )

    @staticmethod
    def _yaw_from_odom(msg: Odometry) -> float:
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _odom_callback(self, msg: Odometry) -> None:
        position = msg.pose.pose.position
        self.latest_pose = (
            float(position.x),
            float(position.y),
            self._yaw_from_odom(msg),
        )
        self.last_odom_time = self._now()

        if (
            not self.ready_announced
            and self.cmd_pub.get_subscription_count() > 0
        ):
            self.ready_announced = True
            self.get_logger().info("Controller ready")
            self._publish_status(
                "ready", True, "Ready to accept a motion command"
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

        if command in ("stop", "cancel", "emergency_stop"):
            self._stop_requested()
            return

        if self.state != self.IDLE:
            self._reject(f"Controller is busy: state={self.state}")
            return

        if self.latest_pose is None or self.last_odom_time is None:
            self._reject("No odometry has been received")
            return

        if self._now() - self.last_odom_time > self.odom_timeout:
            self._reject("Odometry is stale")
            return

        if self.cmd_pub.get_subscription_count() < 1:
            self._reject("No subscriber is connected to /cmd_vel")
            return

        self.command_id = str(data.get("id", ""))
        self.reset_forward_active = False

        try:
            if command == "reset_forward":
                self._start_reset_forward(float(data["distance"]))
                return
            if command == "set_origin":
                self._set_or_validate_origin()
                return
            if command == "align":
                self._start_alignment()
                return
            if command == "compensate_heading":
                right_degrees = float(data["right_degrees"])
                self._start_heading_compensation(right_degrees)
                return
            if command == "move_to":
                axis = str(data.get("axis", "")).strip().lower()
                if axis not in ("x", "y"):
                    raise ValueError("move_to requires axis='x' or axis='y'")
                target_x = float(data["target_x"])
                target_y = float(data["target_y"])
                speed = abs(float(data.get("speed", self.default_speed)))
                self._start_absolute_motion(axis, target_x, target_y, speed)
                return
            axis, distance = self._resolve_command(command, data)
            speed = abs(float(data.get("speed", self.default_speed)))
            self._start_motion(axis, distance, speed)
        except (KeyError, TypeError, ValueError) as exc:
            self._reject(str(exc))

    def _start_reset_forward(self, distance: float) -> None:
        """在G5附近沿修正后的车头低速前进，不自动执行在其他位置。"""
        if not math.isfinite(distance) or not 0.005 <= distance <= 0.05:
            raise ValueError("reset_forward distance must be in [0.005, 0.05] m")
        x, y, heading_error = self._global_logical_pose()
        if math.hypot(x, y) > 0.05 or abs(heading_error) > self.heading_tolerance:
            raise ValueError("reset_forward requires aligned chassis within 5 cm of G5")
        self.reset_forward_active = True
        try:
            # 厘米级微调采用低速；不会跟随常规路线的1 m/s速度上限。
            self._start_motion("x", distance, self.minimum_x_speed)
        except (ValueError, TypeError):
            self.reset_forward_active = False
            raise

    def _commit_reset_translation(self) -> None:
        """只在停车复验成功后保留实际微调位移，不旋转场地坐标轴。"""
        assert self.origin_pose is not None
        assert self.start_pose is not None and self.latest_pose is not None
        dx = self.latest_pose[0] - self.start_pose[0]
        dy = self.latest_pose[1] - self.start_pose[1]
        ox, oy, yaw = self.origin_pose
        self.origin_pose = (ox + dx, oy + dy, yaw)
        cx, cy = self.manual_translation_world
        self.manual_translation_world = (cx + dx, cy + dy)
        self.pending_message += (
            f"; manual route translation retained: dx={dx:+.5f}, dy={dy:+.5f} m"
        )

    def _set_or_validate_origin(self) -> None:
        """在 G5 锁定场地原点；重复调用只能验证，不能悄悄重置。"""
        assert self.latest_pose is not None
        if self.origin_pose is None:
            self.reference_yaw = normalize_angle(
                self.latest_pose[2] + self.initial_heading_correction
            )
            self.heading_target_yaw = self.reference_yaw
            self.origin_pose = (
                self.latest_pose[0],
                self.latest_pose[1],
                self.reference_yaw,
            )
            self.active_axis = "origin"
            self.target_distance = 0.0
            self.last_progress = 0.0
            self.last_cross_track = 0.0
            self.last_heading_error = normalize_angle(
                self.latest_pose[2] - self.reference_yaw
            )
            self.get_logger().warning(
                "Field origin locked at the current G5 grasping pose"
            )
            self._publish_status(
                "succeeded", True, "Field origin locked at G5"
            )
            return

        logical_x, logical_y, heading_error = self._global_logical_pose()
        if (
            abs(logical_x) > self.x_position_tolerance
            or abs(logical_y) > self.y_position_tolerance
            or abs(heading_error) > self.heading_tolerance
        ):
            raise ValueError(
                "Existing field origin is not at the current chassis pose: "
                f"logical=({logical_x:.4f},{logical_y:.4f}) m, "
                f"heading={math.degrees(heading_error):+.2f} deg. "
                "Return the robot to G5 or restart only after manual recovery; "
                "the origin was not reset."
            )
        self.active_axis = "origin"
        self.last_progress = logical_x
        self.last_cross_track = logical_y
        self.last_heading_error = heading_error
        self._publish_status(
            "succeeded", True, "Existing field origin verified at G5"
        )

    def _resolve_command(
        self, command: str, data: Dict[str, Any]
    ) -> Tuple[str, float]:
        if command == "move":
            axis = str(data.get("axis", "")).strip().lower()
            if axis not in ("x", "y"):
                raise ValueError("move requires axis='x' or axis='y'")
            if "distance" not in data:
                raise ValueError("move requires distance in metres")
            logical_distance = float(data["distance"])
            return axis, self._physical_distance(axis, logical_distance)

        named_commands = {
            "next_column": (
                self.column_axis,
                self.column_sign * self.column_spacing,
            ),
            "previous_column": (
                self.column_axis,
                -self.column_sign * self.column_spacing,
            ),
            "next_row": (
                self.row_axis,
                self.row_sign * self.row_spacing,
            ),
            "previous_row": (
                self.row_axis,
                -self.row_sign * self.row_spacing,
            ),
        }

        if command not in named_commands:
            raise ValueError(
                "Unknown command. Use move, next_column, "
                "previous_column, next_row, previous_row, align, "
                "compensate_heading, or stop"
            )

        axis, logical_distance = named_commands[command]
        return axis, self._physical_distance(axis, logical_distance)

    def _physical_distance(self, axis: str, logical_distance: float) -> float:
        """Apply real-robot axis direction calibration to a logical move."""
        if axis == "x":
            return self.x_command_sign * logical_distance
        if axis == "y":
            return self.y_command_sign * logical_distance
        raise ValueError(f"Unsupported physical axis: {axis}")

    def _start_motion(
        self, axis: str, distance: float, speed: float
    ) -> None:
        if not math.isfinite(distance) or abs(distance) < 0.001:
            raise ValueError(
                "distance must be finite and at least 0.001 m"
            )
        if abs(distance) > self.maximum_distance:
            raise ValueError(
                f"Requested distance {distance:.3f} m exceeds "
                f"maximum_distance {self.maximum_distance:.3f} m"
            )
        if not math.isfinite(speed):
            raise ValueError("speed must be finite")
        axis_minimum_speed = self._axis_minimum_speed(axis)
        if not axis_minimum_speed <= speed <= self.maximum_speed:
            raise ValueError(
                f"speed must be between {axis_minimum_speed:.3f} "
                f"and {self.maximum_speed:.3f} m/s"
            )

        self.start_pose = self.latest_pose
        self._ensure_reference_yaw()
        self.absolute_mode = False
        self.target_position = None
        self.active_axis = axis
        self.target_distance = distance
        self.active_speed = speed
        self.motion_started = self._now()
        self.last_progress_time = self.motion_started
        self.best_directed_progress = 0.0
        self.best_abs_error = abs(distance)
        self.terminal_boost_active = False
        self.stable_count = 0
        self.last_progress = 0.0
        self.last_cross_track = 0.0
        self.last_heading_error = 0.0

        estimated_time = abs(distance) / speed
        self.motion_timeout = min(
            self.maximum_motion_time,
            max(
                self.minimum_motion_timeout,
                estimated_time * self.timeout_scale + self.timeout_margin,
            ),
        )

        self.state = self.MOVING
        self.get_logger().warn(
            f"Motion started: axis={axis}, distance={distance:.3f} m, "
            f"speed_limit={speed:.3f} m/s, "
            f"timeout={self.motion_timeout:.1f} s"
        )
        self._publish_status(
            "started", True, "Closed-loop translation started"
        )

    def _start_absolute_motion(
        self, axis: str, target_x: float, target_y: float, speed: float
    ) -> None:
        """移动到固定场地坐标，避免每段从当前位置重新累计误差。"""
        if self.origin_pose is None:
            raise ValueError("Field origin is not locked; send set_origin first")
        if not all(math.isfinite(v) for v in (target_x, target_y, speed)):
            raise ValueError("move_to target and speed must be finite")
        axis_minimum_speed = self._axis_minimum_speed(axis)
        if not axis_minimum_speed <= speed <= self.maximum_speed:
            raise ValueError(
                f"speed must be between {axis_minimum_speed:.3f} "
                f"and {self.maximum_speed:.3f} m/s"
            )

        current_x, current_y, _ = self._global_logical_pose()
        start_axis = current_x if axis == "x" else current_y
        target_axis = target_x if axis == "x" else target_y
        distance = target_axis - start_axis
        if abs(distance) > self.maximum_distance:
            raise ValueError(
                f"Requested absolute segment {distance:.3f} m exceeds "
                f"maximum_distance {self.maximum_distance:.3f} m"
            )

        self.start_pose = self.latest_pose
        self.active_axis = axis
        self.absolute_mode = True
        self.target_position = (target_x, target_y)
        self.target_distance = distance
        self.active_speed = speed
        self.motion_started = self._now()
        self.last_progress_time = self.motion_started
        self.best_directed_progress = 0.0
        self.best_abs_error = abs(distance)
        self.terminal_boost_active = False
        self.stable_count = 0
        self.last_progress = 0.0
        self.last_cross_track = 0.0
        self.last_heading_error = 0.0

        estimated_time = max(
            abs(distance), self._axis_position_tolerance(axis)
        ) / speed
        self.motion_timeout = min(
            self.maximum_motion_time,
            max(
                self.minimum_motion_timeout,
                estimated_time * self.timeout_scale + self.timeout_margin,
            ),
        )
        self.state = self.MOVING
        self.get_logger().warning(
            f"Absolute motion started: axis={axis}, target="
            f"({target_x:.3f},{target_y:.3f}) m, timeout={self.motion_timeout:.1f} s"
        )
        self._publish_status(
            "started", True, "Absolute field-coordinate motion started"
        )

    def _ensure_reference_yaw(self) -> None:
        assert self.latest_pose is not None
        if self.reference_yaw is None:
            self.reference_yaw = normalize_angle(
                self.latest_pose[2] + self.initial_heading_correction
            )
            self.get_logger().info(
                "Global grid heading locked at "
                f"{math.degrees(self.reference_yaw):.2f} deg"
            )
        if self.heading_target_yaw is None:
            self.heading_target_yaw = self.reference_yaw

    def _start_alignment(self) -> None:
        assert self.latest_pose is not None
        self._ensure_reference_yaw()
        self.start_pose = self.latest_pose
        self.active_axis = "heading"
        self.absolute_mode = False
        self.target_position = None
        self.target_distance = 0.0
        self.active_speed = 0.0
        self.motion_started = self._now()
        self.motion_timeout = min(self.maximum_motion_time, 12.0)
        self.stable_count = 0
        self.last_progress = 0.0
        self.last_cross_track = 0.0
        self.last_heading_error = normalize_angle(
            self.latest_pose[2] - self.heading_target_yaw
        )
        self.last_feedback_time = 0.0
        self.state = self.MOVING
        self.get_logger().warn(
            "Heading alignment started: error="
            f"{math.degrees(self.last_heading_error):+.2f} deg"
        )
        self._publish_status(
            "started", True, "Closed-loop heading alignment started"
        )

    def _start_heading_compensation(self, right_degrees: float) -> None:
        """在当前位置顺时针修正，并在成功后更新后续航向保持目标。"""
        assert self.latest_pose is not None
        self._ensure_reference_yaw()
        if not math.isfinite(right_degrees):
            raise ValueError("right_degrees must be finite")
        requested = math.radians(right_degrees)
        if requested <= 0.0:
            raise ValueError("right_degrees must be greater than zero")
        if requested > self.maximum_heading_compensation:
            raise ValueError(
                f"right_degrees exceeds maximum "
                f"{math.degrees(self.maximum_heading_compensation):.1f} deg"
            )

        self.start_pose = self.latest_pose
        self.active_axis = "heading_compensation"
        self.absolute_mode = False
        self.target_position = None
        # ROS约定angular.z为正时逆时针（左转），因此向右为负角度。
        self.target_distance = -requested
        self.compensation_right_deg = right_degrees
        self.compensation_target_yaw = normalize_angle(
            self.latest_pose[2] - requested
        )
        self.active_speed = 0.0
        self.motion_started = self._now()
        self.motion_timeout = min(self.maximum_motion_time, 12.0)
        self.stable_count = 0
        self.last_progress = 0.0
        self.last_cross_track = 0.0
        self.last_heading_error = normalize_angle(
            self.latest_pose[2] - self.compensation_target_yaw
        )
        self.last_feedback_time = 0.0
        self.state = self.MOVING
        self.get_logger().warning(
            f"Cycle heading compensation started: right={right_degrees:.2f} deg"
        )
        self._publish_status(
            "started",
            True,
            f"Clockwise heading compensation started: {right_degrees:.2f} deg",
        )

    def _control_callback(self) -> None:
        now = self._now()

        if self.state == self.IDLE:
            if (
                not self.ready_announced
                and self.latest_pose is not None
                and self.cmd_pub.get_subscription_count() > 0
            ):
                self.ready_announced = True
                self.get_logger().info("Controller ready")
                self._publish_status(
                    "ready", True, "Ready to accept a motion command"
                )
            return

        if self.state == self.STOPPING:
            self._publish_zero()
            if now - self.stop_started >= self.stop_hold:
                if self.pending_success:
                    verified, detail = self._verify_stopped_target(now)
                    if not verified:
                        self.pending_success = False
                        self.pending_message = (
                            "Post-stop verification failed: " + detail
                        )
                # 只有补偿动作通过停止后复验，才更新后续航向目标。
                # 失败、取消或超时绝不能改变目标，否则会掩盖异常。
                if (
                    self.pending_success
                    and self.active_axis == "heading_compensation"
                    and self.latest_pose is not None
                ):
                    self.heading_target_yaw = self.latest_pose[2]
                    self.compensation_count += 1
                    self.last_heading_error = 0.0
                    self.pending_message += (
                        f"; new heading target accepted after clockwise "
                        f"correction #{self.compensation_count}"
                    )
                if self.pending_success and getattr(self, "reset_forward_active", False):
                    self._commit_reset_translation()
                self.state = self.IDLE
                event = "succeeded" if self.pending_success else "failed"
                self._publish_status(
                    event, self.pending_success, self.pending_message
                )
                if self.pending_success:
                    self.get_logger().info(self.pending_message)
                else:
                    self.get_logger().error(self.pending_message)
            return

        if self.last_odom_time is None:
            self._begin_stop(False, "Odometry is unavailable")
            return

        if now - self.last_odom_time > self.odom_timeout:
            self._begin_stop(False, "Odometry timeout; motion aborted")
            return

        if self.cmd_pub.get_subscription_count() < 1:
            self._begin_stop(
                False, "The /cmd_vel subscriber disappeared"
            )
            return

        if self.start_pose is None or self.latest_pose is None:
            self._begin_stop(False, "Motion pose is unavailable")
            return

        dx_body, dy_body, heading_error = self._relative_pose()

        # 总超时必须先于航向恢复分支检查，避免持续转向时绕过硬超时。
        elapsed = now - self.motion_started
        if elapsed > self.motion_timeout:
            self._begin_stop(
                False,
                "Motion hard timeout: "
                f"elapsed={elapsed:.1f} s, progress={self.last_progress:.4f} m",
            )
            return

        if self.active_axis == "heading":
            self._control_alignment(now, heading_error)
            return
        if self.active_axis == "heading_compensation":
            self._control_heading_compensation(now)
            return

        # Heading-first safety gate. Translation remains exactly zero while
        # the chassis is more than the allowed angle away from the global
        # grid heading. Once recovered, normal mecanum X/Y translation resumes
        # with continuous small angular stabilization.
        if abs(heading_error) > self.max_translation_heading_error:
            self.stable_count = 0
            command = Twist()
            command.angular.z = clamp(
                -self.heading_kp * heading_error,
                -self.maximum_angular_speed,
                self.maximum_angular_speed,
            )
            self.cmd_pub.publish(command)
            if now - self.last_feedback_time >= 1.0 / self.feedback_rate:
                self.last_feedback_time = now
                self._publish_status(
                    "heading_recovery",
                    True,
                    "Linear motion paused; correcting heading error="
                    f"{math.degrees(heading_error):+.2f} deg",
                )
            return

        if self.absolute_mode:
            assert self.target_position is not None
            logical_x, logical_y, heading_error = self._global_logical_pose()
            start_x, start_y, _ = self._logical_pose_from(self.start_pose)
            if self.active_axis == "x":
                progress = logical_x - start_x
                error = self.target_position[0] - logical_x
                cross_track = logical_y - self.target_position[1]
            else:
                progress = logical_y - start_y
                error = self.target_position[1] - logical_y
                cross_track = logical_x - self.target_position[0]
        elif self.active_axis == "x":
            progress = dx_body
            cross_track = dy_body
            error = self.target_distance - progress
        else:
            progress = dy_body
            cross_track = dx_body
            error = self.target_distance - progress
        self.last_progress = progress
        self.last_cross_track = cross_track
        self.last_heading_error = heading_error

        # 以“到目标的绝对误差是否缩小”判断进展。这样即使越过目标后
        # 反向修正，也会被视为有效进展，不会被旧的单向进度误判停滞。
        abs_error = abs(error)
        if abs_error <= self.best_abs_error - self.progress_epsilon:
            self.best_abs_error = abs_error
            self.last_progress_time = now

        position_tolerance = self._axis_position_tolerance(self.active_axis)
        # 人工复位只允许向前：首次接近或跨过目标立即进入停车保持，
        # 不等待连续5帧，也不按负误差倒车，避免厘米级往返振荡。
        # 此处只是请求停车，成功仍须通过停车后的新鲜反馈复验。
        if getattr(self, "reset_forward_active", False) and error <= position_tolerance:
            self._begin_stop(
                True,
                f"One-way reset stopped: requested={self.target_distance:.4f} m, "
                f"progress={progress:.4f} m; awaiting post-stop verification",
            )
            return
        no_progress_time = now - self.last_progress_time
        if (
            no_progress_time > self.stall_timeout
            and abs_error > position_tolerance
        ):
            self._begin_stop(
                False,
                "Motion stalled: "
                f"no effective progress for {no_progress_time:.1f} s, "
                f"progress={progress:.4f} m, error={error:.4f} m",
            )
            return

        position_ok = abs_error <= position_tolerance
        cross_ok = abs(cross_track) <= self.cross_track_tolerance
        heading_ok = abs(heading_error) <= self.heading_tolerance

        if position_ok and cross_ok and heading_ok:
            self.stable_count += 1
            self._publish_zero()
            if self.stable_count >= self.stable_cycles_required:
                self._begin_stop(
                    True,
                    "Target reached: "
                    f"progress={progress:.4f} m, "
                    f"error={error:.4f} m, "
                    f"cross={cross_track:.4f} m, "
                    f"heading={math.degrees(heading_error):.2f} deg",
                )
            return

        self.stable_count = 0

        # 各轴独立停机：主轴到达容差后不得再输出 minimum_speed。
        if position_ok:
            along_speed = 0.0
            self.terminal_boost_active = False
        else:
            speed_floor = self._axis_minimum_speed(self.active_axis)
            self.terminal_boost_active = (
                self.active_axis == "y"
                and abs_error <= self.terminal_boost_error
                and no_progress_time >= self.terminal_boost_delay
            )
            if self.terminal_boost_active:
                speed_floor = min(
                    self.y_terminal_boost_speed, self.active_speed
                )
            along_speed = clamp(
                self.position_kp * abs_error,
                speed_floor,
                self.active_speed,
            )
            along_speed = math.copysign(along_speed, error)

        cross_speed = 0.0 if cross_ok else clamp(
            -self.cross_track_kp * cross_track,
            -self.maximum_cross_speed,
            self.maximum_cross_speed,
        )

        if self.active_axis == "x":
            vx_initial = along_speed
            vy_initial = cross_speed
        else:
            vx_initial = cross_speed
            vy_initial = along_speed

        if self.absolute_mode:
            # 绝对目标使用逻辑场地坐标，先映射为 EP 物理轴方向。
            vx_initial *= self.x_command_sign
            vy_initial *= self.y_command_sign

        # Convert velocity expressed in the initial chassis frame into the
        # current chassis frame before publishing Twist.
        # 速度转换必须使用移动坐标系与车身的夹角，不能把航向目标误差
        # 当作场地夹角；否则每轮补偿之后绝对坐标运动会方向偏斜。
        movement_yaw = (
            self.heading_target_yaw
            if getattr(self, "reset_forward_active", False)
            else self.reference_yaw
        )
        frame_error = normalize_angle(self.latest_pose[2] - movement_yaw)
        cos_yaw = math.cos(frame_error)
        sin_yaw = math.sin(frame_error)
        vx_current = cos_yaw * vx_initial + sin_yaw * vy_initial
        vy_current = -sin_yaw * vx_initial + cos_yaw * vy_initial

        angular_speed = 0.0 if heading_ok else clamp(
            -self.heading_kp * heading_error,
            -self.maximum_angular_speed,
            self.maximum_angular_speed,
        )

        command = Twist()
        command.linear.x = vx_current
        command.linear.y = vy_current
        command.angular.z = angular_speed
        self.cmd_pub.publish(command)

        if now - self.last_feedback_time >= 1.0 / self.feedback_rate:
            self.last_feedback_time = now
            self._publish_status(
                "moving",
                True,
                f"progress={progress:.4f}, error={error:.4f}",
            )

    def _control_alignment(
        self, now: float, heading_error: float
    ) -> None:
        self.last_progress = 0.0
        self.last_cross_track = 0.0
        self.last_heading_error = heading_error

        if now - self.motion_started > self.motion_timeout:
            self._begin_stop(
                False,
                "Heading alignment timeout: error="
                f"{math.degrees(heading_error):+.2f} deg",
            )
            return

        if abs(heading_error) <= self.heading_tolerance:
            self.stable_count += 1
            self._publish_zero()
            if self.stable_count >= self.stable_cycles_required:
                self._begin_stop(
                    True,
                    "Heading aligned: error="
                    f"{math.degrees(heading_error):+.2f} deg",
                )
            return

        self.stable_count = 0
        command = Twist()
        command.angular.z = clamp(
            -self.heading_kp * heading_error,
            -self.maximum_angular_speed,
            self.maximum_angular_speed,
        )
        self.cmd_pub.publish(command)

        if now - self.last_feedback_time >= 1.0 / self.feedback_rate:
            self.last_feedback_time = now
            self._publish_status(
                "moving",
                True,
                "aligning heading: error="
                f"{math.degrees(heading_error):+.2f} deg",
            )

    def _control_heading_compensation(self, now: float) -> None:
        """执行一次顺时针相对转动；成功后由STOPPING阶段提交新目标。"""
        assert self.latest_pose is not None
        if self.compensation_target_yaw is None:
            self._begin_stop(
                False, "Heading compensation target is unavailable"
            )
            return

        error = normalize_angle(
            self.latest_pose[2] - self.compensation_target_yaw
        )
        self.last_progress = normalize_angle(
            self.latest_pose[2] - self.start_pose[2]
        )
        self.last_cross_track = 0.0
        self.last_heading_error = error

        if abs(error) <= self.compensation_heading_tolerance:
            self.stable_count += 1
            self._publish_zero()
            if self.stable_count >= self.stable_cycles_required:
                self._begin_stop(
                    True,
                    "Clockwise heading compensation reached: "
                    f"requested={self.compensation_right_deg:.2f} deg, "
                    f"residual={math.degrees(error):+.2f} deg",
                )
            return

        self.stable_count = 0
        angular_speed = clamp(
            self.heading_kp * abs(error),
            self.minimum_compensation_angular_speed,
            self.maximum_angular_speed,
        )
        command = Twist()
        command.angular.z = math.copysign(angular_speed, -error)
        self.cmd_pub.publish(command)

        if now - self.last_feedback_time >= 1.0 / self.feedback_rate:
            self.last_feedback_time = now
            self._publish_status(
                "moving",
                True,
                "clockwise compensation: remaining="
                f"{math.degrees(error):+.2f} deg",
            )

    def _relative_pose(self) -> Tuple[float, float, float]:
        assert self.start_pose is not None
        assert self.latest_pose is not None

        x0, y0, _ = self.start_pose
        x1, y1, yaw1 = self.latest_pose
        dx_world = x1 - x0
        dy_world = y1 - y0

        self._ensure_reference_yaw()
        assert self.reference_yaw is not None
        assert self.heading_target_yaw is not None
        reference_yaw = self.reference_yaw
        if getattr(self, "reset_forward_active", False):
            reference_yaw = self.heading_target_yaw

        dx_body = (
            math.cos(reference_yaw) * dx_world
            + math.sin(reference_yaw) * dy_world
        )
        dy_body = (
            -math.sin(reference_yaw) * dx_world
            + math.cos(reference_yaw) * dy_world
        )
        heading_error = normalize_angle(yaw1 - self.heading_target_yaw)

        return dx_body, dy_body, heading_error

    def _logical_pose_from(
        self, pose: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        if (
            self.origin_pose is None
            or self.reference_yaw is None
            or self.heading_target_yaw is None
        ):
            raise ValueError("Field origin is unavailable")
        x0, y0, _ = self.origin_pose
        x1, y1, yaw1 = pose
        dx_world = x1 - x0
        dy_world = y1 - y0
        physical_x = (
            math.cos(self.reference_yaw) * dx_world
            + math.sin(self.reference_yaw) * dy_world
        )
        physical_y = (
            -math.sin(self.reference_yaw) * dx_world
            + math.cos(self.reference_yaw) * dy_world
        )
        return (
            self.x_command_sign * physical_x,
            self.y_command_sign * physical_y,
            normalize_angle(yaw1 - self.heading_target_yaw),
        )

    def _global_logical_pose(self) -> Tuple[float, float, float]:
        if self.latest_pose is None:
            raise ValueError("Odometry pose is unavailable")
        return self._logical_pose_from(self.latest_pose)

    def _stop_requested(self) -> None:
        if self.state == self.IDLE:
            self._publish_stop_burst()
            self._publish_status(
                "stopped", True, "Robot is already stopped"
            )
            return

        self._begin_stop(False, "Motion cancelled by stop command")

    def _verify_stopped_target(self, now: float) -> Tuple[bool, str]:
        """停止保持结束后再次核验，避免惯性漂移仍被报告为成功。"""
        if (
            self.latest_pose is None
            or self.last_odom_time is None
            or now - self.last_odom_time > self.odom_timeout
        ):
            return False, "odometry is unavailable or stale"

        if self.active_axis == "heading":
            if self.heading_target_yaw is None:
                return False, "heading target is unavailable"
            heading_error = normalize_angle(
                self.latest_pose[2] - self.heading_target_yaw
            )
            return (
                abs(heading_error) <= self.heading_tolerance,
                f"heading error={math.degrees(heading_error):+.2f} deg",
            )

        if self.active_axis == "heading_compensation":
            if self.compensation_target_yaw is None:
                return False, "compensation heading target is unavailable"
            heading_error = normalize_angle(
                self.latest_pose[2] - self.compensation_target_yaw
            )
            return (
                abs(heading_error)
                <= self.compensation_post_stop_tolerance,
                "compensation heading error="
                f"{math.degrees(heading_error):+.2f} deg",
            )

        if self.absolute_mode:
            if self.target_position is None:
                return False, "absolute target is unavailable"
            x, y, heading_error = self._global_logical_pose()
            if self.active_axis == "x":
                along_error = self.target_position[0] - x
                cross_error = y - self.target_position[1]
            else:
                along_error = self.target_position[1] - y
                cross_error = x - self.target_position[0]
        else:
            dx_body, dy_body, heading_error = self._relative_pose()
            if self.active_axis == "x":
                along_error = self.target_distance - dx_body
                cross_error = dy_body
            else:
                along_error = self.target_distance - dy_body
                cross_error = dx_body

        if getattr(self, "reset_forward_active", False):
            # 微调停止后允许有限滑行：5 cm目标容许±12 mm，1 cm目标
            # 容许±5 mm。超过范围直接失败，绝不反向修正或提交原点偏移。
            reset_tolerance = min(0.015, 0.75 * self.target_distance)
            valid = (
                abs(along_error) <= reset_tolerance
                and abs(cross_error) <= self.post_stop_cross_track_tolerance
                and abs(heading_error) <= self.post_stop_heading_tolerance
            )
            detail = (
                f"reset requested={self.target_distance:.4f} m, "
                f"actual={self.target_distance - along_error:.4f} m, "
                f"error={along_error:+.4f} m, tolerance={reset_tolerance:.4f} m, "
                f"cross={cross_error:+.4f} m, "
                f"heading={math.degrees(heading_error):+.2f} deg"
            )
            if valid:
                self.pending_message = "One-way reset verified: " + detail
            return valid, detail

        valid = (
            abs(along_error)
            <= (
                self._axis_position_tolerance(self.active_axis)
                + self.post_stop_position_margin
            )
            and abs(cross_error) <= self.post_stop_cross_track_tolerance
            and abs(heading_error) <= self.post_stop_heading_tolerance
        )
        return (
            valid,
            f"along={along_error:+.4f} m, cross={cross_error:+.4f} m, "
            f"heading={math.degrees(heading_error):+.2f} deg",
        )

    def _begin_stop(self, success: bool, message: str) -> None:
        if self.state == self.STOPPING:
            return

        self.pending_success = success
        self.pending_message = message
        self.stop_started = self._now()
        self.state = self.STOPPING
        self._publish_stop_burst()

        if success:
            self.get_logger().info("Target reached; holding stop")
        else:
            self.get_logger().error(message)

    def _reject(self, message: str) -> None:
        self.get_logger().error(f"Command rejected: {message}")
        self._publish_status("rejected", False, message)

    def _publish_zero(self) -> None:
        self.cmd_pub.publish(Twist())

    def _publish_stop_burst(self) -> None:
        for _ in range(8):
            self._publish_zero()

    def _publish_status(
        self, event: str, success: bool, message: str
    ) -> None:
        payload = {
            "software_version": self.SOFTWARE_VERSION,
            "motion_protocol": "fixed_g5_reset_v4",
            "manual_translation_world": getattr(self, "manual_translation_world", (0.0, 0.0)),
            "reset_forward_active": getattr(self, "reset_forward_active", False),
            "event": event,
            "success": success,
            "state": self.state,
            "id": self.command_id,
            "axis": self.active_axis,
            "target_distance": round(self.target_distance, 6),
            "progress": round(self.last_progress, 6),
            "cross_track": round(self.last_cross_track, 6),
            "heading_error_deg": round(
                math.degrees(self.last_heading_error), 4
            ),
            "reference_yaw_deg": (
                None
                if self.reference_yaw is None
                else round(math.degrees(self.reference_yaw), 4)
            ),
            "heading_target_yaw_deg": (
                None
                if self.heading_target_yaw is None
                else round(math.degrees(self.heading_target_yaw), 4)
            ),
            "compensation_target_yaw_deg": (
                None
                if self.compensation_target_yaw is None
                else round(
                    math.degrees(self.compensation_target_yaw), 4
                )
            ),
            "compensation_right_deg": round(
                self.compensation_right_deg, 4
            ),
            "compensation_post_stop_tolerance_deg": round(
                math.degrees(self.compensation_post_stop_tolerance), 4
            ),
            "post_stop_position_margin": round(
                self.post_stop_position_margin, 6
            ),
            "post_stop_cross_track_tolerance": round(
                self.post_stop_cross_track_tolerance, 6
            ),
            "post_stop_heading_tolerance_deg": round(
                math.degrees(self.post_stop_heading_tolerance), 4
            ),
            "compensation_count": self.compensation_count,
            "origin_locked": self.origin_pose is not None,
            "logical_position": self._logical_position_payload(),
            "target_position": (
                None
                if self.target_position is None
                else {
                    "x": round(self.target_position[0], 6),
                    "y": round(self.target_position[1], 6),
                }
            ),
            "active_position_tolerance": (
                None
                if self.active_axis not in ("x", "y")
                else round(
                    self._axis_position_tolerance(self.active_axis), 6
                )
            ),
            "terminal_boost_active": self.terminal_boost_active,
            "message": message,
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)

    def _logical_position_payload(self) -> Optional[Dict[str, float]]:
        if self.origin_pose is None or self.latest_pose is None:
            return None
        try:
            x, y, _ = self._global_logical_pose()
        except ValueError:
            return None
        return {"x": round(x, 6), "y": round(y, 6)}

    def safe_shutdown(self) -> None:
        self._publish_stop_burst()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[ChassisMotionNode] = None

    try:
        node = ChassisMotionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().warn(
                "Ctrl+C received; sending stop commands"
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


#!/usr/bin/env python3
"""Coordinate vision, chassis and arm nodes for two-stage object sorting.

Grid: G1 G2 G3 / G4 G5 G6. Order: G4 -> G1 -> G5 -> G2 -> G6 -> G3.
The robot must start at the chassis pose aligned with G5.

Phase 1 homes the arm and moves it to the observation pose. Phase 2 freezes
the current stable grid result and starts the physical sorting sequence.
"""

import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String


class SortingCoordinatorNode(Node):
    SOFTWARE_VERSION = "2026-09-21-per-grid-reset-v23"
    IDLE = "IDLE"
    CAPTURING_ORIGIN = "CAPTURING_ORIGIN"
    HOMING_FOR_CAMERA = "HOMING_FOR_CAMERA"
    POSITIONING_CAMERA = "POSITIONING_CAMERA"
    WAITING_SNAPSHOT = "WAITING_SNAPSHOT"
    EXECUTING = "EXECUTING"
    WAITING_CHILD = "WAITING_CHILD"
    SETTLING = "SETTLING"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"

    def __init__(self) -> None:
        super().__init__("sorting_coordinator")

        self.declare_parameter("grid_state_topic", "/grid_states")
        self.declare_parameter("chassis_command_topic", "/chassis_motion/command")
        self.declare_parameter("chassis_status_topic", "/chassis_motion/status")
        self.declare_parameter("arm_command_topic", "/arm_task/command")
        self.declare_parameter("arm_status_topic", "/arm_task/status")
        # Phase 1: home arm and enter observation pose.
        self.declare_parameter("start_topic", "/sorting/phase1")
        # Phase 2: freeze the current stable snapshot and start sorting.
        self.declare_parameter("execute_topic", "/sorting/phase2")
        self.declare_parameter("stop_topic", "/sorting/stop")
        self.declare_parameter("status_topic", "/sorting/status")

        self.declare_parameter("rows", 2)
        self.declare_parameter("columns", 3)
        self.declare_parameter("grid_ids", ["G1", "G2", "G3", "G4", "G5", "G6"])
        # Process the near row before the front cell in each column so the
        # lowered gripper cannot strike an untouched near-row object.
        self.declare_parameter("task_order", ["G4", "G1", "G5", "G2", "G6", "G3"])
        self.declare_parameter("row_spacing", 0.20)
        self.declare_parameter("column_spacing", 0.20)
        # The real robot starts in front of the rear-centre cell G5.
        self.declare_parameter("initial_grid", "G5")
        # chassis_speed is retained for compatibility with older launch files.
        self.declare_parameter("chassis_speed", 1.0)
        self.declare_parameter("chassis_x_speed", 1.0)
        self.declare_parameter("chassis_y_speed", 1.0)
        # 正值表示每个物体放置完并回到G5后，额外向右（顺时针）修正。
        # 设为0可完全关闭，兼容历史启动文件与单独节点测试。
        self.declare_parameter("cycle_heading_compensation_deg", 2.0)
        # 按格号绑定，不因空格跳过而改变对应关系；统一角度仅作为默认值。
        for grid in ("g1", "g2", "g3", "g4", "g5", "g6"):
            self.declare_parameter(
                f"{grid}_reset_angle_deg",
                float(self.get_parameter("cycle_heading_compensation_deg").value),
            )
            self.declare_parameter(f"{grid}_reset_forward_m", 0.01)
        self.grid_reset_angles = {
            f"G{i}": self._float(f"g{i}_reset_angle_deg") for i in range(1, 7)
        }
        self.grid_reset_distances = {
            f"G{i}": self._float(f"g{i}_reset_forward_m") for i in range(1, 7)
        }
        # The rear safety lane is derived as one row spacing behind G5.
        # Each bin entry is 0.15 m outside the corresponding outer grid column.
        self.declare_parameter("bin_outside_margin", 0.15)

        # The final scene contains four bottles and two apples. Each class bin
        # supports up to four positions at offsets 0.30, 0.20, 0.10 and 0.00 m
        # from its entry; the entry itself is already 0.15 m outside the grid.
        self.declare_parameter("bin_slot_count", 4)
        self.declare_parameter("bin_first_offset", 0.30)
        self.declare_parameter("bin_slot_spacing", 0.10)
        self.declare_parameter("bottle_slot_y_direction", 1)
        self.declare_parameter("apple_slot_y_direction", -1)

        self.declare_parameter("allowed_classes", ["bottle", "apple"])
        self.declare_parameter("force_g5_bottle", True)
        self.declare_parameter("snapshot_confirmation_frames", 3)
        self.declare_parameter("snapshot_timeout", 20.0)
        self.declare_parameter("snapshot_freshness_timeout", 1.5)
        self.declare_parameter("chassis_timeout", 40.0)
        self.declare_parameter("arm_timeout", 75.0)
        self.declare_parameter("settle_time", 0.50)
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("position_epsilon", 0.001)
        # Keep each generated move below chassis_motion.maximum_distance.
        self.declare_parameter("max_chassis_segment", 0.49)
        self.declare_parameter("return_to_initial_grid", False)
        # 1 processes G4, the first cell in task_order; 6 runs all cells.
        self.declare_parameter("object_limit", 6)

        self.grid_state_topic = self._str("grid_state_topic")
        self.chassis_command_topic = self._str("chassis_command_topic")
        self.chassis_status_topic = self._str("chassis_status_topic")
        self.arm_command_topic = self._str("arm_command_topic")
        self.arm_status_topic = self._str("arm_status_topic")
        self.start_topic = self._str("start_topic")
        self.execute_topic = self._str("execute_topic")
        self.stop_topic = self._str("stop_topic")
        self.status_topic = self._str("status_topic")

        self.rows = self._int("rows")
        self.columns = self._int("columns")
        self.grid_ids = list(self.get_parameter("grid_ids").value)
        self.task_order = list(self.get_parameter("task_order").value)
        self.row_spacing = self._float("row_spacing")
        self.column_spacing = self._float("column_spacing")
        self.initial_grid = self._str("initial_grid")
        self.chassis_speed = self._float("chassis_speed")
        self.chassis_x_speed = self._float("chassis_x_speed")
        self.chassis_y_speed = self._float("chassis_y_speed")
        self.cycle_heading_compensation_deg = self._float(
            "cycle_heading_compensation_deg"
        )
        self.bin_outside_margin = self._float("bin_outside_margin")
        self.safe_lane_x = -self.row_spacing
        self.bottle_edge = (self.safe_lane_x, self.column_spacing)
        self.apple_edge = (self.safe_lane_x, -self.column_spacing)
        self.bottle_bin = (
            self.safe_lane_x,
            self.column_spacing + self.bin_outside_margin,
        )
        self.apple_bin = (
            self.safe_lane_x,
            -self.column_spacing - self.bin_outside_margin,
        )
        self.bin_slot_count = self._int("bin_slot_count")
        self.bin_first_offset = self._float("bin_first_offset")
        self.bin_slot_spacing = self._float("bin_slot_spacing")
        self.bottle_slot_y_direction = self._int("bottle_slot_y_direction")
        self.apple_slot_y_direction = self._int("apple_slot_y_direction")
        self.allowed_classes = {
            str(value).strip().lower()
            for value in self.get_parameter("allowed_classes").value
        }
        self.force_g5_bottle = self._bool("force_g5_bottle")
        self.snapshot_confirmation_frames = self._int("snapshot_confirmation_frames")
        self.snapshot_timeout = self._float("snapshot_timeout")
        self.snapshot_freshness_timeout = self._float(
            "snapshot_freshness_timeout"
        )
        self.chassis_timeout = self._float("chassis_timeout")
        self.arm_timeout = self._float("arm_timeout")
        self.settle_time = self._float("settle_time")
        self.control_rate = self._float("control_rate")
        self.position_epsilon = self._float("position_epsilon")
        self.max_chassis_segment = self._float("max_chassis_segment")
        self.return_to_initial_grid = self._bool("return_to_initial_grid")
        self.object_limit = self._int("object_limit")

        self._validate()
        self.grid_positions = self._make_grid_positions()

        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.chassis_command_pub = self.create_publisher(
            String, self.chassis_command_topic, 10
        )
        self.arm_command_pub = self.create_publisher(String, self.arm_command_topic, 10)
        self.create_subscription(String, self.grid_state_topic, self._grid_callback, 10)
        self.create_subscription(
            String, self.chassis_status_topic, self._chassis_status_callback, 10
        )
        self.create_subscription(String, self.arm_status_topic, self._arm_status_callback, 10)
        self.create_subscription(Bool, self.start_topic, self._start_callback, 10)
        self.create_subscription(
            Bool, self.execute_topic, self._execute_callback, 10
        )
        self.create_subscription(Bool, self.stop_topic, self._stop_callback, 10)

        self.state = self.IDLE
        self.run_number = 0
        self.snapshot_deadline = 0.0
        self.frozen_snapshot: Dict[str, str] = {}
        self.snapshot_candidate: Optional[Dict[str, str]] = None
        self.snapshot_candidate_count = 0
        self.snapshot_candidate_time = 0.0
        self.snapshot_ready_announced = False
        self.snapshot_not_before_ros_ns = 0
        self.plan: List[Dict[str, Any]] = []
        self.plan_index = -1
        self.active_primitive: Optional[Dict[str, Any]] = None
        self.active_child = ""
        self.active_command_id = ""
        self.child_deadline = 0.0
        self.settle_deadline = 0.0
        self.current_position = self.grid_positions[self.initial_grid]
        self.current_position_source = "assumed_before_origin_lock"
        self.completed_grids: List[str] = []
        self.skipped_grids: List[str] = []
        self.current_grid = self.initial_grid
        self.current_class = ""

        self.create_timer(1.0 / self.control_rate, self._timer_callback)
        self.get_logger().info(
            f"Sorting coordinator {self.SOFTWARE_VERSION} ready; "
            "no automatic motion will occur"
        )
        self.get_logger().info("Task order: " + " -> ".join(self.task_order))
        self.get_logger().info(
            f"Initial grid={self.initial_grid}; object limit={self.object_limit}; "
            f"bottle bin={self.bottle_bin}; apple bin={self.apple_bin}"
        )
        self.get_logger().info(
            f"Mandatory rear safety lane: x={self.safe_lane_x:.3f} m"
        )
        self.get_logger().info(
            f"Chassis speeds: x={self.chassis_x_speed:.3f} m/s, "
            f"y={self.chassis_y_speed:.3f} m/s"
        )
        self.get_logger().warning(
            "Post-cycle clockwise compensation: "
            f"{self.cycle_heading_compensation_deg:.2f} deg"
        )
        self.get_logger().info(
            f"Bin slots: count={self.bin_slot_count}, first offset="
            f"{self.bin_first_offset:.3f} m, spacing="
            f"{self.bin_slot_spacing:.3f} m"
        )
        self._status("ready", True, "Waiting for operator start command")

    def _str(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _float(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _int(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _bool(self, name: str) -> bool:
        return bool(self.get_parameter(name).value)

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    def _validate(self) -> None:
        for grid in self.grid_reset_angles:
            angle = self.grid_reset_angles[grid]
            distance = self.grid_reset_distances[grid]
            if not math.isfinite(angle) or not 0.0 <= angle <= 15.0:
                raise ValueError(f"{grid} reset angle must be finite and in [0, 15] deg")
            if not math.isfinite(distance) or not (
                distance == 0.0 or 0.005 <= distance <= 0.05
            ):
                raise ValueError(f"{grid} reset forward must be 0 or in [0.005, 0.05] m")
        if self.rows != 2 or self.columns != 3:
            raise ValueError("The mission requires a 2-row x 3-column grid")
        if len(self.grid_ids) != 6 or len(set(self.grid_ids)) != 6:
            raise ValueError("grid_ids must contain six unique IDs")
        if len(self.task_order) != 6 or set(self.task_order) != set(self.grid_ids):
            raise ValueError("task_order must contain each grid ID exactly once")
        if self.initial_grid not in self.grid_ids:
            raise ValueError("initial_grid is not in grid_ids")
        if self.row_spacing <= 0.0 or self.column_spacing <= 0.0:
            raise ValueError("Grid spacing must be positive")
        for name, speed in (
            ("chassis_speed", self.chassis_speed),
            ("chassis_x_speed", self.chassis_x_speed),
            ("chassis_y_speed", self.chassis_y_speed),
        ):
            if not 0.0 < speed <= 1.0:
                raise ValueError(f"{name} must be in (0, 1.0] m/s")
        if not math.isfinite(self.cycle_heading_compensation_deg):
            raise ValueError("cycle_heading_compensation_deg must be finite")
        if not 0.0 <= self.cycle_heading_compensation_deg <= 15.0:
            raise ValueError(
                "cycle_heading_compensation_deg must be in [0, 15]"
            )
        if self.allowed_classes != {"bottle", "apple"}:
            raise ValueError("allowed_classes must contain bottle and apple")
        if self.force_g5_bottle and "G5" not in self.grid_ids:
            raise ValueError("force_g5_bottle requires G5 in grid_ids")
        if self.snapshot_confirmation_frames < 1:
            raise ValueError("snapshot_confirmation_frames must be positive")
        if self.snapshot_freshness_timeout <= 0.0:
            raise ValueError("snapshot_freshness_timeout must be positive")
        if min(self.snapshot_timeout, self.chassis_timeout, self.arm_timeout) <= 0.0:
            raise ValueError("Timeout values must be positive")
        if self.settle_time < 0.0:
            raise ValueError("settle_time cannot be negative")
        if not 5.0 <= self.control_rate <= 100.0:
            raise ValueError("control_rate must be between 5 and 100 Hz")
        if self.position_epsilon <= 0.0:
            raise ValueError("position_epsilon must be positive")
        if not 0.01 <= self.max_chassis_segment <= 0.50:
            raise ValueError("max_chassis_segment must be in [0.01, 0.50] m")
        if not 1 <= self.object_limit <= 6:
            raise ValueError("object_limit must be between 1 and 6")
        for name, point in (("bottle_bin", self.bottle_bin), ("apple_bin", self.apple_bin)):
            if not all(math.isfinite(value) for value in point):
                raise ValueError(f"{name} contains an invalid coordinate")
        if not math.isfinite(self.safe_lane_x):
            raise ValueError("safe_lane_x must be finite")
        if self.safe_lane_x >= min(x for x, _ in self.grid_positions_preview()):
            raise ValueError("safe_lane_x must be behind both grid rows")
        for name, point in (("bottle_bin", self.bottle_bin), ("apple_bin", self.apple_bin)):
            if abs(point[0] - self.safe_lane_x) > self.position_epsilon:
                raise ValueError(
                    f"{name}_x must equal safe_lane_x so placement remains "
                    "on the rear safety lane"
                )
        if self.bin_slot_count < 1:
            raise ValueError("bin_slot_count must be positive")
        if self.bin_outside_margin <= 0.0:
            raise ValueError("bin_outside_margin must be positive")
        if self.bin_first_offset <= 0.0 or self.bin_slot_spacing <= 0.0:
            raise ValueError("Bin slot offset and spacing must be positive")
        last_slot_offset = (
            self.bin_first_offset
            - (self.bin_slot_count - 1) * self.bin_slot_spacing
        )
        if last_slot_offset < -self.position_epsilon:
            raise ValueError("The last bin slot offset cannot be negative")
        if self.bottle_slot_y_direction not in (-1, 1):
            raise ValueError("bottle_slot_y_direction must be -1 or 1")
        if self.apple_slot_y_direction not in (-1, 1):
            raise ValueError("apple_slot_y_direction must be -1 or 1")

    def grid_positions_preview(self) -> List[Tuple[float, float]]:
        """Return coordinates without depending on self.grid_positions."""
        raw = [
            (
                (self.rows - 1 - index // self.columns) * self.row_spacing,
                (index % self.columns) * self.column_spacing,
            )
            for index in range(len(self.grid_ids))
        ]
        initial_index = self.grid_ids.index(self.initial_grid)
        origin_x, origin_y = raw[initial_index]
        return [(x - origin_x, y - origin_y) for x, y in raw]

    def _make_grid_positions(self) -> Dict[str, Tuple[float, float]]:
        # Image labels are G1-G3 on the front row and G4-G6 on the rear row.
        # Positive logical X points from the rear row toward the front row;
        # positive logical Y points from column 1 toward column 3.
        raw_positions = {
            grid_id: (
                (self.rows - 1 - index // self.columns) * self.row_spacing,
                (index % self.columns) * self.column_spacing,
            )
            for index, grid_id in enumerate(self.grid_ids)
        }
        origin_x, origin_y = raw_positions[self.initial_grid]
        return {
            grid_id: (x - origin_x, y - origin_y)
            for grid_id, (x, y) in raw_positions.items()
        }

    def _required_grids(self) -> List[str]:
        return self.task_order[: self.object_limit]

    def _bin_slot_position(
        self, object_class: str, zero_based_index: int
    ) -> Tuple[
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
        str,
        int,
        float,
    ]:
        """Return slot, entry, outer-grid edge, bin name, number and offset."""
        if not 0 <= zero_based_index < self.bin_slot_count:
            raise ValueError(
                f"No free {object_class} slot: requested index "
                f"{zero_based_index}, capacity={self.bin_slot_count}"
            )

        if object_class == "bottle":
            base_x, base_y = self.bottle_bin
            edge_position = self.bottle_edge
            bin_name = "bottle_bin"
            y_direction = self.bottle_slot_y_direction
        elif object_class == "apple":
            base_x, base_y = self.apple_bin
            edge_position = self.apple_edge
            bin_name = "apple_bin"
            y_direction = self.apple_slot_y_direction
        else:
            raise ValueError(f"Unsupported class for bin slot: {object_class}")

        # Fill from the far outside edge back toward the grid: 0.30, 0.20,
        # 0.10 and 0.00 m beyond the entry. Clamp the fourth floating-point
        # result to exact zero instead of leaving a tiny negative residue.
        slot_offset = max(
            0.0,
            self.bin_first_offset
            - zero_based_index * self.bin_slot_spacing,
        )
        entry_position = (base_x, base_y)
        slot_position = (
            base_x,
            base_y + y_direction * slot_offset,
        )
        return (
            slot_position,
            entry_position,
            edge_position,
            bin_name,
            zero_based_index + 1,
            slot_offset,
        )

    def _grid_callback(self, msg: String) -> None:
        try:
            snapshot = self._extract_snapshot(json.loads(msg.data))
        except (json.JSONDecodeError, TypeError, ValueError, KeyError):
            snapshot = None

        if snapshot is None:
            self.snapshot_candidate = None
            self.snapshot_candidate_count = 0
            self.snapshot_candidate_time = 0.0
            self.snapshot_ready_announced = False
            return

        if snapshot == self.snapshot_candidate:
            self.snapshot_candidate_count += 1
        else:
            self.snapshot_candidate = snapshot
            self.snapshot_candidate_count = 1
            self.snapshot_ready_announced = False
        self.snapshot_candidate_time = self._now()

        if (
            self.state == self.WAITING_SNAPSHOT
            and self.snapshot_candidate_count >= self.snapshot_confirmation_frames
            and not self.snapshot_ready_announced
        ):
            self.snapshot_ready_announced = True
            snapshot_text = ", ".join(
                f"{key}={value}" for key, value in snapshot.items()
            )
            self.get_logger().info(
                f"Snapshot ready for phase 2: {snapshot_text}"
            )
            self._status(
                "snapshot_ready",
                True,
                f"Stable recognition ready; send phase 2: {snapshot_text}",
            )

    def _extract_snapshot(
        self, payload: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        grids = payload.get("grids")
        if not isinstance(grids, dict):
            return None

        stamp = payload.get("stamp")
        if self.snapshot_not_before_ros_ns > 0:
            if not isinstance(stamp, dict):
                return None
            try:
                stamp_ns = int(stamp.get("sec", 0)) * 1_000_000_000 + int(
                    stamp.get("nanosec", 0)
                )
            except (TypeError, ValueError):
                return None
            if stamp_ns < self.snapshot_not_before_ros_ns:
                return None

        snapshot: Dict[str, str] = {}
        # G5 is a known bottle in this experiment and deliberately bypasses
        # vision stability checks. Other cells still require stable labels.
        for grid_id in self._required_grids():
            if self.force_g5_bottle and grid_id == "G5":
                snapshot[grid_id] = "bottle"
                continue
            cell = grids.get(grid_id)
            if not isinstance(cell, dict) or cell.get("stable") is not True:
                return None
            label = str(cell.get("stable_label", "empty")).strip().lower()
            if not label or label == "pending":
                return None
            snapshot[grid_id] = label
        return snapshot

    def _start_callback(self, msg: Bool) -> None:
        if not msg.data:
            return
        if self.state not in (self.IDLE, self.COMPLETED, self.STOPPED, self.ERROR):
            self._status("rejected", False, f"Mission is busy: {self.state}")
            return
        if self.chassis_command_pub.get_subscription_count() < 1:
            self._status("rejected", False, "chassis_motion_node is not connected")
            return
        if self.arm_command_pub.get_subscription_count() < 1:
            self._status("rejected", False, "arm_task_node is not connected")
            return

        self.run_number += 1
        self.frozen_snapshot = {}
        self.plan = []
        self.plan_index = -1
        self.active_primitive = None
        self.active_child = ""
        self.active_command_id = ""
        self.completed_grids = []
        self.skipped_grids = []
        self.current_grid = self.initial_grid
        self.current_class = ""
        self.current_position = self.grid_positions[self.initial_grid]
        self.current_position_source = "pending_origin_lock"
        # Discard detections collected before the arm reaches its observation
        # pose. Only frames captured after observe succeeds may be frozen.
        self.snapshot_candidate = None
        self.snapshot_candidate_count = 0
        self.snapshot_candidate_time = 0.0
        self.snapshot_ready_announced = False
        # 在任何机械臂动作之前锁定 G5 场地原点。重复运行只允许验证
        # 已有原点，不允许把失败后的当前位置悄悄当成新的 G5。
        self.active_child = "chassis"
        self.active_command_id = f"sort_r{self.run_number}_set_origin"
        self.child_deadline = self._now() + self.chassis_timeout
        self.state = self.CAPTURING_ORIGIN

        self._publish_json(
            self.chassis_command_pub,
            {"command": "set_origin", "id": self.active_command_id},
        )
        self.get_logger().warning(
            f"Run {self.run_number} phase 1: locking/verifying the G5 field origin"
        )
        self._status(
            "phase1_started",
            True,
            "Phase 1 started: locking/verifying G5 origin before arm motion",
        )

    def _execute_callback(self, msg: Bool) -> None:
        """Freeze the current stable labels only on an explicit phase-2 command."""
        if not msg.data:
            return
        if self.state != self.WAITING_SNAPSHOT:
            self._status(
                "phase2_rejected",
                False,
                f"Phase 2 requires WAITING_SNAPSHOT; current state={self.state}",
            )
            return
        if self.snapshot_candidate is None:
            self._status(
                "phase2_rejected",
                False,
                "No complete stable grid snapshot is currently available",
            )
            return
        if self.snapshot_candidate_count < self.snapshot_confirmation_frames:
            self._status(
                "phase2_rejected",
                False,
                "Recognition is not stable yet: "
                f"{self.snapshot_candidate_count}/"
                f"{self.snapshot_confirmation_frames} matching frames",
            )
            return
        snapshot_age = self._now() - self.snapshot_candidate_time
        if snapshot_age > self.snapshot_freshness_timeout:
            self._status(
                "phase2_rejected",
                False,
                f"Latest complete snapshot is stale ({snapshot_age:.2f} s old)",
            )
            return

        snapshot = dict(self.snapshot_candidate)
        self.get_logger().warning(
            f"Run {self.run_number} phase 2: freezing current recognition result"
        )
        self._status(
            "phase2_accepted",
            True,
            "Phase 2 accepted: freezing current stable snapshot",
        )
        self._freeze_snapshot(snapshot)

    def _begin_snapshot_wait(self) -> None:
        """Collect labels after the camera pose and wait indefinitely for phase 2."""
        self.snapshot_candidate = None
        self.snapshot_candidate_count = 0
        self.snapshot_candidate_time = 0.0
        self.snapshot_ready_announced = False
        self.snapshot_deadline = self._now() + self.snapshot_timeout
        # 只接受观察位动作成功之后采集的相机帧。
        self.snapshot_not_before_ros_ns = self.get_clock().now().nanoseconds
        self.active_child = ""
        self.active_command_id = ""
        self.state = self.WAITING_SNAPSHOT

        required = self._required_grids()
        self.get_logger().warning(
            f"Run {self.run_number}: camera pose ready; "
            f"waiting for stable labels and phase-2 command in {required}"
        )
        self._status(
            "waiting_snapshot",
            True,
            "Phase 1 complete; adjust/observe the image, then publish phase 2. "
            f"Required cells: {required}",
        )

    def _stop_callback(self, msg: Bool) -> None:
        if msg.data:
            self._stop_mission("Operator stop command received")

    def _freeze_snapshot(self, snapshot: Dict[str, str]) -> None:
        if self.state != self.WAITING_SNAPSHOT:
            return
        self.frozen_snapshot = dict(snapshot)
        try:
            self.plan = self._build_plan()
        except (KeyError, TypeError, ValueError) as exc:
            # A bad snapshot or invalid placement configuration must stop the
            # mission cleanly rather than terminate the ROS 2 process.
            self._fail_mission(f"Mission planning failed: {exc}")
            return
        self.plan_index = -1
        self.state = self.EXECUTING
        snapshot_text = ", ".join(f"{key}={value}" for key, value in snapshot.items())
        self.get_logger().info(f"Frozen snapshot: {snapshot_text}")
        self.get_logger().info(f"Mission plan has {len(self.plan)} primitives")
        self._status("snapshot_frozen", True, f"Frozen snapshot: {snapshot_text}")
        self._advance_plan()

    def _build_plan(self) -> List[Dict[str, Any]]:
        plan: List[Dict[str, Any]] = []
        position = self.grid_positions[self.initial_grid]
        slot_counts = {"bottle": 0, "apple": 0}
        actionable = any(
            self.frozen_snapshot[grid_id] in self.allowed_classes
            for grid_id in self._required_grids()
        )
        if actionable:
            self._append_chassis_align(
                plan,
                f"Initial chassis alignment at {self.initial_grid}",
                self.initial_grid,
                "",
            )
            self._append_arm(plan, "prepare", "Prepare arm and open gripper")

        for grid_id in self._required_grids():
            object_class = self.frozen_snapshot[grid_id]
            if object_class not in self.allowed_classes:
                plan.append(
                    {
                        "kind": "mark_skipped",
                        "grid_id": grid_id,
                        "object_class": object_class,
                        "skip_reason": "empty_or_unrecognized",
                        "description": f"Skip {grid_id}: {object_class}",
                    }
                )
                continue

            # The compact physical bin provides four collision-free positions
            # per class. If vision reports more than four objects
            # of one class, keep the node alive and record the excess cells as
            # bin_full instead of raising from _bin_slot_position().
            if slot_counts[object_class] >= self.bin_slot_count:
                plan.append(
                    {
                        "kind": "mark_skipped",
                        "grid_id": grid_id,
                        "object_class": object_class,
                        "skip_reason": "bin_full",
                        "target_bin": f"{object_class}_bin",
                        "description": (
                            f"Skip {grid_id}: {object_class} bin is full "
                            f"({self.bin_slot_count} slots)"
                        ),
                    }
                )
                continue
            grid_position = self.grid_positions[grid_id]
            # Change columns while still on the rear row, then approach the
            # selected cell. This avoids sweeping across unprocessed objects
            # in the front row.
            position = self._append_move_to(
                plan, position, grid_position, ("y", "x"),
                f"Move to {grid_id}", grid_id, object_class
            )
            self._append_arm(
                plan, "pick", f"Pick {object_class} from {grid_id}",
                grid_id, object_class
            )

            # SAFETY-CRITICAL: retreat in X as a dedicated primitive.  Do not
            # combine this with the trip to a classification bin.  For the
            # With the current 0.20 m row spacing, the front row retreats
            # 0.40 m and the rear row 0.20 m to reach the safety lane.
            safety_position = (self.safe_lane_x, grid_position[1])
            position = self._append_move_to(
                plan, position, safety_position, ("x", "y"),
                f"MANDATORY RETREAT after picking {grid_id}",
                grid_id, object_class,
                segment_limit=self.row_spacing,
            )

            (
                bin_position,
                bin_entry_position,
                bin_edge_position,
                bin_name,
                slot_number,
                slot_offset,
            ) = self._bin_slot_position(
                object_class, slot_counts[object_class]
            )
            slot_counts[object_class] += 1
            slot_metadata = {
                "target_bin": bin_name,
                "target_slot": slot_number,
                "target_slot_position": bin_position,
                "target_bin_entry": bin_entry_position,
                "target_bin_edge": bin_edge_position,
                "target_slot_offset": slot_offset,
            }

            # Phase 1: stay on the rear safety lane and reach the corresponding
            # outer grid column. For a bottle picked from G4, this is exactly
            # two column spacings from G4 to the G6 lateral position.
            position = self._append_move_to(
                plan, position, bin_edge_position, ("y", "x"),
                f"Carry {grid_id} along safety lane to outer grid edge",
                grid_id, object_class, slot_metadata,
                segment_limit=self.column_spacing,
            )

            # Phase 2: move exactly 0.15 m outside the outer grid column.
            position = self._append_move_to(
                plan, position, bin_entry_position, ("y", "x"),
                f"Move {self.bin_outside_margin:.2f} m outside to {bin_name} entry",
                grid_id, object_class, slot_metadata
            )

            # Phase 3: use collision-free class slots at
            # 0.30/0.20/0.10/0.00 m from the bin entry.
            position = self._append_move_to(
                plan, position, bin_position, ("y", "x"),
                f"Move outward {slot_offset:.2f} m to {bin_name} slot {slot_number}",
                grid_id, object_class, slot_metadata
            )
            self._append_arm(
                plan, "place",
                f"Place {object_class} in {bin_name} slot {slot_number}",
                grid_id, object_class, slot_metadata
            )
            # Reverse the slot offset first. Keeping this as its own movement
            # makes the 0.60/0.50/... m return visible in the result log.
            position = self._append_move_to(
                plan, position, bin_entry_position, ("y", "x"),
                f"Return inward {slot_offset:.2f} m from {bin_name} slot {slot_number}",
                grid_id, object_class, slot_metadata
            )

            # Reverse the fixed 0.15 m outside margin as a separate movement.
            position = self._append_move_to(
                plan, position, bin_edge_position, ("y", "x"),
                f"Return {self.bin_outside_margin:.2f} m to outer grid edge",
                grid_id, object_class, slot_metadata
            )

            # Move one column spacing from G4/G6 lateral position to the point
            # directly behind G5, while remaining on the safety lane.
            safe_zero_position = (self.safe_lane_x, 0.0)
            position = self._append_move_to(
                plan, position, safe_zero_position, ("y", "x"),
                f"Return on safety lane to the point behind {self.initial_grid}",
                grid_id, object_class, slot_metadata,
                segment_limit=self.column_spacing,
            )

            # Finally move forward one configured row spacing. Only after this
            # primitive is the chassis truly back at the G5 grasping pose.
            zero_position = self.grid_positions[self.initial_grid]
            position = self._append_move_to(
                plan, position, zero_position, ("x", "y"),
                f"Move forward from safety lane to zero point {self.initial_grid}",
                grid_id, object_class, slot_metadata,
                segment_limit=self.row_spacing,
            )
            self._append_chassis_align(
                plan,
                f"Align chassis at zero point after {grid_id}",
                grid_id,
                object_class,
                slot_metadata,
            )
            reset_angle = self.grid_reset_angles[grid_id]
            if reset_angle > 0.0:
                self._append_chassis_compensation(
                    plan,
                    reset_angle,
                    (
                        f"After {grid_id}, correct clockwise at "
                        f"{self.initial_grid} by "
                        f"{reset_angle:.2f} deg"
                    ),
                    grid_id,
                    object_class,
                    slot_metadata,
                )
            # 仅放置、回到G5、对齐、右转均成功之后执行；最后一格也执行。
            distance = self.grid_reset_distances[grid_id]
            if distance > 0.0:
                plan.append({
                    "kind": "chassis_reset_forward",
                    "grid_id": grid_id,
                    "object_class": object_class,
                    "distance": distance,
                    **slot_metadata,
                    "description": (
                        f"After {grid_id}, advance {distance:.3f} m at G5; "
                        "retain measured manual translation correction"
                    ),
                })
            plan.append(
                {
                    "kind": "mark_complete",
                    "grid_id": grid_id,
                    "object_class": object_class,
                    **slot_metadata,
                    "description": f"Mark {grid_id} completed",
                }
            )

        if self.return_to_initial_grid:
            self._append_move_to(
                plan, position, self.grid_positions[self.initial_grid],
                ("x", "y"), "Return to initial grid", self.initial_grid, ""
            )
        return plan

    def _append_move_to(
        self,
        plan: List[Dict[str, Any]],
        current: Tuple[float, float],
        target: Tuple[float, float],
        axis_order: Tuple[str, str],
        description: str,
        grid_id: str,
        object_class: str,
        metadata: Optional[Dict[str, Any]] = None,
        segment_limit: Optional[float] = None,
    ) -> Tuple[float, float]:
        x, y = current
        target_x, target_y = target
        for axis in axis_order:
            if axis not in ("x", "y"):
                raise ValueError(f"Unsupported axis: {axis}")
            remaining = (target_x - x) if axis == "x" else (target_y - y)
            part = 0
            while abs(remaining) > self.position_epsilon:
                part += 1
                effective_limit = self.max_chassis_segment
                if segment_limit is not None:
                    effective_limit = min(effective_limit, segment_limit)
                step = math.copysign(
                    min(abs(remaining), effective_limit),
                    remaining,
                )
                if axis == "x":
                    x += step
                    remaining = target_x - x
                else:
                    y += step
                    remaining = target_y - y
                primitive = {
                    "kind": "chassis",
                    "axis": axis,
                    "distance": step,
                    "target_position": (x, y),
                    "grid_id": grid_id,
                    "object_class": object_class,
                    "description": (
                        f"{description}: {axis} {step:+.3f} m "
                        f"(segment {part})"
                    ),
                }
                if metadata:
                    primitive.update(metadata)
                plan.append(primitive)
        return x, y

    @staticmethod
    def _append_arm(
        plan: List[Dict[str, Any]],
        command: str,
        description: str,
        grid_id: str = "",
        object_class: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        primitive = {
            "kind": "arm",
            "command": command,
            "grid_id": grid_id,
            "object_class": object_class,
            "description": description,
        }
        if metadata:
            primitive.update(metadata)
        plan.append(primitive)

    @staticmethod
    def _append_chassis_align(
        plan: List[Dict[str, Any]],
        description: str,
        grid_id: str = "",
        object_class: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        primitive = {
            "kind": "chassis_align",
            "grid_id": grid_id,
            "object_class": object_class,
            "description": description,
        }
        if metadata:
            primitive.update(metadata)
        plan.append(primitive)

    @staticmethod
    def _append_chassis_compensation(
        plan: List[Dict[str, Any]],
        right_degrees: float,
        description: str,
        grid_id: str = "",
        object_class: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        primitive = {
            "kind": "chassis_compensation",
            "right_degrees": right_degrees,
            "grid_id": grid_id,
            "object_class": object_class,
            "description": description,
        }
        if metadata:
            primitive.update(metadata)
        plan.append(primitive)

    def _advance_plan(self) -> None:
        if self.state not in (self.EXECUTING, self.SETTLING):
            return
        self.plan_index += 1

        if self.plan_index >= len(self.plan):
            self.state = self.COMPLETED
            self.active_primitive = None
            self.active_child = ""
            self.active_command_id = ""
            message = (
                f"Mission finished: {len(self.completed_grids)} sorted, "
                f"{len(self.skipped_grids)} skipped"
            )
            self.get_logger().info(message)
            self._status("completed", True, message)
            return

        self.active_primitive = self.plan[self.plan_index]
        if self.active_primitive["kind"] in ("mark_complete", "mark_skipped"):
            grid_id = str(self.active_primitive["grid_id"])
            object_class = str(self.active_primitive["object_class"])
            if self.active_primitive["kind"] == "mark_skipped":
                self.skipped_grids.append(grid_id)
                skip_reason = str(
                    self.active_primitive.get("skip_reason", "unspecified")
                )
                message = (
                    f"Skipped {grid_id}: {object_class}; "
                    f"reason={skip_reason}"
                )
                event = "grid_skipped"
            else:
                self.completed_grids.append(grid_id)
                message = (
                    f"Completed {grid_id}: {object_class} "
                    f"({len(self.completed_grids)} completed, "
                    f"{len(self.skipped_grids)} skipped)"
                )
                event = "grid_completed"
            self.current_grid = grid_id
            self.current_class = object_class
            self.get_logger().info(message)
            self._status(event, True, message)
            self.state = self.EXECUTING
            self._advance_plan()
            return

        self._dispatch(self.active_primitive)

    def _dispatch(self, primitive: Dict[str, Any]) -> None:
        number = self.plan_index + 1
        self.active_command_id = f"sort_r{self.run_number}_p{number:03d}"
        self.current_grid = str(primitive.get("grid_id", ""))
        self.current_class = str(primitive.get("object_class", ""))

        if primitive["kind"] == "chassis":
            axis = str(primitive["axis"])
            speed = (
                self.chassis_x_speed
                if axis == "x"
                else self.chassis_y_speed
            )
            payload = {
                "command": "move_to",
                "axis": axis,
                "target_x": round(float(primitive["target_position"][0]), 6),
                "target_y": round(float(primitive["target_position"][1]), 6),
                "speed": speed,
                "id": self.active_command_id,
            }
            self.active_child = "chassis"
            self.child_deadline = self._now() + self.chassis_timeout
            self._publish_json(self.chassis_command_pub, payload)
        elif primitive["kind"] == "chassis_align":
            payload = {
                "command": "align",
                "id": self.active_command_id,
            }
            self.active_child = "chassis"
            self.child_deadline = self._now() + self.chassis_timeout
            self._publish_json(self.chassis_command_pub, payload)
        elif primitive["kind"] == "chassis_compensation":
            payload = {
                "command": "compensate_heading",
                "right_degrees": round(
                    float(primitive["right_degrees"]), 6
                ),
                "id": self.active_command_id,
            }
            self.active_child = "chassis"
            self.child_deadline = self._now() + self.chassis_timeout
            self._publish_json(self.chassis_command_pub, payload)
        elif primitive["kind"] == "chassis_reset_forward":
            payload = {
                "command": "reset_forward",
                "distance": float(primitive["distance"]),
                "id": self.active_command_id,
            }
            self.active_child = "chassis"
            self.child_deadline = self._now() + self.chassis_timeout
            self._publish_json(self.chassis_command_pub, payload)
        elif primitive["kind"] == "arm":
            payload = {
                "command": str(primitive["command"]),
                "id": self.active_command_id,
            }
            self.active_child = "arm"
            self.child_deadline = self._now() + self.arm_timeout
            self._publish_json(self.arm_command_pub, payload)
        else:
            self._fail_mission(f"Unsupported primitive: {primitive['kind']}")
            return

        self.state = self.WAITING_CHILD
        description = str(primitive.get("description", ""))
        self.get_logger().info(f"Plan {number}/{len(self.plan)}: {description}")
        self._status("primitive_started", True, description)

    def _chassis_status_callback(self, msg: String) -> None:
        self._handle_child_status("chassis", msg)

    def _arm_status_callback(self, msg: String) -> None:
        self._handle_child_status("arm", msg)

    def _handle_child_status(self, child: str, msg: String) -> None:
        if (
            self.state not in (
                self.CAPTURING_ORIGIN,
                self.HOMING_FOR_CAMERA,
                self.POSITIONING_CAMERA,
                self.WAITING_CHILD,
            )
            or child != self.active_child
        ):
            return
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if str(payload.get("id", "")) != self.active_command_id:
            return

        event = str(payload.get("event", "")).lower()
        success = bool(payload.get("success", False))
        if child == "chassis" and event in (
            "started",
            "succeeded",
            "failed",
            "rejected",
            "stopped",
        ):
            protocol = str(payload.get("motion_protocol", ""))
            if protocol != "fixed_g5_reset_v4":
                self._fail_mission(
                    "chassis_motion version/protocol mismatch: expected "
                    "fixed_g5_reset_v4. Rebuild and source the current "
                    "experiment3_ws before running."
                )
                return
        if event == "succeeded" and success:
            if self.state == self.CAPTURING_ORIGIN:
                actual = payload.get("logical_position")
                if not isinstance(actual, dict):
                    self._fail_mission(
                        "chassis origin response did not contain measured logical_position"
                    )
                    return
                self.current_position = (
                    float(actual.get("x", 0.0)),
                    float(actual.get("y", 0.0)),
                )
                self.current_position_source = "odometry"
                self.active_child = "arm"
                self.active_command_id = f"sort_r{self.run_number}_camera_home"
                self.child_deadline = self._now() + self.arm_timeout
                self.state = self.HOMING_FOR_CAMERA
                self._publish_json(
                    self.arm_command_pub,
                    {"command": "home", "id": self.active_command_id},
                )
                self._status(
                    "origin_verified",
                    True,
                    "G5 field origin verified; homing arm before observation",
                )
                return
            if self.state == self.HOMING_FOR_CAMERA:
                self.active_child = "arm"
                self.active_command_id = f"sort_r{self.run_number}_observe"
                self.child_deadline = self._now() + self.arm_timeout
                self.state = self.POSITIONING_CAMERA
                self._publish_json(
                    self.arm_command_pub,
                    {"command": "observe", "id": self.active_command_id},
                )
                self.get_logger().info(
                    "Arm home reached; moving to fixed camera observation pose"
                )
                self._status(
                    "positioning_camera",
                    True,
                    "Arm home reached; moving to camera observation pose",
                )
                return
            if self.state == self.POSITIONING_CAMERA:
                self.get_logger().info(
                    "Camera observation pose reached; starting snapshot window"
                )
                self._status(
                    "camera_positioned",
                    True,
                    "Arm reached camera observation pose",
                )
                self._begin_snapshot_wait()
                return
            if (
                child == "chassis"
                and self.active_primitive is not None
                and (
                    "target_position" in self.active_primitive
                    or self.active_primitive["kind"] == "chassis_reset_forward"
                )
            ):
                actual = payload.get("logical_position")
                if not isinstance(actual, dict):
                    self._fail_mission(
                        "chassis success lacked measured logical_position; "
                        "planned coordinates were not recorded as actual pose"
                    )
                    return
                self.current_position = (
                    float(actual["x"]), float(actual["y"])
                )
                self.current_position_source = "odometry"
            self.state = self.SETTLING
            self.settle_deadline = self._now() + self.settle_time
            self._status(
                "primitive_succeeded",
                True,
                str(payload.get("message", f"{child} command succeeded")),
            )
        elif event in ("failed", "rejected", "stopped"):
            reason = str(payload.get("message", f"{child} command failed"))
            self._fail_mission(f"{child} reported {event}: {reason}")

    def _timer_callback(self) -> None:
        now = self._now()
        if self.state == self.CAPTURING_ORIGIN and now >= self.child_deadline:
            self._fail_mission(
                f"Chassis origin command timeout: {self.active_command_id}"
            )
        elif self.state == self.HOMING_FOR_CAMERA and now >= self.child_deadline:
            self._fail_mission(
                f"Arm home command timeout: {self.active_command_id}"
            )
        elif self.state == self.POSITIONING_CAMERA and now >= self.child_deadline:
            self._fail_mission(
                f"Arm observation command timeout: {self.active_command_id}"
            )
        elif self.state == self.WAITING_CHILD and now >= self.child_deadline:
            self._fail_mission(
                f"{self.active_child} command timeout: {self.active_command_id}"
            )
        elif self.state == self.SETTLING and now >= self.settle_deadline:
            self.state = self.EXECUTING
            self._advance_plan()
        elif (
            self.state == self.WAITING_SNAPSHOT
            and self.snapshot_deadline > 0.0
            and now >= self.snapshot_deadline
            and (
                self.snapshot_candidate is None
                or self.snapshot_candidate_count < self.snapshot_confirmation_frames
            )
        ):
            self._fail_mission(
                f"Stable recognition snapshot timeout after {self.snapshot_timeout:.1f} s"
            )

    def _stop_mission(self, reason: str) -> None:
        self._send_child_stops()
        self.plan = []
        self.active_primitive = None
        self.active_child = ""
        self.active_command_id = ""
        self.state = self.STOPPED
        self.get_logger().warning(reason)
        self._status("stopped", True, reason)

    def _fail_mission(self, reason: str) -> None:
        if self.state == self.ERROR:
            return
        self._send_child_stops()
        self.state = self.ERROR
        self.active_child = ""
        self.get_logger().error(reason)
        self._status("failed", False, reason)

    def _send_child_stops(self) -> None:
        suffix = int(self._now() * 1000)
        self._publish_json(
            self.chassis_command_pub,
            {"command": "stop", "id": f"sorting_stop_chassis_{suffix}"},
        )
        self._publish_json(
            self.arm_command_pub,
            {"command": "stop", "id": f"sorting_stop_arm_{suffix}"},
        )

    @staticmethod
    def _publish_json(publisher, payload: Dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        publisher.publish(msg)

    def _status(self, event: str, success: bool, message: str) -> None:
        active = self.active_primitive or {}
        slot_position = active.get("target_slot_position")
        if isinstance(slot_position, (list, tuple)) and len(slot_position) == 2:
            slot_position_payload: Optional[Dict[str, float]] = {
                "x": round(float(slot_position[0]), 6),
                "y": round(float(slot_position[1]), 6),
            }
        else:
            slot_position_payload = None
        payload = {
            "software_version": self.SOFTWARE_VERSION,
            "grid_reset_angles_deg": self.grid_reset_angles,
            "grid_reset_forward_m": self.grid_reset_distances,
            "event": event,
            "success": success,
            "state": self.state,
            "run": self.run_number,
            "object_limit": self.object_limit,
            "cycle_heading_compensation_deg": round(
                self.cycle_heading_compensation_deg, 4
            ),
            "plan_index": self.plan_index,
            "plan_count": len(self.plan),
            "active_command_id": self.active_command_id,
            "active_child": self.active_child,
            "current_grid": self.current_grid,
            "current_class": self.current_class,
            "target_bin": str(active.get("target_bin", "")),
            "target_slot": active.get("target_slot"),
            "target_slot_position": slot_position_payload,
            "skip_reason": str(active.get("skip_reason", "")),
            "current_position": {
                "x": round(self.current_position[0], 6),
                "y": round(self.current_position[1], 6),
            },
            "current_position_source": self.current_position_source,
            "completed_grids": list(self.completed_grids),
            "skipped_grids": list(self.skipped_grids),
            "snapshot": {
                grid_id: self.frozen_snapshot.get(grid_id, "")
                for grid_id in self.grid_ids
            },
            "live_snapshot": {
                grid_id: (
                    self.snapshot_candidate.get(grid_id, "")
                    if self.snapshot_candidate is not None
                    else ""
                )
                for grid_id in self.grid_ids
            },
            "snapshot_matching_frames": self.snapshot_candidate_count,
            "snapshot_ready": (
                self.snapshot_candidate is not None
                and self.snapshot_candidate_count
                >= self.snapshot_confirmation_frames
                and self._now() - self.snapshot_candidate_time
                <= self.snapshot_freshness_timeout
            ),
            "message": message,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)

    def safe_shutdown(self) -> None:
        if self.state in (
            self.HOMING_FOR_CAMERA,
            self.CAPTURING_ORIGIN,
            self.POSITIONING_CAMERA,
            self.WAITING_SNAPSHOT,
            self.EXECUTING,
            self.WAITING_CHILD,
            self.SETTLING,
        ):
            self._send_child_stops()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[SortingCoordinatorNode] = None
    try:
        node = SortingCoordinatorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().warning(
                "Ctrl+C received; stopping the sorting mission"
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


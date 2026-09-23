#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task3 抓取节点 —— 网格驱动版.

和之前唯一的区别: 多了一层极简的控制逻辑

    for grid_id in 1..6:
        result = vision_state[grid_id]
        if result == 'empty':     continue
        if result == 'cube':      走 Grid N 已经调好的固定路线, 投到方块料盒
        if result == 'cylinder':  走 Grid N 已经调好的固定路线, 投到圆柱料盒

不重新规划路线, 不根据 bbox 反算机械臂坐标, 不做 Homography.
每个网格的 car_y / arm_x / forward_x 全部是配置文件里写死的固定值,
视觉只决定 "跳过 / 走方块支线 / 走圆柱支线" 这三选一.
"""

import json
import math
import os
import threading
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from std_srvs.srv import Trigger

from robomaster_msgs.action import GripperControl, Move, MoveArm

# 状态机 + 报告收集器 (纯 Python, 不依赖 ROS)
from ep_task3_pick.state_machine import (
    StateMachine, RunReporter, load_yaml, StateMachineError,
)


CUBE_ALIASES = ('CUBE', '方块', 'BOX', 'A', 'APPLE')
CYL_ALIASES = ('CYL', 'CYLINDER', '圆柱', 'B', 'BOTTLE')

EMPTY = 'empty'


def wait_future(future, timeout, poll=0.02):
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if future.done():
            return future.result()
        time.sleep(poll)
    return None


def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def norm_kind(text):
    t = str(text).strip().upper()
    if t in CUBE_ALIASES:
        return 'CUBE'
    if t in CYL_ALIASES:
        return 'CYL'
    return 'CUBE' if ('CUBE' in t or '方' in t) else 'CYL'


class PickNode(Node):
    def __init__(self):
        super().__init__('ep_task3_pick')

        # ---------- 运动参数 (和原来完全一致) ----------
        self.declare_parameter('chassis_forward_x', 0.08)
        self.declare_parameter('drop_forward_x', 0.0)
        self.declare_parameter('arm_base_offset', 0.15)
        self.declare_parameter('arm_x_retract', 0.07)
        self.declare_parameter('drop_arm_x', 0.12)
        self.declare_parameter('safe_z', 0.15)
        self.declare_parameter('pick_z', 0.08)
        self.declare_parameter('place_z', 0.08)
        self.declare_parameter('drop_y_cube', -0.35)
        self.declare_parameter('drop_y_cyl', 0.35)
        self.declare_parameter('drop_step_cube', 0.03)
        self.declare_parameter('drop_step_cyl', 0.03)
        self.declare_parameter('zero_dwell', 0.5)
        self.declare_parameter('odom_report', True)
        self.declare_parameter('odom_correct', False)
        self.declare_parameter('odom_correct_tol', 0.015)
        self.declare_parameter('odom_correct_max', 0.15)
        self.declare_parameter('odom_y_sign', 1.0)
        self.declare_parameter('yaw_warn_deg', 5.0)
        self.declare_parameter('y_sign', -1.0)
        self.declare_parameter('x_sign', 1.0)
        self.declare_parameter('move_linear_speed', 0.3)
        self.declare_parameter('action_timeout', 15.0)
        self.declare_parameter('action_cooldown', 0.4)
        self.declare_parameter('gripper_cooldown', 1.0)
        self.declare_parameter('gripper_power', 0.5)
        self.declare_parameter('server_wait', 20.0)
        self.declare_parameter('arm_x_min', 0.05)
        self.declare_parameter('arm_x_max', 0.22)

        # ---------- 物体表 (向后兼容: 没写 grid_* 时用它推导固定路线) ----------
        self.declare_parameter('obj_names', [
            'Cube_1', 'Cube_2', 'Cube_3',
            'Cylinder_1', 'Cylinder_2', 'Cylinder_3'
        ])
        self.declare_parameter('obj_x', [0.38, 0.28, 0.38, 0.28, 0.38, 0.28])
        self.declare_parameter('obj_y', [0.00, -0.12, -0.12, 0.12, 0.12, 0.00])
        self.declare_parameter('obj_kind', [
            'CUBE', 'CUBE', 'CUBE', 'CYL', 'CYL', 'CYL'
        ])
        self.declare_parameter('order_names', ['auto'])

        # ---------- 六个网格的固定路线 ----------
        self.declare_parameter('grid_ids', [1, 2, 3, 4, 5, 6])
        self.declare_parameter('grid_car_y', [0.0])
        self.declare_parameter('grid_arm_x', [0.0])
        self.declare_parameter('grid_forward_x', [0.0])

        # ---------- 视觉 ----------
        self.declare_parameter('use_vision', True)
        self.declare_parameter('vision_topic', '/vision/grid_state')
        self.declare_parameter('vision_refresh_service', '/vision/refresh')
        self.declare_parameter('vision_wait', 20.0)
        self.declare_parameter('vision_timeout', 6.0)
        self.declare_parameter('vision_require_stable', True)
        self.declare_parameter('vision_min_score', 0.0)
        self.declare_parameter('vision_required', False)
        self.declare_parameter('vision_settle', 0.6)
        # 新增: 框内白色像素占比超过该阈值一律视为空, 不抓 (85%)
        self.declare_parameter('empty_white_ratio', 0.85)

        # ---------- 随机布局真值校验 ----------
        self.declare_parameter('verify_vision', True)
        self.declare_parameter('ground_truth_file',
                               '/tmp/ep_task3_ground_truth.json')
        self.declare_parameter('truth_max_age', 900.0)

        # 开局一次性扫完六格, 空格子完全不动车 (顶部相机本来就同时看得到)
        self.declare_parameter('prescan', True)

        # ---------- 状态机 ----------
        # 留空 = 自动去 package share 找 config/state_machine.yaml, 找不到就走旧流程
        # 显式设为 "none" / "disabled" = 强制走旧流程
        self.declare_parameter('state_machine_file', '')
        self.declare_parameter('report_dir', '/tmp/ep_task3_reports')

        self.forward_x = float(self.get_parameter('chassis_forward_x').value)
        self.drop_forward_x = float(self.get_parameter('drop_forward_x').value)
        self.arm_base_offset = float(self.get_parameter('arm_base_offset').value)
        self.arm_x_retract = float(self.get_parameter('arm_x_retract').value)
        self.drop_arm_x = float(self.get_parameter('drop_arm_x').value)
        self.safe_z = float(self.get_parameter('safe_z').value)
        self.pick_z = float(self.get_parameter('pick_z').value)
        self.place_z = float(self.get_parameter('place_z').value)
        self.drop_y_cube = float(self.get_parameter('drop_y_cube').value)
        self.drop_y_cyl = float(self.get_parameter('drop_y_cyl').value)
        self.drop_step_cube = float(self.get_parameter('drop_step_cube').value)
        self.drop_step_cyl = float(self.get_parameter('drop_step_cyl').value)
        self.cube_drop_count = 0
        self.cyl_drop_count = 0
        self.zero_dwell = float(self.get_parameter('zero_dwell').value)
        self.odom_report = bool(self.get_parameter('odom_report').value)
        self.odom_correct = bool(self.get_parameter('odom_correct').value)
        self.odom_tol = float(self.get_parameter('odom_correct_tol').value)
        self.odom_max = float(self.get_parameter('odom_correct_max').value)
        self.odom_y_sign = float(self.get_parameter('odom_y_sign').value)
        self.yaw_warn = math.radians(float(self.get_parameter('yaw_warn_deg').value))
        self.y_sign = float(self.get_parameter('y_sign').value)
        self.x_sign = float(self.get_parameter('x_sign').value)
        self.move_speed = float(self.get_parameter('move_linear_speed').value)
        self.action_timeout = float(self.get_parameter('action_timeout').value)
        self.action_cooldown = float(self.get_parameter('action_cooldown').value)
        self.gripper_cooldown = float(self.get_parameter('gripper_cooldown').value)
        self.gripper_power = float(self.get_parameter('gripper_power').value)
        self.arm_x_min = float(self.get_parameter('arm_x_min').value)
        self.arm_x_max = float(self.get_parameter('arm_x_max').value)

        self.use_vision = bool(self.get_parameter('use_vision').value)
        self.vision_wait = float(self.get_parameter('vision_wait').value)
        self.vision_timeout = float(self.get_parameter('vision_timeout').value)
        self.vision_stable = bool(self.get_parameter('vision_require_stable').value)
        self.vision_min_score = float(self.get_parameter('vision_min_score').value)
        self.vision_required = bool(self.get_parameter('vision_required').value)
        self.vision_settle = float(self.get_parameter('vision_settle').value)
        self.empty_white_ratio = float(
            self.get_parameter('empty_white_ratio').value)

        self.car_x = 0.0
        self.car_y = 0.0
        self.have_odom = False
        self.odom_zero = None
        self.odom_cur = None

        self.vision_lock = threading.Lock()
        self.vision_data = None
        self.vision_recv = 0.0

        self.verify_vision = bool(self.get_parameter('verify_vision').value)
        self.truth = self.load_ground_truth()
        self.verify_log = []

        self.grids = self.build_grids()
        self.print_plan()

        group = ReentrantCallbackGroup()
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(
            Odometry, '/odom', self.on_odom,
            qos_profile_sensor_data, callback_group=group
        )
        self.move_client = ActionClient(self, Move, '/move', callback_group=group)
        self.arm_client = ActionClient(self, MoveArm, '/move_arm', callback_group=group)
        self.gripper_client = ActionClient(
            self, GripperControl, '/gripper', callback_group=group
        )

        self.refresh_client = None
        if self.use_vision:
            topic = str(self.get_parameter('vision_topic').value)
            self.create_subscription(
                String, topic, self.on_vision, 10, callback_group=group)
            service = str(self.get_parameter('vision_refresh_service').value)
            self.refresh_client = self.create_client(
                Trigger, service, callback_group=group)
            self.get_logger().info(f'视觉已启用, 订阅 {topic}')
        else:
            self.get_logger().warn(
                '视觉已关闭 (use_vision=false), 按 obj_kind 静态表执行')

        self.get_logger().info('RoboMaster Task3 Pick & Place Node started')

    # ================================================================
    # 固定路线表
    # ================================================================

    def build_grids(self):
        """建立 Grid 1..6 的固定路线.

        优先用 grid_car_y / grid_arm_x / grid_forward_x;
        这三个留空时, 从 obj_x / obj_y 推导 —— 推导出来的顺序
        和原来 order_names=auto 的抓取顺序完全一致, 保证你已经
        调好的动作一点不变.
        """
        ids = [int(v) for v in self.get_parameter('grid_ids').value]

        car_y = [float(v) for v in self.get_parameter('grid_car_y').value]
        arm_x = [float(v) for v in self.get_parameter('grid_arm_x').value]
        fwd_x = [float(v) for v in self.get_parameter('grid_forward_x').value]

        explicit = (len(car_y) == len(ids) and len(arm_x) == len(ids))

        if explicit:
            if len(fwd_x) != len(ids):
                fwd_x = [self.forward_x] * len(ids)
            grids = []
            for i, gid in enumerate(ids):
                grids.append({
                    'id': gid,
                    'name': f'Grid{gid}',
                    'car_y': car_y[i],
                    'arm_x': arm_x[i],
                    'forward_x': fwd_x[i],
                    'fallback_kind': None,
                })
            self.get_logger().info('固定路线来源: grid_* 显式配置')
        else:
            grids = self.grids_from_objects(ids)
            self.get_logger().info('固定路线来源: 由 obj_x / obj_y 推导')

        for g in grids:
            if not (self.arm_x_min <= g['arm_x'] <= self.arm_x_max):
                self.get_logger().warning(
                    f'⚠ Grid {g["id"]} arm_x={g["arm_x"]:.3f} '
                    f'超出 [{self.arm_x_min:.3f}, {self.arm_x_max:.3f}]')

        return grids

    def grids_from_objects(self, ids):
        names = [str(v).strip() for v in self.get_parameter('obj_names').value]
        xs = [float(v) for v in self.get_parameter('obj_x').value]
        ys = [float(v) for v in self.get_parameter('obj_y').value]
        kinds = [norm_kind(v) for v in self.get_parameter('obj_kind').value]

        n = len(names)
        if not (len(xs) == len(ys) == len(kinds) == n):
            raise ValueError(
                f'物体表长度不一致: names={n}, x={len(xs)}, '
                f'y={len(ys)}, kind={len(kinds)}')

        rows = sorted({round(x, 4) for x in xs})
        row_map = {x: i + 1 for i, x in enumerate(rows)}

        items = []
        for i in range(n):
            items.append({
                'name': names[i],
                'x': xs[i],
                'y': ys[i],
                'kind': kinds[i],
                'row': row_map[round(xs[i], 4)],
            })

        items.sort(key=lambda it: (it['row'], round(it['y'], 4), it['name']))

        grids = []
        for i, it in enumerate(items):
            gid = ids[i] if i < len(ids) else i + 1
            grids.append({
                'id': gid,
                'name': it['name'],
                'car_y': it['y'],
                'arm_x': it['x'] - self.arm_base_offset - self.forward_x,
                'forward_x': self.forward_x,
                'fallback_kind': it['kind'],
                'row': it['row'],
                'world_x': it['x'],
            })
        return grids

    def print_plan(self):
        log = self.get_logger()
        log.info('========== 六网格固定路线 ==========')
        for g in self.grids:
            row = g.get('row', '-')
            log.info(
                f'Grid {g["id"]} ({g["name"]}) 第{row}行 '
                f'car_y={g["car_y"]:+.3f} arm_x={g["arm_x"]:.3f} '
                f'forward={g["forward_x"]:.3f} '
                f'静态类别={g.get("fallback_kind") or "-"}')
        log.info(f'方块料盒 Y={self.drop_y_cube:+.2f}  '
                 f'圆柱料盒 Y={self.drop_y_cyl:+.2f}')
        log.info('====================================')

    # ================================================================
    # 随机布局真值校验
    # ================================================================

    def load_ground_truth(self):
        """读 scene_randomizer 写的真值文件, 没有就返回 None (不校验)."""
        if not self.verify_vision:
            return None

        path = str(self.get_parameter('ground_truth_file').value)
        if not path or not os.path.isfile(path):
            self.get_logger().info(
                '未找到随机布局真值文件, 本次不做视觉校验')
            return None

        try:
            with open(path, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
        except Exception as error:
            self.get_logger().warning(f'真值文件读失败: {error}')
            return None

        age = time.time() - float(data.get('stamp', 0.0))
        max_age = float(self.get_parameter('truth_max_age').value)
        if max_age > 0.0 and age > max_age:
            self.get_logger().warning(
                f'真值文件已过期 ({age / 60.0:.1f} 分钟前), 忽略')
            return None

        grids = {str(k): str(v).upper() for k, v in
                 (data.get('grids') or {}).items()}
        counts = data.get('counts', {})
        self.get_logger().info(
            f'✓ 载入随机布局真值 (seed={data.get("seed", "-")}, '
            f'方体 {counts.get("CUBE", "?")} / 圆柱 {counts.get("CYL", "?")} / '
            f'空 {counts.get("EMPTY", "?")})')
        return grids

    def truth_of(self, grid_id):
        if not self.truth:
            return None
        value = self.truth.get(str(grid_id))
        if value is None:
            return None
        if value in ('EMPTY', 'NONE', ''):
            return None
        return 'CUBE' if value.startswith('CUBE') else 'CYL'

    def record_verify(self, grid_id, seen):
        """把这一格的视觉判断和真值对拍."""
        if not self.truth or not self.use_vision:
            return
        want = self.truth_of(grid_id)
        ok = (want == seen)
        self.verify_log.append((grid_id, want, seen, ok))
        text_w = '空' if want is None else ('方体' if want == 'CUBE' else '圆柱')
        text_s = '空' if seen is None else ('方体' if seen == 'CUBE' else '圆柱')
        if ok:
            self.get_logger().info(
                f'  ✅ 校验 Grid {grid_id}: 真值={text_w} 视觉={text_s}')
        else:
            self.get_logger().error(
                f'  ❌ 校验 Grid {grid_id}: 真值={text_w} 视觉={text_s} (判错)')

    def print_verify_report(self):
        if not self.verify_log:
            return
        log = self.get_logger()
        log.info('========== 视觉 vs 真值 ==========')
        good = 0
        for gid, want, seen, ok in self.verify_log:
            text_w = '空  ' if want is None else (
                '方体' if want == 'CUBE' else '圆柱')
            text_s = '空  ' if seen is None else (
                '方体' if seen == 'CUBE' else '圆柱')
            log.info(f'Grid {gid}  真值={text_w}  视觉={text_s}  '
                     f'{"OK" if ok else "MISMATCH"}')
            good += 1 if ok else 0
        total = len(self.verify_log)
        rate = 100.0 * good / max(total, 1)
        if good == total:
            log.info(f'✅ 视觉判别全部正确 {good}/{total} (100%)')
        else:
            log.error(f'❌ 视觉判别 {good}/{total} 正确 ({rate:.1f}%)')
        log.info('==================================')

    # ================================================================
    # 视觉
    # ================================================================

    def on_vision(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        with self.vision_lock:
            self.vision_data = data
            self.vision_recv = time.monotonic()

    def wait_for_vision(self):
        if not self.use_vision:
            return False
        deadline = time.monotonic() + self.vision_wait
        while time.monotonic() < deadline:
            with self.vision_lock:
                if self.vision_data is not None:
                    backend = self.vision_data.get('backend', '?')
                    self.get_logger().info(
                        f'✓ /vision/grid_state 就绪 (后端={backend})')
                    return True
            time.sleep(0.1)
        self.get_logger().warning('等不到 /vision/grid_state')
        return False

    def call_refresh(self):
        if self.refresh_client is None:
            return
        if not self.refresh_client.service_is_ready():
            return
        try:
            self.refresh_client.call_async(Trigger.Request())
        except Exception:
            pass

    def query_grid(self, grid_id):
        """问视觉: 这个网格有没有东西? 是什么类别?

        返回 'CUBE' / 'CYL' / None(空).
        """
        if not self.use_vision:
            return self.fallback_kind(grid_id)

        self.call_refresh()
        start = time.monotonic()
        deadline = start + self.vision_timeout
        latest = None

        while time.monotonic() < deadline:
            with self.vision_lock:
                data = self.vision_data
                recv = self.vision_recv
            if data is not None and recv >= start:
                entry = (data.get('grids') or {}).get(str(grid_id))
                if entry is not None:
                    latest = entry
                    if (not self.vision_stable) or entry.get('stable'):
                        return self.entry_to_kind(grid_id, entry)
            time.sleep(0.05)

        if latest is not None:
            self.get_logger().warning(
                f'Grid {grid_id}: 视觉结果未稳定, 采用当前值')
            return self.entry_to_kind(grid_id, latest)

        if self.vision_required:
            raise RuntimeError(f'Grid {grid_id}: 拿不到视觉结果')

        self.get_logger().warning(
            f'Grid {grid_id}: 拿不到视觉结果, 退回静态表')
        return self.fallback_kind(grid_id)

    def entry_to_kind(self, grid_id, entry):
        cls = str(entry.get('class', EMPTY)).lower()
        score = float(entry.get('score', 0.0))
        label = entry.get('label', '')
        # 视觉端每一格发过来的白色像素占比 (0~1), 没有就当 0
        white = float(entry.get('white', 0.0))

        # === 强空判据: 框内白色 > 阈值 (默认 85%) 一律跳过, 绝不抓 ===
        if white > self.empty_white_ratio:
            self.get_logger().info(
                f'  视觉: Grid {grid_id} → 空 '
                f'(white={white:.2f} > {self.empty_white_ratio:.2f}, 跳过)')
            return None

        if cls == EMPTY:
            self.get_logger().info(
                f'  视觉: Grid {grid_id} → 空 (white={white:.2f})')
            return None

        if score < self.vision_min_score:
            self.get_logger().warning(
                f'  视觉: Grid {grid_id} 置信度 {score:.2f} 偏低, 当作空')
            return None

        kind = 'CUBE' if cls.startswith('cube') else 'CYL'
        shown = '方块料盒' if kind == 'CUBE' else '圆柱料盒'
        self.get_logger().info(
            f'  视觉: Grid {grid_id} → {cls} '
            f'({label or "-"}, {score:.2f}) → {shown}')
        return kind

    def fallback_kind(self, grid_id):
        """拿不到视觉时怎么办.

        随机布局下 obj_kind 那张静态表已经失效, 绝不能拿它去抓 ——
        会伸手去抓空格子. 所以:
            - 视觉开着但没结果 → 当成空, 直接跳过 (最安全)
            - 视觉整个关掉      → 按真值表走, 方便脱离视觉单独调运动
            - 没有随机真值      → 维持原来的静态表行为
        """
        if self.truth:
            if not self.use_vision:
                self.get_logger().info(
                    f'Grid {grid_id}: 视觉已关闭, 按随机布局真值执行')
                return self.truth_of(grid_id)
            self.get_logger().warning(
                f'Grid {grid_id}: 随机布局下拿不到视觉, 当作空跳过')
            return None

        for g in self.grids:
            if g['id'] == grid_id:
                return g.get('fallback_kind')
        return None

    # ================================================================
    # 底层动作 (以下全部保持原样)
    # ================================================================

    def send_goal(self, client, goal, label):
        self.get_logger().info(f'→ {label}')
        send_future = client.send_goal_async(goal)
        handle = wait_future(send_future, self.action_timeout)
        if handle is None:
            raise RuntimeError(f'{label}: 发送目标超时')
        if not handle.accepted:
            raise RuntimeError(f'{label}: 目标被驱动拒绝')

        result = wait_future(handle.get_result_async(), self.action_timeout)
        if result is None:
            try:
                handle.cancel_goal_async()
            except Exception:
                pass
            raise RuntimeError(f'{label}: 执行超时')

        if result.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(f'{label}: Action 未成功, 状态={result.status}')

        return result.result

    def move_chassis(self, dx, dy, label='底盘', track=True):
        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            return

        goal = Move.Goal()
        goal.x = float(dx) * self.x_sign
        goal.y = float(dy) * self.y_sign
        goal.theta = 0.0
        goal.linear_speed = float(self.move_speed)
        goal.angular_speed = 0.5236

        self.send_goal(self.move_client, goal,
                       f'{label} (dx={dx:+.3f}, dy={dy:+.3f})')

        if track:
            self.car_x += float(dx)
            self.car_y += float(dy)

        time.sleep(self.action_cooldown)

    def move_y_to(self, target_y, label='沿Y平移'):
        self.move_chassis(0.0, float(target_y) - self.car_y,
                          f'{label} → Y={target_y:+.3f}')

    def forward(self, dist, label='前进'):
        self.move_chassis(abs(float(dist)), 0.0,
                          f'{label} {abs(float(dist)):.3f} m')

    def backward(self, dist, label='后退'):
        self.move_chassis(-abs(float(dist)), 0.0,
                          f'{label} {abs(float(dist)):.3f} m')

    def retreat_only(self, tag=''):
        if abs(self.car_x) > 1e-4:
            back = self.car_x
            self.get_logger().info(f'◎ 后退(不归零) {tag}, 距离={back:.3f} m')
            self.backward(back, '后退')

    def retreat_and_return_zero(self, tag=''):
        self.get_logger().info(
            f'◎ 回零 {tag}, 当前 x={self.car_x:+.3f}, y={self.car_y:+.3f}')

        if abs(self.car_x) > 1e-4:
            self.backward(self.car_x, f'回零-后退(={self.car_x:.3f})')

        if abs(self.car_y) > 1e-4:
            self.move_y_to(0.0, '回零-沿Y归中')

        self.check_zero_by_odom()

        if self.zero_dwell > 0.0:
            time.sleep(self.zero_dwell)

    def check_zero_by_odom(self):
        if not (self.odom_report or self.odom_correct):
            return
        if self.odom_zero is None or self.odom_cur is None:
            return

        x0, y0, yaw0 = self.odom_zero
        x1, y1, yaw1 = self.odom_cur
        err_x = x0 - x1
        err_y = (y0 - y1) * self.odom_y_sign
        err = math.hypot(err_x, err_y)

        dyaw = yaw1 - yaw0
        dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))

        self.get_logger().info(
            f'[odom] 回零残差 dx={err_x:+.3f} dy={err_y:+.3f} '
            f'|e|={err:.3f} m 偏航={math.degrees(dyaw):+.1f}°')

        if abs(dyaw) > self.yaw_warn:
            self.get_logger().warning(f'⚠ 底盘偏航 {math.degrees(dyaw):+.1f}°')

        if not self.odom_correct or err < self.odom_tol:
            return

        if err > self.odom_max:
            self.get_logger().warning(
                f'⚠ 残差 {err:.3f} m 超过上限 {self.odom_max:.3f} m，放弃补偿')
            return

        self.move_chassis(err_x, err_y, '回零-补偿', track=False)

    def move_arm_to(self, x, z, label='手臂'):
        goal = MoveArm.Goal()
        goal.x = float(x)
        goal.z = float(z)
        goal.relative = False

        self.send_goal(self.arm_client, goal,
                       f'{label} → (x={x:.3f}, z={z:.3f})')

        time.sleep(self.action_cooldown)

    def arm_lift_retract(self, label='爪子抬起并缩回'):
        self.move_arm_to(self.arm_x_retract, self.safe_z, label)

    def arm_lower(self, label='放下爪子'):
        self.move_arm_to(self.arm_x_retract, self.pick_z, label)

    def gripper(self, state, label='抓夹'):
        goal = GripperControl.Goal()
        goal.target_state = int(state)
        goal.power = float(self.gripper_power)

        text = {
            GripperControl.Goal.OPEN: '张开',
            GripperControl.Goal.CLOSE: '闭合',
            GripperControl.Goal.PAUSE: '暂停',
        }.get(int(state), str(state))

        self.send_goal(self.gripper_client, goal, f'{label} → {text}')

        time.sleep(self.gripper_cooldown)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        yaw = yaw_from_quat(msg.pose.pose.orientation)

        self.odom_cur = (p.x, p.y, yaw)

        if not self.have_odom:
            self.odom_zero = (p.x, p.y, yaw)
            self.have_odom = True
            self.get_logger().info(
                f'✓ /odom 就绪 ({p.x:+.3f}, {p.y:+.3f}, '
                f'{math.degrees(yaw):+.1f}°)')

    def wait_for_odom(self, timeout=15.0):
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            if self.have_odom:
                return True
            time.sleep(0.05)
        return False

    def wait_for_servers(self):
        timeout = float(self.get_parameter('server_wait').value)

        for client, name in (
            (self.move_client, '/move'),
            (self.arm_client, '/move_arm'),
            (self.gripper_client, '/gripper'),
        ):
            self.get_logger().info(f'等待 Action 服务器 {name} ...')
            if not client.wait_for_server(timeout_sec=timeout):
                raise RuntimeError(f'{name} 未上线，请检查 robomaster_ros 驱动')
            self.get_logger().info(f'✓ {name} 就绪')

    # ================================================================
    # 一个网格的固定路线
    # ================================================================

    def execute_fixed_route(self, index, grid, kind):
        """走这个网格 **原来已经调好的** 固定路线.

        kind 只影响最后投到哪个料盒, 不影响取件段的任何坐标.
        """
        gid = grid['id']
        car_y = grid['car_y']
        arm_x = grid['arm_x']
        fwd = grid['forward_x']
        if kind == 'CUBE':
            drop_y = self.drop_y_cube + self.cube_drop_count * self.drop_step_cube
            self.cube_drop_count += 1
        else:
            drop_y = self.drop_y_cyl - self.cyl_drop_count * self.drop_step_cyl
            self.cyl_drop_count += 1

        box = '方块料盒' if kind == 'CUBE' else '圆柱料盒'

        self.get_logger().info(
            f'===== [{index}] Grid {gid} → {kind} → {box} '
            f'(car_y={car_y:+.3f}, arm_x={arm_x:.3f}) =====')

        self.arm_lift_retract('[0] 抬起缩回')
        self.gripper(GripperControl.Goal.OPEN, '[0] 张开')

        self.move_y_to(car_y, f'[1] 对准 Grid {gid}')

        self.forward(fwd, '[2] 前进')

        self.arm_lower('[3] 放下')

        self.move_arm_to(arm_x, self.pick_z, f'[4] 伸出 Grid {gid}')

        self.gripper(GripperControl.Goal.CLOSE, f'[5] 抓取 Grid {gid}')

        self.move_arm_to(arm_x, self.safe_z, f'[6] 抬起 Grid {gid}')

        self.arm_lift_retract('[7] 缩回')

        self.retreat_only(f'[8] Grid {gid}')

        self.move_y_to(drop_y, f'[9] 去{box}')

        if self.drop_forward_x > 1e-4:
            self.forward(self.drop_forward_x, '[9b] 投放前进')

        self.move_arm_to(self.arm_x_retract, self.place_z, '[10] 放下')

        self.move_arm_to(self.drop_arm_x, self.place_z, '[11] 伸出')

        self.gripper(GripperControl.Goal.OPEN, f'[12] 松开 → {box}')

        time.sleep(0.4)

        self.move_arm_to(self.arm_x_retract, self.place_z, '[13] 缩回')

        self.arm_lift_retract('[14] 抬起')

        self.retreat_and_return_zero(f'[15] Grid {gid}')

        self.get_logger().info(f'===== Grid {gid} 完成，已回零 =====')

    def execute_fixed_cube_route(self, index, grid):
        self.execute_fixed_route(index, grid, 'CUBE')

    def execute_fixed_cylinder_route(self, index, grid):
        self.execute_fixed_route(index, grid, 'CYL')

    # ================================================================
    # 主流程
    # ================================================================

    def vision_snapshot(self):
        """刷新一次视觉, 等一帧同时包含六个网格的稳定结果.

        比逐格 query_grid 少 6 次 refresh + 6 次超时等待, 预扫描更快.
        拿不到就返回 None, 调用方退回逐格查询.
        """
        self.call_refresh()
        start = time.monotonic()
        deadline = start + self.vision_timeout
        latest = None
        want = [str(g['id']) for g in self.grids]

        while time.monotonic() < deadline:
            with self.vision_lock:
                data = self.vision_data
                recv = self.vision_recv
            if data is not None and recv >= start:
                grids = data.get('grids') or {}
                if all(k in grids for k in want):
                    latest = grids
                    if (not self.vision_stable) or all(
                            grids[k].get('stable') for k in want):
                        return grids
            time.sleep(0.05)

        if latest is not None:
            self.get_logger().warning('预扫描: 视觉未完全稳定, 采用当前帧')
        return latest

    def scan_all_grids(self):
        """开局把六个网格一次性判完, 返回 [(grid, kind), ...].

        顶部相机是固定俯视的, 六个格子同时在画面里, 判别根本不需要把车开过去.
        所以空格子从头到尾零动作: 不对准, 不前进, 不伸手, 直接轮到下一个有物体的
        网格按顺序执行它自己的固定路线.
        """
        plan = []

        if not bool(self.get_parameter('prescan').value):
            # 老行为: 走到一个判一个
            for grid in self.grids:
                if self.vision_settle > 0.0:
                    time.sleep(self.vision_settle)
                kind = self.query_grid(grid['id'])
                self.record_verify(grid['id'], kind)
                plan.append((grid, kind))
            return plan

        self.get_logger().info('======== 视觉预扫描 (车不动) ========')
        if self.use_vision and self.vision_settle > 0.0:
            time.sleep(self.vision_settle)

        snapshot = self.vision_snapshot() if self.use_vision else None

        for grid in self.grids:
            gid = grid['id']
            entry = (snapshot or {}).get(str(gid))
            if entry is not None:
                kind = self.entry_to_kind(gid, entry)
            else:
                kind = self.query_grid(gid)
            self.record_verify(gid, kind)
            plan.append((grid, kind))

        self.get_logger().info('======== 本次执行计划 ========')
        order = 0
        for grid, kind in plan:
            if kind is None:
                self.get_logger().info(
                    f"  Grid {grid['id']}: 空   → 跳过 (不靠近, 零动作)")
            else:
                order += 1
                box = '方块料盒' if kind == 'CUBE' else '圆柱料盒'
                text = '方体' if kind == 'CUBE' else '圆柱'
                self.get_logger().info(
                    f"  Grid {grid['id']}: {text} → 第 {order} 个抓取, 投{box}")
        self.get_logger().info(
            f'  共 {order} 个要抓, '
            f'{sum(1 for _, k in plan if k is None)} 个空格跳过')
        self.get_logger().info('=============================')
        return plan

    def run(self):
        # ---------- 状态机 dispatch ----------
        sm_file = self._resolve_state_machine_file()
        if sm_file:
            self.get_logger().info(f'✓ 使用状态机配置: {sm_file}')
            return self._run_with_state_machine(sm_file)
        self.get_logger().info('未加载状态机配置, 走旧流程')
        # ---------- ↓↓↓ 以下为旧流程, 一个字没动 ↓↓↓ ----------
        self.get_logger().info('== 等待 /odom ==')
        if not self.wait_for_odom(20.0):
            self.get_logger().warning('收不到 /odom，继续执行')

        self.get_logger().info('== 等待 Action 服务器 ==')
        self.wait_for_servers()

        if self.use_vision:
            self.get_logger().info('== 等待视觉 ==')
            self.wait_for_vision()

        self.get_logger().info('== 初始化 ==')
        self.arm_lift_retract('初始化: 手臂归位')
        self.gripper(GripperControl.Goal.OPEN, '初始化: 张开')

        plan = self.scan_all_grids()
        todo = [(grid, kind) for grid, kind in plan if kind is not None]
        skipped = len(plan) - len(todo)
        picked = 0

        for index, (grid, kind) in enumerate(todo, 1):
            if kind == 'CUBE':
                self.execute_fixed_cube_route(index, grid)
            else:
                self.execute_fixed_cylinder_route(index, grid)
            picked += 1

        self.get_logger().info('== 任务结束 ==')
        self.retreat_and_return_zero('任务结束')
        self.arm_lift_retract('任务结束: 手臂归位')
        self.get_logger().info(
            f'✅ 完成: 抓取 {picked} 个, 跳过 {skipped} 个空网格')
        self.print_verify_report()

    # ================================================================
    # 状态机集成 (下面这一段是新增的, 不影响旧流程)
    # ================================================================

    def _resolve_state_machine_file(self):
        """决定用哪个状态机 yaml. 返回路径字符串, 或 '' 表示走旧流程."""
        raw = str(self.get_parameter('state_machine_file').value).strip()
        if raw.lower() in ('none', 'disabled', 'off', 'false', '0'):
            return ''
        if raw:
            return raw if os.path.isfile(raw) else ''
        # 默认: 去 package share 找
        try:
            from ament_index_python.packages import get_package_share_directory
            default = os.path.join(
                get_package_share_directory('ep_task3_pick'),
                'config', 'state_machine.yaml')
            if os.path.isfile(default):
                return default
        except Exception:
            pass
        return ''

    def _run_with_state_machine(self, sm_file):
        # 1) 读配置
        try:
            spec = load_yaml(sm_file)
        except Exception as error:
            self.get_logger().error(
                f'状态机配置读失败 {sm_file}: {error}, 回退旧流程')
            return self.run_legacy_after_sm_failure()

        reports = (spec.get('state_machine') or spec).get('reports') or {}
        rdir = reports.get('dir') or str(self.get_parameter('report_dir').value)

        # 2) 建 reporter + 状态机
        self._reporter = RunReporter(
            report_dir=rdir,
            cls_file=reports.get('classification_file', 'classification_results.txt'),
            exc_file=reports.get('exception_file', 'exception_log.txt'),
            log_file=reports.get('task_log_file', 'task_log.txt'),
            also_json=bool(reports.get('also_write_json', True)),
            logger=self.get_logger(),
        )
        self._sm = StateMachine(
            spec, target=self, reporter=self._reporter,
            logger=self.get_logger())

        # 3) 运行时状态
        self.sm_current_grid_id = None
        self._sm_plan = []
        self._sm_current = None
        self._sm_pick_count = 0
        self._sm_skip_count = 0
        self._sm_recover_count = 0
        self._sm_max_recover = 3

        # 4) 跑
        status = 'OK'
        try:
            self._sm.run()
        except Exception as error:
            self.get_logger().error(f'状态机异常终止: {error}')
            self._reporter.add_exception('__sm__', error, action='abort')
            status = 'ABORTED'

        # 5) 落盘 (finalize/fault handler 里已经调过 flush; 这里兜底再刷一次)
        self._reporter.set_summary(
            status=status,
            picked=self._sm_pick_count,
            skipped=self._sm_skip_count,
            recoveries=self._sm_recover_count,
        )
        self._reporter.flush()
        self.get_logger().info(f'📄 报告已写入: {self._reporter.dir}')
        self.get_logger().info(
            f'   - {self._reporter.cls_path}')
        self.get_logger().info(
            f'   - {self._reporter.exc_path}')
        self.get_logger().info(
            f'   - {self._reporter.log_path}')

    def run_legacy_after_sm_failure(self):
        """状态机 yaml 加载失败时, 走原来的 run() 主体一次 (跳过 dispatch)."""
        self.get_logger().info('== 等待 /odom ==')
        if not self.wait_for_odom(20.0):
            self.get_logger().warning('收不到 /odom，继续执行')
        self.get_logger().info('== 等待 Action 服务器 ==')
        self.wait_for_servers()
        if self.use_vision:
            self.get_logger().info('== 等待视觉 ==')
            self.wait_for_vision()
        self.get_logger().info('== 初始化 ==')
        self.arm_lift_retract('初始化: 手臂归位')
        self.gripper(GripperControl.Goal.OPEN, '初始化: 张开')
        plan = self.scan_all_grids()
        todo = [(g, k) for g, k in plan if k is not None]
        skipped = len(plan) - len(todo)
        picked = 0
        for index, (g, k) in enumerate(todo, 1):
            if k == 'CUBE':
                self.execute_fixed_cube_route(index, g)
            else:
                self.execute_fixed_cylinder_route(index, g)
            picked += 1
        self.retreat_and_return_zero('任务结束')
        self.arm_lift_retract('任务结束: 手臂归位')
        self.get_logger().info(
            f'✅ 完成: 抓取 {picked} 个, 跳过 {skipped} 个空网格')
        self.print_verify_report()

    # ---------- SM handlers: 每个方法返回 event 字符串 ----------

    def sm_init(self):
        """INIT: 等 odom / Action / 视觉, 归位手臂 + 张爪."""
        self.get_logger().info('== 等待 /odom ==')
        if not self.wait_for_odom(20.0):
            self.get_logger().warning('收不到 /odom, 继续执行')
        self.get_logger().info('== 等待 Action 服务器 ==')
        self.wait_for_servers()
        if self.use_vision:
            self.get_logger().info('== 等待视觉 ==')
            self.wait_for_vision()
        self.get_logger().info('== 初始化 ==')
        self.arm_lift_retract('初始化: 手臂归位')
        self.gripper(GripperControl.Goal.OPEN, '初始化: 张开')
        return 'ready'

    def sm_prescan(self):
        """PRESCAN: 顶视相机一次判完 6 格, 建立待办队列."""
        plan = self.scan_all_grids()
        # 把 record_verify 里已经采集的 verify_log 同步进 reporter, 即使真值文件缺失
        # 也把视觉判定挂进来, 方便报告里看
        truth_map = self.truth or {}
        for grid, kind in plan:
            gid = grid['id']
            if truth_map:
                want = self.truth_of(gid)
                truth_value = want if want is not None else (
                    None if str(truth_map.get(str(gid), '')).upper() in
                    ('EMPTY', 'NONE', '') else 'UNKNOWN')
            else:
                truth_value = 'UNKNOWN'
            self._reporter.add_classification(gid, truth_value, kind,
                                              note='prescan')
        self._sm_plan = list(plan)
        self.get_logger().info(f'[SM] 计划就绪, 共 {len(self._sm_plan)} 格')
        return 'plan_ready'

    def sm_next_grid(self):
        """NEXT_GRID: 取下一个网格, 决定去 EXECUTE / SKIP / FINALIZE."""
        if not self._sm_plan:
            self._sm_current = None
            self.sm_current_grid_id = None
            return 'done_all'
        grid, kind = self._sm_plan.pop(0)
        self._sm_current = (grid, kind)
        self.sm_current_grid_id = grid['id']
        if kind is None:
            return 'empty'
        return 'have_target'

    def sm_execute(self):
        """EXECUTE: 走该 Grid 的固定路线. 抛异常会被 SM 抓到 -> RECOVER."""
        grid, kind = self._sm_current
        index = self._sm_pick_count + 1
        if kind == 'CUBE':
            self.execute_fixed_cube_route(index, grid)
        else:
            self.execute_fixed_cylinder_route(index, grid)
        self._sm_pick_count += 1
        return 'picked'

    def sm_skip(self):
        """SKIP: 空格零动作."""
        grid, _ = self._sm_current
        self.get_logger().info(
            f'[SM] Grid {grid["id"]}: 空格, 跳过 (零动作)')
        self._sm_skip_count += 1
        return 'skipped'

    def sm_recover(self):
        """RECOVER: 松爪 + 手臂归位 + 底盘回零. 超次数直接 fatal."""
        self._sm_recover_count += 1
        gid = self.sm_current_grid_id
        self.get_logger().warning(
            f'[SM] RECOVER #{self._sm_recover_count} '
            f'(Grid {gid}), 尝试回到安全状态')

        # 逐步兜底: 每一步失败都吞掉, 尽力恢复
        try:
            self.gripper(GripperControl.Goal.OPEN, 'RECOVER: 松爪')
        except Exception as e:
            self.get_logger().warning(f'RECOVER 松爪失败: {e}')
        try:
            self.arm_lift_retract('RECOVER: 手臂归位')
        except Exception as e:
            self.get_logger().warning(f'RECOVER 手臂归位失败: {e}')
        try:
            self.retreat_and_return_zero('RECOVER: 底盘回零')
        except Exception as e:
            self.get_logger().warning(f'RECOVER 底盘回零失败: {e}')

        if self._sm_recover_count >= self._sm_max_recover:
            self.get_logger().error(
                f'RECOVER 已尝试 {self._sm_recover_count} 次, 放弃 -> FAULT')
            return 'fatal'
        return 'recovered'

    def sm_finalize(self):
        """FINALIZE: 结束扫尾 + 打印真值报告 + 触发落盘 (flush 在 _run_with_state_machine 兜底)."""
        try:
            self.retreat_and_return_zero('任务结束')
        except Exception as e:
            self.get_logger().warning(f'FINALIZE 回零失败: {e}')
        try:
            self.arm_lift_retract('任务结束: 手臂归位')
        except Exception as e:
            self.get_logger().warning(f'FINALIZE 手臂归位失败: {e}')
        self.get_logger().info(
            f'✅ 完成: 抓取 {self._sm_pick_count} 个, '
            f'跳过 {self._sm_skip_count} 个空网格, '
            f'恢复 {self._sm_recover_count} 次')
        self.print_verify_report()
        self._reporter.set_summary(
            status='OK',
            picked=self._sm_pick_count,
            skipped=self._sm_skip_count,
            recoveries=self._sm_recover_count,
        )
        self._reporter.flush()
        return 'done'

    def sm_fault(self):
        """FAULT: 出大问题, 停车 + 落盘所有报告."""
        self.get_logger().error('[SM] FAULT: 任务终止')
        try:
            self.cmd_vel_pub.publish(Twist())
        except Exception:
            pass
        self._reporter.set_summary(
            status='FAULT',
            picked=self._sm_pick_count,
            skipped=self._sm_skip_count,
            recoveries=self._sm_recover_count,
        )
        self._reporter.flush()
        return 'shutdown'


def main(args=None):
    rclpy.init(args=args)

    node = None
    executor = MultiThreadedExecutor(num_threads=4)

    try:
        node = PickNode()
        executor.add_node(node)

        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        try:
            node.run()
        except Exception as error:
            node.get_logger().error(f'任务失败: {error}')
            try:
                node.cmd_vel_pub.publish(Twist())
            except Exception:
                pass

    except KeyboardInterrupt:
        pass

    except Exception as error:
        if node is not None:
            node.get_logger().error(f'节点异常: {error}')
        else:
            print(f'节点异常: {error}')

    finally:
        try:
            executor.shutdown()
        except Exception:
            pass

        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

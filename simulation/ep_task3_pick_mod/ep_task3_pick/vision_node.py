#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vision_node —— 六网格颜色/形状判别节点 (稳定版).

本次修改 (针对 backend=color 时置信度框和数字闪烁的问题):
    1. push_vote 加入非对称迟滞:
       - 首次达到 min_votes 才建立稳定态;
       - 已经稳定后, 要切换到"另一类别"或"empty"
         需要 switch_min_votes 票 (默认 = vote_window - 1),
         也就是接近全票才允许翻转 —— 这样 blank_ratio
         在阈值附近轻微抖动不会让框和数字闪。
    2. 新增 white_ratio 独立计算 (HSV: S<white_s_max, V>white_v_min),
       并做指数滑动平均。这个"白色占比"更贴近人眼直觉:
       白桌面上没物体时白像素占比很高。
    3. white_ratio 作为独立的强空判据:
       white > empty_white_ratio (默认 0.85) 直接判 empty,
       同时通过 /vision/grid_state 的 entry['white'] 字段
       发布给 pick_node, 让抓取节点也用 85% 白判空。
    4. 默认参数放宽: vote_window 15, min_votes 8, switch 14,
       empty_blank_ratio 抬到 0.93 让空/物体过渡更干脆。

输入:
    /top_camera/image_raw       sensor_msgs/Image

输出:
    /top_camera/detections      vision_msgs/Detection2DArray
    /vision/grid_state          std_msgs/String   JSON, 给 pick_node 用
                                  每个 grid 多了 'white' 字段
    /top_camera/image_annotated sensor_msgs/Image 调试用

服务:
    /vision/refresh              std_srvs/Trigger  清空投票, 重新判别
"""

import json
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

try:
    from vision_msgs.msg import (
        BoundingBox2D,
        Detection2D,
        Detection2DArray,
        ObjectHypothesisWithPose,
    )
    VISION_MSGS_OK = True
except ImportError:
    VISION_MSGS_OK = False


EMPTY = 'empty'
CUBE = 'cube'
CYL = 'cylinder'

COLOR_GRID = (90, 90, 90)
COLOR_EMPTY = (120, 120, 120)
COLOR_CUBE = (255, 120, 40)
COLOR_CYL = (40, 120, 255)


# =====================================================================
# vision_msgs 版本兼容
# =====================================================================

def fill_bbox(bbox, cx, cy, w, h):
    center = bbox.center
    if hasattr(center, 'position'):
        center.position.x = float(cx)
        center.position.y = float(cy)
    else:
        center.x = float(cx)
        center.y = float(cy)
    try:
        center.theta = 0.0
    except Exception:
        pass
    bbox.size_x = float(w)
    bbox.size_y = float(h)


def make_hypothesis(class_name, score):
    hyp = ObjectHypothesisWithPose()
    if hasattr(hyp, 'hypothesis'):
        hyp.hypothesis.class_id = str(class_name)
        hyp.hypothesis.score = float(score)
        return hyp
    try:
        hyp.id = str(class_name)
    except Exception:
        hyp.id = 0
    hyp.score = float(score)
    return hyp


# =====================================================================
# ROI 工具
# =====================================================================

def split_table(table_roi, rows, cols, shrink):
    x0, y0, x1, y1 = table_roi
    cell_w = (x1 - x0) / float(cols)
    cell_h = (y1 - y0) / float(rows)
    mx = cell_w * shrink * 0.5
    my = cell_h * shrink * 0.5

    cells = []
    for r in range(rows):
        for c in range(cols):
            cx0 = x0 + c * cell_w
            cy0 = y0 + r * cell_h
            cells.append([
                cx0 + mx,
                cy0 + my,
                cx0 + cell_w - mx,
                cy0 + cell_h - my,
            ])
    return cells


def to_pixels(roi, width, height):
    x0 = int(round(roi[0] * width))
    y0 = int(round(roi[1] * height))
    x1 = int(round(roi[2] * width))
    y1 = int(round(roi[3] * height))
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return x0, y0, x1, y1


# =====================================================================
# 节点
# =====================================================================

class VisionNode(Node):

    def __init__(self):
        super().__init__('vision_node')

        self.declare_parameter('image_topic', '/top_camera/image_raw')
        self.declare_parameter('backend', 'auto')

        # ---- 网格定义 --------------------------------------------------
        self.declare_parameter('grid_rois', [0.0])
        self.declare_parameter('table_roi', [0.06, 0.10, 0.94, 0.90])
        self.declare_parameter('grid_rows', 2)
        self.declare_parameter('grid_cols', 3)
        self.declare_parameter('grid_shrink', 0.10)
        self.declare_parameter('grid_index_map', [0, 1, 2, 3, 4, 5])

        # ---- YOLO ------------------------------------------------------
        self.declare_parameter('yolo_model', 'yolov8n.pt')
        self.declare_parameter('yolo_conf', 0.25)
        self.declare_parameter('yolo_iou', 0.45)
        self.declare_parameter('yolo_imgsz', 640)
        self.declare_parameter('yolo_device', 'cpu')
        self.declare_parameter('yolo_period', 0.20)
        self.declare_parameter('class_cube', [
            'apple', 'orange', 'sports ball', 'donut'
        ])
        self.declare_parameter('class_cyl', [
            'bottle', 'vase', 'cup', 'wine glass', 'cell phone'
        ])

        # ---- 颜色/形状后备 ---------------------------------------------
        self.declare_parameter('cube_hsv_lo', [100.0, 110.0, 60.0])
        self.declare_parameter('cube_hsv_hi', [130.0, 255.0, 255.0])
        self.declare_parameter('cyl_hsv_lo', [0.0, 110.0, 60.0])
        self.declare_parameter('cyl_hsv_hi', [10.0, 255.0, 255.0])
        self.declare_parameter('cyl_hsv_lo2', [170.0, 110.0, 60.0])
        self.declare_parameter('cyl_hsv_hi2', [180.0, 255.0, 255.0])
        self.declare_parameter('color_min_area_ratio', 0.02)

        # ---- 空白判断 --------------------------------------------------
        self.declare_parameter('empty_check', True)
        # blank_ratio = 1 - 彩色物体像素占比。抬到 0.93 避免和 white 判据冲突。
        self.declare_parameter('empty_blank_ratio', 0.93)
        self.declare_parameter('shape_extent_cube', 0.86)

        # ---- 新增: 白色占比判据 (核心的空判据) ------------------------
        # 白色像素定义 (桌面色): HSV 中 S <= white_s_max 且 V >= white_v_min
        self.declare_parameter('white_s_max', 60.0)
        self.declare_parameter('white_v_min', 170.0)
        # 单格框内白色像素比例超过此阈值即判为空 (0.85 = 85%)
        self.declare_parameter('empty_white_ratio', 0.85)
        # 白色比例的 EMA 平滑系数, 越大越跟随即时值 (0.4 折中)
        self.declare_parameter('white_smooth_alpha', 0.4)

        # ---- 稳定性 (加强版, 抗闪烁) -----------------------------------
        # vote_window 加长, 首次稳定门槛 min_votes 提高,
        # switch_min_votes 是稳定后要"改主意"所需的票数
        # (默认取 vote_window - 1, 也就是近乎全票才允许翻转)。
        self.declare_parameter('vote_window', 15)
        self.declare_parameter('min_votes', 8)
        self.declare_parameter('switch_min_votes', 0)  # 0 = 自动 = window-1
        self.declare_parameter('publish_annotated', True)
        self.declare_parameter('annotated_scale', 1.0)
        self.declare_parameter('state_rate', 5.0)
        self.declare_parameter('log_period', 5.0)

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.backend_req = str(self.get_parameter('backend').value).lower()
        self.index_map = [
            int(v) for v in self.get_parameter('grid_index_map').value
        ]
        self.vote_window = max(1, int(self.get_parameter('vote_window').value))
        self.min_votes = max(1, int(self.get_parameter('min_votes').value))

        raw_switch = int(self.get_parameter('switch_min_votes').value)
        if raw_switch <= 0:
            # 默认: 近乎全票 (window - 1)，最小也要严格大于 min_votes
            self.switch_min_votes = max(self.min_votes + 1,
                                        self.vote_window - 1)
        else:
            self.switch_min_votes = min(self.vote_window,
                                        max(self.min_votes, raw_switch))

        self.want_annotated = bool(
            self.get_parameter('publish_annotated').value
        )
        self.annot_scale = float(self.get_parameter('annotated_scale').value)
        self.log_period = float(self.get_parameter('log_period').value)

        self.cube_names = {
            str(v).lower() for v in self.get_parameter('class_cube').value
        }
        self.cyl_names = {
            str(v).lower() for v in self.get_parameter('class_cyl').value
        }

        self.white_alpha = float(
            self.get_parameter('white_smooth_alpha').value)
        self.white_alpha = min(1.0, max(0.05, self.white_alpha))

        self.rois = self.build_rois()
        self.n_grid = len(self.rois)

        self.lock = threading.Lock()
        self.votes = {i + 1: [] for i in range(self.n_grid)}
        self.state = {
            i + 1: {
                'class': EMPTY,
                'label': '',
                'score': 0.0,
                'stable': False,
                'votes': 0,
            }
            for i in range(self.n_grid)
        }

        # 锁定后的显示结果。稳定后不再让 bbox / score 随每帧轻微变化。
        self.display = {
            i + 1: {
                'kind': EMPTY,
                'label': '',
                'score': 0.0,
                'bbox': None,
            }
            for i in range(self.n_grid)
        }

        self.seq = 0
        self.frames = 0
        self.last_log = time.monotonic()
        self.last_yolo = 0.0
        self.last_boxes = []
        self.yolo_fresh = False

        self.model = None
        self.backend = self.setup_backend()

        self.det_pub = None
        if VISION_MSGS_OK:
            self.det_pub = self.create_publisher(
                Detection2DArray,
                '/top_camera/detections',
                10,
            )
        else:
            self.get_logger().warn(
                '没装 vision_msgs, 不发布 /top_camera/detections. '
                '安装: sudo apt install ros-$ROS_DISTRO-vision-msgs'
            )

        # 每格的原始 blank_ratio (色掩码取反) 与 平滑的 white_ratio
        self.blank_ratio = {g: 1.0 for g in range(1, self.n_grid + 1)}
        self.white_ratio = {g: 1.0 for g in range(1, self.n_grid + 1)}

        self.state_pub = self.create_publisher(
            String,
            '/vision/grid_state',
            10,
        )
        self.annot_pub = self.create_publisher(
            Image,
            '/top_camera/image_annotated',
            qos_profile_sensor_data,
        )

        self.create_subscription(
            Image,
            self.image_topic,
            self.on_image,
            qos_profile_sensor_data,
        )

        self.create_service(
            Trigger,
            '/vision/refresh',
            self.on_refresh,
        )

        rate = float(self.get_parameter('state_rate').value)
        self.create_timer(1.0 / max(rate, 0.5), self.publish_state)

        self.get_logger().info(
            f'vision_node 启动 | 后端={self.backend} | '
            f'{self.n_grid} 个网格 | 订阅 {self.image_topic}'
        )
        self.get_logger().info(
            f'稳定策略: window={self.vote_window}, min={self.min_votes}, '
            f'switch(迟滞)={self.switch_min_votes} '
            f'| 空判据: white>{float(self.get_parameter("empty_white_ratio").value):.2f} '
            f'或 blank>{float(self.get_parameter("empty_blank_ratio").value):.2f}'
        )
        for gid in range(1, self.n_grid + 1):
            roi = self.rois[gid - 1]
            self.get_logger().info(
                f'  Grid {gid}: ROI=[{roi[0]:.3f}, {roi[1]:.3f}, '
                f'{roi[2]:.3f}, {roi[3]:.3f}]'
            )

    # ---------------------------------------------------------- 初始化

    def build_rois(self):
        raw = [
            float(v) for v in self.get_parameter('grid_rois').value
        ]
        if len(raw) >= 8 and len(raw) % 4 == 0:
            cells = [
                raw[i:i + 4] for i in range(0, len(raw), 4)
            ]
            self.get_logger().info('网格来源: grid_rois (手工标定)')
            if len(self.index_map) == len(cells):
                cells = [cells[i] for i in self.index_map]
            return cells

        table = [
            float(v) for v in self.get_parameter('table_roi').value
        ]
        rows = int(self.get_parameter('grid_rows').value)
        cols = int(self.get_parameter('grid_cols').value)
        shrink = float(self.get_parameter('grid_shrink').value)
        cells = split_table(table, rows, cols, shrink)

        if len(self.index_map) == len(cells):
            cells = [cells[i] for i in self.index_map]

        self.get_logger().info(
            f'网格来源: table_roi 自动均分 {rows}x{cols}'
        )
        return cells

    def setup_backend(self):
        if self.backend_req == 'none':
            return 'none'
        if self.backend_req == 'color':
            return 'color'

        try:
            from ultralytics import YOLO
        except ImportError:
            if self.backend_req == 'yolo':
                self.get_logger().error(
                    '要求 backend=yolo 但没装 ultralytics, 退回 color.\n'
                    '  pip3 install ultralytics'
                )
            else:
                self.get_logger().warn(
                    '没装 ultralytics, 使用 color 后端'
                )
            return 'color'

        path = str(self.get_parameter('yolo_model').value)
        try:
            self.model = YOLO(path)
            device = str(self.get_parameter('yolo_device').value)
            self.model.to(device)
            self.get_logger().info(
                f'✓ YOLO 已加载: {path} (device={device})'
            )
            return 'yolo'
        except Exception as error:
            self.get_logger().error(
                f'加载 YOLO 模型失败 ({error}), 退回 color 后端'
            )
            return 'color'

    # ---------------------------------------------------------- 识别

    def run_yolo(self, bgr):
        period = float(self.get_parameter('yolo_period').value)
        now = time.monotonic()
        if now - self.last_yolo < period:
            self.yolo_fresh = False
            return self.last_boxes

        self.last_yolo = now
        self.yolo_fresh = True

        conf = float(self.get_parameter('yolo_conf').value)
        iou = float(self.get_parameter('yolo_iou').value)
        imgsz = int(self.get_parameter('yolo_imgsz').value)

        try:
            result = self.model.predict(
                source=bgr,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False,
            )[0]
        except Exception as error:
            self.get_logger().warn(f'YOLO 推理失败: {error}')
            self.yolo_fresh = False
            return self.last_boxes

        names = result.names
        boxes = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            name = str(names.get(cls_id, cls_id)).lower()
            score = float(box.conf[0])
            x0, y0, x1, y1 = [
                float(v) for v in box.xyxy[0].tolist()
            ]

            if name in self.cube_names:
                kind = CUBE
            elif name in self.cyl_names:
                kind = CYL
            else:
                continue

            boxes.append((
                0.5 * (x0 + x1),
                0.5 * (y0 + y1),
                x0, y0, x1, y1,
                kind,
                name,
                score,
            ))

        self.last_boxes = boxes
        return boxes

    def compute_white_ratio(self, hsv):
        """独立的白色像素占比 (HSV: S 低 + V 高). 与色掩码解耦。"""
        s_max = float(self.get_parameter('white_s_max').value)
        v_min = float(self.get_parameter('white_v_min').value)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        white_mask = (s <= s_max) & (v >= v_min)
        total = white_mask.size
        if total <= 0:
            return 1.0
        return float(np.count_nonzero(white_mask)) / float(total)

    def classify_color(self, patch, gid=0):
        """HSV 颜色 + 形状. 返回 (kind, score, bbox_in_patch) 或 None(空)."""
        h, w = patch.shape[:2]
        if h < 2 or w < 2:
            return None

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

        # ---- 独立白色占比 + EMA 平滑, 抹掉单帧毛刺 -----------------
        white_now = self.compute_white_ratio(hsv)
        prev_w = self.white_ratio.get(gid, white_now)
        smoothed = (1.0 - self.white_alpha) * prev_w + self.white_alpha * white_now
        self.white_ratio[gid] = smoothed

        def rng(name):
            return np.array(
                [
                    float(v)
                    for v in self.get_parameter(name).value
                ],
                dtype=np.uint8,
            )

        mask_cube = cv2.inRange(
            hsv,
            rng('cube_hsv_lo'),
            rng('cube_hsv_hi'),
        )
        mask_cyl = cv2.bitwise_or(
            cv2.inRange(
                hsv,
                rng('cyl_hsv_lo'),
                rng('cyl_hsv_hi'),
            ),
            cv2.inRange(
                hsv,
                rng('cyl_hsv_lo2'),
                rng('cyl_hsv_hi2'),
            ),
        )

        kernel = np.ones((3, 3), np.uint8)
        mask_cube = cv2.morphologyEx(
            mask_cube,
            cv2.MORPH_OPEN,
            kernel,
        )
        mask_cyl = cv2.morphologyEx(
            mask_cyl,
            cv2.MORPH_OPEN,
            kernel,
        )

        # ---------------- 空白闸门 ----------------
        fg = cv2.bitwise_or(mask_cube, mask_cyl)
        fg = cv2.morphologyEx(
            fg,
            cv2.MORPH_CLOSE,
            kernel,
        )
        fill = float(cv2.countNonZero(fg)) / float(max(w * h, 1))
        blank = 1.0 - fill
        self.blank_ratio[gid] = blank

        if bool(self.get_parameter('empty_check').value):
            # 强空判据 1: 平滑后的白色占比 > 85% (核心, 稳定)
            white_limit = float(
                self.get_parameter('empty_white_ratio').value)
            if smoothed > white_limit:
                return None
            # 弱空判据 2: 色掩码补集 (保底)
            limit = float(
                self.get_parameter('empty_blank_ratio').value
            )
            if blank > limit:
                return None

        min_area = float(
            self.get_parameter('color_min_area_ratio').value
        ) * w * h
        best = None

        for mask, kind in (
            (mask_cube, CUBE),
            (mask_cyl, CYL),
        ):
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                if best is None or area > best[1]:
                    bx, by, bw, bh = cv2.boundingRect(contour)
                    best = (
                        kind,
                        area,
                        (bx, by, bw, bh),
                    )

        if best is None:
            return self.classify_shape(patch, min_area)

        kind, area, rect = best
        score = min(
            1.0,
            area / float(w * h) * 4.0,
        )
        return kind, score, rect

    def classify_shape(self, patch, min_area):
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < min_area:
            return None

        bx, by, bw, bh = cv2.boundingRect(contour)
        extent = area / float(max(bw * bh, 1))
        threshold = float(
            self.get_parameter('shape_extent_cube').value
        )
        kind = CUBE if extent >= threshold else CYL
        return (kind, float(min(1.0, extent)), (bx, by, bw, bh))

    # ---------------------------------------------------------- 主回调

    def on_image(self, msg):
        if msg.encoding not in ('bgr8', 'rgb8'):
            self.get_logger().warn(
                f'暂不支持的图像编码 {msg.encoding}, 需要 bgr8 / rgb8'
            )
            return

        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3,
        )
        if msg.encoding == 'rgb8':
            img = img[:, :, ::-1]
        img = np.ascontiguousarray(img)

        height, width = img.shape[:2]
        yolo_boxes = (
            self.run_yolo(img)
            if self.backend == 'yolo'
            else []
        )

        detections = []
        results = {}

        for gid in range(1, self.n_grid + 1):
            x0, y0, x1, y1 = to_pixels(
                self.rois[gid - 1], width, height,
            )
            kind = EMPTY
            label = ''
            score = 0.0
            bbox = None

            if self.backend == 'yolo':
                best = None
                for box in yolo_boxes:
                    cx, cy = box[0], box[1]
                    if x0 <= cx < x1 and y0 <= cy < y1:
                        if best is None or box[8] > best[8]:
                            best = box
                if best is not None:
                    kind = best[6]
                    label = best[7]
                    score = best[8]
                    bbox = (best[2], best[3], best[4], best[5])

            elif self.backend == 'color':
                out = self.classify_color(img[y0:y1, x0:x1], gid)
                if out is not None:
                    kind, score, rect = out
                    label = kind
                    bbox = (
                        x0 + rect[0],
                        y0 + rect[1],
                        x0 + rect[0] + rect[2],
                        y0 + rect[1] + rect[3],
                    )

            if self.backend != 'yolo' or self.yolo_fresh:
                self.push_vote(gid, kind, label, score)

            self.update_stable_display(gid, bbox)

            with self.lock:
                shown_kind = self.display[gid]['kind']
                shown_label = self.display[gid]['label']
                shown_score = self.display[gid]['score']
                shown_bbox = self.display[gid]['bbox']

            results[gid] = (
                shown_kind,
                shown_label,
                shown_score,
                shown_bbox,
                (x0, y0, x1, y1),
            )

            if shown_bbox is not None:
                detections.append((
                    gid,
                    shown_kind,
                    shown_label,
                    shown_score,
                    shown_bbox,
                ))

        self.publish_detections(msg.header, detections)

        if self.want_annotated:
            self.publish_annotated(msg.header, img, results)

        self.frames += 1
        now = time.monotonic()
        if self.log_period > 0 and now - self.last_log > self.log_period:
            self.last_log = now
            with self.lock:
                text = ' '.join(
                    f'G{g}={self.state[g]["class"][:4]}'
                    f'(w={self.white_ratio.get(g, 0.0):.2f})'
                    for g in range(1, self.n_grid + 1)
                )
            self.get_logger().info(f'[vision] {text}')

    def push_vote(self, gid, kind, label, score):
        """滑动窗口投票 + 非对称迟滞。

        关键改动：稳定后要"改主意"（切到其它类别或 empty），
        需要 switch_min_votes 张票 (默认接近全票)，而不再是刚过半的
        min_votes。这样 blank_ratio 在阈值附近抖动时不会让稳定态翻转。
        """
        with self.lock:
            buf = self.votes[gid]
            buf.append((kind, label, float(score)))
            while len(buf) > self.vote_window:
                buf.pop(0)

            counts = {}
            for k, _, _ in buf:
                counts[k] = counts.get(k, 0) + 1

            if not counts:
                return

            old = self.state[gid]

            if old['stable']:
                # 已稳定：查当前 stable_kind 是否被推翻。
                cur_kind = old['class']
                cur_count = counts.get(cur_kind, 0)
                # 找出票数最多的"其它"类别
                other = None
                other_count = -1
                for k, c in counts.items():
                    if k == cur_kind:
                        continue
                    if c > other_count:
                        other = k
                        other_count = c

                # 若其它类别没到 switch 门槛, 保持现状 —— 不刷新 label/score
                if other is None or other_count < self.switch_min_votes:
                    return

                # 走到这里：确实要切到 other 类别 (可能是 EMPTY 也可能是另一物体)
                winner_kind = other
                winner_count = other_count
            else:
                # 还没稳定过：用 min_votes 阈值首次锁定
                winner_kind, winner_count = max(
                    counts.items(),
                    key=lambda kv: kv[1],
                )
                if winner_count < self.min_votes:
                    return

            winner_scores = [s for k, _, s in buf if k == winner_kind]
            winner_labels = [ln for k, ln, _ in buf
                             if k == winner_kind and ln]

            new_score = (
                float(np.mean(winner_scores))
                if winner_scores else 0.0
            )
            new_label = winner_labels[-1] if winner_labels else ''

            self.state[gid] = {
                'class': winner_kind,
                'label': new_label,
                'score': new_score,
                'stable': True,
                'votes': int(winner_count),
            }

    def update_stable_display(self, gid, current_bbox):
        """稳定后锁定 bbox / label / score, 后续同类不动。"""
        with self.lock:
            state = self.state[gid]
            display = self.display[gid]

            if not state['stable']:
                return

            stable_kind = state['class']

            if stable_kind == EMPTY:
                # 只有真的切到 EMPTY 稳定态时才清框, 平时不动
                if display['kind'] != EMPTY:
                    display['kind'] = EMPTY
                    display['label'] = ''
                    display['score'] = 0.0
                    display['bbox'] = None
                return

            # 第一次确认该类别，锁定当前 bbox 和稳定 score
            if display['kind'] != stable_kind:
                if current_bbox is not None:
                    display['kind'] = stable_kind
                    display['label'] = state['label']
                    display['score'] = float(state['score'])
                    display['bbox'] = tuple(current_bbox)
                return

            # 同一稳定类别：完全保持现有显示，不跟随当前帧微抖
            if display['bbox'] is None and current_bbox is not None:
                display['label'] = state['label']
                display['score'] = float(state['score'])
                display['bbox'] = tuple(current_bbox)

    # ---------------------------------------------------------- 发布

    def publish_detections(self, header, detections):
        if self.det_pub is None:
            return

        array = Detection2DArray()
        array.header = header

        for gid, kind, label, score, bbox in detections:
            det = Detection2D()
            det.header = header
            det.bbox = BoundingBox2D()

            x0, y0, x1, y1 = bbox
            fill_bbox(
                det.bbox,
                0.5 * (x0 + x1),
                0.5 * (y0 + y1),
                x1 - x0,
                y1 - y0,
            )
            det.results = [make_hypothesis(label or kind, score)]
            try:
                det.id = f'grid{gid}'
            except Exception:
                pass
            array.detections.append(det)

        self.det_pub.publish(array)

    def publish_annotated(self, header, img, results):
        view = img.copy()

        for gid, (kind, label, score, bbox, roi) in results.items():
            x0, y0, x1, y1 = roi
            color = {
                CUBE: COLOR_CUBE,
                CYL: COLOR_CYL,
            }.get(kind, COLOR_EMPTY)

            cv2.rectangle(view, (x0, y0), (x1, y1), COLOR_GRID, 1)
            cv2.rectangle(view, (x0, y0), (x1, y1), color, 2)

            shown = kind
            shown_score = score
            with self.lock:
                stable = self.state[gid]['stable']

            # 顶栏：G{id}:{类别} {score} (label)  + 白色占比
            wr = self.white_ratio.get(gid, 0.0)
            tag = f'G{gid}:{shown}'
            if shown != EMPTY:
                tag += f' {shown_score:.2f}'
                if label and label != shown:
                    tag += f' ({label})'
            tag += f' w{wr:.2f}'
            if not stable:
                tag += '?'

            cv2.rectangle(
                view,
                (x0, y0),
                (x0 + 8 + 7 * len(tag), y0 + 16),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                view, tag, (x0 + 4, y0 + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42, color, 1, cv2.LINE_AA,
            )

            if bbox is not None:
                bx0, by0, bx1, by1 = [int(v) for v in bbox]
                cv2.rectangle(
                    view, (bx0, by0), (bx1, by1),
                    (0, 255, 0), 1,
                )

        cv2.putText(
            view,
            f'backend={self.backend}',
            (6, view.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (0, 255, 0), 1, cv2.LINE_AA,
        )

        if abs(self.annot_scale - 1.0) > 1e-3:
            view = cv2.resize(
                view, None,
                fx=self.annot_scale, fy=self.annot_scale,
                interpolation=cv2.INTER_NEAREST,
            )

        out = Image()
        out.header = header
        out.height, out.width = view.shape[:2]
        out.encoding = 'bgr8'
        out.is_bigendian = 0
        out.step = int(out.width * 3)
        out.data = np.ascontiguousarray(view).tobytes()
        self.annot_pub.publish(out)

    def state_json(self):
        with self.lock:
            grids = {
                str(g): dict(self.state[g])
                for g in range(1, self.n_grid + 1)
            }

        for g, entry in grids.items():
            entry['blank'] = round(
                self.blank_ratio.get(int(g), 1.0), 3,
            )
            # 新增: 平滑后的白色占比, 供 pick_node 判空
            entry['white'] = round(
                self.white_ratio.get(int(g), 1.0), 3,
            )

        return json.dumps(
            {
                'stamp': time.time(),
                'seq': self.seq,
                'frames': self.frames,
                'backend': self.backend,
                'empty_white_ratio': float(
                    self.get_parameter('empty_white_ratio').value),
                'grids': grids,
            },
            ensure_ascii=False,
        )

    def publish_state(self):
        self.seq += 1
        msg = String()
        msg.data = self.state_json()
        self.state_pub.publish(msg)

    def on_refresh(self, request, response):
        with self.lock:
            for gid in self.votes:
                self.votes[gid] = []

            for gid in self.state:
                self.state[gid] = {
                    'class': EMPTY,
                    'label': '',
                    'score': 0.0,
                    'stable': False,
                    'votes': 0,
                }

            for gid in self.display:
                self.display[gid] = {
                    'kind': EMPTY,
                    'label': '',
                    'score': 0.0,
                    'bbox': None,
                }

            # 白色比例的 EMA 也重置, 避免旧值污染刚清空的判决
            for gid in self.white_ratio:
                self.white_ratio[gid] = 1.0

        self.get_logger().info(
            '[vision] 收到 /vision/refresh, 投票和显示结果已清空'
        )
        response.success = True
        response.message = self.state_json()
        return response


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VisionNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as error:
        print(f'vision_node 异常: {error}', file=sys.stderr)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

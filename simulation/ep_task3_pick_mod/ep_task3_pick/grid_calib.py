#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""六网格 ROI 标定小工具.

订阅 /top_camera/image_raw, 弹一个窗口, 你在画面上点 4 个角
(顺序: 左上 → 右上 → 右下 → 左下, 圈住整个桌台区域),
它自动切成 2 行 x 3 列 = 6 个网格, 并把可以直接粘进
config/vision.yaml 的 grid_rois 打印出来.

用法:
    ros2 run ep_task3_pick grid_calib
    ros2 run ep_task3_pick grid_calib --ros-args -p rows:=2 -p cols:=3

按键:
    鼠标左键  点角点 (4 个)
    r         清空重来
    p         打印 YAML
    s         保存到 ./grid_rois.yaml
    1..6      高亮某个网格, 确认编号对不对
    q / ESC   退出
"""

import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

WINDOW = 'grid_calib  (click 4 corners: TL -> TR -> BR -> BL)'


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


class GridCalib(Node):

    def __init__(self):
        super().__init__('grid_calib')

        self.declare_parameter('image_topic', '/top_camera/image_raw')
        self.declare_parameter('rows', 2)
        self.declare_parameter('cols', 3)
        self.declare_parameter('shrink', 0.10)
        self.declare_parameter('scale', 1.5)
        self.declare_parameter('output', 'grid_rois.yaml')

        self.rows = int(self.get_parameter('rows').value)
        self.cols = int(self.get_parameter('cols').value)
        self.shrink = float(self.get_parameter('shrink').value)
        self.scale = float(self.get_parameter('scale').value)
        self.output = str(self.get_parameter('output').value)

        self.frame = None
        self.corners = []
        self.highlight = -1

        topic = str(self.get_parameter('image_topic').value)
        self.create_subscription(
            Image, topic, self.on_image, qos_profile_sensor_data)

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self.on_mouse)

        self.create_timer(0.03, self.on_tick)
        self.get_logger().info(
            f'grid_calib 启动, 订阅 {topic}. '
            f'点 4 个角 (左上→右上→右下→左下), 按 p 打印 YAML')

    # ---------------------------------------------------------- 输入

    def on_image(self, msg):
        if msg.encoding not in ('bgr8', 'rgb8'):
            return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        if msg.encoding == 'rgb8':
            img = img[:, :, ::-1]
        self.frame = np.ascontiguousarray(img)

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self.corners) >= 4:
            self.corners = []
        self.corners.append((x / self.scale, y / self.scale))
        self.get_logger().info(
            f'角点 {len(self.corners)}/4: '
            f'({x / self.scale:.0f}, {y / self.scale:.0f})')

    # ---------------------------------------------------------- 计算

    def cells(self):
        """双线性插值把四边形切成 rows x cols 个格子, 返回像素级 ROI."""
        if len(self.corners) != 4:
            return []
        tl, tr, br, bl = self.corners
        out = []
        for r in range(self.rows):
            for c in range(self.cols):
                u0 = c / float(self.cols)
                u1 = (c + 1) / float(self.cols)
                v0 = r / float(self.rows)
                v1 = (r + 1) / float(self.rows)

                pts = []
                for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
                    top = lerp(tl, tr, u)
                    bot = lerp(bl, br, u)
                    pts.append(lerp(top, bot, v))

                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                mx = (x1 - x0) * self.shrink * 0.5
                my = (y1 - y0) * self.shrink * 0.5
                out.append((x0 + mx, y0 + my, x1 - mx, y1 - my))
        return out

    def yaml_text(self):
        if self.frame is None or len(self.corners) != 4:
            return '# 还没点满 4 个角'
        h, w = self.frame.shape[:2]
        lines = ['    grid_rois:']
        for i, (x0, y0, x1, y1) in enumerate(self.cells(), 1):
            lines.append(
                f'      - {x0 / w:.4f}\n'
                f'      - {y0 / h:.4f}\n'
                f'      - {x1 / w:.4f}\n'
                f'      - {y1 / h:.4f}    # Grid {i}')
        return '\n'.join(lines)

    # ---------------------------------------------------------- 显示

    def on_tick(self):
        if self.frame is None:
            blank = np.zeros((240, 480, 3), np.uint8)
            cv2.putText(blank, 'waiting for /top_camera/image_raw ...',
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 200, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, blank)
            cv2.waitKey(1)
            return

        view = self.frame.copy()

        for i, (x, y) in enumerate(self.corners):
            cv2.circle(view, (int(x), int(y)), 4, (0, 255, 255), -1)
            cv2.putText(view, str(i + 1), (int(x) + 6, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if len(self.corners) == 4:
            pts = np.array([[int(p[0]), int(p[1])] for p in self.corners])
            cv2.polylines(view, [pts], True, (0, 255, 255), 1)

            for i, (x0, y0, x1, y1) in enumerate(self.cells(), 1):
                color = (0, 255, 0) if i != self.highlight else (0, 0, 255)
                thick = 1 if i != self.highlight else 3
                cv2.rectangle(view, (int(x0), int(y0)), (int(x1), int(y1)),
                              color, thick)
                cv2.putText(view, f'G{i}', (int(x0) + 3, int(y0) + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                            cv2.LINE_AA)
        else:
            cv2.putText(view,
                        f'click corners {len(self.corners)}/4 '
                        f'(TL,TR,BR,BL)',
                        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 200, 255), 1, cv2.LINE_AA)

        if abs(self.scale - 1.0) > 1e-3:
            view = cv2.resize(view, None, fx=self.scale, fy=self.scale,
                              interpolation=cv2.INTER_NEAREST)

        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            raise SystemExit(0)
        if key == ord('r'):
            self.corners = []
            self.highlight = -1
        if key == ord('p'):
            print('\n# ---- 粘到 config/vision.yaml 的 ros__parameters 下 ----')
            print(self.yaml_text())
            print()
        if key == ord('s'):
            with open(self.output, 'w', encoding='utf-8') as handle:
                handle.write('vision_node:\n  ros__parameters:\n')
                handle.write(self.yaml_text())
                handle.write('\n')
            self.get_logger().info(f'已保存 {self.output}')
        if ord('1') <= key <= ord('6'):
            self.highlight = key - ord('0')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GridCalib()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as error:
        print(f'grid_calib 异常: {error}', file=sys.stderr)
    finally:
        cv2.destroyAllWindows()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TopCamera 取流节点.

把 CoppeliaSim 场景里的 Vision sensor (俗称 TopCamera) 通过
ZMQ Remote API 抓下来, 发布成标准 ROS2 话题:

    /top_camera/image_raw     sensor_msgs/Image        (bgr8)
    /top_camera/camera_info   sensor_msgs/CameraInfo

这个节点 **只负责取流**, 不做任何识别, 也不参与运动规划.

为什么不用 robomaster_ros 的 camera 模块?
    robomaster_ros 发布的是小车自己头上的相机 (/camera/image_raw),
    走的是 robomaster_sim 插件的 UDP 视频流.
    TopCamera 是你在场景里自己加的一个独立 Vision sensor,
    robomaster_sim 不认识它, 所以只能用 CoppeliaSim 自带的
    ZMQ Remote API (默认端口 23000) 直接去读.

依赖:
    pip3 install coppeliasim-zmqremoteapi-client numpy

运行:
    ros2 run ep_task3_pick top_camera_node
    ros2 run ep_task3_pick top_camera_node --ros-args -p sensor_path:=/TopCamera
    ros2 run ep_task3_pick top_camera_node --ros-args -p list_sensors:=true
"""

import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def import_remote_api():
    """兼容新老两种 ZMQ Remote API 客户端写法."""
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        return RemoteAPIClient
    except ImportError:
        pass
    try:
        from zmqRemoteAPI import RemoteAPIClient
        return RemoteAPIClient
    except ImportError:
        return None


class TopCameraNode(Node):

    def __init__(self):
        super().__init__('top_camera_node')

        self.declare_parameter('host', '127.0.0.1')
        self.declare_parameter('port', 23000)
        self.declare_parameter('sensor_path', '/TopCamera')
        self.declare_parameter('sensor_name_hint', 'top')
        self.declare_parameter('frame_id', 'top_camera_optical_frame')
        self.declare_parameter('publish_rate', 15.0)
        self.declare_parameter('flip_vertical', True)
        self.declare_parameter('flip_horizontal', False)
        self.declare_parameter('rgb_to_bgr', True)
        self.declare_parameter('handle_explicitly', False)
        self.declare_parameter('publish_camera_info', True)
        self.declare_parameter('reconnect_period', 3.0)
        self.declare_parameter('list_sensors', False)

        self.host = str(self.get_parameter('host').value)
        self.port = int(self.get_parameter('port').value)
        self.sensor_path = str(self.get_parameter('sensor_path').value)
        self.name_hint = str(self.get_parameter('sensor_name_hint').value).lower()
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.rate = float(self.get_parameter('publish_rate').value)
        self.flip_v = bool(self.get_parameter('flip_vertical').value)
        self.flip_h = bool(self.get_parameter('flip_horizontal').value)
        self.to_bgr = bool(self.get_parameter('rgb_to_bgr').value)
        self.explicit = bool(self.get_parameter('handle_explicitly').value)
        self.want_info = bool(self.get_parameter('publish_camera_info').value)
        self.reconnect_period = float(self.get_parameter('reconnect_period').value)

        self.sim = None
        self.client = None
        self.sensor = -1
        self.res = (0, 0)
        self.info_msg = None
        self.fail_count = 0
        self.frame_count = 0
        self.last_report = time.monotonic()

        self.image_pub = self.create_publisher(
            Image, '/top_camera/image_raw', qos_profile_sensor_data)
        self.info_pub = self.create_publisher(
            CameraInfo, '/top_camera/camera_info', qos_profile_sensor_data)

        if not self.connect():
            self.get_logger().warn('首次连接 CoppeliaSim 失败, 将持续重试')

        if bool(self.get_parameter('list_sensors').value):
            self.list_sensors()
            raise SystemExit(0)

        period = 1.0 / max(self.rate, 0.1)
        self.timer = self.create_timer(period, self.on_timer)
        self.get_logger().info(
            f'TopCamera 取流节点启动, {self.rate:.1f} Hz → /top_camera/image_raw')

    # ---------------------------------------------------------- 连接

    def connect(self):
        cls = import_remote_api()
        if cls is None:
            self.get_logger().error(
                '找不到 ZMQ Remote API 客户端, 请先执行:\n'
                '  pip3 install coppeliasim-zmqremoteapi-client')
            return False

        try:
            self.client = cls(self.host, self.port)
            if hasattr(self.client, 'require'):
                self.sim = self.client.require('sim')
            else:
                self.sim = self.client.getObject('sim')
        except Exception as error:
            self.get_logger().warn(
                f'连接 CoppeliaSim {self.host}:{self.port} 失败: {error}')
            self.sim = None
            return False

        self.sensor, path = self.resolve_sensor()
        if self.sensor < 0:
            return False

        try:
            res = self.sim.getVisionSensorResolution(self.sensor)
            self.res = (int(res[0]), int(res[1]))
        except Exception:
            self.res = (0, 0)

        self.get_logger().info(
            f'✓ 已连接 CoppeliaSim, 使用视觉传感器 {path}, 分辨率={self.res}')
        self.info_msg = None
        return True

    def resolve_sensor(self):
        """优先按路径; 找不到就按名字关键字; 再不行就用第一个."""
        if self.sensor_path:
            try:
                handle = self.sim.getObject(self.sensor_path)
                return handle, self.sensor_path
            except Exception:
                self.get_logger().warn(
                    f'场景里找不到 {self.sensor_path}, 尝试按名字匹配 '
                    f'"{self.name_hint}"')

        best = (-1, '')
        first = (-1, '')
        idx = 0
        while True:
            try:
                handle = self.sim.getObjects(
                    idx, self.sim.object_visionsensor_type)
            except Exception:
                break
            if handle is None or handle < 0:
                break
            path = self.alias_of(handle)
            if first[0] < 0:
                first = (handle, path)
            if self.name_hint and self.name_hint in path.lower():
                best = (handle, path)
                break
            idx += 1

        if best[0] >= 0:
            return best
        if first[0] >= 0:
            self.get_logger().warn(
                f'没有名字含 "{self.name_hint}" 的传感器, 退回第一个: {first[1]}')
            return first

        self.get_logger().error(
            '场景里一个 Vision sensor 都没有. '
            '请在 CoppeliaSim 里加一个俯视的 Vision sensor 并命名为 TopCamera')
        return -1, ''

    def alias_of(self, handle):
        try:
            return self.sim.getObjectAlias(handle, 2)
        except Exception:
            try:
                return self.sim.getObjectAlias(handle)
            except Exception:
                return f'<handle {handle}>'

    def list_sensors(self):
        if self.sim is None:
            print('未连接 CoppeliaSim')
            return
        print('场景里的 Vision sensor:')
        idx = 0
        found = 0
        while True:
            try:
                handle = self.sim.getObjects(
                    idx, self.sim.object_visionsensor_type)
            except Exception:
                break
            if handle is None or handle < 0:
                break
            try:
                res = self.sim.getVisionSensorResolution(handle)
            except Exception:
                res = '?'
            print(f'  [{idx}] {self.alias_of(handle)}  分辨率={res}')
            found += 1
            idx += 1
        if found == 0:
            print('  (一个都没有. 注意类型必须是 Vision sensor, 不是普通 Camera)')

    # ---------------------------------------------------------- 取图

    def grab(self):
        if self.explicit:
            try:
                self.sim.handleVisionSensor(self.sensor)
            except Exception:
                pass

        buf = None
        res = None
        try:
            buf, res = self.sim.getVisionSensorImg(self.sensor)
        except AttributeError:
            buf, res = self.sim.getVisionSensorCharImage(self.sensor)

        if buf is None or res is None:
            return None

        width, height = int(res[0]), int(res[1])
        if isinstance(buf, str):
            buf = buf.encode('latin-1')

        img = np.frombuffer(buf, dtype=np.uint8)
        if img.size != width * height * 3:
            return None

        img = img.reshape(height, width, 3)
        if self.to_bgr:
            img = img[:, :, ::-1]
        if self.flip_v:
            img = img[::-1, :, :]
        if self.flip_h:
            img = img[:, ::-1, :]

        self.res = (width, height)
        return np.ascontiguousarray(img)

    # ---------------------------------------------------------- 发布

    def make_camera_info(self, width, height, stamp):
        if self.info_msg is None:
            msg = CameraInfo()
            msg.header.frame_id = self.frame_id
            msg.width = int(width)
            msg.height = int(height)
            msg.distortion_model = 'plumb_bob'
            msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

            fov = 1.0472
            try:
                fov = float(self.sim.getObjectFloatParam(
                    self.sensor,
                    self.sim.visionfloatparam_perspective_angle))
            except Exception:
                pass

            longer = float(max(width, height))
            f = 0.5 * longer / max(np.tan(fov * 0.5), 1e-6)
            cx = width * 0.5
            cy = height * 0.5

            msg.k = [f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0]
            msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            msg.p = [f, 0.0, cx, 0.0, 0.0, f, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            self.info_msg = msg

        self.info_msg.header.stamp = stamp
        return self.info_msg

    def on_timer(self):
        if self.sim is None or self.sensor < 0:
            if self.fail_count % max(int(self.rate * self.reconnect_period), 1) == 0:
                self.connect()
            self.fail_count += 1
            return

        try:
            img = self.grab()
        except Exception as error:
            self.fail_count += 1
            if self.fail_count % 30 == 1:
                self.get_logger().warn(f'取图失败 ({error}), 尝试重连')
            self.sim = None
            return

        if img is None:
            self.fail_count += 1
            if self.fail_count % 60 == 1:
                self.get_logger().warn(
                    '取到空帧 —— CoppeliaSim 里点"开始仿真"了吗? '
                    '传感器是不是被遮住了?')
            return

        self.fail_count = 0
        height, width = img.shape[:2]
        stamp = self.get_clock().now().to_msg()

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.height = int(height)
        msg.width = int(width)
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = int(width * 3)
        msg.data = img.tobytes()
        self.image_pub.publish(msg)

        if self.want_info:
            self.info_pub.publish(
                self.make_camera_info(width, height, stamp))

        self.frame_count += 1
        now = time.monotonic()
        if now - self.last_report > 10.0:
            fps = self.frame_count / (now - self.last_report)
            self.get_logger().info(
                f'[TopCamera] {width}x{height} @ {fps:.1f} fps')
            self.frame_count = 0
            self.last_report = now


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TopCameraNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as error:
        print(f'top_camera_node 异常: {error}', file=sys.stderr)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

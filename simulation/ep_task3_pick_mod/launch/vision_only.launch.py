#!/usr/bin/env python3
"""第一步用的 launch: 只起视觉, 先把视频流和 6 个网格看清楚.

前提: CoppeliaSim 已经手动打开 final.ttt 并且 **已经点了开始仿真**.

用法:
    # 只画网格, 不识别 (确认视频流 + 网格位置)
    ros2 launch ep_task3_pick vision_only.launch.py

    # 打开 YOLO 真实识别
    ros2 launch ep_task3_pick vision_only.launch.py backend:=yolo

    # 指定传感器路径
    ros2 launch ep_task3_pick vision_only.launch.py sensor_path:=/TopCamera

    # 不自动开 rqt_image_view
    ros2 launch ep_task3_pick vision_only.launch.py view:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('ep_task3_pick')
    vision_config = os.path.join(share, 'config', 'vision.yaml')

    args = [
        DeclareLaunchArgument('backend', default_value='none',
                              description='none / color / yolo / auto'),
        DeclareLaunchArgument('sensor_path', default_value='/TopCamera',
                              description='CoppeliaSim 里俯视相机的路径'),
        DeclareLaunchArgument('view', default_value='true',
                              description='是否自动打开 rqt_image_view'),
        DeclareLaunchArgument('config_file', default_value=vision_config),
    ]

    backend = LaunchConfiguration('backend')
    sensor_path = LaunchConfiguration('sensor_path')
    config_file = LaunchConfiguration('config_file')

    top_camera = Node(
        package='ep_task3_pick',
        executable='top_camera_node',
        name='top_camera_node',
        output='screen',
        emulate_tty=True,
        parameters=[config_file, {'sensor_path': sensor_path}],
    )

    vision = Node(
        package='ep_task3_pick',
        executable='vision_node',
        name='vision_node',
        output='screen',
        emulate_tty=True,
        parameters=[config_file, {'backend': backend}],
    )

    viewer = ExecuteProcess(
        cmd=['ros2', 'run', 'rqt_image_view', 'rqt_image_view',
             '/top_camera/image_annotated'],
        output='log',
        condition=IfCondition(LaunchConfiguration('view')),
    )

    return LaunchDescription(args + [top_camera, vision, viewer])

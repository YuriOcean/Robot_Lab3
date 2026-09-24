#!/usr/bin/env python3
"""Launch the complete RoboMaster EP object-sorting system.

YOLO is deliberately started by the project virtual-environment interpreter,
because an ament_python console script can retain /usr/bin/python3 in its
shebang even when colcon is called from an activated virtual environment.
"""

import os

from ament_index_python.packages import (
    get_package_prefix,
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    # Configuration revision: 2026-09-21-per-grid-reset-v31
    package_name = "ep_object_sorting"
    package_share = get_package_share_directory(package_name)
    package_prefix = get_package_prefix(package_name)
    robomaster_share = get_package_share_directory("robomaster_ros")

    grid_config = os.path.join(package_share, "config", "grid_mapper.yaml")
    driver_launch = os.path.join(robomaster_share, "launch", "ep.launch")
    yolo_executable = os.path.join(
        package_prefix,
        "lib",
        package_name,
        "yolo_detector",
    )

    connection_type = LaunchConfiguration("connection_type")
    object_limit = LaunchConfiguration("object_limit")
    venv_python = LaunchConfiguration("venv_python")
    observe_x = LaunchConfiguration("observe_x")
    observe_z = LaunchConfiguration("observe_z")
    pick_x = LaunchConfiguration("pick_x")
    initial_heading_correction_deg = LaunchConfiguration(
        "initial_heading_correction_deg"
    )
    cycle_heading_compensation_deg = LaunchConfiguration(
        "cycle_heading_compensation_deg"
    )
    apple_confidence_threshold = LaunchConfiguration(
        "apple_confidence_threshold"
    )
    apple_stability_frames = LaunchConfiguration(
        "apple_stability_frames"
    )
    row_spacing = LaunchConfiguration("row_spacing")
    column_spacing = LaunchConfiguration("column_spacing")
    bin_outside_margin = LaunchConfiguration("bin_outside_margin")
    chassis_x_speed = LaunchConfiguration("chassis_x_speed")
    chassis_y_speed = LaunchConfiguration("chassis_y_speed")
    maximum_chassis_speed = LaunchConfiguration("maximum_chassis_speed")
    # 六格各自两项参数，名称与格号对应，不随任务跳过而重新编号。
    reset_arguments = []
    reset_parameters = {}
    for index in range(1, 7):
        angle_name = f"g{index}_reset_angle_deg"
        forward_name = f"g{index}_reset_forward_m"
        reset_arguments.extend([
            DeclareLaunchArgument(
                angle_name, default_value=cycle_heading_compensation_deg,
                description=f"G{index}放置并回到G5后右转角度，0~15度",
            ),
            DeclareLaunchArgument(
                forward_name, default_value="0.01",
                description=f"G{index}右转后前进距离，0或0.005~0.05米",
            ),
        ])
        reset_parameters[angle_name] = ParameterValue(
            LaunchConfiguration(angle_name), value_type=float
        )
        reset_parameters[forward_name] = ParameterValue(
            LaunchConfiguration(forward_name), value_type=float
        )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "connection_type",
                default_value="ap",
                description="RoboMaster connection type: ap, sta, or rndis",
            ),
            DeclareLaunchArgument(
                "object_limit",
                default_value="6",
                description="Number of grid cells to process in task order",
            ),
            DeclareLaunchArgument(
                "venv_python",
                default_value="/home/lemon/experiment3_ws/.venv/bin/python3",
                description="Python executable containing ultralytics",
            ),
            DeclareLaunchArgument(
                "observe_x",
                default_value="0.106",
                description="Final fixed arm X position for the grid snapshot",
            ),
            DeclareLaunchArgument(
                "observe_z",
                default_value="0.030",
                description="Final fixed arm Z position for the grid snapshot",
            ),
            DeclareLaunchArgument(
                "pick_x",
                default_value="0.200",
                description=(
                    "Arm X extension used during picking, metres; hardware "
                    "software limit is 0.075 to 0.212"
                ),
            ),
            DeclareLaunchArgument(
                "initial_heading_correction_deg",
                default_value="0.0",
                description=(
                    "Optional correction from startup heading; zero locks the "
                    "robot's initial physical heading as forward"
                ),
            ),
            DeclareLaunchArgument(
                "cycle_heading_compensation_deg",
                default_value="2.0",
                description=(
                    "Clockwise correction after each placed object returns "
                    "to G5; set 0 to disable, valid range 0 to 15 degrees"
                ),
            ),
            DeclareLaunchArgument(
                "apple_confidence_threshold",
                default_value="0.15",
                description=(
                    "YOLO and grid confidence threshold used only for apple; "
                    "bottle remains at 0.25"
                ),
            ),
            DeclareLaunchArgument(
                "apple_stability_frames",
                default_value="2",
                description=(
                    "Consecutive grid frames required to confirm apple; "
                    "bottle and empty cells remain at 3"
                ),
            ),
            DeclareLaunchArgument(
                "arm_motion_step",
                default_value="0.020",
                description=(
                    "Deprecated compatibility argument; direct arm motion "
                    "does not use segmented steps"
                ),
            ),
            DeclareLaunchArgument(
                "arm_segment_cooldown",
                default_value="0.30",
                description=(
                    "Deprecated compatibility argument; direct arm motion "
                    "does not use segment cooldown"
                ),
            ),
            DeclareLaunchArgument(
                "row_spacing",
                default_value="0.25",
                description="Physical spacing between the two grid rows, metres",
            ),
            DeclareLaunchArgument(
                "column_spacing",
                default_value="0.25",
                description="Physical spacing between adjacent grid columns, metres",
            ),
            DeclareLaunchArgument(
                "bin_outside_margin",
                default_value="0.15",
                description="Distance from the outer grid column to bin entry, metres",
            ),
            DeclareLaunchArgument(
                "chassis_x_speed",
                default_value="1.0",
                description=(
                    "Requested X-axis translation speed limit in m/s; "
                    "valid range is greater than 0 and at most 1.0"
                ),
            ),
            DeclareLaunchArgument(
                "chassis_y_speed",
                default_value="1.0",
                description=(
                    "Requested Y-axis translation speed limit in m/s; "
                    "valid range is greater than 0 and at most 1.0"
                ),
            ),
            DeclareLaunchArgument(
                "maximum_chassis_speed",
                default_value="1.0",
                description=(
                    "Hard chassis translation speed limit in m/s; "
                    "valid range is at most 1.0"
                ),
            ),

            *reset_arguments,
            IncludeLaunchDescription(
                AnyLaunchDescriptionSource(driver_launch),
                launch_arguments={
                    "conn_type": connection_type,
                    # This robomaster_ros release uses integer stream modes:
                    # OFF=0, ON=1, ON_DEMAND=2, DISABLED=-1.
                    "video": "1",
                    "video_resolution": "360",
                    "video_raw": "1",
                    "video_h264": "0",
                    "video_ffmpeg": "0",
                    "audio": "0",
                    "audio_raw": "0",
                    "audio_opus": "0",
                    "audio_level": "0",
                }.items(),
            ),

            # Explicit venv Python fixes ModuleNotFoundError: ultralytics.
            # Detector defaults select yolo11n.pt and bottle/apple.
            ExecuteProcess(
                cmd=[
                    venv_python,
                    "-u",
                    yolo_executable,
                    "--ros-args",
                    "-p",
                    "input_topic:=/camera/image_color",
                    "-p",
                    [
                        "apple_confidence_threshold:=",
                        apple_confidence_threshold,
                    ],
                    "-p",
                    "bottle_confidence_threshold:=0.25",
                ],
                name="yolo_detector_process",
                output="screen",
            ),

            Node(
                package=package_name,
                executable="grid_mapper",
                name="grid_mapper",
                output="screen",
                parameters=[
                    grid_config,
                    {
                        "input_image_topic": "/camera/image_color",
                        "roi_x_min": 0,
                        "roi_y_min": 0,
                        "roi_x_max": 640,
                        "roi_y_max": 360,
                        "force_g5_bottle": True,
                        "forced_g5_confidence": 1.0,
                        "minimum_confidence": 0.25,
                        "bottle_minimum_confidence": 0.25,
                        "apple_minimum_confidence": ParameterValue(
                            apple_confidence_threshold,
                            value_type=float,
                        ),
                        "stability_frames": 3,
                        "apple_stability_frames": ParameterValue(
                            apple_stability_frames,
                            value_type=int,
                        ),
                    },
                ],
            ),

            Node(
                package=package_name,
                executable="chassis_motion",
                name="chassis_motion",
                output="screen",
                # Pass the two axis values through rcl's command-line parser.
                # The embedded YAML quotes prevent `y` from becoming Boolean true.
                ros_arguments=[
                    "-p",
                    "column_axis:='y'",
                    "-p",
                    "row_axis:='x'",
                ],
                parameters=[
                    {
                        "column_spacing": ParameterValue(
                            column_spacing, value_type=float
                        ),
                        "row_spacing": ParameterValue(
                            row_spacing, value_type=float
                        ),
                        "column_sign": 1,
                        "row_sign": 1,
                        "x_command_sign": 1,
                        "y_command_sign": -1,
                        "default_speed": 0.06,
                        "minimum_speed": 0.045,
                        "minimum_x_speed": 0.045,
                        "minimum_y_speed": 0.060,
                        "y_terminal_boost_speed": 0.080,
                        "terminal_boost_error": 0.040,
                        "terminal_boost_delay": 1.0,
                        "maximum_speed": ParameterValue(
                            maximum_chassis_speed,
                            value_type=float,
                        ),
                        "heading_kp": 4.0,
                        "maximum_angular_speed": 0.30,
                        "heading_tolerance_deg": 1.0,
                        "post_stop_position_margin": 0.005,
                        "post_stop_cross_track_tolerance": 0.035,
                        "post_stop_heading_tolerance_deg": 2.0,
                        "compensation_heading_tolerance_deg": 0.50,
                        "compensation_post_stop_tolerance_deg": 1.5,
                        "minimum_compensation_angular_speed": 0.10,
                        "maximum_heading_compensation_deg": 15.0,
                        "max_translation_heading_error_deg": 5.0,
                        "initial_heading_correction_deg": ParameterValue(
                            initial_heading_correction_deg,
                            value_type=float,
                        ),
                        "position_tolerance": 0.008,
                        "x_position_tolerance": 0.008,
                        "y_position_tolerance": 0.012,
                        "minimum_motion_timeout": 20.0,
                        "maximum_motion_time": 35.0,
                        "stall_timeout": 12.0,
                        "progress_epsilon": 0.0010,
                    }
                ],
            ),

            Node(
                package=package_name,
                executable="arm_task",
                name="arm_task",
                output="screen",
                parameters=[
                    {
                        "observe_x": ParameterValue(
                            observe_x, value_type=float
                        ),
                        "observe_z": ParameterValue(
                            observe_z, value_type=float
                        ),
                        "pick_x": ParameterValue(
                            pick_x, value_type=float
                        ),
                        "z_min": 0.005,
                        "pick_z": 0.030,
                    }
                ],
            ),

            Node(
                package=package_name,
                executable="sorting_coordinator",
                name="sorting_coordinator",
                output="screen",
                parameters=[
                    {
                        "object_limit": ParameterValue(object_limit, value_type=int),
                        **reset_parameters,
                        "start_topic": "/sorting/phase1",
                        "execute_topic": "/sorting/phase2",
                        "task_order": ["G4", "G1", "G5", "G2", "G6", "G3"],
                        "row_spacing": ParameterValue(
                            row_spacing, value_type=float
                        ),
                        "column_spacing": ParameterValue(
                            column_spacing, value_type=float
                        ),
                        "initial_grid": "G5",
                        "force_g5_bottle": True,
                        "bin_outside_margin": ParameterValue(
                            bin_outside_margin, value_type=float
                        ),
                        # chassis_speed仅为旧版本兼容参数；正式调度使用
                        # 下方两个分轴速度上限。
                        "chassis_speed": ParameterValue(
                            chassis_x_speed,
                            value_type=float,
                        ),
                        "chassis_x_speed": ParameterValue(
                            chassis_x_speed,
                            value_type=float,
                        ),
                        "chassis_y_speed": ParameterValue(
                            chassis_y_speed,
                            value_type=float,
                        ),
                        "cycle_heading_compensation_deg": ParameterValue(
                            cycle_heading_compensation_deg,
                            value_type=float,
                        ),
                        "bin_slot_count": 4,
                        "bin_first_offset": 0.30,
                        "bin_slot_spacing": 0.10,
                        "bottle_slot_y_direction": 1,
                        "apple_slot_y_direction": -1,
                        "max_chassis_segment": 0.49,
                        "chassis_timeout": 40.0,
                        "arm_timeout": 120.0,
                    }
                ],
            ),

            Node(
                package=package_name,
                executable="sorting_result_logger",
                name="sorting_result_logger",
                output="screen",
                parameters=[
                    {
                        "log_directory": (
                            "/home/lemon/experiment3_ws/result_logs"
                        ),
                        "task_order": ["G4", "G1", "G5", "G2", "G6", "G3"],
                    }
                ],
            ),

            Node(
                package="image_view",
                executable="image_view",
                name="grid_debug_view",
                output="screen",
                remappings=[("image", "/grid/debug_image")],
            ),
        ]
    )


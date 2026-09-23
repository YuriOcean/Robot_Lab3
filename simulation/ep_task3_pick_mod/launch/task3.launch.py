#!/usr/bin/env python3
"""Task3 一键启动 launch (加了视觉).

流程:
    1. 启动 CoppeliaSim + final.ttt
    2. 等 CoppeliaSim (23000) 和 robomaster_sim (40921) 就绪
    3. **只**启动 robomaster_ros 驱动 (ep.launch), 等 /move, /move_arm,
       /gripper Action 就绪 —— 这段时间严禁并发拉 top_camera_node,
       否则 ZMQ 抢死 CoppeliaSim 主线程, robomaster_sim 握手回调排不上队,
       DJI SDK scan_robot_ip 3秒硬超时, Client 初始化就报
       `DEFAULT_CONN_PROTO` 那个次生 AttributeError.
    4. Action 就绪后, 再拉起 top_camera_node + vision_node
    5. 等 /vision/grid_state 有人发布
    6. 启动 pick_node, 按 Grid 1..6 顺序执行
    7. pick_node 退出后自动关闭仿真

用法:
    ros2 launch ep_task3_pick task3.launch.py
    ros2 launch ep_task3_pick task3.launch.py backend:=yolo
    ros2 launch ep_task3_pick task3.launch.py use_vision:=false   # 退回静态表
    ros2 launch ep_task3_pick task3.launch.py view:=true          # 顺便开图像窗口
    ros2 launch ep_task3_pick task3.launch.py random:=true        # 六个网格随机布置
    ros2 launch ep_task3_pick task3.launch.py random:=true seed:=42
    ros2 launch ep_task3_pick task3.launch.py random:=true layout:=CUBE,CYL,EMPTY,CYL,CUBE,CYL
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


READY_CHECK_SIM = '''
echo "[READY] 等待 CoppeliaSim (23000) 和 robomaster_sim (40921) ..."
count=0
while true; do
    ports="$(ss -lnt 2>/dev/null)"
    if echo "$ports" | grep -qE ':23000[[:space:]]' && \\
       echo "$ports" | grep -qE ':40921[[:space:]]'; then
        echo "[READY] CoppeliaSim 与 robomaster_sim 都已就绪"
        exit 0
    fi
    sleep 1
    count=$((count + 1))
    if [ "$count" -ge 90 ]; then
        echo "[READY] ERROR: 等待 CoppeliaSim 超时"
        exit 1
    fi
done
'''

READY_CHECK_ACTIONS = '''
echo "[READY] 等待 /move, /move_arm, /gripper Action 就绪 ..."
count=0
while true; do
    actions="$(ros2 action list 2>/dev/null)"
    if echo "$actions" | grep -qx '/move' && \\
       echo "$actions" | grep -qx '/move_arm' && \\
       echo "$actions" | grep -qx '/gripper'; then
        echo "[READY] 所有 Action 服务器已就绪"
        exit 0
    fi
    sleep 1
    count=$((count + 1))
    if [ "$count" -ge 90 ]; then
        echo "[READY] ERROR: 等待 Action 超时"
        exit 1
    fi
done
'''

READY_CHECK_VISION = '''
echo "[READY] 等待 /vision/grid_state ..."
count=0
while true; do
    if ros2 topic info /vision/grid_state 2>/dev/null \\
        | grep -qE 'Publisher count: [1-9]'; then
        echo "[READY] vision_node 已在发布网格状态"
        exit 0
    fi
    sleep 1
    count=$((count + 1))
    if [ "$count" -ge 60 ]; then
        echo "[READY] WARN: 等不到视觉, pick_node 会退回静态表"
        exit 0
    fi
done
'''


def generate_launch_description():
    home = os.path.expanduser('~')

    default_scene = os.path.join(
        home, 'robomaster2/robomaster_sim/scenes/random.ttt')
    default_coppelia = os.path.join(
        home,
        'CoppeliaSim/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04/coppeliaSim.sh',
    )

    share = get_package_share_directory('ep_task3_pick')
    config_file = os.path.join(share, 'config', 'objects.yaml')
    vision_config = os.path.join(share, 'config', 'vision.yaml')

    args = [
        DeclareLaunchArgument('scene_file', default_value=default_scene),
        DeclareLaunchArgument('coppelia_executable',
                              default_value=default_coppelia),
        DeclareLaunchArgument('backend', default_value='auto',
                              description='none / color / yolo / auto'),
        DeclareLaunchArgument('sensor_path', default_value='/TopCamera'),
        DeclareLaunchArgument('use_vision', default_value='true'),
        DeclareLaunchArgument('view', default_value='false'),

        # ---- 随机布局 ----
        DeclareLaunchArgument(
            'random', default_value='false',
            description='true = 每次运行随机布置六个网格 (空<=2, 方体>=1, 圆柱>=1)'),
        DeclareLaunchArgument(
            'seed', default_value='-1',
            description='随机种子, -1 = 每次都不一样'),
        DeclareLaunchArgument(
            'layout', default_value='',
            description='手工指定布局, 如 CUBE,CYL,EMPTY,CYL,CUBE,CYL'),
        DeclareLaunchArgument(
            'recapture', default_value='false',
            description='true = 忽略旧原型, 重新记忆物体中心和尺寸'),
        DeclareLaunchArgument(
            'settle', default_value='1.5',
            description='放完物体后等待落稳的秒数'),
        DeclareLaunchArgument(
            'truth_file', default_value='/tmp/ep_task3_ground_truth.json'),
        DeclareLaunchArgument(
            'clear_mode', default_value='region',
            description='随机前清空范围: region(工作区矩形) / slots(中心点附近) / all'),
        DeclareLaunchArgument(
            'auto_start', default_value='false',
            description='仿真没在跑时自动点开始 (正常 launch 流程用不到)'),
    ]

    scene_file = LaunchConfiguration('scene_file')
    coppelia_executable = LaunchConfiguration('coppelia_executable')
    backend = LaunchConfiguration('backend')
    sensor_path = LaunchConfiguration('sensor_path')
    use_vision = LaunchConfiguration('use_vision')
    random_flag = LaunchConfiguration('random')
    seed = LaunchConfiguration('seed')
    layout = LaunchConfiguration('layout')
    recapture = LaunchConfiguration('recapture')
    settle = LaunchConfiguration('settle')
    truth_file = LaunchConfiguration('truth_file')
    clear_mode = LaunchConfiguration('clear_mode')
    auto_start = LaunchConfiguration('auto_start')

    # ---------------------------------------------------------------- 1
    coppelia_process = ExecuteProcess(
        cmd=[coppelia_executable, '-f', scene_file],
        output='log',
        emulate_tty=False,
    )

    wait_sim = ExecuteProcess(cmd=['bash', '-c', READY_CHECK_SIM],
                              output='screen')

    # ---------------------------------------------------------------- 2 驱动
    robomaster_share = get_package_share_directory('robomaster_ros')
    driver_launch_file = os.path.join(
        robomaster_share, 'launch', 'ep.launch')

    driver_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(driver_launch_file),
        launch_arguments={
            'conn_type': 'sta',
            'video_raw': '0',
            'video_h264': '0',
            'video_ffmpeg': '0',
            'audio_raw': '0',
            'audio_opus': '0',
            'audio_level': '0',
        }.items(),
    )

    # ---------------------------------------------------------------- 视觉
    # publish_rate 从 15 降到 5:
    # 640x480 RGB 单帧 ~900KB, 走 ZMQ 时会占住 CoppeliaSim 主线程数百毫秒.
    # 15Hz 会把主线程持续压满, 拖慢整个仿真, 也可能间歇性影响机器人控制回路.
    # 识别方块/圆柱这类静态目标, 5Hz 足够, 如需实时可再往上调.
    top_camera = Node(
        package='ep_task3_pick',
        executable='top_camera_node',
        name='top_camera_node',
        output='screen',
        emulate_tty=True,
        parameters=[vision_config, {
            'sensor_path': sensor_path,
            'publish_rate': 5.0,
        }],
    )

    vision_node = Node(
        package='ep_task3_pick',
        executable='vision_node',
        name='vision_node',
        output='screen',
        emulate_tty=True,
        parameters=[vision_config, {'backend': backend}],
    )

    viewer = ExecuteProcess(
        cmd=['ros2', 'run', 'rqt_image_view', 'rqt_image_view',
             '/top_camera/image_annotated'],
        output='log',
        condition=IfCondition(LaunchConfiguration('view')),
    )

    # ---------------------------------------------------------------- 随机布局
    # 一个"闸门"进程: random:=true 就跑 scene_randomizer, 否则只删旧真值文件.
    # 无论哪种情况它都会退出, 后面的事件链不受影响.
    randomize_gate = ExecuteProcess(
        cmd=['bash', '-c', [
            'TRUTH="', truth_file, '"; rm -f "$TRUTH";\n',
            'RAND="', random_flag, '";\n',
            'LAYOUT="', layout, '";\n',
            'if [ "$RAND" = "true" ] || [ "$RAND" = "True" ] '
            '|| [ "$RAND" = "1" ]; then\n',
            '  EXTRA="";\n',
            '  if [ -n "$LAYOUT" ]; then EXTRA="-p layout:=$LAYOUT"; fi;\n',
            '  echo "[RANDOM] 随机布置六个网格 ...";\n',
            '  ros2 run ep_task3_pick scene_randomizer --ros-args',
            ' -p mode:=random',
            ' -p seed:=', seed,
            ' -p recapture:=', recapture,
            ' -p settle:=', settle,
            ' -p clear_mode:=', clear_mode,
            ' -p auto_start:=', auto_start,
            ' -p truth_file:="$TRUTH" $EXTRA',
            ' || echo "[RANDOM] ######## 随机化失败! '
            '本次仍是场景原始布局, 且不做视觉校验 ########";\n',
            'else\n',
            '  echo "[RANDOM] 未启用 (random:=false), 使用场景原始布局";\n',
            'fi\n',
        ]],
        output='screen',
    )

    wait_actions = ExecuteProcess(cmd=['bash', '-c', READY_CHECK_ACTIONS],
                                  output='screen')
    wait_vision = ExecuteProcess(cmd=['bash', '-c', READY_CHECK_VISION],
                                 output='screen')

    # ---------------------------------------------------------------- 3 抓取
    pick_node = Node(
        package='ep_task3_pick',
        executable='pick_node',
        name='ep_task3_pick',
        output='screen',
        emulate_tty=True,
        parameters=[config_file, {
            'use_vision': use_vision,
            'ground_truth_file': truth_file,
        }],
    )

    # ---------------------------------------------------------------- 串联
    #
    # 关键改动 (相对旧版本):
    #
    #   旧: wait_sim ─┬─► driver_launch
    #                ├─► top_camera        ← 和握手并发, 会踩死主线程
    #                ├─► vision_node
    #                ├─► viewer
    #                └─► wait_actions
    #
    #   新: wait_sim ──► driver_launch + wait_actions
    #       wait_actions ──► top_camera + vision_node + viewer + wait_vision
    #       wait_vision  ──► pick_node
    #
    # 让 top_camera 严格晚于 "Action 已经就绪" 这个事件, 也就是驱动侧握手
    # 已经完全跑完, 之后 ZMQ 再怎么占主线程都不会破坏协议连接了.

    start_driver = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_sim,
            on_exit=[driver_launch, wait_actions],
        )
    )

    start_randomize = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_actions,
            on_exit=[randomize_gate],
        )
    )

    start_vision_pipeline = RegisterEventHandler(
        OnProcessExit(
            target_action=randomize_gate,
            on_exit=[top_camera, vision_node, viewer, wait_vision],
        )
    )

    start_pick = RegisterEventHandler(
        OnProcessExit(target_action=wait_vision, on_exit=[pick_node])
    )

    shutdown_when_done = RegisterEventHandler(
        OnProcessExit(
            target_action=pick_node,
            on_exit=[EmitEvent(event=Shutdown(
                reason='Task3 抓取任务完成, 关闭全部子进程'))],
        )
    )

    return LaunchDescription(args + [
        coppelia_process,
        wait_sim,
        start_driver,
        start_randomize,
        start_vision_pipeline,
        start_pick,
        shutdown_when_done,
    ])

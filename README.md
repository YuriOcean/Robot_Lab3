# RoboMaster EP Object Sorting
基于 **ROS 2 Humble + RoboMaster EP + CoppeliaSim** 的桌面物体自动分类整理项目。
项目包含两套可独立运行的实现：
- **仿真版 `ep_task3_pick`**：使用 CoppeliaSim + ZMQ Remote API 获取顶部相机图像，完成六网格视觉判别、固定路线抓取、分类投放与状态机管理。
- **实机版 `ep_object_sorting`**：通过 `robomaster_ros` 接入 RoboMaster EP，使用 YOLO 对 `bottle / apple` 进行识别，并协调底盘、机械臂和夹爪完成分类整理。

## 1. 功能
- 2 × 3 六网格桌面布局
- 空 / 方体 / 圆柱状态判断
- 实机 YOLO `bottle / apple` 识别
- 仿真顶部相机实时取流
- 多帧投票稳定视觉结果
- 底盘与机械臂协同控制
- 抓取失败后的恢复流程
- 状态机管理完整任务流程
- 随机仿真场景生成
- 仿真视觉结果与真值自动比对
- 分类结果、异常信息和任务过程自动记录
---
## 2. 项目结构

建议将两个 ROS 2 package 放入同一个工作空间的 `src/` 目录：

```text
ros2_ws/
└── src/
    ├── ep_task3_pick/
    │   ├── ep_task3_pick/
    │   │   ├── top_camera_node.py
    │   │   ├── vision_node.py
    │   │   ├── pick_node.py
    │   │   ├── state_machine.py
    │   │   ├── scene_randomizer.py
    │   │   └── grid_calib.py
    │   ├── config/
    │   │   ├── objects.yaml
    │   │   ├── vision.yaml
    │   │   └── state_machine.yaml
    │   ├── launch/
    │   │   ├── task3.launch.py
    │   │   └── vision_only.launch.py
    │   ├── tools/
    │   │   └── make_real_objects.py
    │   └── vision_view.py
    │
    └── ep_object_sorting/
        ├── ep_object_sorting/
        │   ├── yolo_detector_node.py
        │   ├── grid_mapper_node.py
        │   ├── chassis_motion_node.py
        │   ├── arm_task_node.py
        │   ├── sorting_coordinator_node.py
        │   └── sorting_result_logger_node.py
        ├── config/
        │   ├── yolo_detector.yaml
        │   └── grid_mapper.yaml
        ├── launch/
        │   └── full_sorting_system.launch.py
        ├── models/
        │   └── yolo11n.pt
        └── scripts/
            └── run_sorting_with_spacing.sh
```
---

## 3. 软件环境

- Ubuntu 22.04
- ROS 2 Humble
- Python 3
- CoppeliaSim 4.10（仿真）
- OpenCV / NumPy / PyYAML
- RoboMaster ROS：`robomaster_ros`、`robomaster_msgs`
- Python：`coppeliasim-zmqremoteapi-client`、`ultralytics`

## 4. 编译

进入 ROS 2 工作空间：

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

只编译仿真包：

```bash
colcon build --symlink-install --packages-select ep_task3_pick
source install/setup.bash
```

只编译实机包：

```bash
colcon build --symlink-install --packages-select ep_object_sorting
source install/setup.bash
```

检查包是否安装成功：

```bash
ros2 pkg list | grep -E 'ep_task3_pick|ep_object_sorting'
```

---

# 5. 仿真运行

## 5.1 一键运行

`ep_task3_pick` 的一键启动流程为：
```bash
ros2 launch ep_task3_pick task3.launch.py
```

如果场景文件和 CoppeliaSim 安装路径不是默认位置，建议显式指定：

```bash
ros2 launch ep_task3_pick task3.launch.py \
scene_file:=/path/to/your_scene.ttt \
coppelia_executable:=/path/to/coppeliaSim.sh
```

---

## 5.2 查看视觉窗口

只启动视觉：

```bash
ros2 launch ep_task3_pick vision_only.launch.py
```

打开图像窗口：

```bash
ros2 launch ep_task3_pick vision_only.launch.py view:=true
```

当前配置默认使用颜色视觉：

```bash
ros2 launch ep_task3_pick vision_only.launch.py backend:=color
```

也保留 YOLO 后端接口：

```bash
ros2 launch ep_task3_pick vision_only.launch.py backend:=yolo
```

---

## 5.3 随机仿真场景
每次运行随机生成六格布局：

```bash
ros2 launch ep_task3_pick task3.launch.py random:=true
```
指定随机种子：

```bash
ros2 launch ep_task3_pick task3.launch.py \
random:=true seed:=42
```
指定固定布局：

```bash
ros2 launch ep_task3_pick task3.launch.py \
random:=true \
layout:=CUBE,CYL,EMPTY,CYL,CUBE,CYL
```
重新记录当前场景原型：

```bash
ros2 launch ep_task3_pick task3.launch.py \
random:=true recapture:=true
```

---

## 5.4 仿真视觉真值校验

启用随机布局后，系统会保存真实布局：

```text
/tmp/ep_task3_ground_truth.json
```

任务结束后会比较：

```text
真实布局  vs  视觉判断
```

并输出每个 Grid 的匹配情况和总体正确率。

报告默认保存到：

```text
/tmp/ep_task3_reports/
```

包括：

```text
classification_results.txt
exception_log.txt
task_log.txt
```

同时可以生成 JSON 报告。

---

## 5.5 仿真核心参数

运动参数主要位于：

```text
config/objects.yaml
```

六个 Grid 的固定位置由以下参数控制：

```yaml
grid_ids:       [1, 2, 3, 4, 5, 6]
grid_car_y:     [-0.12, 0.00, 0.12, -0.12, 0.00, 0.12]
grid_arm_x:     [0.11, 0.11, 0.11, 0.21, 0.21, 0.21]
grid_forward_x: [0.08, 0.08, 0.08, 0.08, 0.08, 0.08]
```

当前六格对应：

```text
Grid 1   (0.28, -0.12)
Grid 2   (0.28,  0.00)
Grid 3   (0.28,  0.12)

Grid 4   (0.38, -0.12)
Grid 5   (0.38,  0.00)
Grid 6   (0.38,  0.12)
```

视觉参数位于：

```text
config/vision.yaml
```

主要包括：

- 顶部相机 `/TopCamera`
- 2 × 3 网格划分
- ROI
- HSV 颜色范围
- 空格判定
- 多帧投票稳定机制
---

# 6. 实机运行

实机版本：

```text
ep_object_sorting
```

主要模块：

```text
camera/image_color
       ↓
yolo_detector
       ↓
/detections
       ↓
grid_mapper
       ↓
/grid_states
       ↓
sorting_coordinator
       ├── chassis_motion
       └── arm_task
               ↓
       sorting_result_logger
```

---

## 6.1 使用 RoboMaster ROS 驱动

先加载 ROS 环境：

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

然后检查 Action：

```bash
ros2 action list
```

正常情况下应包含：

```text
/move
/move_arm
/gripper
```

---

## 6.2 启动完整实机系统

最直接的方式：

```bash
ros2 launch ep_object_sorting full_sorting_system.launch.py
```

默认连接方式：

```text
connection_type:=ap
```

也可以显式指定：

```bash
ros2 launch ep_object_sorting full_sorting_system.launch.py \
connection_type:=ap
```

支持的连接类型由驱动提供：

```text
ap
sta
rndis
```

---

## 6.3 指定 YOLO Python 环境

YOLO 节点通过单独的 Python 解释器启动，用于确保 `ultralytics` 位于正确环境中。

例如：

```bash
ros2 launch ep_object_sorting full_sorting_system.launch.py \
connection_type:=ap \
venv_python:=/path/to/.venv/bin/python3
```

推荐先创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics
```

模型文件位于：

```text
ep_object_sorting/models/yolo11n.pt
```

当前识别目标：

```text
bottle
apple
```

---

## 6.4 实机快速测试脚本

项目提供：

```text
scripts/run_sorting_with_spacing.sh
```

用法：

```bash
bash scripts/run_sorting_with_spacing.sh ROW_M COLUMN_M [HEADING_DEG] [OBJECT_LIMIT]
```

例如：

```bash
bash scripts/run_sorting_with_spacing.sh 0.32 0.26 0 6
```

参数含义：

```text
0.32   行间距 / m
0.26   列间距 / m
0      初始航向修正 / deg
6      本次处理网格数量
```

脚本也支持通过环境变量指定工作空间：

```bash
export EXPERIMENT3_WORKSPACE=/path/to/experiment3_ws
export ROBOMASTER_ARM_WORKSPACE=/path/to/arm_ws

bash scripts/run_sorting_with_spacing.sh 0.32 0.26 0 6
```

---

# 7. 主要 ROS Topic / Action

## 仿真

图像：

```text
/top_camera/image_raw
/top_camera/image_annotated
```

视觉状态：

```text
/vision/grid_state
```

控制接口：

```text
/move
/move_arm
/gripper
```

---

## 实机

相机：

```text
/camera/image_color
```

YOLO：

```text
/detections
/yolo/annotated_image
```

网格：

```text
/grid_detections
/grid_states
/grid/debug_image
```

任务状态：

```text
/sorting/status
/arm_task/status
/chassis_motion/status
```

---

# 8. 常用调试命令

查看所有节点：

```bash
ros2 node list
```

查看所有 Topic：

```bash
ros2 topic list
```

查看视觉状态：

```bash
ros2 topic echo /vision/grid_state
```

查看实机分类状态：

```bash
ros2 topic echo /grid_states
```

查看任务状态：

```bash
ros2 topic echo /sorting/status
```

查看 Action：

```bash
ros2 action list
```

查看某个节点信息：

```bash
ros2 node info /vision_node
```

查看实时图像：

```bash
ros2 run rqt_image_view rqt_image_view
```

# 9. 运动与异常处理

系统同时对以下情况进行处理：

- Action 服务未就绪
- 视觉结果超时
- 目标位置超出软件限制
- 底盘运动超时
- 机械臂运动超时
- 抓取流程失败
- 状态机异常
- 任务结束后的安全回零

---

# 10. 参数配置位置

不建议直接修改 Python 代码中的固定参数，优先通过 YAML 或 launch 参数调整。

### 仿真

```text
ep_task3_pick/config/objects.yaml
ep_task3_pick/config/vision.yaml
ep_task3_pick/config/state_machine.yaml
```

### 实机

```text
ep_object_sorting/config/yolo_detector.yaml
ep_object_sorting/config/grid_mapper.yaml
```

---

# 11. 注意事项

### 仿真

CoppeliaSim 的 ZMQ Remote API 默认使用：

```text
127.0.0.1:23000
```

顶部视觉传感器默认：

```text
/TopCamera
```

如果场景名称或路径不同，请通过：

```bash
sensor_path:=/YourCamera
```

进行修改。

随机场景模式是在仿真运行过程中动态创建和删除物体，不需要修改原始 `.ttt` 场景文件。

### 实机

启动正式任务前建议确认：

```bash
ros2 action list
```

能够看到：

```text
/move
/move_arm
/gripper
```

并确认机器人处于安全、可随时急停的环境。

---

# 12. 推荐运行顺序

## 仿真验证

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 launch ep_task3_pick vision_only.launch.py view:=true
```

确认相机和六网格正常后，再运行：

```bash
ros2 launch ep_task3_pick task3.launch.py
```

随机测试：

```bash
ros2 launch ep_task3_pick task3.launch.py \
random:=true seed:=42 view:=true
```

---

## 实机验证

先确认 ROS 驱动和 Action：

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 action list
```

然后启动完整系统：

```bash
ros2 launch ep_object_sorting full_sorting_system.launch.py \
connection_type:=ap \
venv_python:=/path/to/.venv/bin/python3
```

需要改变网格间距或测试数量时：

```bash
bash scripts/run_sorting_with_spacing.sh 0.32 0.26 0 6
```

---

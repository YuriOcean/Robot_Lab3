# 随机布局 + 视觉真值校验

## 改了什么

| 文件 | 改动 |
|---|---|
| `ep_task3_pick/scene_randomizer.py` | **新增**。记住六个网格上物体的中心和尺寸，并按约束随机重建 |
| `launch/task3.launch.py` | 新增 `random / seed / layout / recapture / settle / truth_file` 参数，事件链插入随机化闸门 |
| `ep_task3_pick/pick_node.py` | 新增真值校验：逐格对拍「视觉判断 vs 真实布局」，结束时打印准确率 |
| `setup.py` | 注册 `scene_randomizer` 可执行 |
| `config/objects.yaml` | 新增 `verify_vision / ground_truth_file / truth_max_age` |

运动参数、固定路线、视觉后端、投放逻辑**一个都没动**。

## 执行顺序（v2）

仿真跑起来之后，随机化按这个顺序做，全部通过 ZMQ 在内存里完成，`.ttt` 文件不受影响：

1. 检查仿真是否在运行 —— **没运行就直接报错退出**（否则新建的物体会被存进场景文件）
2. 读取/记忆原型：六个网格的中心 `(x, y, z)` 和包围盒尺寸
3. 克隆一个方体 + 一个圆柱当模板，藏到 `(5, 5, 1)` 且可见层设 0，相机拍不到
4. **清空工作区**：把六个网格张成的矩形 + 12 cm 余量内的旧物体全部删掉
5. 按随机布局重新生成，空格子什么都不放
6. 自检一遍：重新扫描场景，确认每格实际内容 == 目标布局
7. 写真值文件

空格子的处理：`vision_node` 判为 `empty` → `pick_node` 直接 `continue` 跳过，机械臂完全不动。
随机布局下如果视觉超时拿不到结果，也**一律当空跳过**，不会退回 `obj_kind` 静态表
（那张表在随机布局下是错的，照着抓会去抓空格子）。只有显式 `use_vision:=false` 时才按真值表执行。

## 空白判定（只作用于 backend=color）

ROI 里既不是蓝也不是红的像素算「空白」。空白超过 `empty_blank_ratio`（默认 0.80）直接判空，
**并且不再退回 `classify_shape` 去猜方圆**——颜色匹配不上时原来会用 Otsu 阈值硬猜，
空桌面上的阴影最容易被它猜成物体，这是空格误检的主要来源。

标注窗口每个框上会多出一个 `b73%` 的读数，就是该格的实时空白率；
`/vision/grid_state` 里每格也多了 `blank` 字段。调阈值就看这个数：

```bash
ros2 launch ep_task3_pick task3.launch.py random:=true backend:=color view:=true
ros2 topic echo /vision/grid_state --once     # 看每格 blank
```

**阈值怎么选**：空格子的 blank 接近 1.00，有物体的格子取决于物体占 ROI 的比例。
离线实测，同样大小的 ROI 里蓝方块 blank≈0.64，而红圆柱因为是圆的只有 ≈0.82 ——
已经越过 0.80 了。所以如果你的圆柱在 ROI 里偏小，务必把阈值放宽：

```yaml
    empty_blank_ratio: 0.90      # 或者在 vision.yaml 里调小 roi_shrink 让框更贴身
```

取值原则：**比「空格子的 blank」小一点，比「最小的那个物体的 blank」大一点**。
两者拉不开就先调 `roi_shrink` 把框收紧，让物体在框里占比更高。

## 颜色约定

生成的物体颜色**写死**为「蓝方块 + 红圆柱」，和 `vision.yaml` 里已有的 HSV 段一一对应：

| | 颜色 | RGB 参数 | vision.yaml 对应段 |
|---|---|---|---|
| 方体 | 蓝 | `cube_color: [0.05, 0.25, 0.95]` | `cube_hsv` H 100~130 |
| 圆柱 | 红 | `cyl_color: [0.95, 0.12, 0.08]` | `cyl_hsv` H 0~10 + 170~180 |

克隆出来的物体会被强制刷色并清掉贴图，所以即使场景里原来混过苹果/瓶子也不会串色。
高光被压到 0.1，避免反光把饱和度冲淡导致 HSV 判别发虚。

如果克隆源是组合体导致上色失败（日志会警告），换成直接建基本几何体：

```bash
ros2 run ep_task3_pick scene_randomizer --ros-args -p mode:=random -p use_primitives:=true
```

这样生成的就是纯蓝长方体 + 纯红圆柱，尺寸仍取自记住的原型。

## 空格子是零动作

顶部相机是固定俯视的，六个网格同时在画面里，判别不需要把车开过去。所以 `pick_node`
**开局先一次性扫完六格**，打印执行计划，然后只对有物体的网格跑固定路线：

```
======== 本次执行计划 ========
  Grid 1: 方体 → 第 1 个抓取, 投方块料盒
  Grid 2: 空   → 跳过 (不靠近, 零动作)
  Grid 3: 圆柱 → 第 2 个抓取, 投圆柱料盒
  ...
  共 4 个要抓, 2 个空格跳过
=============================
```

空格子从头到尾不对准、不前进、不伸手，直接轮到下一个有物体的网格。
每个网格的路线本来就是绝对坐标（`move_y_to` 绝对 y，结束 `retreat_and_return_zero` 回零），
跳过不会影响后面网格的定位精度。

想回到「走一格判一格」的老行为：`-p prescan:=false`。

## 为什么物体不会"飞"

克隆体是在模板藏身处 `(5,5,1)` 诞生的。如果先打开动力学再瞬移到网格上，物理引擎会把这 5 米
位移当成一帧内的运动，算出几十 m/s 的初速度，物体就会飞出去撞车。所以生成流程严格按：

1. `freeze()` —— 静态 + 不参与碰撞
2. `setObjectPosition()` —— 此时是静态体，瞬移不产生速度
3. 六个格子全部摆好之后，`activate()` 统一打开动力学，并调 `resetDynamicObject()`
   让引擎按新位姿重建刚体、速度清零

相关参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `spawn_lift` | `0.002` | 落位时抬高 2 mm，避免和桌面穿模被弹开 |
| `spawn_static` | `false` | `true` = 物体完全冻死不受力（只测视觉时很方便，但抓不起来） |

只想验证视觉、不想让物体被碰倒：

```bash
ros2 run ep_task3_pick scene_randomizer --ros-args -p mode:=random -p spawn_static:=true
```

## 工作原理

1. **记忆**：第一次运行时扫描场景，量出六个网格上物体的中心 `(x, y, z)`、包围盒尺寸、颜色、类别，
   存进 `~/.ros/ep_task3_scene_proto.json`。以后所有随机布局都复用这批中心点和尺寸。
2. **随机**：在这 6 个中心点上随机放 圆柱 / 方体 / 空，约束为 空 ≤ 2、方体 ≥ 1、圆柱 ≥ 1。
   物体不是重新建模，而是把场景里原有的方体/圆柱 **克隆** 一份，所以外观、尺寸、材质完全一致，
   YOLO / color 后端照样认得出。
3. **真值**：把本次真实布局写进 `/tmp/ep_task3_ground_truth.json`。
   `pick_node` 读到后会逐格把视觉判断和真值对拍，结束时打印：

```
========== 视觉 vs 真值 ==========
Grid 1  真值=方体  视觉=方体  OK
Grid 2  真值=空    视觉=空    OK
Grid 3  真值=圆柱  视觉=方体  MISMATCH
...
❌ 视觉判别 5/6 正确 (83.3%)
```

`random:=false` 时真值文件会被删掉，`pick_node` 自动跳过校验，行为和以前完全一样。

## 编译

```bash
cd ~/ros2_ws
colcon build --packages-select ep_task3_pick --symlink-install
source install/setup.bash
```

## 命令行

### 一键跑（推荐）

```bash
# 老样子，固定布局
ros2 launch ep_task3_pick task3.launch.py

# 每次运行都随机布置六个网格
ros2 launch ep_task3_pick task3.launch.py random:=true

# 可复现：同一个种子 = 同一套布局
ros2 launch ep_task3_pick task3.launch.py random:=true seed:=42

# 随机 + YOLO + 开图像窗口
ros2 launch ep_task3_pick task3.launch.py random:=true backend:=yolo view:=true

# 手工指定布局（顺序 = Grid 1..6）
ros2 launch ep_task3_pick task3.launch.py random:=true \
    layout:=CUBE,CYL,EMPTY,CYL,CUBE,CYL

# 换了场景 / 挪过物体，强制重新记忆一次中心和尺寸
ros2 launch ep_task3_pick task3.launch.py random:=true recapture:=true
```

launch 参数一览：

| 参数 | 默认 | 说明 |
|---|---|---|
| `random` | `false` | `true` = 启用随机布局 |
| `seed` | `-1` | 随机种子，`-1` 每次不同 |
| `layout` | 空 | 手工布局，逗号分隔 6 项，给了就不随机 |
| `recapture` | `false` | 忽略旧原型，重新记忆 |
| `settle` | `1.5` | 放完物体等落稳的秒数 |
| `truth_file` | `/tmp/ep_task3_ground_truth.json` | 真值文件路径 |

### 单独跑随机化（CoppeliaSim 已在仿真中）

```bash
# 只记忆，不改场景
ros2 run ep_task3_pick scene_randomizer --ros-args -p mode:=capture

# 看看记住了什么
ros2 run ep_task3_pick scene_randomizer --ros-args -p mode:=show

# 随机一次
ros2 run ep_task3_pick scene_randomizer --ros-args -p mode:=random
ros2 run ep_task3_pick scene_randomizer --ros-args -p mode:=random -p seed:=7

# 强制恰好 2 个空
ros2 run ep_task3_pick scene_randomizer --ros-args -p mode:=random -p force_empty:=2

# 手工布局
ros2 run ep_task3_pick scene_randomizer --ros-args \
    -p mode:=random -p layout:=CYL,CUBE,CUBE,EMPTY,CYL,CYL

# 看看场景里有哪些物体、被认成什么类别（排错第一步）
ros2 run ep_task3_pick scene_randomizer --ros-args -p mode:=list

# 只清空工作区，不生成
ros2 run ep_task3_pick scene_randomizer --ros-args -p mode:=clear

# 只打印不改场景
ros2 run ep_task3_pick scene_randomizer --ros-args -p mode:=random -p dry_run:=true

# 还原成最初记住的那套布局
ros2 run ep_task3_pick scene_randomizer --ros-args -p mode:=restore
```

`layout` 的写法很宽松：`CUBE/cube/box/apple/方` → 方体；`CYL/cylinder/bottle/圆` → 圆柱；
`EMPTY/empty/none/-/空` → 空。

### 常用调参

| 参数 | 默认 | 说明 |
|---|---|---|
| `max_empty` | `2` | 最多几个空 |
| `min_cube` / `min_cyl` | `1` | 至少几个方体 / 圆柱 |
| `force_empty` | `-1` | 强制空格数，`-1` = 随机 |
| `clear_mode` | `region` | 清空范围：`region` 工作区矩形 / `slots` 中心点附近 / `all` 全场 |
| `clear_margin` | `0.12` | `region` 模式下矩形外扩多少米 |
| `slot_radius` | `0.07` | 判定「在这个网格上」的半径，物体挨得近就调小 |
| `cube_names` / `cyl_names` | 空 | 名字对不上时显式指定，如 `-p cube_names:=Shape_1,Shape_4` |
| `require_running` | `true` | 仿真没跑就报错 |
| `auto_start` | `false` | 仿真没跑时自动点开始 |
| `max_object_size` | `0.30` | 大于这个尺寸的 shape 不当成待抓物（挡掉桌面/料盒） |
| `sort_x_sign` | `1.0` | 近排在 -X 方向就填 `-1.0` |
| `sort_y_sign` | `1.0` | 左右反了就填 `-1.0` |
| `grid_order` | `[1,2,3,4,5,6]` | 记忆时的编号重排，彻底对不上时用它 |
| `proto_file` | `~/.ros/ep_task3_scene_proto.json` | 原型文件位置 |

**Grid 编号怎么核对**：先跑 `-p mode:=capture`，日志会打印每个 Grid 的中心坐标；
和 `objects.yaml` 里 `grid_car_y / grid_arm_x` 的那六个点对一遍。
如果近远排反了就加 `-p sort_x_sign:=-1.0`，左右反了就加 `-p sort_y_sign:=-1.0`，
改完重新 `-p mode:=capture -p recapture:=true` 一次即可。

## 注意

* 随机化是在**仿真运行中**动态建/删物体的，CoppeliaSim 停止仿真后场景会自动恢复原状，
  不会污染你的 `.ttt` 文件。
* 原型文件只需要生成一次；之后即使场景里物体被抓走了，也能照着原型重建。
* 若场景里方体/圆柱的名字不含 `Cube/Cuboid/Apple` 或 `Cylinder/Bottle`，
  `capture` 会报「找不到方体/圆柱」，改名或先跑 `tools/make_real_objects.py` 即可。

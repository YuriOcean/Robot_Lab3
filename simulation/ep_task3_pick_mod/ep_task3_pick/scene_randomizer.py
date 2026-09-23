#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""场景随机化节点 —— 仿真运行中清空并重新随机布置六个网格.

执行顺序 (全部在仿真运行中通过 ZMQ 完成, 不会修改 .ttt 文件):

    1. 检查仿真是否已经在跑      —— 没跑就报错退出 (否则改动会被存进场景)
    2. 记住/读取原型             —— 六个网格的中心 (x, y, z) 和包围盒尺寸
    3. 克隆一个方体 + 一个圆柱当模板, 藏到相机拍不到的地方
    4. **清空整个工作区**        —— 桌面上的旧物体一律删掉
    5. 按随机布局重新生成        —— 空 <= 2, 方体 >= 1, 圆柱 >= 1
    6. 写真值文件                —— pick_node 拿它和视觉结果对拍

停止仿真后 CoppeliaSim 会自动还原场景, 所以这套流程对 .ttt 零影响.

依赖:
    pip3 install coppeliasim-zmqremoteapi-client
"""

import json
import math
import os
import random
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


CUBE = 'CUBE'
CYL = 'CYL'
EMPTY = 'EMPTY'

CUBE_WORDS = ('cube', 'box', 'apple', 'cuboid', '方')
CYL_WORDS = ('cylinder', 'cyl', 'bottle', 'can', '圆')
EMPTY_WORDS = ('empty', 'none', 'nil', 'null', '-', 'x', '空')

# 找待抓物时要排除的名字 (地板 / 桌子 / 料盒 / 机器人本体 / 自己藏的模板)
EXCLUDE_WORDS = (
    '_proto',
    'floor', 'table', 'wall', 'plane', 'ground', 'desk',
    'robomaster', 'robot', 'ep_', 'gripper', 'arm', 'wheel',
    'camera', 'sensor', 'bin', 'tray', 'basket', 'crate',
)


# ====================================================================
# 纯函数
# ====================================================================

def import_remote_api():
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


def norm_kind(text):
    """把各种写法归一成 CUBE / CYL / EMPTY."""
    t = str(text).strip().lower()
    if not t:
        return EMPTY
    for w in EMPTY_WORDS:
        if t == w or (len(w) > 1 and t.startswith(w)):
            return EMPTY
    for w in CYL_WORDS:
        if w in t:
            return CYL
    for w in CUBE_WORDS:
        if w in t:
            return CUBE
    return EMPTY


def kind_cn(kind):
    return {CUBE: '方体', CYL: '圆柱', EMPTY: '空'}.get(kind, kind)


def split_names(text):
    return [v.strip() for v in str(text).replace(';', ',').split(',')
            if v.strip()]


def cluster(values, tol=0.05):
    """把一串 X 坐标聚成若干行, 返回每行代表值 (升序)."""
    rows = []
    for v in sorted(values):
        if rows and abs(v - rows[-1][-1]) <= tol:
            rows[-1].append(v)
        else:
            rows.append([v])
    return [sum(r) / len(r) for r in rows]


def row_index(rows, x):
    best, dist = 0, float('inf')
    for i, r in enumerate(rows):
        d = abs(x - r)
        if d < dist:
            best, dist = i, d
    return best


def sample_layout(rng, n_slots=6, max_empty=2, min_cube=1, min_cyl=1,
                  force_empty=-1):
    """随机布局: 空 <= max_empty, 方体 >= min_cube, 圆柱 >= min_cyl."""
    for _ in range(1000):
        if force_empty >= 0:
            n_empty = min(int(force_empty), max_empty)
        else:
            n_empty = rng.randint(0, max_empty)

        k = n_slots - n_empty
        if k < min_cube + min_cyl:
            continue

        n_cube = rng.randint(min_cube, k - min_cyl)
        layout = [CUBE] * n_cube + [CYL] * (k - n_cube) + [EMPTY] * n_empty
        rng.shuffle(layout)
        return layout

    return [CUBE, CYL] * (n_slots // 2)


def check_layout(layout, max_empty=2, min_cube=1, min_cyl=1):
    problems = []
    if layout.count(EMPTY) > max_empty:
        problems.append(f'空 {layout.count(EMPTY)} 个 > {max_empty}')
    if layout.count(CUBE) < min_cube:
        problems.append(f'方体 {layout.count(CUBE)} 个 < {min_cube}')
    if layout.count(CYL) < min_cyl:
        problems.append(f'圆柱 {layout.count(CYL)} 个 < {min_cyl}')
    return problems


# ====================================================================
# 节点
# ====================================================================

class SceneRandomizer(Node):

    def __init__(self):
        super().__init__('scene_randomizer')

        home = os.path.expanduser('~')

        # mode: random / capture / show / clear / restore / list
        self.declare_parameter('mode', 'random')
        self.declare_parameter('host', '127.0.0.1')
        self.declare_parameter('port', 23000)

        self.declare_parameter(
            'proto_file',
            os.path.join(home, '.ros', 'ep_task3_scene_proto.json'))
        self.declare_parameter('truth_file',
                               '/tmp/ep_task3_ground_truth.json')

        # ---- 随机约束 ----
        self.declare_parameter('seed', -1)
        self.declare_parameter('layout', '')
        self.declare_parameter('max_empty', 2)
        self.declare_parameter('min_cube', 1)
        self.declare_parameter('min_cyl', 1)
        self.declare_parameter('force_empty', -1)

        # ---- 仿真状态 ----
        self.declare_parameter('require_running', True)
        self.declare_parameter('auto_start', False)
        self.declare_parameter('start_wait', 2.0)

        # ---- 清空范围 ----
        # region: 清掉六个网格张成的矩形 + margin 内的所有待抓物 (默认, 最稳)
        # slots : 只清每个中心点 slot_radius 内的
        # all   : 清掉全场所有"像待抓物"的 shape
        self.declare_parameter('clear_mode', 'region')
        self.declare_parameter('clear_margin', 0.12)
        self.declare_parameter('slot_radius', 0.07)
        self.declare_parameter('max_object_size', 0.30)

        # ---- 物体识别 (名字对不上时用它显式指定) ----
        self.declare_parameter('cube_names', '')
        self.declare_parameter('cyl_names', '')

        # ---- 网格编号 ----
        self.declare_parameter('grid_order', [1, 2, 3, 4, 5, 6])
        self.declare_parameter('sort_x_sign', 1.0)
        self.declare_parameter('sort_y_sign', 1.0)

        self.declare_parameter('settle', 1.5)
        self.declare_parameter('spawn_lift', 0.002)   # 落位时抬高一点, 避免穿模弹飞
        self.declare_parameter('spawn_static', False)  # true = 物体冻死不受力

        # ---- 颜色: 只有红圆柱 + 蓝方块, 不许串色 ----
        self.declare_parameter('force_color', True)
        self.declare_parameter('cube_color', [0.05, 0.25, 0.95])   # 蓝
        self.declare_parameter('cyl_color', [0.95, 0.12, 0.08])    # 红
        self.declare_parameter('use_primitives', False)  # true = 不克隆, 直接建基本体
        self.declare_parameter('recapture', False)
        self.declare_parameter('dry_run', False)
        self.declare_parameter('keep_alive', False)

        self.mode = str(self.get_parameter('mode').value).strip().lower()
        self.host = str(self.get_parameter('host').value)
        self.port = int(self.get_parameter('port').value)
        self.proto_file = str(self.get_parameter('proto_file').value)
        self.truth_file = str(self.get_parameter('truth_file').value)
        self.clear_mode = str(self.get_parameter('clear_mode').value).lower()
        self.clear_margin = float(self.get_parameter('clear_margin').value)
        self.slot_radius = float(self.get_parameter('slot_radius').value)
        self.max_size = float(self.get_parameter('max_object_size').value)
        self.settle = float(self.get_parameter('settle').value)
        self.dry_run = bool(self.get_parameter('dry_run').value)

        self.cube_names = [n.lower() for n in
                           split_names(self.get_parameter('cube_names').value)]
        self.cyl_names = [n.lower() for n in
                          split_names(self.get_parameter('cyl_names').value)]

        self.sim = None
        self.client = None

        self.truth_pub = self.create_publisher(String, '/scene/ground_truth', 1)

    # ================================================================
    # 连接 + 仿真状态
    # ================================================================

    def connect(self):
        cls = import_remote_api()
        if cls is None:
            raise RuntimeError(
                '缺少 ZMQ 客户端, 请先: '
                'pip3 install coppeliasim-zmqremoteapi-client')

        self.client = cls(self.host, self.port)
        if hasattr(self.client, 'require'):
            self.sim = self.client.require('sim')
        else:
            self.sim = self.client.getObject('sim')
        self.get_logger().info(f'✓ 已连接 CoppeliaSim {self.host}:{self.port}')

        self.ensure_running()

    def ensure_running(self):
        """必须在仿真运行中改场景, 否则改动会被存进 .ttt."""
        try:
            state = self.sim.getSimulationState()
            stopped = (state == self.sim.simulation_stopped)
        except Exception as error:
            self.get_logger().warning(f'读不到仿真状态 ({error}), 继续')
            return

        if not stopped:
            self.get_logger().info('✓ 仿真正在运行, 可以安全地改场景')
            return

        if bool(self.get_parameter('auto_start').value):
            self.get_logger().warning('仿真未运行, 自动点开始 ...')
            self.sim.startSimulation()
            time.sleep(float(self.get_parameter('start_wait').value))
            return

        if bool(self.get_parameter('require_running').value):
            raise RuntimeError(
                '仿真没有运行. 随机化必须在仿真运行中做, 否则物体会被写进场景文件. '
                '请先在 CoppeliaSim 里点"开始仿真" (或加 -p auto_start:=true).')

        self.get_logger().warning('⚠ 仿真未运行, 改动可能被存进 .ttt')

    # ================================================================
    # 场景访问
    # ================================================================

    def alias_of(self, handle):
        try:
            return self.sim.getObjectAlias(handle, 2)
        except Exception:
            try:
                return self.sim.getObjectAlias(handle)
            except Exception:
                return f'<{handle}>'

    def set_alias(self, handle, name):
        for fn in ('setObjectAlias', 'setObjectName'):
            try:
                getattr(self.sim, fn)(handle, name)
                return
            except Exception:
                continue

    def bbox_size(self, handle):
        try:
            lo = [self.sim.getObjectFloatParam(handle, p) for p in (
                self.sim.objfloatparam_objbbox_min_x,
                self.sim.objfloatparam_objbbox_min_y,
                self.sim.objfloatparam_objbbox_min_z)]
            hi = [self.sim.getObjectFloatParam(handle, p) for p in (
                self.sim.objfloatparam_objbbox_max_x,
                self.sim.objfloatparam_objbbox_max_y,
                self.sim.objfloatparam_objbbox_max_z)]
            size = [float(hi[i] - lo[i]) for i in range(3)]
            if min(size) > 1e-6:
                return tuple(size)
        except Exception:
            pass

        try:
            out = self.sim.getShapeBB(handle)
            if out and len(out) == 3 and not hasattr(out[0], '__len__'):
                return (float(out[0]), float(out[1]), float(out[2]))
        except Exception:
            pass

        return (0.05, 0.05, 0.05)

    def color_of(self, handle):
        try:
            ok, rgb = self.sim.getShapeColor(
                handle, None, self.sim.colorcomponent_ambient_diffuse)
            if ok and rgb:
                return [float(v) for v in rgb[:3]]
        except Exception:
            pass
        return [0.6, 0.6, 0.6]

    def all_shapes(self):
        out = []
        idx = 0
        while True:
            try:
                handle = self.sim.getObjects(idx, self.sim.object_shape_type)
            except Exception:
                break
            if handle is None or handle < 0:
                break
            out.append(handle)
            idx += 1
        return out

    def kind_of_name(self, name):
        low = name.lower()
        short = low.rsplit('/', 1)[-1]
        if self.cube_names and (short in self.cube_names
                                or low in self.cube_names):
            return CUBE
        if self.cyl_names and (short in self.cyl_names
                               or low in self.cyl_names):
            return CYL
        if self.cube_names or self.cyl_names:
            return EMPTY          # 显式给了名单就只认名单
        return norm_kind(name)

    def candidate_objects(self):
        """场景里所有"看起来像待抓物"的 shape."""
        result = []
        for handle in self.all_shapes():
            name = self.alias_of(handle)
            low = name.lower()
            if any(w in low for w in EXCLUDE_WORDS):
                continue
            size = self.bbox_size(handle)
            if max(size) > self.max_size:
                continue
            try:
                x, y, z = self.sim.getObjectPosition(handle, -1)
            except Exception:
                continue
            result.append({
                'handle': handle,
                'name': name,
                'x': float(x), 'y': float(y), 'z': float(z),
                'size': [round(float(v), 5) for v in size],
                'color': self.color_of(handle),
                'kind': self.kind_of_name(name),
            })
        return result

    def remove(self, handle):
        for fn, arg in (('removeObjects', [handle]), ('removeObject', handle)):
            try:
                getattr(self.sim, fn)(arg)
                return True
            except Exception:
                continue
        return False

    def list_scene(self):
        objs = self.candidate_objects()
        log = self.get_logger()
        log.info(f'====== 场景里像待抓物的 shape ({len(objs)} 个) ======')
        for o in objs:
            log.info(f"  {o['name']:<24} {kind_cn(o['kind']):<4} "
                     f"({o['x']:+.3f}, {o['y']:+.3f}, {o['z']:+.3f}) "
                     f"尺寸={tuple(o['size'])}")
        if not objs:
            log.warning(
                '一个都没找到. 名字里要含 Cube/Cuboid/Apple 或 Cylinder/Bottle, '
                '否则请用 -p cube_names:=A,B -p cyl_names:=C,D 显式指定.')
        log.info('=============================================')
        return objs

    # ================================================================
    # 记忆
    # ================================================================

    def capture(self):
        objs = [o for o in self.candidate_objects() if o['kind'] in (CUBE, CYL)]

        if not objs:
            self.list_scene()
            raise RuntimeError('场景里找不到方体/圆柱, 无法记忆')

        if len(objs) != 6:
            self.get_logger().warning(
                f'只找到 {len(objs)} 个物体 (期望 6 个), 仍按找到的记忆')

        x_sign = float(self.get_parameter('sort_x_sign').value)
        y_sign = float(self.get_parameter('sort_y_sign').value)

        rows = cluster([o['x'] for o in objs], tol=0.05)
        if x_sign < 0:
            rows = list(reversed(rows))
        for o in objs:
            o['row'] = row_index(rows, o['x'])
        objs.sort(key=lambda o: (o['row'], y_sign * o['y']))

        order = [int(v) for v in self.get_parameter('grid_order').value]
        if len(order) == len(objs) and sorted(order) == list(
                range(1, len(objs) + 1)):
            objs = [objs[i - 1] for i in order]

        slots = [{
            'grid': i,
            'src_name': o['name'],
            'kind': o['kind'],
            'x': round(o['x'], 5),
            'y': round(o['y'], 5),
            'z': round(o['z'], 5),
            'size': o['size'],
            'color': [round(c, 4) for c in o['color']],
        } for i, o in enumerate(objs, 1)]

        proto = {'captured_at': time.time(), 'slots': slots}

        if not self.dry_run:
            os.makedirs(os.path.dirname(self.proto_file) or '.', exist_ok=True)
            with open(self.proto_file, 'w', encoding='utf-8') as fp:
                json.dump(proto, fp, ensure_ascii=False, indent=2)
            self.get_logger().info(f'✓ 原型已保存: {self.proto_file}')

        self.print_proto(proto)
        return proto

    def print_proto(self, proto):
        log = self.get_logger()
        log.info('========== 已记住的六个网格 ==========')
        for s in proto['slots']:
            sx, sy, sz = s['size']
            log.info(
                f"Grid {s['grid']}  原名={s['src_name']:<14} "
                f"{kind_cn(s['kind'])}  中心=({s['x']:+.3f}, {s['y']:+.3f}, "
                f"{s['z']:+.3f})  尺寸=({sx:.3f}, {sy:.3f}, {sz:.3f})")
        log.info('=====================================')

    def load_proto(self):
        recapture = bool(self.get_parameter('recapture').value)
        if os.path.isfile(self.proto_file) and not recapture:
            try:
                with open(self.proto_file, 'r', encoding='utf-8') as fp:
                    proto = json.load(fp)
                if proto.get('slots'):
                    self.get_logger().info(
                        f'✓ 读到原型 {self.proto_file} '
                        f'({len(proto["slots"])} 个网格)')
                    return proto
            except Exception as error:
                self.get_logger().warning(f'原型读失败: {error}, 重新记忆')

        self.get_logger().info('没有可用原型, 先从当前场景记忆一次')
        return self.capture()

    # ================================================================
    # 模板 + 清空 + 生成
    # ================================================================

    def stash_templates(self, proto):
        """克隆一个方体和一个圆柱藏到 (5, 5, 1), 可见层设 0, 相机拍不到."""
        objs = [o for o in self.candidate_objects() if o['kind'] in (CUBE, CYL)]
        templates = {}

        for kind in (CUBE, CYL):
            src = next((o for o in objs if o['kind'] == kind), None)
            if src is None:
                self.get_logger().warning(
                    f'场景里没有现成的{kind_cn(kind)}, 该类将按记住的尺寸补建')
                continue

            copy_handle = self.copy_object(src['handle'])
            if copy_handle is None:
                continue

            try:
                self.sim.setObjectInt32Param(
                    copy_handle, self.sim.objintparam_visibility_layer, 0)
                self.sim.setObjectInt32Param(
                    copy_handle, self.sim.shapeintparam_static, 1)
                self.sim.setObjectInt32Param(
                    copy_handle, self.sim.shapeintparam_respondable, 0)
                self.sim.setObjectPosition(copy_handle, -1, [5.0, 5.0, 1.0])
            except Exception:
                pass

            self.set_alias(copy_handle, f'_Proto{kind}')
            templates[kind] = {'handle': copy_handle, 'z': src['z'],
                               'size': src['size'], 'color': src['color']}
            self.get_logger().info(f'  模板 {kind_cn(kind)} ← {src["name"]}')

        return templates

    def copy_object(self, handle):
        for options in (1, 0):
            try:
                out = self.sim.copyPasteObjects([handle], options)
                if out:
                    return int(out[0])
            except Exception:
                continue
        self.get_logger().warning('copyPasteObjects 失败')
        return None

    def workspace_rect(self, proto):
        xs = [s['x'] for s in proto['slots']]
        ys = [s['y'] for s in proto['slots']]
        m = self.clear_margin
        return (min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m)

    def clear_scene(self, proto):
        """仿真运行中清空工作区里的所有旧物体."""
        x0, y0, x1, y1 = self.workspace_rect(proto)
        objs = self.candidate_objects()
        removed, kept = 0, []

        for o in objs:
            if self.clear_mode == 'all':
                hit = True
            elif self.clear_mode == 'slots':
                hit = any(math.hypot(o['x'] - s['x'], o['y'] - s['y'])
                          <= self.slot_radius for s in proto['slots'])
            else:   # region
                hit = (x0 <= o['x'] <= x1) and (y0 <= o['y'] <= y1)

            if hit:
                if self.remove(o['handle']):
                    removed += 1
            else:
                kept.append(o['name'])

        self.get_logger().info(
            f'清空工作区 ({self.clear_mode}): 移除 {removed} 个旧物体'
            + (f', 保留 {len(kept)} 个 ({", ".join(kept[:4])}...)' if kept
               else ''))

        if removed == 0:
            self.get_logger().warning(
                '⚠ 一个都没删掉 —— 名字可能对不上, 先跑 -p mode:=list 看看')
        return removed

    def build_primitive(self, kind, slot):
        """模板不可用时的兜底: 按记住的尺寸建一个基本几何体."""
        sx, sy, sz = slot['size']
        if kind == CUBE:
            shape_type = self.sim.primitiveshape_cuboid
        else:
            shape_type = self.sim.primitiveshape_cylinder
            sx = sy = max(sx, sy)

        handle = self.sim.createPrimitiveShape(
            shape_type, [float(sx), float(sy), float(sz)], 0)
        try:
            if bool(self.get_parameter('force_color').value):
                name = 'cube_color' if kind == CUBE else 'cyl_color'
                rgb = [float(v) for v in self.get_parameter(name).value][:3]
            else:
                rgb = list(slot['color'])
            self.sim.setShapeColor(
                handle, None, self.sim.colorcomponent_ambient_diffuse, rgb)
            self.sim.computeMassAndInertia(handle, 500.0)
            self.sim.setShapeMass(handle, 0.035)
        except Exception:
            pass
        self.freeze(handle)          # 摆好位置之前一律冻住
        return handle

    def paint(self, handle, kind):
        """强制刷成蓝方块 / 红圆柱, 避免克隆源材质串色.

        HSV 上对应 vision.yaml 里的 cube_hsv(蓝 100~130) 和 cyl_hsv(红 0~10/170~180).
        """
        if not bool(self.get_parameter('force_color').value):
            return

        name = 'cube_color' if kind == CUBE else 'cyl_color'
        rgb = [float(v) for v in self.get_parameter(name).value][:3]

        # 贴图会盖住颜色, 先清掉
        try:
            self.sim.setShapeTexture(
                handle, -1, self.sim.texturemap_plane, 0, [1.0, 1.0])
        except Exception:
            pass

        ok = False
        for comp in ('colorcomponent_ambient_diffuse',
                     'colorcomponent_diffuse'):
            try:
                self.sim.setShapeColor(handle, None,
                                       getattr(self.sim, comp), rgb)
                ok = True
            except Exception:
                continue

        # 高光会把饱和度冲淡, 让 HSV 判别变糊
        for comp, val in (('colorcomponent_specular', [0.1, 0.1, 0.1]),
                          ('colorcomponent_emission', [0.0, 0.0, 0.0])):
            try:
                self.sim.setShapeColor(handle, None,
                                       getattr(self.sim, comp), val)
            except Exception:
                pass

        if not ok:
            self.get_logger().warning(
                f'{kind_cn(kind)} 上色失败 (可能是组合体), '
                f'可改用 -p use_primitives:=true')

    def freeze(self, handle):
        """把物体变成静态 + 不参与碰撞. 摆位置之前必须先冻住."""
        try:
            self.sim.setObjectInt32Param(
                handle, self.sim.shapeintparam_static, 1)
            self.sim.setObjectInt32Param(
                handle, self.sim.shapeintparam_respondable, 0)
        except Exception:
            pass

    def activate(self, handle):
        """摆好之后再打开动力学, 并让引擎按新位姿重建刚体 (速度清零).

        顺序反了的话, 物体会被当成"一帧之内位移了 5 米", 求解器算出几十 m/s
        的初速度, 直接飞出去撞车 —— 这是最容易踩的坑.
        """
        if bool(self.get_parameter('spawn_static').value):
            try:
                self.sim.setObjectInt32Param(
                    handle, self.sim.shapeintparam_respondable, 1)
            except Exception:
                pass
            return

        try:
            self.sim.setObjectInt32Param(
                handle, self.sim.shapeintparam_respondable, 1)
            self.sim.setObjectInt32Param(
                handle, self.sim.shapeintparam_static, 0)
        except Exception:
            pass

        # 速度清零: 优先用 resetDynamicObject, 老版本退回 init_velocity 参数
        done = False
        for fn in ('resetDynamicObject', 'resetDynamicObjects'):
            try:
                getattr(self.sim, fn)(handle)
                done = True
                break
            except Exception:
                continue

        if not done:
            for name in ('shapefloatparam_init_velocity_x',
                         'shapefloatparam_init_velocity_y',
                         'shapefloatparam_init_velocity_z',
                         'shapefloatparam_init_ang_velocity_x',
                         'shapefloatparam_init_ang_velocity_y',
                         'shapefloatparam_init_ang_velocity_z'):
                try:
                    self.sim.setObjectFloatParam(
                        handle, getattr(self.sim, name), 0.0)
                except Exception:
                    pass

    def spawn(self, kind, slot, templates):
        """生成一个物体. 全程保持静态, 动力学留到最后统一打开."""
        tpl = templates.get(kind)
        handle, z = None, slot['z']

        if not bool(self.get_parameter('use_primitives').value) \
                and tpl is not None:
            handle = self.copy_object(tpl['handle'])
            if handle is not None:
                z = tpl['z']

        if handle is None:
            handle = self.build_primitive(kind, slot)
            z = slot['z']

        # ① 上色: 蓝方块 / 红圆柱, 不许串色
        self.paint(handle, kind)

        # ② 先冻住 (克隆体从模板那里继承了静态属性, 这里再保一道)
        self.freeze(handle)

        # ③ 再摆位置 —— 此时它还是静态的, 瞬移不会产生速度
        z += float(self.get_parameter('spawn_lift').value)
        self.sim.setObjectPosition(
            handle, -1, [float(slot['x']), float(slot['y']), float(z)])
        try:
            self.sim.setObjectOrientation(handle, -1, [0.0, 0.0, 0.0])
        except Exception:
            pass

        try:
            self.sim.setObjectInt32Param(
                handle, self.sim.objintparam_visibility_layer, 1)
        except Exception:
            pass

        name = ('Cube' if kind == CUBE else 'Cylinder') + f'_{slot["grid"]}'
        self.set_alias(handle, name)
        return handle, name

    def apply_layout(self, proto, layout):
        # 顺序很重要: 先留模板, 再清空, 最后生成
        templates = self.stash_templates(proto)
        self.clear_scene(proto)

        spawned = []
        for slot, kind in zip(proto['slots'], layout):
            if kind == EMPTY:
                self.get_logger().info(f"  Grid {slot['grid']}: 空, 不生成")
                continue
            handle, name = self.spawn(kind, slot, templates)
            spawned.append(handle)
            self.get_logger().info(
                f"  Grid {slot['grid']}: 生成 {kind_cn(kind)} → {name}")

        # 全部摆好之后再统一打开动力学, 中间不会有任何瞬移产生的速度
        for handle in spawned:
            self.activate(handle)
        if spawned:
            self.get_logger().info(
                f'✓ {len(spawned)} 个物体已就位并启用动力学 (速度已清零)')

        for tpl in templates.values():
            self.remove(tpl['handle'])

        if self.settle > 0.0:
            time.sleep(self.settle)

        self.verify_scene(proto, layout)

    def verify_scene(self, proto, layout):
        """生成完再扫一遍, 确认场景里真的就是这套布局."""
        objs = self.candidate_objects()
        log = self.get_logger()
        bad = 0

        for slot, kind in zip(proto['slots'], layout):
            near = [o for o in objs
                    if math.hypot(o['x'] - slot['x'], o['y'] - slot['y'])
                    <= self.slot_radius]
            got = near[0]['kind'] if near else EMPTY
            if got != kind:
                bad += 1
                log.error(f"  ✗ Grid {slot['grid']}: 想要{kind_cn(kind)}, "
                          f"实际{kind_cn(got)}")

        if bad == 0:
            log.info('✓ 场景自检通过, 六个网格和目标布局一致')
        else:
            log.error(f'✗ 场景自检有 {bad} 个网格对不上')

    # ================================================================
    # 真值
    # ================================================================

    def write_truth(self, layout, proto, seed):
        truth = {
            'stamp': time.time(),
            'seed': seed,
            'source': 'scene_randomizer',
            'grids': {str(s['grid']): k
                      for s, k in zip(proto['slots'], layout)},
            'counts': {
                'CUBE': layout.count(CUBE),
                'CYL': layout.count(CYL),
                'EMPTY': layout.count(EMPTY),
            },
        }

        if not self.dry_run:
            try:
                os.makedirs(os.path.dirname(self.truth_file) or '.',
                            exist_ok=True)
                with open(self.truth_file, 'w', encoding='utf-8') as fp:
                    json.dump(truth, fp, ensure_ascii=False, indent=2)
                self.get_logger().info(f'✓ 真值已写入 {self.truth_file}')
            except Exception as error:
                self.get_logger().warning(f'真值写入失败: {error}')

        msg = String()
        msg.data = json.dumps(truth, ensure_ascii=False)
        self.truth_pub.publish(msg)
        return truth

    # ================================================================
    # 主流程
    # ================================================================

    def decide_layout(self, proto):
        n = len(proto['slots'])
        max_empty = int(self.get_parameter('max_empty').value)
        min_cube = int(self.get_parameter('min_cube').value)
        min_cyl = int(self.get_parameter('min_cyl').value)

        if self.mode == 'restore':
            self.get_logger().info('模式 restore: 还原成最初记住的布局')
            return [s['kind'] for s in proto['slots']], -1

        text = str(self.get_parameter('layout').value).strip()
        if text:
            layout = [norm_kind(v) for v in
                      text.replace(';', ',').split(',') if v.strip()]
            if len(layout) != n:
                raise RuntimeError(f'layout 给了 {len(layout)} 项, 需要 {n} 项')
            bad = check_layout(layout, max_empty, min_cube, min_cyl)
            if bad:
                self.get_logger().warning(
                    '⚠ 手工布局不满足约束: ' + '; '.join(bad))
            self.get_logger().info(f'模式 manual: {layout}')
            return layout, -1

        seed = int(self.get_parameter('seed').value)
        if seed < 0:
            seed = int(time.time() * 1000) % 100000
        rng = random.Random(seed)
        layout = sample_layout(
            rng, n_slots=n, max_empty=max_empty, min_cube=min_cube,
            min_cyl=min_cyl,
            force_empty=int(self.get_parameter('force_empty').value))
        self.get_logger().info(f'模式 random: seed={seed}')
        return layout, seed

    def run(self):
        self.connect()

        if self.mode == 'list':
            self.list_scene()
            return

        if self.mode == 'capture':
            self.capture()
            return

        proto = self.load_proto()

        if self.mode == 'show':
            self.print_proto(proto)
            return

        if self.mode == 'clear':
            self.clear_scene(proto)
            return

        layout, seed = self.decide_layout(proto)

        log = self.get_logger()
        log.info('========== 本次布局 ==========')
        for s, k in zip(proto['slots'], layout):
            log.info(f"Grid {s['grid']}  ({s['x']:+.3f}, {s['y']:+.3f})  "
                     f"→ {kind_cn(k)}")
        log.info(f'方体 {layout.count(CUBE)} / 圆柱 {layout.count(CYL)} / '
                 f'空 {layout.count(EMPTY)}')
        log.info('==============================')

        if self.dry_run:
            log.info('(dry_run, 场景没有改动)')
            self.write_truth(layout, proto, seed)
            return

        self.apply_layout(proto, layout)
        self.write_truth(layout, proto, seed)
        log.info('✅ 场景已随机布置完成')


def main(args=None):
    rclpy.init(args=args)
    node = None
    code = 0
    try:
        node = SceneRandomizer()
        node.run()
        if bool(node.get_parameter('keep_alive').value):
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        code = 1
        if node is not None:
            node.get_logger().error(f'场景随机化失败: {error}')
        else:
            print(f'场景随机化失败: {error}', file=sys.stderr)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()

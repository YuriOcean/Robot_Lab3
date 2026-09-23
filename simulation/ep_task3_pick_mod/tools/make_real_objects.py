#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把场景里的方块 / 圆柱换成"像真东西"的苹果 / 瓶子.

中心点 (x, y) 一律保持不变, 底面贴合原来物体的底面,
大小按抓夹能夹住的尺寸给 (瓶身直径 ~45 mm, 苹果直径 ~55 mm).

这样做的目的只有一个: 让 YOLO 能真的识别出 apple / bottle.
纯色方块和圆柱 YOLO 是认不出来的.

用法 (CoppeliaSim 打开场景, **先别点开始仿真**):

    # 先看看场景里现在有什么
    python3 make_real_objects.py --list

    # 用内置的程序化模型替换 (不需要任何外部文件)
    python3 make_real_objects.py

    # 用你自己下载的网格模型替换 (识别率更高)
    python3 make_real_objects.py \\
        --apple-mesh ~/models/apple.obj \\
        --bottle-mesh ~/models/bottle.obj

    # 只换一部分 / 改名字
    python3 make_real_objects.py --cubes Cube_1,Cube_2 --cylinders Cylinder_1

替换完记得在 CoppeliaSim 里 File → Save scene 存一下.

依赖:
    pip3 install coppeliasim-zmqremoteapi-client
"""

import argparse
import math
import sys


# ---------------------------------------------------------------- 尺寸
# 单位: 米. 想让抓夹更好夹就把 *_DIAM 调小一点.
APPLE_DIAM = 0.055
APPLE_STEM_D = 0.006
APPLE_STEM_H = 0.018

BOTTLE_BODY_D = 0.045
BOTTLE_BODY_H = 0.085
BOTTLE_LABEL_H = 0.035
BOTTLE_NECK_D = 0.022
BOTTLE_NECK_H = 0.028
BOTTLE_CAP_D = 0.026
BOTTLE_CAP_H = 0.012

APPLE_MASS = 0.03
BOTTLE_MASS = 0.04
DENSITY = 500.0          # kg/m^3, 只用来算惯量张量, 质量随后单独设

COLOR_APPLE = [0.78, 0.10, 0.08]
COLOR_LEAF = [0.15, 0.45, 0.12]
COLOR_STEM = [0.35, 0.24, 0.12]
COLOR_BOTTLE = [0.20, 0.55, 0.28]
COLOR_LABEL = [0.95, 0.95, 0.92]
COLOR_CAP = [0.85, 0.75, 0.15]


def connect(host, port):
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    except ImportError:
        try:
            from zmqRemoteAPI import RemoteAPIClient
        except ImportError:
            sys.exit('请先: pip3 install coppeliasim-zmqremoteapi-client')

    client = RemoteAPIClient(host, port)
    sim = client.require('sim') if hasattr(client, 'require') \
        else client.getObject('sim')
    return client, sim


# ---------------------------------------------------------------- 兼容层

def alias_of(sim, handle):
    try:
        return sim.getObjectAlias(handle, 2)
    except Exception:
        try:
            return sim.getObjectAlias(handle)
        except Exception:
            return f'<{handle}>'


def set_alias(sim, handle, name):
    for fn in ('setObjectAlias', 'setObjectName'):
        try:
            getattr(sim, fn)(handle, name)
            return
        except Exception:
            continue


def find(sim, name):
    for candidate in (name, '/' + name.lstrip('/')):
        try:
            return sim.getObject(candidate)
        except Exception:
            continue
    return -1


def shape_size(sim, handle):
    """返回 (sx, sy, sz). 兼容 getShapeBB 的几种返回形式."""
    try:
        out = sim.getShapeBB(handle)
    except Exception:
        return (0.04, 0.04, 0.04)

    if out is None:
        return (0.04, 0.04, 0.04)

    # 新版: 直接返回 [sx, sy, sz]
    if len(out) == 3 and not hasattr(out[0], '__len__'):
        return (float(out[0]), float(out[1]), float(out[2]))

    # 老版: 返回 (bbMin, bbMax)
    try:
        bb_min, bb_max = out[0], out[1]
        return (float(bb_max[0] - bb_min[0]),
                float(bb_max[1] - bb_min[1]),
                float(bb_max[2] - bb_min[2]))
    except Exception:
        return (0.04, 0.04, 0.04)


def bbox_z(sim, handle):
    """返回 (min_z, max_z): 世界坐标下这个物体的底面和顶面高度."""
    _, _, z = sim.getObjectPosition(handle, -1)
    try:
        lo = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_z)
        hi = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_z)
        if hi > lo:
            return z + lo, z + hi
    except Exception:
        pass
    half = shape_size(sim, handle)[2] * 0.5
    return z - half, z + half


def remove(sim, handle):
    try:
        sim.removeObjects([handle])
        return
    except Exception:
        pass
    try:
        sim.removeObject(handle)
    except Exception as error:
        print(f'  ⚠ 删除失败: {error}')


def make_primitive(sim, kind, size, color, position):
    shape_type = {
        'cuboid': sim.primitiveshape_cuboid,
        'sphere': sim.primitiveshape_spheroid,
        'cylinder': sim.primitiveshape_cylinder,
        'cone': sim.primitiveshape_cone,
    }[kind]

    handle = sim.createPrimitiveShape(shape_type, list(size), 0)
    sim.setObjectPosition(handle, -1, list(position))
    try:
        sim.setShapeColor(handle, None,
                          sim.colorcomponent_ambient_diffuse, list(color))
    except Exception:
        pass
    return handle


def finalize(sim, handle, name, mass, dynamic=True):
    set_alias(sim, handle, name)
    try:
        sim.setObjectInt32Param(handle, sim.shapeintparam_respondable, 1)
        sim.setObjectInt32Param(handle, sim.shapeintparam_static,
                                0 if dynamic else 1)
    except Exception:
        pass

    # 先按密度算一遍惯量张量, 再把质量精确设成想要的值
    try:
        sim.computeMassAndInertia(handle, DENSITY)
    except Exception:
        pass
    try:
        sim.setShapeMass(handle, float(mass))
    except Exception:
        pass


# ---------------------------------------------------------------- 建模

def build_apple(sim, x, y, bottom):
    """红苹果: 球体 + 褐色果梗 + 一片绿叶."""
    r = APPLE_DIAM * 0.5
    cz = bottom + r

    parts = [make_primitive(
        sim, 'sphere',
        [APPLE_DIAM, APPLE_DIAM, APPLE_DIAM * 0.92],
        COLOR_APPLE, [x, y, cz])]

    parts.append(make_primitive(
        sim, 'cylinder',
        [APPLE_STEM_D, APPLE_STEM_D, APPLE_STEM_H],
        COLOR_STEM,
        [x, y, cz + r * 0.92 + APPLE_STEM_H * 0.4]))

    leaf = make_primitive(
        sim, 'sphere',
        [0.022, 0.010, 0.003], COLOR_LEAF,
        [x + 0.010, y, cz + r * 0.92 + APPLE_STEM_H * 0.7])
    try:
        sim.setObjectOrientation(leaf, -1, [0.0, 0.0, math.radians(25)])
    except Exception:
        pass
    parts.append(leaf)

    try:
        handle = sim.groupShapes(parts)
    except Exception:
        handle = parts[0]
    return handle


def build_bottle(sim, x, y, bottom):
    """绿色饮料瓶: 瓶身 + 白标签 + 瓶颈 + 瓶盖."""
    body_cz = bottom + BOTTLE_BODY_H * 0.5

    parts = [make_primitive(
        sim, 'cylinder',
        [BOTTLE_BODY_D, BOTTLE_BODY_D, BOTTLE_BODY_H],
        COLOR_BOTTLE, [x, y, body_cz])]

    # 白色标签: 稍微粗一点点, 贴在瓶身中段. 对 YOLO 帮助很大.
    parts.append(make_primitive(
        sim, 'cylinder',
        [BOTTLE_BODY_D * 1.02, BOTTLE_BODY_D * 1.02, BOTTLE_LABEL_H],
        COLOR_LABEL, [x, y, bottom + BOTTLE_BODY_H * 0.45]))

    # 肩部
    parts.append(make_primitive(
        sim, 'cone',
        [BOTTLE_BODY_D, BOTTLE_BODY_D, 0.020],
        COLOR_BOTTLE, [x, y, bottom + BOTTLE_BODY_H + 0.010]))

    neck_z = bottom + BOTTLE_BODY_H + 0.020 + BOTTLE_NECK_H * 0.5
    parts.append(make_primitive(
        sim, 'cylinder',
        [BOTTLE_NECK_D, BOTTLE_NECK_D, BOTTLE_NECK_H],
        COLOR_BOTTLE, [x, y, neck_z]))

    parts.append(make_primitive(
        sim, 'cylinder',
        [BOTTLE_CAP_D, BOTTLE_CAP_D, BOTTLE_CAP_H],
        COLOR_CAP,
        [x, y, neck_z + BOTTLE_NECK_H * 0.5 + BOTTLE_CAP_H * 0.5]))

    try:
        handle = sim.groupShapes(parts)
    except Exception:
        handle = parts[0]
    return handle


def import_mesh(sim, path, x, y, bottom, target_height):
    """导入外部网格并缩放到目标高度, 底面贴合 bottom."""
    ext = path.lower().rsplit('.', 1)[-1]
    fmt = {'obj': 0, 'dxf': 1, '3ds': 2, 'stl': 4, 'dae': 5, 'ply': 6}.get(ext)
    if fmt is None:
        raise ValueError(f'不支持的网格格式: .{ext}')

    handle = sim.importShape(fmt, path, 0, 0.0001, 1.0)

    height = shape_size(sim, handle)[2]

    if height > 1e-6:
        scale = target_height / height
        try:
            sim.scaleObject(handle, scale, scale, scale, 0)
        except Exception:
            pass

    sim.setObjectPosition(handle, -1, [x, y, bottom + target_height * 0.5])
    low, _ = bbox_z(sim, handle)
    _, _, z = sim.getObjectPosition(handle, -1)
    sim.setObjectPosition(handle, -1, [x, y, z + (bottom - low)])
    return handle


# ---------------------------------------------------------------- 主流程

def replace_one(sim, old_name, new_name, kind, args):
    handle = find(sim, old_name)
    if handle < 0:
        print(f'  - {old_name}: 场景里没有, 跳过')
        return None

    x, y, _ = sim.getObjectPosition(handle, -1)
    bottom, top = bbox_z(sim, handle)
    print(f'  - {old_name}: 中心=({x:+.3f}, {y:+.3f}) '
          f'底面 z={bottom:.3f} 高={top - bottom:.3f}')

    if args.dry_run:
        return None

    remove(sim, handle)

    if kind == 'apple':
        mesh = args.apple_mesh
        height = APPLE_DIAM
        mass = APPLE_MASS
        builder = build_apple
    else:
        mesh = args.bottle_mesh
        height = (BOTTLE_BODY_H + 0.020 + BOTTLE_NECK_H + BOTTLE_CAP_H)
        mass = BOTTLE_MASS
        builder = build_bottle

    if mesh:
        new_handle = import_mesh(sim, mesh, x, y, bottom, height)
    else:
        new_handle = builder(sim, x, y, bottom)

    finalize(sim, new_handle, new_name, mass, dynamic=not args.static)
    print(f'    → {new_name} 已创建')
    return new_handle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=23000)
    parser.add_argument('--cubes', default='Cube_1,Cube_2,Cube_3',
                        help='要换成苹果的物体名, 逗号分隔')
    parser.add_argument('--cylinders',
                        default='Cylinder_1,Cylinder_2,Cylinder_3',
                        help='要换成瓶子的物体名, 逗号分隔')
    parser.add_argument('--apple-mesh', default='',
                        help='苹果网格文件 (.obj/.stl/.dae), 留空用内置模型')
    parser.add_argument('--bottle-mesh', default='',
                        help='瓶子网格文件, 留空用内置模型')
    parser.add_argument('--static', action='store_true',
                        help='创建成静态物体 (不参与动力学)')
    parser.add_argument('--dry-run', action='store_true',
                        help='只打印, 不真的改场景')
    parser.add_argument('--list', action='store_true',
                        help='列出场景里所有 shape 的名字和位置')
    args = parser.parse_args()

    _client, sim = connect(args.host, args.port)
    print(f'✓ 已连接 CoppeliaSim {args.host}:{args.port}')

    try:
        state = sim.getSimulationState()
        if state != sim.simulation_stopped:
            print('⚠ 仿真正在运行. 建议先停止仿真再替换物体.')
    except Exception:
        pass

    if args.list:
        print('场景里的 shape:')
        idx = 0
        while True:
            handle = sim.getObjects(idx, sim.object_shape_type)
            if handle is None or handle < 0:
                break
            x, y, z = sim.getObjectPosition(handle, -1)
            print(f'  [{idx:2d}] {alias_of(sim, handle):<28} '
                  f'({x:+.3f}, {y:+.3f}, {z:+.3f})')
            idx += 1
        return

    cubes = [v.strip() for v in args.cubes.split(',') if v.strip()]
    cylinders = [v.strip() for v in args.cylinders.split(',') if v.strip()]

    print('\n方块 → 苹果 (Apple, 走方块料盒):')
    for i, name in enumerate(cubes, 1):
        replace_one(sim, name, f'Apple_{i}', 'apple', args)

    print('\n圆柱 → 瓶子 (Bottle, 走圆柱料盒):')
    for i, name in enumerate(cylinders, 1):
        replace_one(sim, name, f'Bottle_{i}', 'bottle', args)

    if args.dry_run:
        print('\n(dry-run, 场景没有被修改)')
    else:
        print('\n完成. 回 CoppeliaSim 里 File → Save scene 保存.')
        print('注意: pick_z / gripper_power 可能要跟着物体高度微调一次.')


if __name__ == '__main__':
    main()

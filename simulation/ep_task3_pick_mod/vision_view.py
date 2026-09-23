#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CoppeliaSim 视觉传感器实时取流 + 简易识别, 独立窗口显示.

不依赖 ROS, 不用编译任何 CoppeliaSim 插件.
走 ZMQ Remote API (CoppeliaSim 4.x 自带, 默认端口 23000).

用法:
    # 1) 先在 CoppeliaSim 里点"开始仿真"
    # 2) 另开一个终端
    python3 vision_view.py
    python3 vision_view.py --sensor /visionSensor
    python3 vision_view.py --list          # 列出场景里所有视觉传感器
    python3 vision_view.py --no-detect     # 只看原始画面

窗口按键:
    q / ESC   退出
    d         开关识别
    s         保存当前帧到 ./shot_xxx.png
    m         开关掩膜视图 (看阈值调得准不准)

依赖:
    pip3 install coppeliasim-zmqremoteapi-client opencv-python numpy
"""

import argparse
import sys
import time

import cv2
import numpy as np


# =========================================================
# 连接 CoppeliaSim (兼容几个版本的客户端写法)
# =========================================================

def connect(host, port):
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    except ImportError:
        try:
            # 老版本: CoppeliaSim 安装目录
            # programming/zmqRemoteApi/clients/python 里的 zmqRemoteAPI.py
            from zmqRemoteAPI import RemoteAPIClient
        except ImportError:
            sys.exit(
                '找不到 ZMQ Remote API 客户端, 先装:\n'
                '  pip3 install coppeliasim-zmqremoteapi-client\n'
                '或把 CoppeliaSim 安装目录下的\n'
                '  programming/zmqRemoteApi/clients/python\n'
                '加进 PYTHONPATH'
            )

    client = RemoteAPIClient(host, port)
    # 新客户端用 require, 老客户端用 getObject
    if hasattr(client, 'require'):
        sim = client.require('sim')
    else:
        sim = client.getObject('sim')
    return client, sim


def list_sensors(sim):
    """列出场景里所有视觉传感器的路径."""
    print('场景里的视觉传感器:')
    found = 0
    idx = 0
    while True:
        h = sim.getObjects(idx, sim.object_visionsensor_type)
        if h < 0:
            break
        try:
            path = sim.getObjectAlias(h, 2)   # 2 = 完整路径
        except Exception:
            path = sim.getObjectAlias(h)
        res = sim.getVisionSensorResolution(h)
        print(f'  [{idx}] {path}   分辨率={res}')
        found += 1
        idx += 1
    if found == 0:
        print('  (一个都没有. 确认传感器类型是 Vision sensor,'
              ' 不是普通 Camera)')
    return found


def resolve_sensor(sim, name):
    """按路径拿句柄; 拿不到就退回"场景里第一个视觉传感器"."""
    if name:
        try:
            return sim.getObject(name), name
        except Exception:
            print(f'⚠ 找不到 {name}, 改用场景里第一个视觉传感器')
    h = sim.getObjects(0, sim.object_visionsensor_type)
    if h < 0:
        sys.exit('场景里没有视觉传感器, 先加一个 Vision sensor')
    try:
        path = sim.getObjectAlias(h, 2)
    except Exception:
        path = '<unnamed>'
    return h, path


def grab(sim, handle):
    """取一帧, 返回 BGR 图 (已翻正)."""
    # 新 API
    try:
        buf, res = sim.getVisionSensorImg(handle)
        w, h = res[0], res[1]
    except AttributeError:
        # 老 API
        buf, res = sim.getVisionSensorCharImage(handle)
        w, h = res[0], res[1]

    if isinstance(buf, str):
        buf = buf.encode('latin-1')

    img = np.frombuffer(buf, dtype=np.uint8)
    if img.size != w * h * 3:
        return None
    img = img.reshape(h, w, 3)
    # CoppeliaSim 的图是 RGB 且上下颠倒的
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.flip(img, 0)


# =========================================================
# 简易识别: 红圆柱 / 蓝方块
# =========================================================

# HSV 阈值. 红色跨 0 度, 所以要两段
RED_RANGES = [((0, 120, 70), (10, 255, 255)),
              ((170, 120, 70), (180, 255, 255))]
BLUE_RANGE = ((100, 120, 70), (130, 255, 255))

MIN_AREA = 300          # 小于这个面积的连通域当噪点丢掉


def color_mask(hsv, ranges):
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect(img):
    """返回 (标注后的图, 掩膜, 检测结果列表)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    red = color_mask(hsv, RED_RANGES)
    blue = color_mask(hsv, [BLUE_RANGE])

    out = img.copy()
    results = []

    for mask, label, box_color in (
        (red,  'CYL(red)',   (0, 0, 255)),
        (blue, 'CUBE(blue)', (255, 0, 0)),
    ):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
            cv2.rectangle(out, (x, y), (x + w, y + h), box_color, 2)
            cv2.circle(out, (cx, cy), 4, box_color, -1)
            cv2.putText(out, f'{label} {int(area)}',
                        (x, max(y - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        box_color, 1, cv2.LINE_AA)
            results.append({
                'label': label, 'cx': cx, 'cy': cy,
                'w': w, 'h': h, 'area': area,
            })

    both = cv2.bitwise_or(red, blue)
    return out, both, results


# =========================================================
# 主循环
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=23000)
    ap.add_argument('--sensor', default='',
                    help='视觉传感器路径, 如 /visionSensor')
    ap.add_argument('--scale', type=float, default=2.0,
                    help='显示放大倍数 (传感器分辨率通常很小)')
    ap.add_argument('--list', action='store_true',
                    help='只列出场景里的视觉传感器然后退出')
    ap.add_argument('--no-detect', action='store_true')
    args = ap.parse_args()

    print(f'连接 CoppeliaSim {args.host}:{args.port} ...')
    _client, sim = connect(args.host, args.port)
    print('✓ 已连接')

    if args.list:
        list_sensors(sim)
        return

    handle, path = resolve_sensor(sim, args.sensor)
    res = sim.getVisionSensorResolution(handle)
    print(f'✓ 使用传感器 {path}, 分辨率 {res}')
    print('窗口按键: q 退出 | d 开关识别 | m 掩膜 | s 截图')

    win = 'CoppeliaSim vision'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    do_detect = not args.no_detect
    show_mask = False
    shot_id = 0
    frames = 0
    t0 = time.time()
    fps = 0.0
    miss = 0

    while True:
        try:
            img = grab(sim, handle)
        except Exception as error:
            print(f'取图失败: {error}')
            time.sleep(0.3)
            continue

        if img is None:
            miss += 1
            if miss % 30 == 1:
                print('⚠ 取到空帧 —— 仿真在运行吗? 传感器被遮挡了吗?')
            time.sleep(0.05)
            continue
        miss = 0

        if do_detect:
            view, mask, results = detect(img)
        else:
            view, mask, results = img.copy(), None, []

        if show_mask and mask is not None:
            view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        frames += 1
        if frames % 10 == 0:
            now = time.time()
            fps = 10.0 / max(now - t0, 1e-6)
            t0 = now

        cv2.putText(view, f'{fps:4.1f} fps  obj={len(results)}',
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0), 1, cv2.LINE_AA)

        if args.scale != 1.0:
            view = cv2.resize(
                view, None, fx=args.scale, fy=args.scale,
                interpolation=cv2.INTER_NEAREST)

        cv2.imshow(win, view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('d'):
            do_detect = not do_detect
            print(f'识别: {"开" if do_detect else "关"}')
        elif key == ord('m'):
            show_mask = not show_mask
        elif key == ord('s'):
            fn = f'shot_{shot_id:03d}.png'
            cv2.imwrite(fn, view)
            print(f'已保存 {fn}')
            shot_id += 1

    cv2.destroyAllWindows()
    print('已退出')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""ep_task3_pick.state_machine.

极简 YAML 驱动状态机 + 三份运行报告的收集与落盘.

设计目标:
    * 不引入 ROS 依赖, 便于单元测试
    * handler 是 target(PickNode) 上的方法, 返回 event 字符串 (或 (event, note))
    * handler 抛异常时, 记录到 exception log 并按 on_exception 映射到事件
"""

import json
import os
import time
import traceback
from datetime import datetime


class StateMachineError(Exception):
    pass


# =============================================================================
# 运行报告收集器
# =============================================================================

class RunReporter:
    """收集分类结果 / 异常 / 状态转移日志, 结束时写三份文件."""

    def __init__(self, report_dir, cls_file, exc_file, log_file,
                 also_json=True, logger=None):
        self.dir = report_dir
        self.cls_path = os.path.join(report_dir, cls_file)
        self.exc_path = os.path.join(report_dir, exc_file)
        self.log_path = os.path.join(report_dir, log_file)
        self.also_json = also_json
        self.logger = logger

        self.classifications = []   # [{grid_id, truth, vision, ok, note}]
        self.exceptions = []        # [{ts, state, grid_id, err_type, msg, tb, action}]
        self.transitions = []       # [{ts, from, to, event, note}]
        self.summary = {}
        self.start_ts = time.time()
        self.end_ts = None

        try:
            os.makedirs(report_dir, exist_ok=True)
        except Exception as e:
            if logger:
                logger.warning(f'报告目录创建失败 {report_dir}: {e}')

    # ---- 采集 ----

    def add_classification(self, grid_id, truth, vision, note=''):
        self.classifications.append({
            'grid_id': int(grid_id),
            'truth': truth,        # 'CUBE' / 'CYL' / None(空) / 'UNKNOWN'(无真值)
            'vision': vision,      # 'CUBE' / 'CYL' / None(空)
            'ok': (truth == vision) if truth != 'UNKNOWN' else None,
            'note': note,
        })

    def add_exception(self, state, err, grid_id=None, action='recover'):
        self.exceptions.append({
            'ts': time.time(),
            'state': state,
            'grid_id': grid_id,
            'err_type': type(err).__name__,
            'msg': str(err),
            'traceback': traceback.format_exc(),
            'action': action,
        })

    def add_transition(self, frm, to, event, note=''):
        self.transitions.append({
            'ts': time.time(),
            'from': frm,
            'to': to,
            'event': event,
            'note': note,
        })

    def set_summary(self, **kw):
        self.summary.update(kw)

    # ---- 渲染 ----

    @staticmethod
    def _fmt_ts(ts):
        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def _kind_zh(k):
        if k == 'UNKNOWN':
            return ' -- '
        if k is None or k == 'EMPTY':
            return ' 空 '
        if k == 'CUBE':
            return '方体'
        if k == 'CYL':
            return '圆柱'
        return str(k)[:4]

    def render_classification(self):
        L = []
        L.append('============================================================')
        L.append('              分  类  结  果   Classification')
        L.append('============================================================')
        L.append(f'开始时间 : {self._fmt_ts(self.start_ts)}')
        if self.end_ts:
            L.append(f'结束时间 : {self._fmt_ts(self.end_ts)}')
            L.append(f'总耗时   : {self.end_ts - self.start_ts:.2f} s')
        L.append('')

        if not self.classifications:
            L.append('(本次无分类结果)')
            L.append('')
            return '\n'.join(L)

        L.append('+--------+--------+--------+----------+----------------------+')
        L.append('| Grid   | 真值   | 视觉   | 判定     | 备注                 |')
        L.append('+--------+--------+--------+----------+----------------------+')

        good = 0
        have_truth = 0
        for c in self.classifications:
            truth_s  = self._kind_zh(c['truth'])
            vision_s = self._kind_zh(c['vision'])
            if c['ok'] is None:
                verdict = '   --   '
            elif c['ok']:
                verdict = '   OK   '
                good += 1
                have_truth += 1
            else:
                verdict = 'MISMATCH'
                have_truth += 1
            note = (c['note'] or '')[:20]
            L.append(
                f'| Grid {c["grid_id"]:<2}|  {truth_s}  |  {vision_s}  |'
                f' {verdict} | {note:<20} |'
            )
        L.append('+--------+--------+--------+----------+----------------------+')
        L.append('')

        if have_truth > 0:
            rate = 100.0 * good / have_truth
            marker = '✅' if good == have_truth else '❌'
            L.append(f'{marker}  视觉判别命中: {good}/{have_truth}  ({rate:.1f}%)')
        else:
            L.append('(未加载真值文件, 无准确率)')

        # 视觉计数
        cnt_v = {'CUBE': 0, 'CYL': 0, 'EMPTY': 0}
        cnt_t = {'CUBE': 0, 'CYL': 0, 'EMPTY': 0, 'UNKNOWN': 0}
        for c in self.classifications:
            v = c['vision']
            if   v is None:    cnt_v['EMPTY'] += 1
            elif v == 'CUBE':  cnt_v['CUBE']  += 1
            elif v == 'CYL':   cnt_v['CYL']   += 1
            t = c['truth']
            if   t is None:         cnt_t['EMPTY']   += 1
            elif t == 'UNKNOWN':    cnt_t['UNKNOWN'] += 1
            elif t == 'CUBE':       cnt_t['CUBE']    += 1
            elif t == 'CYL':        cnt_t['CYL']     += 1
        L.append(f'视觉计数 : 方体 {cnt_v["CUBE"]}  圆柱 {cnt_v["CYL"]}  空 {cnt_v["EMPTY"]}')
        if cnt_t['UNKNOWN'] == 0:
            L.append(f'真值计数 : 方体 {cnt_t["CUBE"]}  圆柱 {cnt_t["CYL"]}  空 {cnt_t["EMPTY"]}')

        if self.summary:
            L.append('')
            L.append('---- 汇总 ----')
            for k, v in self.summary.items():
                L.append(f'  {k}: {v}')
        L.append('')
        return '\n'.join(L)

    def render_exceptions(self):
        L = []
        L.append('============================================================')
        L.append('             异 常 测 试 记 录   Exception Log')
        L.append('============================================================')
        L.append(f'生成时间 : {self._fmt_ts(time.time())}')
        L.append(f'异常总数 : {len(self.exceptions)}')
        L.append('')

        if not self.exceptions:
            L.append('本次执行未捕获任何异常. ✅')
            L.append('')
            return '\n'.join(L)

        for i, e in enumerate(self.exceptions, 1):
            L.append(f'------- #{i} ----------------------------------------------')
            L.append(f'  时间   : {self._fmt_ts(e["ts"])}')
            L.append(f'  状态   : {e["state"]}')
            if e['grid_id'] is not None:
                L.append(f'  网格   : Grid {e["grid_id"]}')
            L.append(f'  类型   : {e["err_type"]}')
            L.append(f'  处理   : {e["action"]}')
            L.append(f'  消息   : {e["msg"]}')
            tb = (e.get('traceback') or '').strip()
            if tb and tb != 'NoneType: None':
                L.append('  栈追踪 :')
                for tl in tb.splitlines()[-8:]:
                    L.append(f'    {tl}')
            L.append('')
        return '\n'.join(L)

    def render_task_log(self):
        L = []
        L.append('============================================================')
        L.append('                任 务 日 志   Task Log')
        L.append('============================================================')
        L.append(f'开始 : {self._fmt_ts(self.start_ts)}')
        if self.end_ts:
            L.append(f'结束 : {self._fmt_ts(self.end_ts)}')
            L.append(f'耗时 : {self.end_ts - self.start_ts:.2f} s')
        L.append(f'状态转移总数 : {len(self.transitions)}')
        L.append('')
        L.append('+-----+---------+-------------+-------------+-------------+----------------------+')
        L.append('|  #  | 相对(s) | From        | Event       | To          | 备注                 |')
        L.append('+-----+---------+-------------+-------------+-------------+----------------------+')
        for i, t in enumerate(self.transitions, 1):
            rel = t['ts'] - self.start_ts
            frm = (t['from'] or '-')[:11]
            evt = (t['event'] or '-')[:11]
            to  = (t['to']   or '-')[:11]
            note = (t['note'] or '')[:20]
            L.append(
                f'| {i:>3} | {rel:>7.2f} | {frm:<11} | {evt:<11} | {to:<11} |'
                f' {note:<20} |'
            )
        L.append('+-----+---------+-------------+-------------+-------------+----------------------+')
        L.append('')
        return '\n'.join(L)

    # ---- 落盘 ----

    def _write(self, path, text):
        try:
            with open(path, 'w', encoding='utf-8') as fp:
                fp.write(text)
            return True
        except Exception as e:
            if self.logger:
                self.logger.warning(f'写文件失败 {path}: {e}')
            return False

    def _dump_json(self, path, data):
        try:
            with open(path, 'w', encoding='utf-8') as fp:
                json.dump(data, fp, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            if self.logger:
                self.logger.warning(f'写 JSON 失败 {path}: {e}')

    def flush(self):
        self.end_ts = time.time()
        ok1 = self._write(self.cls_path, self.render_classification())
        ok2 = self._write(self.exc_path, self.render_exceptions())
        ok3 = self._write(self.log_path, self.render_task_log())
        if self.also_json:
            self._dump_json(os.path.splitext(self.cls_path)[0] + '.json', {
                'start': self.start_ts, 'end': self.end_ts,
                'summary': self.summary, 'items': self.classifications,
            })
            self._dump_json(os.path.splitext(self.exc_path)[0] + '.json', {
                'start': self.start_ts, 'end': self.end_ts,
                'exceptions': self.exceptions,
            })
            self._dump_json(os.path.splitext(self.log_path)[0] + '.json', {
                'start': self.start_ts, 'end': self.end_ts,
                'transitions': self.transitions,
            })
        return ok1 and ok2 and ok3


# =============================================================================
# 状态机
# =============================================================================

class StateMachine:
    """YAML 驱动的极简状态机.

    每个状态可以有:
        handler         : target 上的方法名, 返回 event (str) 或 (event, note)
        transitions     : {event: next_state}
        on_exception    : 覆盖 default_exception, 值是事件名或状态名
        terminal        : true 则进入即结束
        description     : 记日志用
    """

    def __init__(self, spec, target, reporter, logger=None):
        sm = spec.get('state_machine', spec)
        self.initial = sm.get('initial')
        self.states = sm.get('states') or {}
        self.default_exception = sm.get('default_exception', 'error')
        self.max_transitions = int(sm.get('max_transitions', 200))
        self.reports_cfg = sm.get('reports') or {}
        self.target = target
        self.reporter = reporter
        self.logger = logger
        if not self.initial or self.initial not in self.states:
            raise StateMachineError(f'非法 initial state: {self.initial}')

    def _log_info(self, msg):
        if self.logger:
            self.logger.info(msg)

    def _log_err(self, msg):
        if self.logger:
            self.logger.error(msg)

    def run(self):
        state = self.initial
        self.reporter.add_transition(None, state, 'START', 'init')
        self._log_info(f'[SM] START -> {state}')
        steps = 0

        while True:
            steps += 1
            if steps > self.max_transitions:
                raise StateMachineError(
                    f'状态转移超过上限 {self.max_transitions}, 疑似死循环')

            spec = self.states.get(state)
            if spec is None:
                raise StateMachineError(f'未知状态: {state}')

            if spec.get('terminal'):
                self._log_info(f'[SM] TERMINAL {state}')
                self.reporter.add_transition(state, state, 'END', 'terminal')
                return

            desc = spec.get('description', '')
            self._log_info(f'[SM] === {state} ===  {desc}')

            handler_name = spec.get('handler')
            transitions = spec.get('transitions') or {}
            event = 'done'
            note = ''

            if handler_name:
                handler = getattr(self.target, handler_name, None)
                if handler is None:
                    raise StateMachineError(
                        f'状态 {state} 的 handler {handler_name} 在 target 上不存在')

                try:
                    ret = handler()
                    if isinstance(ret, tuple) and len(ret) == 2:
                        event, note = ret
                    elif isinstance(ret, str):
                        event = ret
                    elif ret is None:
                        event = 'done'
                    else:
                        event = str(ret)
                except Exception as e:
                    on_exc = spec.get('on_exception') or self.default_exception
                    grid_id = getattr(self.target, 'sm_current_grid_id', None)

                    # on_exc 优先当作事件名, 找不到再当作状态名直接跳
                    if on_exc in transitions:
                        event = on_exc
                        note = f'{type(e).__name__}: {e}'
                        self.reporter.add_exception(
                            state, e, grid_id=grid_id, action=event)
                        self._log_err(
                            f'[SM] {state} 异常 (event={event}): {e}')
                    else:
                        # 直接跳到 on_exc 状态
                        self.reporter.add_exception(
                            state, e, grid_id=grid_id, action=on_exc)
                        self.reporter.add_transition(
                            state, on_exc, 'exception',
                            f'{type(e).__name__}: {e}')
                        self._log_err(
                            f'[SM] {state} 异常 -> {on_exc}: {e}')
                        state = on_exc
                        continue

            nxt = transitions.get(event) or transitions.get('*')
            if nxt is None:
                raise StateMachineError(
                    f'状态 {state} 未定义 event={event!r} 的下一站; '
                    f'可选: {list(transitions.keys())}')

            self.reporter.add_transition(state, nxt, event, note)
            self._log_info(f'[SM] {state} --[{event}]--> {nxt}')
            state = nxt


# =============================================================================
# 便捷函数
# =============================================================================

def load_yaml(path):
    """延迟导入 pyyaml, 缺依赖时错误信息更清晰."""
    import yaml
    with open(path, 'r', encoding='utf-8') as fp:
        return yaml.safe_load(fp)

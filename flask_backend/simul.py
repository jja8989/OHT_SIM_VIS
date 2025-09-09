import numpy as np
import random
import networkit as nk
import json
import threading
import gzip
import base64
import math
import multiprocessing
import copy
from queue import Queue
import math


import time
random.seed(time.time())


def compress_data(data):
    json_data = json.dumps(data).encode('utf-8')
    compressed_data = gzip.compress(json_data)
    return base64.b64encode(compressed_data).decode('utf-8')

class node():
    def __init__(self, id, coord):

        self.id = id
        self.coord = np.array(coord)

        self.incoming_edges = []
        self.outgoing_edges = []
        
from collections import deque
import math

class RollingAvgWithIdle:
    """entry/exit 기반 롤링 평균속도 + 유휴 구간 가상 통과(1대/τ)"""
    def __init__(self, window_sec: float, length: float, max_speed: float):
        self.W = float(window_sec)
        self.L = float(length)
        self.V = float(max_speed)
        self.tau = self.L / max(self.V, 1e-12)

        # 덱에 [start, end, dist, dur]
        self.win = deque()
        self.sum_dist = 0.0
        self.sum_time = 0.0
        
        self.wip_win = deque()    # [s, e, dist, dur]
        self.wip_sum_dist = 0.0
        self.wip_sum_time = 0.0
        self.wip_anchor = None    # 마지막 WIP 적산 시각

        self.last_avg = 0.0
        self.impute_speed = self.V 

        # 유휴 가상 통과 관리
        self.last_event_time = -math.inf       # 마지막 실제/가상 통과의 end(=exit) 시각
        self.last_impute_anchor = -math.inf    # 마지막 가상 통과가 만들어진 시각

    def _append(self, s: float, e: float, d: float, u: float):
        self.win.append([s, e, d, u])
        self.sum_dist += d
        self.sum_time += u
        
    def _wip_append(self, s, e, d, u):
        self.wip_win.append([s, e, d, u])
        self.wip_sum_dist += d
        self.wip_sum_time += u
        
    def _wip_prune_left(self, t0: float):
        while self.wip_win and self.wip_win[0][1] <= t0:
            _, _, d, u = self.wip_win.popleft()
            self.wip_sum_dist -= d
            self.wip_sum_time -= u
        if self.wip_win:
            s, e, d, u = self.wip_win[0]
            if s < t0 < e and u > 1e-12:
                cut = t0 - s
                frac = cut / u
                d_cut = d * frac
                u_cut = u * frac
                self.wip_win[0][0] = t0
                self.wip_win[0][2] -= d_cut
                self.wip_win[0][3] -= u_cut
                self.wip_sum_dist -= d_cut
                self.wip_sum_time -= u_cut

    def wip_clear(self, now: float | None = None):
        self.wip_win.clear()
        self.wip_sum_dist = 0.0
        self.wip_sum_time = 0.0
        self.wip_anchor = now  # None 또는 now로 재설정
        
    # def wip_accumulate(self, now: float, v_wip: float):
    #     """연속 적산: 직전 anchor 이후 Δt를 v_wip로 누적."""
    #     if self.wip_anchor is None:
    #         self.wip_anchor = now
    #         return
    #     dt = now - self.wip_anchor
    #     if dt <= 0:
    #         return
    #     dt = float(dt)
    #     v = max(0.0, min(float(v_wip), self.V))
    #     dist = v * dt
    #     self._wip_append(now - dt, now, dist, dt)
    #     self.wip_anchor = now
            
    def wip_accumulate(self, now: float, v_wip: float):
        if self.wip_anchor is None:
            self.wip_anchor = now
            return
        dt = now - self.wip_anchor
        if dt <= 0:
            return
        v = max(0.0, min(float(v_wip), self.V))
        dist = v * dt

        # ★ 마지막 WIP 세그먼트 연장
        if self.wip_win and abs(self.wip_win[-1][1] - self.wip_anchor) < 1e-12:
            self.wip_win[-1][1] += dt
            self.wip_win[-1][2] += dist
            self.wip_win[-1][3] += dt
            self.wip_sum_dist += dist
            self.wip_sum_time += dt
        else:
            self._wip_append(now - dt, now, dist, dt)

        self.wip_anchor = now


    def seed_carry_avg(self, now: float, v_carry: float):
        """직전 윈도우 W를 v_carry로 가득 채운 1개 합산 세그먼트로 시딩"""
        dur  = self.W
        dist = max(0.0, float(v_carry)) * dur
        start = now - dur
        end   = now
        # 기존 창 초기화 후 carry로 채움
        self.win.clear()
        self.sum_dist = dist
        self.sum_time = dur
        self.win.append([start, end, dist, dur])
        self.last_avg = float(v_carry)
        # 기준점은 now로 고정 (idle 보간 폭주/점프 방지)
        self.last_event_time    = end
        self.last_impute_anchor = end
        
        self.wip_clear(now)
        self.last_event_time    = end
        self.last_impute_anchor = end
        
    def _prune_left(self, t0: float):
        # 완전히 창 밖인 세그먼트 제거
        while self.win and self.win[0][1] <= t0:
            _, _, d, u = self.win.popleft()
            self.sum_dist -= d
            self.sum_time -= u
        # 왼쪽 경계가 세그먼트 내부를 자르면 비례 절단
        if self.win:
            s, e, d, u = self.win[0]
            if s < t0 < e and u > 1e-12:
                cut = t0 - s
                frac = cut / u
                d_cut = d * frac
                u_cut = u * frac
                self.win[0][0] = t0
                self.win[0][2] -= d_cut
                self.win[0][3] -= u_cut
                self.sum_dist -= d_cut
                self.sum_time -= u_cut

    def add_completion(self, entry: float, exit: float):
        dur = max(exit - entry, 1e-12)
        self._append(entry, exit, self.L, dur)
        self.last_event_time = exit
        self.last_impute_anchor = max(self.last_impute_anchor, exit)
        self.impute_speed = self.V
        self.wip_clear(exit)



    def impute_idle_until(self, now: float, is_idle: bool):
        if not is_idle or self.tau <= 1e-12:
            return
        # ★ 추가: 아직 어떤 이벤트도 없으면(둘 다 -inf) 이번 호출을 '기준점 설정'으로만 사용
        if self.last_impute_anchor == -math.inf and self.last_event_time == -math.inf:
            self.last_impute_anchor = now
            return

        anchor = max(self.last_impute_anchor, self.last_event_time)
        
        dur = now - anchor
        if dur <= 0: return
        dist = self.impute_speed * dur
        self._append(now - dur, now, dist, dur)
        self.last_impute_anchor = now
        self.last_event_time = max(self.last_event_time, now)
        
        # gap = now - anchor
        # n = int(gap // self.tau)
        # if n <= 0:
        #     return

        # dur  = n * self.tau
        # dist = self.impute_speed * dur   # ← Vmax 또는 keep 등 '지정한 보간 속도'
        # start = now - dur
        # end   = now
        
        # self._append(start, end, dist, dur)
        # self.last_impute_anchor = end
        # self.last_event_time = max(self.last_event_time, end)
        
    def get_avg_with_wip(self, now: float):
        t0 = now - self.W
        self._prune_left(t0)
        self._wip_prune_left(t0)
        tot_time = self.sum_time + self.wip_sum_time
        if tot_time > 1e-12:
            v = (self.sum_dist + self.wip_sum_dist) / tot_time
            v = max(0.0, min(v, self.V))
            self.last_avg = v
            return v
        return self.last_avg

    def get_avg(self, now: float):
        t0 = now - self.W
        self._prune_left(t0)
        if self.sum_time > 1e-12:
            v = self.sum_dist / self.sum_time
            v = max(0.0, min(v, self.V))
            self.last_avg = v
            return v
        return self.last_avg
        
    # def warm_reset(self, now: float):
    #     """
    #     modi_rail 등으로 entry/exit 기록을 비운 직후 호출.
    #     기존 평균을 표시로 유지하고, 가상통과 기준점을 now로 고정.
    #     """
    #     # 외부에서 기록/포인터 비웠다면 여기선 avg만 warm-start
    #     if self._rt is None:
    #         # _rt는 다음 calculate에서 생성될 테니 화면 표시만 유지
    #         self.avg_speed = getattr(self, "avg_speed", self.max_speed)
    #         return

    #     keep = self._rt.last_avg  # 기존 평균 보존

    #     # 롤링 창 초기화
    #     self._rt.win.clear()
    #     self._rt.sum_dist = 0.0
    #     self._rt.sum_time = 0.0
    #     self._rt.last_avg = keep

    #     # ★ 중요: 첫 idle 보간에서 gap=∞ 방지
    #     self._rt.last_event_time = -math.inf      # 리셋 이후 실제 통과는 아직 없음
    #     self._rt.last_impute_anchor = float(now)  # 앵커를 now로

    #     # UI 표시값 동기화(선택)
    #     self.avg_speed = keep
        
    def seed_freeflow(self, now: float):
        """초기 안정화를 위해 윈도우를 free-flow(V)로 채운 '합산 세그먼트' 1개 주입."""
        if self.tau <= 1e-12:
            return  # V가 0에 가까우면 시딩 무의미
        n = max(1, int(math.ceil(self.W / self.tau)))  # 윈도우를 덮도록 필요한 '대수'
        start = now - n * self.tau
        end   = now
        dist  = n * self.L
        dur   = n * self.tau
        self._append(start, end, dist, dur)
        # 앵커 고정: 이후 idle 보간이 폭주하지 않도록 기준을 now로 맞춘다
        self.last_event_time   = end
        self.last_impute_anchor = end
        self.last_avg = self.V  # 초기 표시값도 안정적으로 V


        
class edge():
    def __init__(self, source, dest, length, max_speed):
        self.id = f'{source.id}-{dest.id}'
        self.source = source.id
        self.dest = dest.id
        self.length = length
        self.unit_vec = (dest.coord - source.coord) / self.length
        
        dx, dy = self.unit_vec
        self.angleDeg = math.degrees(math.atan2(dy, dx)) + 90
        
        self.max_speed = max_speed
        self.OHTs = []  
        
        self.count = 0  
        self.avg_speed = max_speed
        self.entry_exit_records = {} 
        
        
        self._last_synced_idx = {}    # {oht_id: 마지막으로 소비한 index}
        self._rt = None
        self.avg_speed = self.max_speed
        
        self._pending_warm_reset = False
        self._pending_keep_avg = None

        self.is_removed = False
        
    def _ensure_rt(self, window_sec: float):
        if (self._rt is None or
            abs(self._rt.W - window_sec) > 1e-9 or
            abs(self._rt.L - self.length) > 1e-9 or
            abs(self._rt.V - self.max_speed) > 1e-9):
            self._rt = RollingAvgWithIdle(window_sec, self.length, self.max_speed)
            
    def _sync_from_records(self, now: float):
        """
        지난 호출 이후 '새로 완결된' (entry, exit)만 집계기에 반영.
        """
        for oht_id, recs in self.entry_exit_records.items():
            last_i = self._last_synced_idx.get(oht_id, -1)
            # 기록이 시간순 append라고 가정 (아니면 정렬 필요)
            i = last_i + 1
            n = len(recs)
            while i < n:
                entry, exit = recs[i]
                if exit is None or exit > now:
                    # 아직 빠져나가지 않았거나 미래면 스킵
                    break
                # 완료 통과만 반영
                self._rt.add_completion(entry, exit)
                i += 1
            # i-1까지 소비했으면 인덱스 업데이트
            self._last_synced_idx[oht_id] = max(self._last_synced_idx.get(oht_id, -1), i - 1)
            
    def request_warm_reset(self, keep_avg=None):
        """
        다음 calculate_avg_speed 호출 시점에 warm reset을 적용하도록 표시만 해둔다.
        keep_avg가 주어지면 그 값을 초기 평균으로 사용(없으면 self.avg_speed).
        """
        self._pending_warm_reset = True
        self._pending_keep_avg = keep_avg
            
    # def _wip_extra(self, current_time: float, time_window: float):
    #     """exit 없이 edge 위에 올라와 있는 OHT들의 '임시' 거리/시간 기여를 계산(덱에는 저장 X)."""
    #     t0 = current_time - time_window

    #     # 1) 속도 추정: 현재 엣지 속도들의 중앙값(막혔으면 0), 상한은 max_speed
    #     speeds = [getattr(oht, "speed", 0.0) for oht in self.OHTs]
    #     if speeds:
    #         speeds.sort()
    #         mid = len(speeds)//2
    #         v_est = (speeds[mid] if len(speeds) % 2 == 1 else 0.5*(speeds[mid-1]+speeds[mid]))
    #         v_est = max(0.0, min(float(v_est), float(self.max_speed)))
    #         # 완전 막힘(거의 0) 노이즈 컷
    #         if max(speeds) < 1e-6:
    #             v_est = 0.0
    #     else:
    #         v_est = 0.0

    #     dist_extra = 0.0
    #     time_extra = 0.0

    #     # 2) 열린 레코드들만( exit=None ) 윈도우 겹친 시간만큼 임시 누적
    #     for oht_id, recs in self.entry_exit_records.items():
    #         if not recs:
    #             continue
    #         entry, exit = recs[-1]
    #         if exit is not None:
    #             continue  # 완료된 건 WIP 아님
    #         start = max(entry, t0)
    #         overlap = current_time - start
    #         if overlap > 0:
    #             time_extra += overlap
    #             dist_extra += v_est * overlap

    #     return dist_extra, time_extra



    def calculate_avg_speed(self, time_window: float, current_time: float):
        self._ensure_rt(time_window)
        if self.is_removed:
            if self._rt is not None:
                self._rt.win.clear()
                self._rt.sum_dist = 0.0
                self._rt.sum_time = 0.0
                self._rt.wip_clear(current_time)
                self._rt.last_avg = 0.0
                self._rt.last_event_time = current_time
                self._rt.last_impute_anchor = current_time
            self.avg_speed = 0.0
            return 0.0
        
        if self._pending_warm_reset:
            keep = self._pending_keep_avg if (self._pending_keep_avg is not None) else self.avg_speed
            # 직전 윈도우를 keep으로 채워 넣고 시작
            self._rt.seed_carry_avg(current_time, keep)
            # 이후 idle 보간은 free-flow로 → 점진적 변화
            self._rt.impute_speed = self._rt.V

            self._pending_warm_reset = False
            self._pending_keep_avg = None

            
        if current_time <= 1e-9 and len(self._rt.win) == 0:
            self._rt.seed_freeflow(current_time)
    
        # 1) 누락분만 동기화(새로 완료된 exit만 반영)
        self._sync_from_records(current_time)
        # 2) 유휴 가상 통과(정체 무시는 is_idle만으로 판단)
        # is_idle = (len(self.OHTs) == 0)
        # self._rt.impute_idle_until(current_time, is_idle=is_idle)
        
        # dist_extra, time_extra = self._wip_extra(current_time, time_window)
        # if self._rt.sum_time + time_extra > 1e-12:
        #     v = (self._rt.sum_dist + dist_extra) / (self._rt.sum_time + time_extra)
        # else:
        #     v = self.avg_speed  # 분모 0이면 직전값 유지
        if len(self.OHTs) > 0:
        # WIP 추정속도: 평균(클립) + EMA
            speeds = [max(0.0, min(float(getattr(oht, "speed", 0.0)), float(self.max_speed))) for oht in self.OHTs]
            if speeds:
                v_est = sum(speeds) / len(speeds)
            else:
                v_est = 0.0
            # 짧은 EMA로 과도한 출렁임 완화(선택)
            if not hasattr(self, "_wip_v_ema"):
                self._wip_v_ema = v_est
            else:
                lam = 0.7     # 0.6~0.85 권장
                self._wip_v_ema = lam*self._wip_v_ema + (1-lam)*v_est
            self._rt.wip_accumulate(current_time, self._wip_v_ema)
            # idle 보간은 하지 않음
        else:
            # OHT 없음 → WIP 리셋 + idle 보간으로 free-flow
            self._rt.wip_clear(current_time)
            self._rt.impute_idle_until(current_time, is_idle=True)

        # 3) 최종 평균 (윈도우 prune 포함)
        v = self._rt.get_avg_with_wip(current_time)
        # 3) 평균 조회(윈도우 Prune 포함)
        # v = self._rt.get_avg(current_time)
        self.avg_speed = v
        return v
                    
    # def prune_old_records(self, current_time, time_window):
    #     t0 = max(0, current_time - time_window)

    #     for oht_id, records in list(self.entry_exit_records.items()):
    #         pruned = []
    #         for entry, exit in records:
    #             if exit is None:
    #                 pruned.append((entry, exit))
    #                 continue
    #             start = max(entry, t0)
    #             end = min(exit, current_time)
    #             if end - start > 0:
    #                 pruned.append((entry, exit))

    #         if pruned:
    #             self.entry_exit_records[oht_id] = pruned
    #         else:
    #             del self.entry_exit_records[oht_id]

    # def calculate_avg_speed(self, time_window, current_time):

    #     self.prune_old_records(current_time, time_window)

    #     t0 = max(0, current_time - time_window)
    #     total_time = 0.0
    #     total_dist = 0.0

    #     for _, records in self.entry_exit_records.items():
    #         for entry, exit in records:
    #             ex = current_time if exit is None else exit
    #             start = max(entry, t0)
    #             end = min(ex, current_time)
    #             overlap = end - start
    #             if overlap <= 0:
    #                 continue
    #             dur = ex - entry
    #             if dur > 1e-6:
    #                 total_time += overlap
    #                 total_dist += self.length * (overlap / dur)

    #     if total_time > 0:
    #         observed_avg = total_dist / total_time
    #     else:
    #         observed_avg = getattr(self, "avg_speed", self.max_speed)

    #     utilization = min(1.0, total_time / max(1e-6, float(time_window)))
    #     blended = utilization * observed_avg + (1.0 - utilization) * self.max_speed

    #     prev = getattr(self, "avg_speed", blended)
    #     alpha = 0.2 
    #     smoothed = alpha * blended + (1 - alpha) * prev

    #     smoothed = max(0.0, min(smoothed, self.max_speed))

    #     self.avg_speed = smoothed
    #     return self.avg_speed
        
    # def calculate_avg_speed(self, time_window, current_time):
    #     self.prune_old_records(current_time, time_window)

    #     t0 = max(0, current_time - time_window)
    #     total_time = 0.0
    #     total_dist = 0.0

    #     # 1) throughput(완료 통과)로만 속도 산출
    #     for _, records in self.entry_exit_records.items():
    #         for entry, exit in records:
    #             if exit is None:
    #                 continue
    #             ex = exit
    #             start = max(entry, t0)
    #             end = min(ex, current_time)
    #             overlap = end - start
    #             if overlap <= 0:
    #                 continue
    #             dur = ex - entry
    #             if dur > 1e-6:
    #                 total_time += overlap
    #                 total_dist += self.length * (overlap / dur)

    #     # 2) 타깃 속도 결정
    #     if total_time > 0:
    #         target = total_dist / total_time               # 정상: 실제 평균속도
    #     else:
    #         # 통과 기록이 전혀 없을 때: OHT가 있으면 정체(0), 없으면 유휴(max)
    #         has_stuck_oht = (len(self.OHTs) > 0)
    #         target = 0.0 if has_stuck_oht else self.max_speed

    #     # 3) 스무딩만 적용 (max와 혼합 금지)
    #     prev = getattr(self, "avg_speed", target)
    #     alpha = 0.2
    #     smoothed = alpha * target + (1 - alpha) * prev
    #     smoothed = max(0.0, min(smoothed, self.max_speed))

    #     self.avg_speed = smoothed
    #     return self.avg_speed

    # def calculate_avg_speed(self, time_window, current_time):
    #     # 컷이면 바로 0 (표시/제어 목적상)
    #     if self.is_removed:
    #         self.avg_speed = 0.0
    #         # 상태 초기화(선택)
    #         self.idle_since = None
    #         self.cong_since = None
    #         self.last_components.update({"flow_time":0,"idle_time":0,"cong_time":time_window,"flow_avg":0})
    #         return 0.0

    #     self.prune_old_records(current_time, time_window)
    #     t0 = max(0, current_time - time_window)

    #     # 1) 실제 통과(throughput)
    #     total_time, total_dist = 0.0, 0.0
    #     for _, recs in self.entry_exit_records.items():
    #         for entry, exit in recs:
    #             if exit is None: 
    #                 continue
    #             start = max(entry, t0); end = min(exit, current_time)
    #             overlap = end - start
    #             if overlap <= 0: 
    #                 continue
    #             dur = exit - entry
    #             if dur > 1e-6:
    #                 total_time += overlap
    #                 total_dist += self.length * (overlap / dur)

    #     flow_time = total_time
    #     flow_avg  = (total_dist / total_time) if total_time > 0 else self.max_speed

    #     idle_time = 0.0
    #     if len(self.OHTs) == 0:
    #         if self.idle_since is None:
    #             self.idle_since = current_time
    #         idle_overlap = max(0.0, current_time - max(t0, self.idle_since))
    #         idle_time = idle_overlap
    #         total_time += idle_overlap
    #         total_dist += self.max_speed * idle_overlap
    #         self.cong_since = None
    #     else:
    #         self.idle_since = None
    #         if flow_time == 0.0:
    #             if self.cong_since is None:
    #                 self.cong_since = current_time
    #             cong_overlap = max(0.0, current_time - max(t0, self.cong_since))
    #             speeds = [oht.speed for oht in self.OHTs]
    #             if speeds:
    #                 v_fill = min(max(0.0, float(np.median(speeds))), self.v_cap_cong)
    #                 if max(speeds) < self.v_block_eps:
    #                     v_fill = 0.0
    #             else:
    #                 v_fill = 0.0
    #             total_time += cong_overlap
    #             total_dist += v_fill * cong_overlap
    #             cong_time = cong_overlap
    #         else:
    #             self.cong_since = None
    #             cong_time = 0.0

    #     # 4) 최종 시간가중 평균
    #     if total_time > 0:
    #         v = total_dist / total_time
    #     else:
    #         v = self.max_speed if len(self.OHTs) == 0 else 0.0

    #     eps = max(1e-3, 1e-4 * self.max_speed)
    #     if v < eps: v = 0.0
    #     elif self.max_speed - v < eps: v = self.max_speed

    #     self.avg_speed = max(0.0, min(v, self.max_speed))
    #     # (선택) 진단용 구성요소 저장
    #     self.last_components.update({
    #         "flow_time": round(flow_time, 3),
    #         "idle_time": round(idle_time, 3),
    #         "cong_time": round(total_time - flow_time - idle_time, 3),
    #         "flow_avg": round(flow_avg, 3)
    #     })
    #     return self.avg_speed



    
class port():
    def __init__(self, name, from_node, to_node, from_dist):

        self.name = name
        self.from_node = from_node
        self.to_node = to_node
        self.from_dist = from_dist
        self.edge = None

        
class OHT():
    def __init__(self, id, from_node, from_dist, speed, acc, rect, path, node_map, edge_map, port_map):
        self.id = id 
        self.from_node = from_node.id 
        self.from_dist = from_dist 
        
        self.node_map = node_map
        self.edge_map = edge_map
        self.port_map = port_map
        
        self.path = path
        self.path_to_start = []
        self.path_to_end = []
        self.edge = path.pop(0) if path else None 
        
        self.pos = (
            from_node.coord + self.edge_map[self.edge].unit_vec * self.from_dist
            if self.edge else from_node.coord
        ) 
        
        self.speed = speed 
        self.acc = acc 
        self.rect = rect 
        
        self.edge.OHTs.append(self) if self.edge else None 
        
        
        self.start_port = None
        self.end_port = None 
        self.wait_time = 0 
        
        self.status = "IDLE" 
    

    def cal_pos(self, time_step):
        delta = self.speed * time_step + 0.5 * self.acc * time_step**2
        if delta < 0:
            delta = 0
        self.from_dist += delta

        from_node = self.node_map[self.from_node]
        edge = self.edge_map[self.edge] if self.edge is not None else None
        
        self.pos = from_node.coord + edge.unit_vec * self.from_dist if self.edge is not None else from_node.coord

        
    def move(self, time_step, current_time):
        
        if self.status == 'ON_REMOVED':
            self.speed = 0
            self.acc = 0
            return
        
        if not self.edge:
            self.speed = 0
            self.acc = 0
            if len(self.path) != 0:
                from_node = self.node_map[self.from_node]
                # from_node.OHT = None
            return


        if self.wait_time > 0:
            self.wait_time -= time_step
            if self.wait_time <= 0: 
                self.wait_time = 0
                if self.status == 'STOP_AT_START':
                    self.status = 'TO_END'
                elif self.status == 'STOP_AT_END':
                    self.status = 'IDLE'
            return 


        if self.is_arrived():
            self.arrive()
            return
        
        from_node = self.node_map[self.from_node]
        edge = self.edge_map[self.edge]
        
                    
        self.col_check(time_step)
            
        while (self.from_dist > edge.length):
            
            self.from_node = edge.dest

            self.from_dist = self.from_dist - edge.length 
            
            if edge:
                try:

                    if len(self.path) > 0:
                
                        exit_record = edge.entry_exit_records.get(self.id, [])
                        if exit_record and exit_record[-1][1] is None:
                            exit_record[-1] = (exit_record[-1][0], current_time)
                        edge.entry_exit_records[self.id] = exit_record
                                            
                        edge.OHTs.remove(self)
                        
                        self.edge = self.path.pop(0)
                        
                        edge = self.edge_map[self.edge]
                        
                        if self not in edge.OHTs:
                            edge.OHTs.append(self)
                            edge.count += 1
                            # if self.id not in edge.entry_exit_records:
                            #     edge.entry_exit_records[self.id] = []
                            #     edge.entry_exit_records[self.id].append((current_time, None))
                            recs = edge.entry_exit_records.setdefault(self.id, [])
                            if (not recs) or (recs[-1][1] is not None):
                                recs.append((current_time, None))
                            
                    else:
                        self.speed = 0
                        self.acc = 0
                        self.from_dist = edge.length
                        self.from_node = edge.source
                        return
                except:
                    print('update error : ', self.edge)

  
            if self.is_arrived():
                self.arrive()
                return
            

        if self.is_arrived():
            self.arrive()
            return

        self.speed = min(max(self.speed + self.acc * time_step, 0), edge.max_speed)
        
    def is_arrived(self):
        
        edge = self.edge_map[self.edge]

        if self.status == "TO_START":
            start_port = self.port_map[self.start_port] if self.start_port else None

            return (start_port 
                    and start_port.from_node == self.from_node and start_port.to_node == edge.dest
                    and self.from_dist >= start_port.from_dist)
        elif self.status == "TO_END":
            end_port = self.port_map[self.end_port] if self.end_port else None

            return (end_port 
                    and end_port.from_node == self.from_node and end_port.to_node == edge.dest
                    and self.from_dist >= end_port.from_dist)
        
    def arrive(self):

        if self.status == "TO_START":
            start_port = self.port_map[self.start_port]
            edge = self.edge_map[self.edge]
            from_node = self.node_map[self.from_node]

            
            self.from_dist = start_port.from_dist
            self.pos = from_node.coord + edge.unit_vec * start_port.from_dist
            self.speed = 0
            self.acc = 0
            self.wait_time = 5
            
            self.status = "STOP_AT_START"
            self.path = self.path_to_end 
            self.path_to_start = [] 
            self.start_port = None

        elif self.status == "TO_END":
            end_port = self.port_map[self.end_port]
            edge = self.edge_map[self.edge]
            from_node = self.node_map[self.from_node]
            
            self.from_dist = end_port.from_dist
            self.pos = from_node.coord + edge.unit_vec * end_port.from_dist
            self.speed = 0
            self.acc = 0
            self.wait_time = 5
            self.end_port = None
            self.status = "STOP_AT_END"
            self.path = []
            self.path_to_end = [] 

        else:
            print(f"OHT {self.id} is idle.")

    
    def col_check(self, time_step):
        
        edge = self.edge_map[self.edge]
        
        emergency_threshold = 1000

        OHTs = edge.OHTs
                
        if len(OHTs) > 1:
            index = OHTs.index(self)

            if index > 0: 
                prev_oht = OHTs[index - 1] 
                dist_diff = prev_oht.from_dist - self.from_dist
                
                
                if 0 < dist_diff < self.rect:
                    emergency_coeff = 1.0 * (dist_diff < emergency_threshold or self.speed == 0)
                    self.acc = emergency_coeff * (-self.speed / time_step) + (1-emergency_coeff) * (-3500)
                    return
                
        # if self.node_map[edge.dest].OHT is not None:
        #     dist_diff = edge.length - self.from_dist
            
        #     if 0 < dist_diff < self.rect:
        #         emergency_coeff = 1.0 * (dist_diff < emergency_threshold)
        #         self.acc = emergency_coeff * (-self.speed / time_step) + (1-emergency_coeff) * (-3500)

        #         return
              
        if len(self.path) > 0:
            next_edge = self.edge_map[self.path[0]]
            try:
                last_oht = next_edge.OHTs[-1]
                rem_diff = edge.length - self.from_dist
                dist_diff = last_oht.from_dist + rem_diff
                if 0 < dist_diff < self.rect:
                    
                    emergency_coeff = 1.0 * (dist_diff < emergency_threshold or self.speed == 0)
                    self.acc = emergency_coeff * (-self.speed / time_step) + (1-emergency_coeff) * (-3500)

                    return
            except:
                pass
        
        incoming_edges = [
            incoming_edge for incoming_edge in self.node_map[edge.dest].incoming_edges
            if incoming_edge != self.edge 
        ]
        
        if len(incoming_edges) == 1: 
            rem_diff = edge.length - self.from_dist 
            try:
                other_oht = self.edge_map[incoming_edges[0]].OHTs[0]
                
                other_diff= self.edge_map[other_oht.edge].length - other_oht.from_dist 
                dist_diff = rem_diff + other_diff
                if 0 < dist_diff < 3 * self.rect and rem_diff > other_diff:
                    
                    emergency_coeff = 1.0 * (dist_diff < emergency_threshold or self.speed == 0)
                    self.acc = emergency_coeff * (-self.speed / time_step) + (1-emergency_coeff) * (-3500)
                    return
                elif rem_diff == other_diff and self.id > other_oht.id: 
                    
                    emergency_coeff = 1.0 * (dist_diff < emergency_threshold or self.speed == 0)
                    self.acc = emergency_coeff * (-self.speed / time_step) + (1-emergency_coeff) * (-3500)
                    return
            except:
                pass

        self.acc = 2000 if self.speed < edge.max_speed else 0

        
        
class AMHS:
    def __init__(self, nodes, edges, ports, num_OHTs, max_jobs, job_list = [], oht_list = []):
        
        self.graph = nk.Graph(directed=True, weighted=True) 
        self.node_map = {}
        self.edge_map = {}
        self.port_map = {}
        
        self.nodes = copy.deepcopy(nodes)


        for node in self.nodes:
            # node.OHT = None
            self.node_map[node.id] = node
            
        self.edges = copy.deepcopy(edges)
        
        for edge in self.edges:
            self.edge_map[edge.id] = edge
            edge.OHTs = []
            edge.count = 0
            edge.avg_speed = edge.max_speed
            edge.entry_exit_records = {} 

        
        self.ports = copy.deepcopy(ports)

        for port in self.ports:
            port.edge = f'{port.from_node}-{port.to_node}'
            self.port_map[port.name] = port
            
        self.OHTs = []
        if job_list != []:
            self.job_queue = [[self.get_port(q[0]), self.get_port(q[1])] for q in job_list]
        else:
            self.job_queue = []
        self.max_jobs = max_jobs
        
        self.node_id_map = {}
        
        
        self.current_time = 0
        self.time_step=0.01
        
                
        self.simulation_running = False  
        self.stop_simulation_event = threading.Event()  
        
                        
        self.back_simulation_running = False
        self.back_stop_simulation_event = threading.Event()
        
        self.original_graph = nk.Graph(directed=True, weighted=True)

        self._create_graph()

        if oht_list != []:
            self.set_initial_OHTs(oht_list)
        else:
            self.initialize_OHTs(num_OHTs)
        
        self.apsp = nk.distance.APSP(self.graph)
        self.apsp.run()
        
        self.original_apsp = nk.distance.APSP(self.original_graph)
        self.original_apsp.run()
        
        self.queue = Queue()
        self.back_queue = Queue()
        


    def _create_graph(self):
        for i, node in enumerate(self.nodes):
            self.graph.addNode()  
            self.original_graph.addNode()
            self.node_id_map[node.id] = i
        
        for edge in self.edges:
            u = self.node_id_map[edge.source] 
            v = self.node_id_map[edge.dest]   
            self.graph.addEdge(u, v, edge.length)
            self.original_graph.addEdge(u, v, edge.length)
            self.node_map[edge.dest].incoming_edges.append(edge.id)
            self.node_map[edge.source].outgoing_edges.append(edge.id)

    def initialize_OHTs(self, num_OHTs):
        available_nodes = self.nodes.copy()
        
        for i in range(num_OHTs):
            if not available_nodes:
                available_nodes = self.nodes.copy() 
            
            start_node = random.choice(available_nodes)
            available_nodes.remove(start_node) 
            
            oht = OHT(
                id=i,
                from_node=start_node,
                from_dist=0,
                speed=0,
                acc=0,
                rect=3000, 
                path=[],
                node_map = self.node_map,
                edge_map = self.edge_map,
                port_map = self.port_map
            )
            # start_node.OHT = oht
            self.OHTs.append(oht)
            
    def set_initial_OHTs(self, oht_list):
        i = 0
        for start_node in oht_list:  
            start_node = self.get_node(start_node)
            oht = OHT(
                id=i,
                from_node=start_node,
                from_dist=0,
                speed=0,
                acc=0,
                rect=3000, 
                path=[],
                node_map = self.node_map,
                edge_map = self.edge_map,
                port_map = self.port_map
            )
            # start_node.OHT = oht
            self.OHTs.append(oht)
            i = i+1
    
    def set_oht(self, oht_origin, oht_new):
        oht_origin.from_node = oht_new['from_node']
        oht_origin.from_dist = oht_new['from_dist']
        oht_origin.edge = f"{oht_new['source']}-{oht_new['dest']}" if oht_new['source'] and oht_new['dest'] else None
        
        oht_origin.start_port = oht_new['startPort'] if oht_new['startPort'] else None
        oht_origin.end_port = oht_new['endPort'] if oht_new['endPort'] else None
        oht_origin.wait_time = oht_new['wait_time']
        
        if oht_origin.edge:
            if oht_origin not in self.edge_map[oht_origin.edge].OHTs:
                self.edge_map[oht_origin.edge].OHTs.append(oht_origin)
                
            if not self.graph.hasEdge(self.node_id_map[oht_new['source']], self.node_id_map[oht_new['dest']]):
                oht_origin.speed = 0
                oht_origin.acc = 0
                oht_origin.status = 'ON_REMOVED'
                oht_origin.cal_pos(0)
                return
            
        oht_origin.speed = oht_new['speed']
        oht_origin.status = oht_new['status']
        
        if oht_origin.status == 'ON_REMOVED':
            if oht_origin.start_port:
                oht_origin.status = "TO_START"
            elif oht_origin.end_port:
                oht_origin.status = "TO_END"
            else:
                oht_origin.status = "IDLE"
                
        # oht_origin.cal_pos(self.time_step)
        oht_origin.cal_pos(0)
             

                
        if oht_origin.start_port != None:
            if oht_origin.edge:
                path_edges_to_start = self.get_path_edges(oht_new['dest'], self.port_map[oht_origin.start_port].to_node)
                path_edges_to_start = self._validate_path(path_edges_to_start)

            else:
                path_edges_to_start = self.get_path_edges(oht_origin.from_node, self.port_map[oht_origin.start_port].to_node)
                path_edges_to_start = self._validate_path(path_edges_to_start)
                
            oht_origin.path_to_start = path_edges_to_start[:]
        
        if oht_origin.end_port != None:
            if oht_origin.status == "TO_END" or oht_origin.status == 'STOP_AT_START':
                if oht_origin.edge:
                    path_edges_to_end = self.get_path_edges(oht_new['dest'], self.port_map[oht_origin.end_port].to_node)
                    path_edges_to_end = self._validate_path(path_edges_to_end)
                else:
                    path_edges_to_end = self.get_path_edges(oht_origin.from_node, self.port_map[oht_origin.end_port].to_node)
                    path_edges_to_end = self._validate_path(path_edges_to_end)
            else:
                path_edges_to_end = self.get_path_edges(self.port_map[oht_origin.start_port].to_node, self.port_map[oht_origin.end_port].to_node)

            oht_origin.path_to_end = path_edges_to_end
        
        if oht_origin.status == "TO_START":
            oht_origin.path = path_edges_to_start[:]
        elif oht_origin.status == "TO_END" or oht_origin.status == "STOP_AT_START":
            oht_origin.path = path_edges_to_end[:]
        else:
            oht_origin.path = []

    def modi_edge(self, source, dest, oht_positions, is_removed):
        if is_removed:
            removed_edge = self.get_edge(source, dest)
            self.edge_map[removed_edge].is_removed = True
            try:       
                self.graph.removeEdge(self.node_id_map[source], self.node_id_map[dest])
                self.apsp = nk.distance.APSP(self.graph)
                self.apsp.run()
            except:
                print(source, dest)
        else:
            edge_to_restore = self.get_edge(source, dest)
            self.edge_map[edge_to_restore].is_removed = False
            if edge_to_restore:
                self.graph.addEdge(self.node_id_map[source], self.node_id_map[dest], self.edge_map[edge_to_restore].length)
                self.apsp = nk.distance.APSP(self.graph)
                self.apsp.run()
            else:
                print(f"Rail {source} -> {dest} not found in original edges.")
        
    
    def reinitialize_simul(self, oht_positions, edge_data):
        for edge in self.edges:
            edge.OHTs.clear()
            edge.entry_exit_records.clear()
            edge._last_synced_idx.clear()   # ← 이것도 같이 초기화 권장

        
        for oht in oht_positions:
            oht_in_amhs = self.get_oht(oht['id'])
            self.set_oht(oht_in_amhs, oht)
            
        edge_map = {f"{edge.source}-{edge.dest}": edge for edge in self.edges}
    
        for edge_info in edge_data:
            edge_key = f"{edge_info['from']}-{edge_info['to']}"
            if edge_key in edge_map:
                edge = edge_map[edge_key]
                edge.count = edge_info.get('count', 0)
                edge.avg_speed = edge_info.get('avg_speed', 0)
            
        for edge in self.edges:
            edge.OHTs.sort(key = lambda oht : -oht.from_dist)
            edge.request_warm_reset(keep_avg=edge.avg_speed)
    
    def generate_job(self):
        for _ in range(500):
            start_port = random.choice(self.ports)
            end_port = random.choice(self.ports)
            while start_port == end_port:  
                end_port = random.choice(self.ports)
            self.job_queue.append((start_port, end_port))
            
    def assign_jobs(self):
        while self.job_queue:
            idle_ohts = [oht for oht in self.OHTs if oht.status == 'IDLE' and not oht.path] 
            
            if not idle_ohts: 
                break
            
            closest_oht = None
            closest_dist = float('inf')
            start_port, end_port = None, None 
            
            for oht in idle_ohts:
                temp_start_port, temp_end_port = self.job_queue[0]  
                dist = self.get_path_distance(oht.from_node, temp_start_port.from_node)
                
                if dist < closest_dist:
                    closest_dist = dist
                    closest_oht = oht
                    start_port, end_port = temp_start_port, temp_end_port  
            
            if closest_oht and start_port and end_port:
                self.job_queue.pop(0) 
                self._assign_job_to_oht(closest_oht, start_port, end_port) 
            else:
                break 

    def _assign_job_to_oht(self, oht, start_port, end_port):
        oht.start_port = start_port.name
        oht.end_port = end_port.name
        oht.status = "TO_START"

        if oht.edge:
            path_edges_to_start = self.get_path_edges(self.edge_map[oht.edge].dest, start_port.to_node)
        else:
            path_edges_to_start = self.get_path_edges(oht.from_node, start_port.to_node)
        
        oht.path_to_start = path_edges_to_start[:]

        path_edges_to_end = self.get_path_edges(start_port.to_node, end_port.to_node)
        oht.path_to_end = path_edges_to_end


        oht.path = path_edges_to_start[:]

        if oht.path:
            if not oht.edge:
                oht.edge = oht.path.pop(0)
                # self.node_map[oht.from_node].OHT = None
                if oht not in self.edge_map[oht.edge].OHTs:
                    self.edge_map[oht.edge].OHTs.append(oht)
                    recs = self.edge_map[oht.edge].entry_exit_records.setdefault(oht.id, [])
                    if (not recs) or (recs[-1][1] is not None):
                        recs.append((self.current_time, None))
                            
    def update_edge_metrics(self, current_time, time_window):
        for edge in self.edges:
            edge.avg_speed = edge.calculate_avg_speed(time_window, current_time)


    def get_node(self, node_id):
        return self.node_map[node_id]

    def get_edge(self, source_id, dest_id):
        edge_key = f'{source_id}-{dest_id}'
        return edge_key
        
    def get_path_distance(self, source_id, dest_id):
        source_idx = self.node_id_map[source_id]
        dest_idx = self.node_id_map[dest_id]

        dist = self.apsp.getDistance(source_idx, dest_idx)
        
        if dist >= 1e100:
            return dist
        else:
            source_idx = self.node_id_map[source_id]
            dest_idx = self.node_id_map[dest_id]            

            dist = self.original_apsp.getDistance(source_idx, dest_idx)                
            return dist
        

    def get_path_edges(self, source_id, dest_id):

        source_idx = self.node_id_map[source_id]
        dest_idx = self.node_id_map[dest_id]
        
        dijkstra = nk.distance.Dijkstra(self.graph, source_idx, storePaths=True, storeNodesSortedByDistance=False, target=dest_idx)
        dijkstra.run()

        path = dijkstra.getPath(dest_idx)
        
        if path:
            return [
                self.get_edge(self.nodes[path[i]].id, self.nodes[path[i+1]].id)
                for i in range(len(path) - 1)
            ]
        else:

            source_idx = self.node_id_map[source_id]
            dest_idx = self.node_id_map[dest_id]
            
            dijkstra = nk.distance.Dijkstra(self.original_graph, source_idx, storePaths=True, storeNodesSortedByDistance=False, target=dest_idx)
            dijkstra.run()

            path = dijkstra.getPath(dest_idx)
            
            return [
                self.get_edge(self.nodes[path[i]].id, self.nodes[path[i+1]].id)
                for i in range(len(path) - 1)
            ]

    
    def get_oht(self, oht_id):
        return next((oht for oht in self.OHTs if oht.id == oht_id), None)
    
    def get_port(self, port_name):
        return self.port_map[port_name]

    def _validate_path(self, path):
        valid_path = []
        for edge in path:
            if self.graph.hasEdge(self.node_id_map[self.edge_map[edge].source], self.node_id_map[self.edge_map[edge].dest]):
                valid_path.append(edge)
            else:
                break
        return valid_path
    
        
    def start_simulation(self, socketio, sid, current_time, max_time = 4000, time_step = 0.1, isAccel = True):
     
        if self.simulation_running:
            print("Simulation is already running. Stopping the current simulation...")
            self.stop_simulation_event.set()
            while self.simulation_running:
                socketio.sleep(0.01)
            return
        
        self.simulation_running = True
        self.stop_simulation_event.clear()
        
        count = 0
        
        if (isAccel):
            _current_time = 0
            
            last_saved_time = -10
            

            while _current_time < current_time:
                if self.stop_simulation_event.is_set():
                    break
                
                if count % 100 == 0:
                    self.generate_job()
                
                self.assign_jobs()
                
                for oht in self.OHTs:
                    oht.move(time_step, _current_time)

                self.update_edge_metrics(_current_time, time_window=60)
                
                for oht in self.OHTs:
                    oht.cal_pos(time_step)

                if _current_time - last_saved_time > 10:
                    edge_data = [(last_saved_time+10, edge.id, edge.avg_speed) for edge in self.edges]
                    self.queue.put(edge_data)
                    last_saved_time += 10
                    
                self.current_time = _current_time

                _current_time += time_step*10
                count += 1
            
        
        last_saved_time = ((current_time - 10) // 10) * 10
        
        edge_metrics_cache = {} 
        

        while current_time <= max_time:
            if self.stop_simulation_event.is_set():
                break
            
            self.current_time = current_time
            
            if count % 100 == 0:
                self.generate_job()
            self.assign_jobs()
                

            oht_positions = []
            for oht in self.OHTs:
                oht.move(time_step, current_time)

            self.update_edge_metrics(current_time, time_window=60)
            
            for oht in self.OHTs:
                oht.cal_pos(time_step)
                
            if current_time - last_saved_time > 10:
                edge_data = [(last_saved_time+10, edge.id, edge.avg_speed) for edge in self.edges]
                self.queue.put(edge_data)
                last_saved_time += 10
                
            if count % 5 == 0:
                for oht in self.OHTs:
                        
                    oht_positions.append({
                        'id': oht.id,
                        'x': oht.pos[0],
                        'y': oht.pos[1],
                        'source': self.edge_map[oht.edge].source if oht.edge else None,
                        'dest': self.edge_map[oht.edge].dest if oht.edge else None,
                        'speed': oht.speed,
                        'status': oht.status,
                        'startPort': oht.start_port if oht.start_port else None,
                        'endPort': oht.end_port if oht.end_port else None,
                        'from_node': oht.from_node if oht.from_node else None,
                        'from_dist': oht.from_dist,
                        'wait_time': oht.wait_time,
                        'angleDeg': self.edge_map[oht.edge].angleDeg if oht.edge else 0

                    })

                updated_edges = []
                for edge in self.edges:
                    key = f"{edge.source}-{edge.dest}"
                    new_metrics = {"count": edge.count, "avg_speed": edge.avg_speed}
                    if edge_metrics_cache.get(key) != new_metrics:
                        edge_metrics_cache[key] = new_metrics
                        updated_edges.append({
                            "from": edge.source,
                            "to": edge.dest,
                            **new_metrics
                        })

                payload = {
                    'time': current_time,
                    'oht_positions': oht_positions,
                    'edges': updated_edges
                }

                compressed_payload = compress_data(payload)
                socketio.emit('updateOHT', {'data': compressed_payload}, to=sid)

            current_time += time_step
            count += 1

        self.simulation_running = False
        print('Simulation ended')
        
    def only_simulation(self, socketio, sid, current_time, max_time = 4000, time_step = 0.1):

        if self.back_simulation_running:
            print("Simulation is already running. Stopping the current simulation...")
            self.back_stop_simulation_event.set()
            while self.back_simulation_running:
                socketio.sleep(0.01) 
            return
                
        self.back_simulation_running = True
        self.back_stop_simulation_event.clear()
        
        _current_time = 0
        
        count = 0
        
        last_saved_time = -10

        while _current_time < current_time:
            if self.back_stop_simulation_event.is_set():
                break
            
            if count % 100 == 0:
                self.generate_job()
            
            self.assign_jobs()
            
            for oht in self.OHTs:
                oht.move(time_step, _current_time)

            self.update_edge_metrics(_current_time, time_window=60)
            
            for oht in self.OHTs:
                oht.cal_pos(time_step)

            if _current_time - last_saved_time > 10:
                edge_data = [(last_saved_time+10, edge.id, edge.avg_speed) for edge in self.edges]
                self.back_queue.put(edge_data)
                last_saved_time += 10
                
            self.current_time = _current_time

            _current_time += time_step*10
            count += 1
            
        
        edge_metrics_cache = {} 

        while current_time <= max_time:
            if self.back_stop_simulation_event.is_set():
                break

            self.current_time = current_time
            
            if count % 100 == 0:
                self.generate_job()
            self.assign_jobs()
                

            for oht in self.OHTs:
                oht.move(time_step, current_time)

            self.update_edge_metrics(current_time, time_window=60)
            
            for oht in self.OHTs:
                oht.cal_pos(time_step)
                
            if current_time - last_saved_time > 10:
                edge_data = [(last_saved_time+10, edge.id, edge.avg_speed) for edge in self.edges]
                self.back_queue.put(edge_data)
                last_saved_time += 10
    
            current_time += time_step
            count += 1
    
        self.back_simulation_running = False
        print('Simulation ended')
        
        socketio.emit("backSimulationFinished", to=sid)

        
        
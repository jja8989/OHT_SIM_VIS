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
from collections import deque

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

class EdieWindow:
    """
    Edie 정의 기반 윈도우 평균속도 집계기.
    - 매 스텝(now)마다: 엣지 위 OHT 수 n, 그들의 평균속도 v_avg를 받아
      dist += v_avg * n * dt, time += n * dt 로 누적 (동시점유 반영)
    - 버킷(기본 1초) 링버퍼로 슬라이딩 윈도우 유지
    - sum_time==0이면 '정의 불가'이므로 last_avg를 그대로 유지
    """
    def __init__(self, window_sec: float, vmax: float, bin_sec: float = 1.0):
        self.W = float(window_sec)
        self.V = float(vmax)
        self.bin = float(bin_sec)
        self.inv_bin = 1.0 / self.bin
        self.N = int(math.ceil(self.W / self.bin)) + 3

        self.bins_dist = [0.0] * self.N
        self.bins_time = [0.0] * self.N
        self.bins_id   = [None] * self.N
        self.sum_dist = 0.0
        self.sum_time = 0.0

        self.head_id = None
        self.anchor = None
        self.last_avg = self.V 
        self._last_eval_bin = -1

    def _bid(self, t: float) -> int:
        return int(t * self.inv_bin + 1e-12)

    def _add_bin(self, b: int, d: float, u: float):
        i = b % self.N
        if self.bins_id[i] != b:
            # 재사용 전에 기존 기여 제거
            self.sum_dist -= self.bins_dist[i]
            self.sum_time -= self.bins_time[i]
            self.bins_dist[i] = 0.0
            self.bins_time[i] = 0.0
            self.bins_id[i]   = b
            if self.head_id is None:
                self.head_id = b
        self.bins_dist[i] += d
        self.bins_time[i] += u
        self.sum_dist     += d
        self.sum_time     += u

    def _drop_older(self, now: float):
        if self.head_id is None:
            return
        min_ok = self._bid(now - self.W)
        h = self.head_id
        while h < min_ok:
            i = h % self.N
            if self.bins_id[i] == h:
                self.sum_dist -= self.bins_dist[i]
                self.sum_time -= self.bins_time[i]
                self.bins_dist[i] = 0.0
                self.bins_time[i] = 0.0
                self.bins_id[i]   = None
            h += 1
        self.head_id = h

    # 외부 API
    def clear(self, now: float | None = None):
        self.bins_dist = [0.0] * self.N
        self.bins_time = [0.0] * self.N
        self.bins_id   = [None] * self.N
        self.sum_dist = 0.0
        self.sum_time = 0.0
        self.head_id = None
        self.anchor = now
        self._last_eval_bin = -1
        # last_avg는 유지(시각화 안정). 초기화까지 원하면 self.last_avg= self.V/0.0로 조정
        
    def reset(self, anchor: float):
        # 창을 완전히 비우고, 다음 적산을 anchor 시각부터 시작
        self.bins_dist = [0.0] * self.N
        self.bins_time = [0.0] * self.N
        self.bins_id   = [None] * self.N
        self.sum_dist = 0.0
        self.sum_time = 0.0
        self.head_id  = None
        self.anchor = anchor
        self._last_eval_bin = -1
        
        
    def seed_freeflow(self, now: float):
        """[now - W, now) 구간을 '가상 1대가 V로 통과'했다고 채우기"""
        # self.reset(now - self.W)
        # # anchor = now - W 상태에서 now까지 누적 → 창이 V로 가득
        # self.accumulate(now, self.V, 1)
        # self.last_avg = self.V
        self.seed_with(now, self.V, n=1.0, fill=1.0)
        
    def seed_with(self, now: float, v: float, n: float = 1.0, fill: float = 1.0):
        """
        [now - fill*W, now) 구간을 'n대가 속도 v로 통과'했다고 가정해 한 번에 채움.
        - fill: 0~1 (1이면 창 전체, 0.3이면 창의 30%만)
        """
        fill = max(0.0, min(1.0, float(fill)))
        v = max(0.0, min(self.V, float(v)))
        if fill <= 0.0 or n <= 0.0:
            self.reset(now)          # 그냥 비우고 앵커만 now
            return
        self.reset(now - fill * self.W)
        self.accumulate(now, v, n)   # 여기서 last_avg도 갱신되도록
        self.last_avg = v            # 즉시 표시 안정


    def accumulate(self, now: float, v_avg: float, n_veh: int):
        """
        anchor~now 구간을 v_avg, n_veh 로 적산하여 윈도우에 반영.
        n_veh==0이면 '엣지 점유가 없었음' → 누적 없음(Edie 정의 그대로).
        """
        if self.anchor is None:
            self.anchor = now
            return
        t1 = self.anchor; t2 = float(now)
        if t2 <= t1:
            return

        if n_veh <= 0:
            # 점유 없으면 누적 없이 기준만 이동
            self.anchor = now
            return

        v = float(v_avg)
        if v < 0.0: v = 0.0
        elif v > self.V: v = self.V

        # 동시점유 반영: dist = v * n * dt, time = n * dt
        b1 = self._bid(t1)
        b2 = self._bid(t2 - 1e-12)
        bw = self.bin

        if b1 == b2:
            dt = t2 - t1
            self._add_bin(b1, v * n_veh * dt, n_veh * dt)
        else:
            dt1 = (b1 + 1) * bw - t1
            if dt1 > 0.0: self._add_bin(b1, v * n_veh * dt1, n_veh * dt1)
            for b in range(b1 + 1, b2):
                self._add_bin(b, v * n_veh * bw, n_veh * bw)
            dt2 = t2 - (b2 * bw)
            if dt2 > 0.0: self._add_bin(b2, v * n_veh * dt2, n_veh * dt2)

        self.anchor = now

    def get_avg(self, now: float, per_second_gate: bool = True):
        if per_second_gate:
            cur_bin = int(now)
            if cur_bin == self._last_eval_bin:
                return self.last_avg

        self._drop_older(now)
        if self.sum_time > 1e-12:
            v = self.sum_dist / self.sum_time
            if v < 0.0: v = 0.0
            elif v > self.V: v = self.V
            self.last_avg = v

        if per_second_gate:
            self._last_eval_bin = int(now)
        return self.last_avg
        
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
        
        self._rt = None
        self.avg_speed = self.max_speed

        self.is_removed = False
    
    def _ensure_rt(self, window_sec: float, bin_sec: float = 1.0):
        if (self._rt is None or
            abs(self._rt.W - window_sec) > 1e-9 or
            abs(self._rt.V - self.max_speed) > 1e-9):
            self._rt = EdieWindow(window_sec, self.max_speed, bin_sec=bin_sec)
            

    def calculate_avg_speed(self, time_window: float, current_time: float):
        self._ensure_rt(time_window, bin_sec=1.0)
        rt = self._rt

        # 제거된 엣지는 0으로
        if self.is_removed:
            rt.clear(current_time)
            self.avg_speed = 0.0
            return 0.0
        
        if (current_time <= 0.15):
            rt.seed_freeflow(current_time)

        # 현재 엣지 위 OHT 평균속도/대수
        n = len(self.OHTs)
        if n > 0:
            maxV = float(self.max_speed)
            ssum = 0.0
            for o in self.OHTs:
                s = getattr(o, "speed", 0.0)
                if s < 0.0: s = 0.0
                elif s > maxV: s = maxV
                ssum += s
            v_avg = ssum / n

            # 실제 점유 누적
            rt.accumulate(current_time, v_avg, n)

        else:
            rt.accumulate(current_time, self.max_speed, 1)

        # 평균 계산
        v = rt.get_avg(current_time, per_second_gate=False)

        self.avg_speed = v
        return v
    
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
        
        if edge is not None:
            self.speed = min(max(self.speed + self.acc * time_step, 0), edge.max_speed)

        
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
                                            
                        edge.OHTs.remove(self)
                        
                        self.edge = self.path.pop(0)
                        
                        edge = self.edge_map[self.edge]
                        
                        if self not in edge.OHTs:
                            edge.OHTs.append(self)
                            edge.count += 1
                            
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

        # self.speed = min(max(self.speed + self.acc * time_step, 0), edge.max_speed)
        
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
        edge_key = self.get_edge(source, dest)
        e = self.edge_map[edge_key]

        if is_removed:
            e.is_removed = True
            try:
                self.graph.removeEdge(self.node_id_map[source], self.node_id_map[dest])
                self.apsp = nk.distance.APSP(self.graph)
                self.apsp.run()
            except:
                print(source, dest)


            if e._rt is not None:
                e._rt.reset(self.current_time)
            e.avg_speed = 0.0 
            
            # removed_edge = self.get_edge(source, dest)
            # self.edge_map[removed_edge].is_removed = True
            # try:       
            #     self.graph.removeEdge(self.node_id_map[source], self.node_id_map[dest])
            #     self.apsp = nk.distance.APSP(self.graph)
            #     self.apsp.run()
            # except:
            #     print(source, dest)
        else:
            e.is_removed = False
            if edge_key:
                self.graph.addEdge(self.node_id_map[source], self.node_id_map[dest], e.length)
                self.apsp = nk.distance.APSP(self.graph); self.apsp.run()
            else:
                print(f"Rail {source} -> {dest} not found in original edges.")

            if e._rt is not None:
                e._rt.reset(self.current_time)
                
            # edge_to_restore = self.get_edge(source, dest)
            # self.edge_map[edge_to_restore].is_removed = False
            # if edge_to_restore:
            #     self.graph.addEdge(self.node_id_map[source], self.node_id_map[dest], self.edge_map[edge_to_restore].length)
            #     self.apsp = nk.distance.APSP(self.graph)
            #     self.apsp.run()
            # else:
            #     print(f"Rail {source} -> {dest} not found in original edges.")
        
    
    def reinitialize_simul(self, oht_positions, edge_data):
        for edge in self.edges:
            edge.OHTs.clear()
            if edge._rt is None:
                edge._ensure_rt(60)
            else:
                edge._rt.reset(self.current_time) 
        
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
            carry_v = edge.avg_speed if edge.avg_speed > 0 else edge.max_speed
            edge._rt.seed_with(self.current_time, carry_v, n=1.0, fill=0.35)
            edge.avg_speed = carry_v
    
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
            source_idx = self.node_id_map[source_id]
            dest_idx = self.node_id_map[dest_id]            
            dist = self.original_apsp.getDistance(source_idx, dest_idx)     
            return dist
        else:
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
                    oht.move(time_step*10, _current_time)

                self.update_edge_metrics(_current_time, time_window=60)
                
                for oht in self.OHTs:
                    oht.cal_pos(time_step*10)

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
                    # key = f"{edge.source}-{edge.dest}"
                    # new_metrics = {"count": edge.count, "avg_speed": edge.avg_speed}
                    # if edge_metrics_cache.get(key) != new_metrics:
                    #     edge_metrics_cache[key] = new_metrics
                    #     updated_edges.append({
                    #         "from": edge.source,
                    #         "to": edge.dest,
                    #         **new_metrics
                    #     })

                    updated_edges.append({
                        "from": edge.source,
                        "to": edge.dest,
                        "count": edge.count,
                        "avg_speed": edge.avg_speed,
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
                oht.move(time_step*10, _current_time)

            self.update_edge_metrics(_current_time, time_window=60)
            
            for oht in self.OHTs:
                oht.cal_pos(time_step*10)

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

        
        
        
        
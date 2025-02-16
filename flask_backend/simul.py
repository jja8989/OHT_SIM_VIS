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


def compress_data(data):
    json_data = json.dumps(data).encode('utf-8')
    compressed_data = gzip.compress(json_data)
    return base64.b64encode(compressed_data).decode('utf-8')

#node 클래스
class node():
    def __init__(self, id, coord):
        #node id
        self.id = id
        #node 좌표 (2차원)
        self.coord = np.array(coord)
        #노드로 들어오는 edge들 => intersection 쪽 충돌 감지 위해 필요
        self.incoming_edges = []
        self.outgoing_edges = []
        self.OHT = None
        
class edge():
    def __init__(self, source, dest, length, max_speed):
        self.id = f'{source.id}-{dest.id}'
        #Edge source
        self.source = source.id
        #edge target(destination)
        self.dest = dest.id
        #edge length
        self.length = length
        #edge의 방향 벡터
        self.unit_vec = (dest.coord - source.coord) / self.length
        #max speed -> 전부 1500
        self.max_speed = max_speed
        #edge 위에 존재하는 OHT 리스트
        self.OHTs = []  # Use a set to track OHTs
        
        self.count = 0  # OHT가 이 Edge에 진입한 횟수
        # self.speeds = []  # 최근 time_window 동안의 속도 저장
        self.avg_speed = max_speed  # time_window 내 평균 속도
        self.entry_exit_records = {} 
        
    def prune_old_records(self, current_time, time_window):
        """
        오래된 데이터를 제거하여 entry_exit_records를 관리합니다.
        current_time 기준으로 time_window 이전 데이터 삭제.
        """
        for oht_id, records in list(self.entry_exit_records.items()):  # dict 복사본으로 반복
            # 유효한 기록만 유지
            self.entry_exit_records[oht_id] = [
                (entry, exit) for entry, exit in records
                if exit is not None and exit >= current_time - time_window
            ]

            # 해당 OHT에 남은 기록이 없다면 제거
            if not self.entry_exit_records[oht_id]:
                del self.entry_exit_records[oht_id]

    def calculate_avg_speed(self, time_window, current_time):
        self.prune_old_records(current_time, time_window)

        total_time_covered = 0  # 기록된 총 이동 시간
        total_distance_covered = 0  # 기록된 총 이동 거리
        recent_speeds = []  # 최근 time_window 내 속도 기록

        # ✅ 기존 `entry_exit_records` 활용 (과거 OHT 통과 속도 반영)
        for oht_id, records in self.entry_exit_records.items():
            relevant_records = [
                (entry, exit) for entry, exit in records
                if exit is not None and entry >= current_time - time_window
            ]
            for entry, exit in relevant_records:
                travel_time = exit - entry
                if travel_time > 0.01:  # 최소 이동 시간
                    speed = self.length / travel_time
                    total_time_covered += travel_time
                    total_distance_covered += speed * travel_time
                    recent_speeds.append(speed)

        # ✅ 현재 edge 위에 있는 OHT 속도 반영 (정지한 OHT 포함)
        if len(self.OHTs) > 0:
            for oht in self.OHTs:
                recent_speeds.append(oht.speed)

        # ✅ time_window 내 Moving Average 기반 평균 속도 계산
        if recent_speeds:
            avg_speed = sum(recent_speeds) / len(recent_speeds)
        else:
            avg_speed = getattr(self, "avg_speed", self.max_speed)  # 최근 데이터가 없으면 이전 속도 유지

        # ✅ OHT가 없을 때 → 점진적으로 `max_speed`로 회복
        prev_avg_speed = getattr(self, "avg_speed", avg_speed)
        if len(self.OHTs) == 0 and not recent_speeds:
            recovery_rate = 0.01  # 회복 속도 (작을수록 천천히 max_speed로 이동)
            avg_speed = prev_avg_speed * (1 - recovery_rate) + self.max_speed * recovery_rate

        # ✅ OHT가 있지만 정체된 경우 → 속도를 서서히 감소
        elif len(self.OHTs) > 0 and all(oht.speed < 1 for oht in self.OHTs):
            decay_rate = 0.01  # 감속 비율
            avg_speed = prev_avg_speed * (1 - decay_rate)

        # ✅ `time_window` 내 변화 반영하는 EMA 적용 (부드러운 변화)
        alpha = 2 / (time_window / 100)  # `time_window` 크기에 따라 smoothing factor 조절
        alpha = max(0.01, min(alpha, 0.2))  # 최소 0.01, 최대 0.2로 제한
        avg_speed = alpha * avg_speed + (1 - alpha) * prev_avg_speed
        
        avg_speed = round(avg_speed, 2)

        # ✅ 속도 저장 (다음 iteration에서 활용)
        self.avg_speed = avg_speed

        return min(avg_speed, self.max_speed)


    
class port():
    def __init__(self, name, from_node, to_node, from_dist):
        #port name
        self.name = name
        #포트 위치 설정을 위한 from_node
        self.from_node = from_node
        #포트 to_node
        self.to_node = to_node
        #from_node로부터의 거리
        self.from_dist = from_dist
        #포트가 존재하는 edge
        self.edge = None

        
class OHT():
    def __init__(self, id, from_node, from_dist, speed, acc, rect, path, node_map, edge_map, port_map):
        self.id = id #oht id
        self.from_node = from_node.id #oht 위치를 계산하기 위한 from_node
        self.from_dist = from_dist #from_node로부터의 거리
        
        self.node_map = node_map
        self.edge_map = edge_map
        self.port_map = port_map
        
        self.path = path #oht가 움직일 경로, (Edge들의 List)
        self.path_to_start = [] #start port로 갈 때 경로
        self.path_to_end = [] #end port로 갈 때 경로
        self.edge = path.pop(0) if path else None #OHT가 위치한 edge
        
        self.pos = (
            from_node.coord + self.edge_map[self.edge.unit_vec] * self.from_dist
            if self.edge else from_node.coord
        ) #OHT의 위치 계산
        
        self.speed = speed #속도
        self.acc = acc #가속도
        self.rect = rect #충돌 감지 범위
        
        self.edge.OHTs.append(self) if self.edge else None #OHT가 위치한 edge의 OHT list에 self 추가 
        
        
        self.start_port = None #출발 포트
        self.end_port = None #도착 포트
        self.wait_time = 0 #loading / unloading시 기다리는 시간
        
        self.status = "IDLE" #STATUS, IDLE / TO_START / TO_END
    
    #위치 계산 method
    def cal_pos(self, time_step):
        self.from_dist = self.from_dist + self.speed * time_step + 1/2 * self.acc * time_step**2
        
        from_node = self.node_map[self.from_node]
        edge = self.edge_map[self.edge] if self.edge != None else None
        
        self.pos = from_node.coord + edge.unit_vec * self.from_dist if self.edge != None else from_node.coord

    #move, 매 time step 마다 실행            
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
                from_node.OHT = None
            return

        #end port 도착하면 wait time 동안 기다리기
        if self.wait_time > 0:
            self.wait_time -= time_step
            if self.wait_time <= 0:  # 대기 시간이 끝나면
                self.wait_time = 0
                if self.status == 'STOP_AT_START':
                    self.status = 'TO_END'
                elif self.status == 'STOP_AT_END':
                    self.status = 'IDLE'
            return  # 대기 중이므로 이동하지 않음
        
        #충돌 감지
        self.col_check(time_step)
        
        #만약 도착했다면 멈추기 
        if self.is_arrived():
            self.arrive()
            return
        
        from_node = self.node_map[self.from_node]
        edge = self.edge_map[self.edge]
            
        #From_dist가 edge의 length보다 길어지면 다음 엣지로 업데이트    
        while (self.from_dist > edge.length):
            
            self.from_node = edge.dest #from_node를 Edge dest로 업데이트

            self.from_dist = self.from_dist - edge.length #from_dist도 업데이트
            
            if edge:
                try:
                     #원래 엣지에서 현재 OHT 제거
                    if len(self.path) > 0:
                
                        exit_record = edge.entry_exit_records.get(self.id, [])
                        if exit_record and exit_record[-1][1] is None:
                            exit_record[-1] = (exit_record[-1][0], current_time)
                        edge.entry_exit_records[self.id] = exit_record
                                            
                        edge.OHTs.remove(self)
                        
                        self.edge = self.path.pop(0) #다음 엣지로 업데이트
                        
                        edge = self.edge_map[self.edge]
                        
                        # print(edge.count)
                        
                        if self not in edge.OHTs:
                            edge.OHTs.append(self) #다음 엣지 안에 OHT 추가
                            # print(self.edge, edge.count)
                            edge.count += 1
                            # print(self.edge, edge.count)
                            if self.id not in edge.entry_exit_records:
                                edge.entry_exit_records[self.id] = []
                                edge.entry_exit_records[self.id].append((current_time, None))
                            
                    else:
                        self.speed = 0
                        self.acc = 0
                        self.from_dist = edge.length
                        self.from_node = edge.source
                        self.status = 'ON_REMOVED'
                        return
                except:
                    print('update error : ', self.edge)

  
            if self.is_arrived():
                self.arrive()
                return
            

        if self.is_arrived():
            self.arrive()
            return
            
        #가속도 고려해서 스피드 계산
        self.speed = min(max(self.speed + self.acc * time_step, 0), edge.max_speed)
        
    def is_arrived(self):
        
        edge = self.edge_map[self.edge]

        #start_port 혹은 end port에 도착했는지 확인, OHT와 port가 같은 엣지, 같은 from_node에 있는지, OHT의 from_dist가 port의 From_dist보다 커지는지
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

        #도착하면 port위치로 OHT를 고정하고, 속도, 가속도 0으로 정지, wait_time 5초 주기
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
            self.path = self.path_to_end  # 경로를 end_port로 변경
            self.path_to_start = []  # start_port 경로 초기화
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
            # print(f"OHT {self.id} arrived at end port: {self.end_port.name}")
            self.status = "STOP_AT_END"
            self.path = []
            self.path_to_end = []  # end_port 경로 초기화

        else:
            print(f"OHT {self.id} is idle.")

    
    #충돌 감지
    def col_check(self, time_step):
        
        edge = self.edge_map[self.edge]

        OHTs = edge.OHTs
                
        if len(OHTs) > 1:
            # 현재 OHT의 인덱스 찾기
            index = OHTs.index(self)

            # 바로 앞에 있는 OHT와만 거리 비교
            if index > 0:  # 첫 번째 OHT가 아니라면 앞의 OHT 존재
                prev_oht = OHTs[index - 1]  # 바로 앞에 있는 OHT
                dist_diff = prev_oht.from_dist - self.from_dist
                
                if 0 < dist_diff < self.rect:  # rect 거리보다 가까우면
                    self.acc = -self.speed / time_step  # 속도 감소 또는 정지
                    return
                
        if self.node_map[edge.dest].OHT is not None:
            dist_diff = edge.length - self.from_dist
            if 0 < dist_diff < self.rect:  # rect 길이보다 가까워지면
                # 가속도를 줄여 충돌 방지
                self.acc = -self.speed/time_step # 속도 감소 또는 정지
                return
         
         #다음 엣지에 있는 OHT들 중 제일 마지막에 있는 친구와 거리 비교       
        if len(self.path) > 0:
            next_edge = self.edge_map[self.path[0]]
            try:
                last_oht = next_edge.OHTs[-1]
                rem_diff = edge.length - self.from_dist
                dist_diff = last_oht.from_dist + rem_diff
                if 0 < dist_diff < self.rect:
                    self.acc = -self.speed/time_step # 속도 감소 또는 정지
                    return
            except:
                pass
        
        #다음 노드에서 들어오는 엣지들 상에 있는 OHT들과 충돌 비교 (intersection이 무조건 최대 2개임)    
        incoming_edges = [
            incoming_edge for incoming_edge in self.node_map[edge.dest].incoming_edges
            if incoming_edge != self.edge # Edges leading to the same destination node
        ]
        
        if len(incoming_edges) == 1: #만약 intersection이 있따면 (다른 edge가 있다면)
            rem_diff = edge.length - self.from_dist #현재 자기 자신의 엣지 상에서 남은 거리 계산
            try:
                other_oht = self.edge_map[incoming_edges[0]].OHTs[0]
                
                other_diff= self.edge_map[other_oht.edge].length - other_oht.from_dist #다른 엣지 위 제일 앞에 있는 OHT의 남은 거리 계산
                dist_diff = rem_diff + other_diff #두 OHT간의 거리 계산
                if 0 < dist_diff < 3 * self.rect and rem_diff > other_diff: #더 가까운 OHT가 먼저 움직이도록, 나머지는 정지. 3*rect로 잡은 이유는 그래야 좀 더 미리 멈춰서
                    self.acc = -self.speed/time_step # 속도 감소 또는 정지
                    return
                elif rem_diff == other_diff and self.id > other_oht.id: #혹시나 거리가 같다면 OHT id가 더 빠른 아이가 움직이도록
                    self.acc = -self.speed/time_step # 속도 감소 또는 정지
                    return
            except:
                pass
            
        if len(self.path) > 0:
            outgoing_edges = [
                outgoing_edge for outgoing_edge in self.node_map[edge.dest].outgoing_edges
                if outgoing_edge != self.path[0] # Edges leading to the same destination node
            ]
        
            if len(outgoing_edges) == 1: #만약 intersection이 있따면 (다른 edge가 있다면)
                rem_diff = edge.length - self.from_dist #현재 자기 자신의 엣지 상에서 남은 거리 계산
                try:
                    other_oht = self.edge_map[outgoing_edges[0]].OHTs[0]
                    other_diff= other_oht.from_dist #다른 엣지 위 제일 앞에 있는 OHT의 남은 거리 계산
                    dist_diff = rem_diff + other_diff #두 OHT간의 거리 계산
                    if 0 < dist_diff <  self.rect: #더 가까운 OHT가 먼저 움직이도록, 나머지는 정지. 3*rect로 잡은 이유는 그래야 좀 더 미리 멈춰서
                        self.acc = -self.speed/time_step # 속도 감소 또는 정지
                        return
                except:
                    pass
        
        #충돌 위험이 없다면 다시 원래 max speed로 가속
        self.acc = (edge.max_speed - self.speed) / time_step
        
        
class AMHS:
    def __init__(self, nodes, edges, ports, num_OHTs, max_jobs, job_list = [], oht_list = []):
        """
        AMHS 초기화.
        - nodes: node 클래스 객체 리스트
        - edges: edge 클래스 객체 리스트
        - num_OHTs: 초기 OHT 수
        - max_jobs: 작업 큐의 최대 크기
        """
        # self.graph = nx.DiGraph()
        # self.original_graph=nx.DiGraph()
        
        # ray.init(ignore_reinit_error=True, num_cpus=10)
        
        self.graph = nk.Graph(directed=True, weighted=True)  # Directed weighted graph
        self.node_map = {}
        self.edge_map = {}
        self.port_map = {}
        
        # self.nodes = nodes[:]  # node 객체 리스트
        self.nodes = copy.deepcopy(nodes)


        for node in self.nodes:
            node.OHT = None
            self.node_map[node.id] = node
            
        # self.edges = edges[:]  # edge 객체 리스트
        self.edges = copy.deepcopy(edges)
        
        for edge in self.edges:
            self.edge_map[edge.id] = edge
            edge.OHTs = []
            edge.count = 0
            edge.avg_speed = edge.max_speed
            edge.entry_exit_records = {} 

        
        # self.ports = ports[:] #port list
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
        
                
        self.simulation_running = False  # 시뮬레이션 상태 관리
        self.stop_simulation_event = threading.Event()  # 스레드 이벤트 처리
        
                        
        self.back_simulation_running = False  # 시뮬레이션 상태 관리
        self.back_stop_simulation_event = threading.Event()  # 스레드 이벤트 처리
        
        self.original_graph = nk.Graph(directed=True, weighted=True)
        # 그래프 생성
        self._create_graph()

        # 초기 OHT 배치
        if oht_list != []:
            self.set_initial_OHTs(oht_list)
        else:
            self.initialize_OHTs(num_OHTs)
        
        self.apsp = nk.distance.APSP(self.graph)
        self.apsp.run()
        
        self.original_apsp = nk.distance.APSP(self.original_graph)
        self.original_apsp.run()
        


    def _create_graph(self):
        """NetworkX 그래프를 nodes와 edges로 생성."""
        for i, node in enumerate(self.nodes):
            self.graph.addNode()  # Add node to the graph
            self.original_graph.addNode()
            self.node_id_map[node.id] = i
        
        for edge in self.edges:
            u = self.node_id_map[edge.source]  # Get index for source node
            v = self.node_id_map[edge.dest]    # Get index for destination node
            self.graph.addEdge(u, v, edge.length)
            self.original_graph.addEdge(u, v, edge.length)
            self.node_map[edge.dest].incoming_edges.append(edge.id) #각 node마다 incoming edge 추가
            self.node_map[edge.source].outgoing_edges.append(edge.id)

    def initialize_OHTs(self, num_OHTs):
        available_nodes = self.nodes.copy()
        
        for i in range(num_OHTs):
            # 노드 중에서 랜덤 선택 (노드 소진 시 다시 전체에서 랜덤)
            if not available_nodes:
                available_nodes = self.nodes.copy()  # 노드 리스트 재생성
            
            start_node = random.choice(available_nodes)
            available_nodes.remove(start_node)  # 선택한 노드는 제거
            
            oht = OHT(
                id=i,
                from_node=start_node,
                from_dist=0,
                speed=0,
                acc=0,
                rect=1000,  # 충돌 판정 거리
                path=[],
                node_map = self.node_map,
                edge_map = self.edge_map,
                port_map = self.port_map
            )
            start_node.OHT = oht
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
                rect=1000,  # 충돌 판정 거리
                path=[],
                node_map = self.node_map,
                edge_map = self.edge_map,
                port_map = self.port_map
            )
            start_node.OHT = oht
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
                oht_origin.cal_pos(self.time_step)
                return
            
        elif oht_origin.from_dist == 0:
            self.node_map[oht_origin.from_node].OHT = oht_origin;
            
        # try:

        # except:
        #     pass
  
            
        oht_origin.speed = oht_new['speed']
        oht_origin.status = oht_new['status']
        
        if oht_origin.status == 'ON_REMOVED':
            if oht_origin.start_port:
                oht_origin.status = "TO_START"
            elif oht_origin.end_port:
                oht_origin.status = "TO_END"
            else:
                oht_origin.status = "IDLE"
                
        oht_origin.cal_pos(self.time_step)
             

                
        if oht_origin.start_port != None:
            if oht_origin.edge:
                path_edges_to_start = self.get_path_edges(oht_new['dest'], self.port_map[oht_origin.start_port].to_node)
                path_edges_to_start = self._validate_path(path_edges_to_start)

            else:
                path_edges_to_start = self.get_path_edges(oht_origin.from_node, self.port_map[oht_origin.start_port].to_node)
                path_edges_to_start = self._validate_path(path_edges_to_start)
                
            oht_origin.path_to_start = path_edges_to_start[:]
        
        # elif oht_origin.status == "TO_END":
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
                # path_edges_to_end = self._validate_path(path_edges_to_end)

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
            try:       
                self.graph.removeEdge(self.node_id_map[source], self.node_id_map[dest])
                self.apsp = nk.distance.APSP(self.graph)
                self.apsp.run()
            except:
                print(source, dest)
        else:
            edge_to_restore = self.get_edge(source, dest)
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
            
        for node in self.nodes:
            node.OHT = None
    
    def generate_job(self):
        """모든 OHT가 Job을 갖도록 작업 생성."""
        for _ in range(500):
            start_port = random.choice(self.ports)
            end_port = random.choice(self.ports)
            while start_port == end_port:  # 시작/목적 포트가 같지 않도록 보장
                end_port = random.choice(self.ports)
            self.job_queue.append((start_port, end_port))
            
    def assign_jobs(self):
        """모든 OHT가 Job을 갖도록 작업 할당."""
        while self.job_queue:  # ✅ Job Queue가 비어있지 않다면 반복
            idle_ohts = [oht for oht in self.OHTs if oht.status == 'IDLE' and not oht.path]  # ✅ IDLE인 OHT 리스트
            
            if not idle_ohts:  # ✅ Idle OHT가 없으면 종료
                break
            
            closest_oht = None
            closest_dist = float('inf')
            start_port, end_port = None, None  # ✅ 가장 가까운 OHT에 배정할 Job
            
            for oht in idle_ohts:
                temp_start_port, temp_end_port = self.job_queue[0]  # ✅ 첫 번째 Job 미리 확인
                dist = self.get_path_distance(oht.from_node, temp_start_port.from_node)
                
                if dist < closest_dist:
                    closest_dist = dist
                    closest_oht = oht
                    start_port, end_port = temp_start_port, temp_end_port  # ✅ 가장 가까운 OHT에 할당할 Job 결정
            
            if closest_oht and start_port and end_port:
                self.job_queue.pop(0)  # ✅ Job Queue에서 제거 (가장 가까운 OHT가 가져감)
                self._assign_job_to_oht(closest_oht, start_port, end_port)  # ✅ Job 할당 로직 분리
            else:
                break  # ✅ 할당할 Job이 없다면 종료

    def _assign_job_to_oht(self, oht, start_port, end_port):
        """OHT에 Job을 실제로 할당하는 함수"""
        oht.start_port = start_port.name
        oht.end_port = end_port.name
        oht.status = "TO_START"

        # Start로 이동하는 경로
        if oht.edge:
            path_edges_to_start = self.get_path_edges(self.edge_map[oht.edge].dest, start_port.to_node)
        else:
            path_edges_to_start = self.get_path_edges(oht.from_node, start_port.to_node)
        
        oht.path_to_start = path_edges_to_start[:]

        # End로 이동하는 경로
        path_edges_to_end = self.get_path_edges(start_port.to_node, end_port.to_node)
        oht.path_to_end = path_edges_to_end

        # 전체 경로를 OHT에 설정
        oht.path = path_edges_to_start[:]

        # Assign the first edge in the path to the OHT's edge
        if oht.path:
            if not oht.edge:
                oht.edge = oht.path.pop(0)
                self.node_map[oht.from_node].OHT = None
                if oht not in self.edge_map[oht.edge].OHTs:
                    self.edge_map[oht.edge].OHTs.append(oht)
                            
    def update_edge_metrics(self, current_time, time_window):
        for edge in self.edges:
            edge.avg_speed = edge.calculate_avg_speed(time_window, current_time)


    def get_node(self, node_id):
        """node id로 node 객체 반환."""
        return self.node_map[node_id]

    def get_edge(self, source_id, dest_id):
        """source와 dest ID로 edge 객체 반환."""
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
        """source와 dest ID로 최단 경로 에지 리스트 반환."""
        # try:
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
        # except:
            # try:
            source_idx = self.node_id_map[source_id]
            dest_idx = self.node_id_map[dest_id]
            
            dijkstra = nk.distance.Dijkstra(self.original_graph, source_idx, storePaths=True, storeNodesSortedByDistance=False, target=dest_idx)
            dijkstra.run()

            path = dijkstra.getPath(dest_idx)
            
            return [
                self.get_edge(self.nodes[path[i]].id, self.nodes[path[i+1]].id)
                for i in range(len(path) - 1)
            ]
            # except:
            #     print(f"No path found even in original_graph for {source_id} -> {dest_id}.")
            #     return []
    
    def get_oht(self, oht_id):
        return next((oht for oht in self.OHTs if oht.id == oht_id), None)
    
    def get_port(self, port_name):
        return self.port_map[port_name]

    def _validate_path(self, path):
        """
        현재 경로에 지워진 edge가 포함된 경우, 해당 edge 직전까지만 반환합니다.

        Parameters:
            path (list): 현재 경로 (edge 객체 리스트)

        Returns:
            list: 수정된 경로 (지워진 edge 직전까지만 포함)
        """
        valid_path = []
        for edge in path:
            if self.graph.hasEdge(self.node_id_map[self.edge_map[edge].source], self.node_id_map[self.edge_map[edge].dest]):
                valid_path.append(edge)
            else:
                break  # 지워진 edge를 발견하면 직전까지만 반환
        return valid_path
    
        
    def start_simulation(self, socketio, current_time, max_time = 4000, time_step = 0.1):
        """시뮬레이션 시작"""        
        if self.simulation_running:
            print("Simulation is already running. Stopping the current simulation...")
            self.stop_simulation_event.set()
            while self.simulation_running:
                socketio.sleep(0.01)  # Wait for the current simulation to stop
            return
        
        self.simulation_running = True
        self.stop_simulation_event.clear()
        
        count = 0
        edge_metrics_cache = {}  # Cache for edge metrics to track changes

        while current_time < max_time:
            if self.stop_simulation_event.is_set():
                break
            
            self.current_time = current_time
            
            if count % 5 == 0:
                self.generate_job()
            self.assign_jobs()
                
            # print(count)

            # Move all OHTs
            oht_positions = []
            for oht in self.OHTs:
                oht.move(time_step, current_time)

            self.update_edge_metrics(current_time, time_window=500)
            
            for oht in self.OHTs:
                oht.cal_pos(time_step)
                
            if count % 1 == 0:
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
                        'wait_time': oht.wait_time
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
                socketio.emit('updateOHT', {'data': compressed_payload})

            # Increment time
            current_time += time_step
            count += 1
            # socketio.sleep(0.00001)

        self.simulation_running = False
        print('Simulation ended')
        
        
    
    def accelerate_simul(self, socketio, current_time, max_time = 4000, time_step = 0.1):
        """시뮬레이션 시작"""        
        if self.simulation_running:
            print("Simulation is already running. Stopping the current simulation...")
            self.stop_simulation_event.set()
            while self.simulation_running:
                socketio.sleep(0.01)  # Wait for the current simulation to stop
            return
        
        self.simulation_running = True
        self.stop_simulation_event.clear()
        
        _current_time = 0
        
        count = 0
        
        accel_factor = 10

        while _current_time < current_time:
            if self.stop_simulation_event.is_set():
                break
            
            if count % 5 == 0:
                self.generate_job()
            
            self.assign_jobs()
            # Move all OHTs
            
            for oht in self.OHTs:
                oht.move(time_step, _current_time)

            self.update_edge_metrics(_current_time, time_window=500)
            
            for oht in self.OHTs:
                oht.cal_pos(time_step)
                
            self.current_time = _current_time

            # Increment time
            _current_time += time_step*10
            count += 1
            
        
        edge_metrics_cache = {} 
            
        
        while _current_time < max_time:
            if self.stop_simulation_event.is_set():
                break
            
            if count % 5 == 0:
                self.generate_job()
            self.assign_jobs()

            self.current_time = _current_time
            
            # Move all OHTs
            oht_positions = []
            for oht in self.OHTs:
                oht.move(time_step, _current_time)

            self.update_edge_metrics(_current_time, time_window=500)
            
            for oht in self.OHTs:
                oht.cal_pos(time_step)

            if count % 1 == 0:
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
                        'wait_time': oht.wait_time
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
                socketio.emit('updateOHT', {'data': compressed_payload})

            # Increment time
            current_time += time_step
            count += 1

        self.simulation_running = False
        print('Simulation ended')
        
        
    def only_simulation(self, socketio, current_time, max_time = 4000, time_step = 0.1):
        """시뮬레이션 시작"""        
        if self.back_simulation_running:
            print("Simulation is already running. Stopping the current simulation...")
            self.back_stop_simulation_event.set()
            while self.back_simulation_running:
                socketio.sleep(0.01)  # Wait for the current simulation to stop
            return
        
        print('only simulationr on running')
        
        self.back_simulation_running = True
        self.back_stop_simulation_event.clear()
        
        count = 0

        while current_time < max_time:
            if self.back_stop_simulation_event.is_set():
                break
            
            # print('only simulationr on running')


            self.current_time = current_time
            
            if count % 5 == 0:
                self.generate_job()
            self.assign_jobs()
                

            for oht in self.OHTs:
                oht.move(time_step, current_time)

            self.update_edge_metrics(current_time, time_window=500)
            
            for oht in self.OHTs:
                oht.cal_pos(time_step)
    
            # Increment time
            current_time += time_step
            count += 1
            # socketio.sleep(0.00001)

        self.back_simulation_running = False
        print('Simulation ended')
        
        socketio.emit("backSimulationFinished")

        
        
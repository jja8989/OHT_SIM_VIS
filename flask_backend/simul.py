#import libraries
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import pdb


#node 클래스
class node():
    def __init__(self, id, coord):
        #node id
        self.id = id
        #node 좌표 (2차원)
        self.coord = np.array(coord)
        #노드로 들어오는 edge들 => intersection 쪽 충돌 감지 위해 필요
        self.incoming_edges = []
        
class edge():
    def __init__(self, source, dest, length, max_speed):
        #Edge source
        self.source = source
        #edge target(destination)
        self.dest = dest
        #edge length
        self.length = length
        #edge의 방향 벡터
        self.unit_vec = (self.dest.coord - self.source.coord) / self.length
        #max speed -> 전부 1500
        self.max_speed = max_speed
        #edge 위에 존재하는 OHT 리스트
        self.OHTs = []  # Use a set to track OHTs
        
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
    def __init__(self, id, from_node, from_dist, speed, acc, rect, path):
        self.id = id #oht id
        self.from_node = from_node #oht 위치를 계산하기 위한 from_node
        self.from_dist = from_dist #from_node로부터의 거리
        
        self.path = path #oht가 움직일 경로, (Edge들의 List)
        self.path_to_start = [] #start port로 갈 때 경로
        self.path_to_end = [] #end port로 갈 때 경로
        self.edge = path.pop(0) if path else None #OHT가 위치한 edge
        
        self.pos = (
            self.from_node.coord + self.edge.unit_vec * self.from_dist
            if self.edge else self.from_node.coord
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
    def cal_pos(self):
        self.pos = self.from_node.coord + self.edge.unit_vec * self.from_dist

    #move, 매 time step 마다 실행            
    def move(self, time_step):
        #node 위에만 있을 떄?? 이거 사실 잘 모르겠구 에러 나는거 해결하려고 이거저거하다가 넣었습니다
        if not self.edge:
            # print('no edge', self.edge)
            self.speed = 0
            self.acc = 0
            return

        #end port 도착하면 wait time 동안 기다리기
        if self.wait_time > 0:
            self.wait_time -= time_step
            if self.wait_time <= 0:  # 대기 시간이 끝나면
                self.wait_time = 0
            return  # 대기 중이므로 이동하지 않음
        
        #충돌 감지
        self.col_check(time_step)
        
        # ori_dist_1 = copy.copy(self.from_dist)
        
        #다음 스텝에 움직인 거리 계산
        self.from_dist = self.from_dist + self.speed * time_step + 1/2 * self.acc * time_step**2
        
        #만약 도착했다면 멈추기 
        if self.is_arrived():
            self.arrive()
            return
        
        # before_node = copy.copy(self.from_node)
        # ori_dist_2 = copy.copy(self.from_dist)
        
        #여기는 디버깅 위한 구간이라서 무시하셔도 됩니다
        if self.from_node != self.edge.source:
            # pdb.set_trace()
            all_path = [(p.source.id, p.dest.id) for p in self.path]

            print(f"Mismatch detected: from_node={self.from_node.id}, edge.source={self.edge.source.id}")
            print(all_path)
            print(self.status)
            print(self.from_dist)
            print(self.edge.length)
            print(self in self.edge.OHTs)
            
        #From_dist가 edge의 length보다 길어지면 다음 엣지로 업데이트    
        while (self.from_dist > self.edge.length):
            
            self.from_node = self.edge.dest #from_node를 Edge dest로 업데이트
            self.from_dist = self.from_dist - self.edge.length #from_dist도 업데이트
            
            if self.edge:
                try:
                    self.edge.OHTs.remove(self) #원래 엣지에서 현재 OHT 제거
                except:
                    print(self.edge.source.id)

            if len(self.path) > 0:
                # before_edge = copy.copy(self.edge)
                
                self.edge = self.path.pop(0) #다음 엣지로 업데이트
                    
                if self not in self.edge.OHTs:
                    self.edge.OHTs.append(self) #다음 엣지 안에 OHT 추가
                    
                # if before_node == self.from_node:
                #     # print(start_path)
                #     print('why this is happening????')
                #     print(self.id)
                #     # print(self.start_port.from_node.id)
                #     # print(self.end_port.from_node.id)
                #     # print(before_edge.source.id, before_edge.dest.id)
                #     # print(before_path)
            
            #마찬가지로 도착했는지 확인 (여기서는 엣지가 업데이트 되었을 때 도착했는지)     
            if self.is_arrived():
                self.arrive()
                return
            
        
        #마찬가지로 도착했는지 확인 (엣지가 업데이트 안돼었을 때 도착했는지)
        if self.is_arrived():
            self.arrive()
            return
        
        
        # after_node = self.from_node
        # before_pos = copy.copy(self.pos)
            
        #포지션 계산 (실제 움직임)    
        self.cal_pos()
        
        # dis = (np.sum(self.pos - before_pos)**2)**0.5
        
        # # print(dis)
        # if self.from_dist > self.edge.length:
        #     print('weird')
            
        # if dis > 1500:
        #     print(dis)
        #     print('before pos : ', before_pos)
        #     print('pos : ' , self.pos)
        #     print('before node : ', before_node.id, before_node.coord)
        #     print('after node : ', after_node.id, after_node.coord)
        #     print(self.edge.length)
        #     print(self.edge.source.id)
        #     print(self.edge.dest.id)
        #     print(self.edge.dest.coord)
        #     # print(ori_dist_1)
        #     # print(ori_dist_2)
        #     print(self.from_dist)
        #     print(self.edge.unit_vec)
        #     print(self.edge.unit_vec * self.from_dist)
        #     print(before_node.coord + self.edge.unit_vec * self.from_dist)
            
        #가속도 고려해서 스피드 계산
        self.speed = min(max(self.speed + self.acc * time_step, 0), self.edge.max_speed)
        
    def is_arrived(self):

        #start_port 혹은 end port에 도착했는지 확인, OHT와 port가 같은 엣지, 같은 from_node에 있는지, OHT의 from_dist가 port의 From_dist보다 커지는지
        if self.status == "TO_START":
            return (self.start_port 
                    and self.start_port.from_node == self.from_node and self.start_port.to_node == self.edge.dest
                    and self.from_dist >= self.start_port.from_dist)
        elif self.status == "TO_END":
            return (self.end_port 
                    and self.end_port.from_node == self.from_node and self.end_port.to_node == self.edge.dest
                    and self.from_dist >= self.end_port.from_dist)
        
    def arrive(self):
        # print('arrived at', self.pos, self.from_node.id)
        # self.from_dist = self.end_port.from_dist
        # self.pos = self.from_node.coord + self.edge.unit_vec * self.end_port.from_dist
        # self.speed = 0
        # self.acc = 0
        # self.wait_time = 500
        # self.end_port = None
        # print('arrived rail : ', self.edge.OHTs, self in self.edge.OHTs)
        
        #도착하면 port위치로 OHT를 고정하고, 속도, 가속도 0으로 정지, wait_time 5초 주기
        if self.status == "TO_START":
            # print(f"OHT {self.id} arrived at start port: {self.start_port.name}")
            self.from_dist = self.start_port.from_dist
            self.pos = self.from_node.coord + self.edge.unit_vec * self.start_port.from_dist
            self.speed = 0
            self.acc = 0
            self.wait_time = 5
            
            self.status = "TO_END"
            self.path = self.path_to_end  # 경로를 end_port로 변경
            self.path_to_start = []  # start_port 경로 초기화
            self.start_port = None

        elif self.status == "TO_END":
            
            self.from_dist = self.end_port.from_dist
            self.pos = self.from_node.coord + self.edge.unit_vec * self.end_port.from_dist
            self.speed = 0
            self.acc = 0
            self.wait_time = 5
            self.end_port = None
            # print(f"OHT {self.id} arrived at end port: {self.end_port.name}")
            self.status = "IDLE"
            self.path = []
            self.path_to_end = []  # end_port 경로 초기화

        else:
            print(f"OHT {self.id} is idle.")

    
    #충돌 감지
    def col_check(self, time_step):
        OHTs = self.edge.OHTs
        
        #같은 edge상에 있는 OHT들과 거리 비교
        for oht in OHTs:
            if oht is not self:  # 자신과의 비교를 피함
                dist_diff = oht.from_dist - self.from_dist
                if 0 < dist_diff < self.rect:  # rect 길이보다 가까워지면
                    # 가속도를 줄여 충돌 방지
                    self.acc = -self.speed/time_step # 속도 감소 또는 정지
                    return
         
         #다음 엣지에 있는 OHT들 중 제일 마지막에 있는 친구와 거리 비교       
        if len(self.path) > 0:
            next_edge = self.path[0]
            try:
                last_oht = next_edge.OHTs[-1]
                rem_diff = self.edge.length - self.from_dist
                dist_diff = last_oht.from_dist + rem_diff
                if 0 < dist_diff < self.rect:
                    self.acc = -self.speed/time_step # 속도 감소 또는 정지
                    return
            except:
                pass
        
        #다음 노드에서 들어오는 엣지들 상에 있는 OHT들과 충돌 비교 (intersection이 무조건 최대 2개임)    
        incoming_edges = [
            edge for edge in self.edge.dest.incoming_edges
            if edge != self.edge # Edges leading to the same destination node
        ]
        
        if len(incoming_edges) == 1: #만약 intersection이 있따면 (다른 edge가 있다면)
            rem_diff = self.edge.length - self.from_dist #현재 자기 자신의 엣지 상에서 남은 거리 계산
            try:
                other_oht = incoming_edges[0].OHTs[0]
                other_diff= other_oht.edge.length - other_oht.from_dist #다른 엣지 위 제일 앞에 있는 OHT의 남은 거리 계산
                dist_diff = rem_diff + other_diff #두 OHT간의 거리 계산
                if 0 < dist_diff < 3 * self.rect and rem_diff > other_diff: #더 가까운 OHT가 먼저 움직이도록, 나머지는 정지. 3*rect로 잡은 이유는 그래야 좀 더 미리 멈춰서
                    self.acc = -self.speed/time_step # 속도 감소 또는 정지
                    return
                elif rem_diff == other_diff and self.id > other_oht.id: #혹시나 거리가 같다면 OHT id가 더 빠른 아이가 움직이도록
                    self.acc = -self.speed/time_step # 속도 감소 또는 정지
                    return
            except:
                pass
        
        #충돌 위험이 없다면 다시 원래 max speed로 가속
        self.acc = (self.edge.max_speed - self.speed) / time_step
        
class AMHS:
    def __init__(self, nodes, edges, ports, num_OHTs, max_jobs):
        """
        AMHS 초기화.
        - nodes: node 클래스 객체 리스트
        - edges: edge 클래스 객체 리스트
        - num_OHTs: 초기 OHT 수
        - max_jobs: 작업 큐의 최대 크기
        """
        self.graph = nx.DiGraph()
        self.nodes = nodes  # node 객체 리스트
        self.edges = edges  # edge 객체 리스트
        for edge in edges:
            edge.OHTs = []
        self.ports = ports #port list
        for p in ports:
            p.edge = self.get_edge(p.from_node, p.to_node)
        self.OHTs = []
        self.job_queue = [] 
        self.max_jobs = max_jobs

        # 그래프 생성
        self._create_graph()

        # 초기 OHT 배치
        self.initialize_OHTs(num_OHTs)

    def _create_graph(self):
        """NetworkX 그래프를 nodes와 edges로 생성."""
        for node in self.nodes:
            self.graph.add_node(node.id, coord=node.coord.tolist())
        
        for edge in self.edges:
            self.graph.add_edge(
                edge.source.id, 
                edge.dest.id, 
                length=edge.length, 
                max_speed=edge.max_speed
            )
            edge.dest.incoming_edges.append(edge) #각 node마다 incoming edge 추가

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
            )
            self.OHTs.append(oht)
            
    def generate_job(self):
        """모든 OHT가 Job을 갖도록 작업 생성."""
        for oht in self.OHTs:
            if not oht.path and len(self.job_queue) < self.max_jobs:
                start_port = random.choice(self.ports)
                end_port = random.choice(self.ports)
                while start_port == end_port:  # 시작/목적 포트가 같지 않도록 보장
                    end_port = random.choice(self.ports)
                self.job_queue.append((start_port, end_port))

    def assign_jobs(self):
        """모든 OHT가 Job을 갖도록 작업 할당."""
        for oht in self.OHTs:
            if not oht.path and self.job_queue and oht.status == 'IDLE':  # OHT가 Idle 상태이고 Job Queue가 비어있지 않은 경우
                start_port, end_port = self.job_queue.pop(0)
                oht.start_port = start_port
                # oht.start_port = next((port for port in self.ports if port.name == 'DIE0137_Port3'), None)
                oht.end_port = end_port
                oht.status = "TO_START"

                # # Start로 이동하는 경로
                if oht.edge:
                    path_edges_to_start = self.get_path_edges(oht.edge.dest.id, start_port.to_node.id)
                else:
                    path_edges_to_start = self.get_path_edges(oht.from_node.id, start_port.to_node.id)
                oht.path_to_start = path_edges_to_start[:]
                
                # # End로 이동하는 경로
                # start_edge = [self.get_edge(start_port.from_node.id, start_port.to_node.id)]
                path_edges_to_end = self.get_path_edges(start_port.to_node.id, end_port.to_node.id)
                oht.path_to_end = path_edges_to_end

                # # 전체 경로를 OHT에 설정
                oht.path = path_edges_to_start[:]

                # Assign the first edge in the path to the OHT's edge
                if oht.path:
                    if not oht.edge:
                        oht.edge = oht.path.pop(0)
                        if oht not in oht.edge.OHTs:
                            oht.edge.OHTs.append(oht)


    def get_node(self, node_id):
        """node id로 node 객체 반환."""
        return next((node for node in self.nodes if node.id == node_id), None)

    def get_edge(self, source_id, dest_id):
        """source와 dest ID로 edge 객체 반환."""
        return next(
            (e for e in self.edges if e.source.id == source_id and e.dest.id == dest_id), 
            None
        )

    def get_path_edges(self, source_id, dest_id):
        """source와 dest ID로 최단 경로 에지 리스트 반환."""
        path_nodes = nx.shortest_path(
            self.graph, source=source_id, target=dest_id, weight='length'
        )
        return [
            self.get_edge(path_nodes[i], path_nodes[i + 1])
            for i in range(len(path_nodes) - 1)
        ]


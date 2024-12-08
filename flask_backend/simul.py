#import libraries
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import pdb


class node():
    def __init__(self, id, coord):
        #node 좌표
        self.id = id
        self.coord = np.array(coord)
        self.incoming_edges = []
        
class edge():
    def __init__(self, source, dest, length, max_speed):
        #Edge source
        self.source = source
        #edge target(destination)
        self.dest = dest
        #edge length
        self.length = length
        self.unit_vec = (self.dest.coord - self.source.coord) / self.length
        self.max_speed = max_speed
        self.OHTs = []  # Use a set to track OHTs
        
class port():
    def __init__(self, name, from_node, to_node, from_dist):
        self.name = name
        self.from_node = from_node
        self.to_node = to_node
        self.from_dist = from_dist
        self.edge = None

        
class OHT():
    def __init__(self, id, from_node, from_dist, speed, acc, rect, path):
        self.id = id
        self.from_node = from_node
        self.from_dist = from_dist
        
        self.path = path
        self.path_to_start = []
        self.path_to_end = []
        self.edge = path.pop(0) if path else None
        
        self.pos = (
            self.from_node.coord + self.edge.unit_vec * self.from_dist
            if self.edge else self.from_node.coord
        )
        self.speed = speed
        self.acc = acc
        self.rect = rect
        
        self.edge.OHTs.append(self) if self.edge else None
        
        
        self.start_port = None
        self.end_port = None
        self.wait_time = 0
        
        self.status = "IDLE"
    
    def cal_pos(self):
        self.pos = self.from_node.coord + self.edge.unit_vec * self.from_dist

            
    def move(self, time_step):
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
        
        
        self.col_check(time_step)
        
        ori_dist_1 = copy.copy(self.from_dist)
        
        self.from_dist = self.from_dist + self.speed * time_step + 1/2 * self.acc * time_step**2
        
        before_node = copy.copy(self.from_node)
        ori_dist_2 = copy.copy(self.from_dist)
        
        if self.from_node != self.edge.source:
            all_path = [(p.source.id, p.dest.id) for p in self.path]

            print(f"Mismatch detected: from_node={self.from_node.id}, edge.source={self.edge.source.id}")
            print(all_path)
            print(self.status)
            print(self.from_dist)
            print(self.edge.length)
            print(self in self.edge.OHTs)
            
        while (self.from_dist > self.edge.length):
            
            self.from_node = self.edge.dest
            self.from_dist = self.from_dist - self.edge.length
            
            if self.edge:
                self.edge.OHTs.remove(self)

            if len(self.path) > 0:
                # before_edge = copy.copy(self.edge)
                
                self.edge = self.path.pop(0)
                    
                if self not in self.edge.OHTs:
                    self.edge.OHTs.append(self)
                    
                if before_node == self.from_node:
                    # print(start_path)
                    print('why this is happening????')
                    print(self.id)
                    # print(self.start_port.from_node.id)
                    # print(self.end_port.from_node.id)
                    # print(before_edge.source.id, before_edge.dest.id)
                    # print(before_path)
                    
            if self.is_arrived():
                self.arrive()
                return
                
        if self.is_arrived():
            self.arrive()
            return
        
        after_node = self.from_node
        before_pos = copy.copy(self.pos)
            
        self.cal_pos()
        
        dis = (np.sum(self.pos - before_pos)**2)**0.5
        
        # print(dis)
        if self.from_dist > self.edge.length:
            print('weird')
            
        if dis > 1500:
            print(dis)
            print('before pos : ', before_pos)
            print('pos : ' , self.pos)
            print('before node : ', before_node.id, before_node.coord)
            print('after node : ', after_node.id, after_node.coord)
            print(self.edge.length)
            print(self.edge.source.id)
            print(self.edge.dest.id)
            print(self.edge.dest.coord)
            # print(ori_dist_1)
            # print(ori_dist_2)
            print(self.from_dist)
            print(self.edge.unit_vec)
            print(self.edge.unit_vec * self.from_dist)
            print(before_node.coord + self.edge.unit_vec * self.from_dist)
            
        self.speed = min(max(self.speed + self.acc * time_step, 0), self.edge.max_speed)
        
    def is_arrived(self):
        # return (self.end_port and self.end_port.from_node == self.from_node and self.from_dist > self.end_port.from_dist)
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

    
    def col_check(self, time_step):
        OHTs = self.edge.OHTs
        
        for oht in OHTs:
            if oht is not self:  # 자신과의 비교를 피함
                dist_diff = oht.from_dist - self.from_dist
                if 0 < dist_diff < self.rect:  # rect 길이보다 가까워지면
                    # 가속도를 줄여 충돌 방지
                    self.acc = -self.speed/time_step # 속도 감소 또는 정지
                    return
                
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
            
        incoming_edges = [
            edge for edge in self.edge.dest.incoming_edges
            if edge != self.edge # Edges leading to the same destination node
        ]
        
        if len(incoming_edges) == 1:
            rem_diff = self.edge.length - self.from_dist
            try:
                other_oht = incoming_edges[0].OHTs[0]
                other_diff= other_oht.edge.length - other_oht.from_dist
                dist_diff = rem_diff + other_diff
                if 0 < dist_diff < self.rect and rem_diff > other_diff:
                    self.acc = -self.speed/time_step # 속도 감소 또는 정지
                    return
                elif rem_diff == other_diff and self.id > other_oht.id:
                    self.acc = -self.speed/time_step # 속도 감소 또는 정지
                    return
            except:
                pass

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
            edge.dest.incoming_edges.append(edge)

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
                while start_port == end_port:  # 시작/목적 노드가 같지 않도록 보장
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


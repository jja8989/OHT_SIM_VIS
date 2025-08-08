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
        self.OHT = None
        
class edge():
    def __init__(self, source, dest, length, max_speed):
        self.id = f'{source.id}-{dest.id}'
        self.source = source.id
        self.dest = dest.id
        self.length = length
        self.unit_vec = (dest.coord - source.coord) / self.length
        self.max_speed = max_speed
        self.OHTs = []  
        
        self.count = 0  
        self.avg_speed = max_speed
        self.entry_exit_records = {} 
        
    def prune_old_records(self, current_time, time_window):

        for oht_id, records in list(self.entry_exit_records.items()):  

            self.entry_exit_records[oht_id] = [
                (entry, exit) for entry, exit in records
                if exit is not None and exit >= current_time - time_window
            ]


            if not self.entry_exit_records[oht_id]:
                del self.entry_exit_records[oht_id]

    def calculate_avg_speed(self, time_window, current_time):
        self.prune_old_records(current_time, time_window)

        total_time_covered = 0
        total_distance_covered = 0  
        recent_speeds = [] 
        
        for oht_id, records in self.entry_exit_records.items():
            relevant_records = [
                (entry, exit) for entry, exit in records
                if exit is not None and entry >= current_time - time_window
            ]
            for entry, exit in relevant_records:
                travel_time = exit - entry
                if travel_time > 0.1: 
                    speed = self.length / travel_time
                    total_time_covered += travel_time
                    total_distance_covered += speed * travel_time
                    recent_speeds.append(speed)


        if len(self.OHTs) > 0:
            for oht in self.OHTs:
                recent_speeds.append(oht.speed)

        if recent_speeds:
            avg_speed = sum(recent_speeds) / len(recent_speeds)
        else:
            avg_speed = getattr(self, "avg_speed", self.max_speed) 
            
        prev_avg_speed = getattr(self, "avg_speed", avg_speed)
        if len(self.OHTs) == 0 and not recent_speeds:
            recovery_rate = 0.1 
            avg_speed = prev_avg_speed * (1 - recovery_rate) + self.max_speed * recovery_rate


        elif len(self.OHTs) > 0 and all(oht.speed < 1 for oht in self.OHTs):
            decay_rate = 0.1 
            avg_speed = prev_avg_speed * (1 - decay_rate)


        alpha = 2 / (time_window / 100) 
        alpha = max(0.1, min(alpha, 0.2)) 
        avg_speed = alpha * avg_speed + (1 - alpha) * prev_avg_speed
        
        avg_speed = round(avg_speed, 2)


        self.avg_speed = avg_speed

        return min(avg_speed, self.max_speed)


    
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
            from_node.coord + self.edge_map[self.edge.unit_vec] * self.from_dist
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
                from_node.OHT = None
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
                            if self.id not in edge.entry_exit_records:
                                edge.entry_exit_records[self.id] = []
                                edge.entry_exit_records[self.id].append((current_time, None))
                            
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
        
        if len(self.path)==0 and self.from_dist == edge.length:
            self.speed = 0
            self.acc = 0
            return
                
        if len(OHTs) > 1:
            index = OHTs.index(self)

            if index > 0: 
                prev_oht = OHTs[index - 1] 
                dist_diff = prev_oht.from_dist - self.from_dist
                
                
                if 0 < dist_diff < self.rect:
                    emergency_coeff = 1.0 * (dist_diff < emergency_threshold or self.speed == 0)
                    self.acc = emergency_coeff * (-self.speed / time_step) + (1-emergency_coeff) * (-3500)
                    # self.acc = -3500
                    # if self.speed == 0:
                    #     self.acc = 0
                    return
                
        if self.node_map[edge.dest].OHT is not None:
            dist_diff = edge.length - self.from_dist
            
            if 0 < dist_diff < self.rect:
                emergency_coeff = 1.0 * (dist_diff < emergency_threshold)
                self.acc = emergency_coeff * (-self.speed / time_step) + (1-emergency_coeff) * (-3500)
            # if 0 < dist_diff < self.rect: 
            #     self.acc = -self.speed/time_step 

            #     # self.acc = -3500
                # if self.speed == 0:
                #     self.acc = 0

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
                    # self.acc = -self.speed/time_step 
                    # self.acc = -3500
                    
                    # if self.speed == 0:
                    #     self.acc = 0

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
                    # self.acc = -3500
                    
                    # if self.speed == 0:
                    #     self.acc = 0
                    
                    emergency_coeff = 1.0 * (dist_diff < emergency_threshold or self.speed == 0)
                    self.acc = emergency_coeff * (-self.speed / time_step) + (1-emergency_coeff) * (-3500)


                    # self.acc = -self.speed/time_step 
                    return
                elif rem_diff == other_diff and self.id > other_oht.id: 
                    
                    emergency_coeff = 1.0 * (dist_diff < emergency_threshold or self.speed == 0)
                    self.acc = emergency_coeff * (-self.speed / time_step) + (1-emergency_coeff) * (-3500)
                    # self.acc = -self.speed/time_step 
                    # self.acc = -3500    
                    # if self.speed == 0:
                        # self.acc = 0

                    return
            except:
                pass
            
        # if len(self.path) > 0:
        #     outgoing_edges = [
        #         outgoing_edge for outgoing_edge in self.node_map[edge.dest].outgoing_edges
        #         if outgoing_edge != self.path[0]
        #     ]
        
        #     if len(outgoing_edges) == 1:
        #         rem_diff = edge.length - self.from_dist 
        #         try:
        #             other_oht = self.edge_map[outgoing_edges[0]].OHTs[0]
        #             other_diff= other_oht.from_dist 
        #             dist_diff = rem_diff + other_diff 
        #             if 0 < dist_diff <  self.rect: 
        #                 emergency_coeff = 1.0 * (dist_diff < emergency_threshold or self.speed == 0)
        #                 self.acc = emergency_coeff * (-self.speed / time_step) + (1-emergency_coeff) * (-3500)
        #                 # self.acc = -self.speed/time_step
        #                 # self.acc = -3500

                        
        #                 # if self.speed == 0:
        #                 #     self.acc = 0
        #                 return
        #         except:
        #             pass

        # self.acc = (edge.max_speed - self.speed) / time_step
        self.acc = 2000 if self.speed < edge.max_speed else 0

        
        
class AMHS:
    def __init__(self, nodes, edges, ports, num_OHTs, max_jobs, job_list = [], oht_list = []):
        
        self.graph = nk.Graph(directed=True, weighted=True) 
        self.node_map = {}
        self.edge_map = {}
        self.port_map = {}
        
        self.nodes = copy.deepcopy(nodes)


        for node in self.nodes:
            node.OHT = None
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
                rect=3000, 
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
                self.node_map[oht.from_node].OHT = None
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
    
        
    def start_simulation(self, socketio, sid, current_time, max_time = 4000, time_step = 0.1):
     
        if self.simulation_running:
            print("Simulation is already running. Stopping the current simulation...")
            self.stop_simulation_event.set()
            while self.simulation_running:
                socketio.sleep(0.01)
            return
        
        self.simulation_running = True
        self.stop_simulation_event.clear()
        
        count = 0
        edge_metrics_cache = {} 
        
        last_saved_time = -10

        while current_time < max_time:
            if self.stop_simulation_event.is_set():
                break
            
            self.current_time = current_time
            
            if count % 5 == 0:
                self.generate_job()
            self.assign_jobs()
                

            oht_positions = []
            for oht in self.OHTs:
                oht.move(time_step, current_time)

            self.update_edge_metrics(current_time, time_window=500)
            
            for oht in self.OHTs:
                oht.cal_pos(time_step)
                
            if current_time - last_saved_time > 10:
                edge_data = [(last_saved_time+10, edge.id, edge.avg_speed) for edge in self.edges]
                self.queue.put(edge_data)
                last_saved_time += 10
                
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
                socketio.emit('updateOHT', {'data': compressed_payload}, to=sid)


            current_time += time_step
            count += 1

        self.simulation_running = False
        print('Simulation ended')
        
        
    
    def accelerate_simul(self, socketio, sid, current_time, max_time = 4000, time_step = 0.1):
     
        if self.simulation_running:
            print("Simulation is already running. Stopping the current simulation...")
            self.stop_simulation_event.set()
            while self.simulation_running:
                socketio.sleep(0.01) 
            return
        
        self.simulation_running = True
        self.stop_simulation_event.clear()
        
        _current_time = 0
        
        count = 0
        
        accel_factor = 10
        
        last_saved_time = -10


        while _current_time < current_time:
            if self.stop_simulation_event.is_set():
                break
            
            if count % 5 == 0:
                self.generate_job()
            
            self.assign_jobs()
            
            for oht in self.OHTs:
                oht.move(time_step, _current_time)

            self.update_edge_metrics(_current_time, time_window=500)
            
            for oht in self.OHTs:
                oht.cal_pos(time_step)

            if _current_time - last_saved_time > 10:
                edge_data = [(last_saved_time+10, edge.id, edge.avg_speed) for edge in self.edges]
                self.queue.put(edge_data)
                last_saved_time += 10
                
            self.current_time = _current_time

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
            
            oht_positions = []
            for oht in self.OHTs:
                oht.move(time_step, _current_time)

            self.update_edge_metrics(_current_time, time_window=500)
            
            for oht in self.OHTs:
                oht.cal_pos(time_step)
                
            
            if _current_time - last_saved_time > 10:
                edge_data = [(last_saved_time+10, edge.id, edge.avg_speed) for edge in self.edges]
                self.queue.put(edge_data)
                last_saved_time += 10

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
        
        last_saved_time = -10

        
        count = 0

        while current_time < max_time:
            if self.back_stop_simulation_event.is_set():
                break

            self.current_time = current_time
            
            if count % 5 == 0:
                self.generate_job()
            self.assign_jobs()
                

            for oht in self.OHTs:
                oht.move(time_step, current_time)

            self.update_edge_metrics(current_time, time_window=500)
            
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

        
        
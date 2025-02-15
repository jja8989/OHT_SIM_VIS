from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import random
import json
import time
import threading
import math
# from simul import *
from simul_parallel import *
from datetime import datetime
import time


import pandas as pd
import io


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")  # Enable CORS for SocketIO

# Load the layout data
with open('fab_oht_layout_2nd.json') as f:
    layout_data = json.load(f)
    # print('load_ended')

@socketio.on('layout')
def layout():
    socketio.emit('layoutData', layout_data)



nodes = [node(n['id'], [n['x'], n['y']]) for n in layout_data['nodes']]
edges = [
    edge(
        source=next(node for node in nodes if node.id == rail['from']),
        dest=next(node for node in nodes if node.id == rail['to']),
        length=math.sqrt(
            (next(node for node in nodes if node.id == rail['to']).coord[0] -
             next(node for node in nodes if node.id == rail['from']).coord[0])**2 +
            (next(node for node in nodes if node.id == rail['to']).coord[1] -
             next(node for node in nodes if node.id == rail['from']).coord[1])**2
        ),
        max_speed=1500  # Default speed = 0.01
    )
    for rail in layout_data['rails']
]

ports = [
    port(
        name = p['name'], 
        from_node = p['from_node'],
        to_node =p['to_node'],
        from_dist = p['distance']
    )
    for p in layout_data['ports']
]

global job_list
global oht_list
oht_list = []
job_list = []
num_oht = 500


@socketio.on('uploadFiles')
def handle_file_upload(data):
    global job_list
    global oht_list

    job_list = []
    oht_list = []

    # job_file과 oht_file을 받아옴
    job_file = data.get('job_file')
    oht_file = data.get('oht_file')
    
    # 파일이 있을 경우 파일을 처리하여 job_list와 oht_list 업데이트
    if job_file:
        job_file_io = io.BytesIO(job_file)  # 받은 job_file을 BytesIO로 읽어옴
        df = pd.read_csv(job_file_io)
        for _, row in df.iterrows():
            start_port_name = row['start_port']
            end_port_name = row['end_port']
            job_list.append([start_port_name, end_port_name])

    if oht_file:
        oht_file_io = io.BytesIO(oht_file)  # 받은 oht_file을 BytesIO로 읽어옴
        df = pd.read_csv(oht_file_io)
        for _, row in df.iterrows():
            start_node = row['start_node']
            oht_list.append(start_node)

    # 파일 처리 완료 후, 클라이언트에 처리된 결과를 전송
    socketio.emit('filesProcessed', {"message": "Files successfully uploaded"})

amhs = None

def run_simulation(max_time):
    global amhs
    global job_list
    global oht_list
    
    amhs = AMHS(nodes=nodes, edges=edges, ports=ports, num_OHTs=500, max_jobs=1000, job_list = job_list, oht_list = oht_list)
    amhs.start_simulation(socketio, 0, max_time)

    
def accel_simulation(current_time, max_time):
    global amhs
    global job_list
    global oht_list
    global num_oht
    
    amhs = AMHS(nodes=nodes, edges=edges, ports=ports, num_OHTs=num_oht, max_jobs=1000, job_list = job_list, oht_list = oht_list)
    amhs.accelerate_simul(socketio, current_time, max_time)
    
    
@socketio.on('startSimulation')
def start_simulation(data):
    global max_time
    global num_oht
    max_time = data.get('max_time', 4000)
    current_time = data.get('current_time', None)
    num_oht = data.get('num_OHTs', 500)

    if not current_time:
        socketio.start_background_task(run_simulation, max_time)
    else:    
        socketio.start_background_task(accel_simulation, current_time, max_time)
        
    global current_simulation_id
    global last_saved_time
    last_saved_time = 0
    
    
@socketio.on('stopSimulation')
def stop_simulation():
    global amhs
    amhs.stop_simulation_event.set()  # Signal all running processes to stop
    socketio.emit('simulationStopped')  # Notify the frontend that the simulation has stopped

@socketio.on('modiRail')
def handle_rail_update(data):    
    global amhs
    global current_simulation_id

    removed_rail_key = data['removedRailKey']
    oht_positions = data['ohtPositions']
    is_removed = data['isRemoved']
    current_time = data['currentTime']  # currentTime 추가
    edge_data = data['edges']
    
    print('????')

    
    amhs.simulation_running = False
    amhs.stop_simulation_event.set()
    
    if amhs.simulation_running:
        amhs.stop_simulation_event.set()
        while amhs.simulation_running:
            socketio.sleep(0.01)  # Wait for the current simulation to stop
    
    socketio.emit('simulationStopped') 
    
    amhs.current_time = current_time

    # Remove the specified rail from the simulation graph
    source, dest = removed_rail_key.split('-')
    amhs.modi_edge(source, dest, oht_positions, is_removed)
    amhs.reinitialize_simul(oht_positions, edge_data)
    
        # DB에서 현재 simulation time 이후 데이터 삭제
    # if current_simulation_id:
    #     clear_future_edge_data(current_simulation_id, current_time)
    
    print('????')
    
    socketio.start_background_task(restart_simulation, amhs, current_time)
    # socketio.start_background_task(save_edge_data) 


def restart_simulation(amhs, current_time):
    """
    Restart the simulation from the given current_time using the existing AMHS object.

    Parameters:
        amhs (AMHS): The current AMHS object to be used for the simulation.
        current_time (float): The time to restart the simulation from.
        max_time (float): The maximum time for the simulation to run. Default is 4000.
        time_step (float): The time step for the simulation loop. Default is 0.01.

    Returns:
        None
    """    
    global max_time
    if amhs.simulation_running:
        amhs.stop_simulation_event.set()
        while amhs.simulation_running:
            socketio.sleep(0.01)  # Wait for the current simulation to stop
        return
    
    amhs.start_simulation(socketio, current_time, max_time)    
    
    
if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)

from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import networkx as nx
import simpy
import random
import json
import time
import threading
import math
from simul import *
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")  # Enable CORS for SocketIO

# Load the layout data
with open('fab_oht_layout_2nd.json') as f:
    layout_data = json.load(f)

@app.route('/layout')
def layout():
    return jsonify(layout_data)



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
        from_node = next((node for node in nodes if node.id == p['from_node'])),
        to_node = next((node for node in nodes if node.id == p['to_node'])),
        from_dist = p['distance']
    )
    for p in layout_data['ports']
]

global job_list
global oht_list
oht_list = []
job_list = []

@app.route('/upload_csv_files', methods=['POST'])
def upload_job():
    global job_list
    global oht_list
    job_list = []
    oht_list = []

    
    job_file = request.files['job_file'] if 'job_file' in request.files else ''
    oht_file = request.files['oht_file'] if 'oht_file' in request.files else ''
    
    print(job_file, oht_file)
       
    if job_file != '':
        df = pd.read_csv(job_file)
        for _, row in df.iterrows():
            start_port_name = row['start_port']
            end_port_name = row['end_port']
            job_list.append([start_port_name, end_port_name])
            
    if oht_file != '':
        df = pd.read_csv(oht_file)
        for _, row in df.iterrows():
            start_node = row['start_node']
            oht_list.append(start_node)

            
    return jsonify({"message": "File successfully uploaded"})


global amhs

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
    
    amhs = AMHS(nodes=nodes, edges=edges, ports=ports, num_OHTs=500, max_jobs=1000, job_list = job_list, oht_list = oht_list)
    amhs.accelerate_simul(socketio, current_time, max_time)

@socketio.on('startSimulation')
def start_simulation(data):
    global max_time
    max_time = data.get('max_time', 4000)
    current_time = data.get('current_time', None)
    print(max_time, current_time)
    if not current_time:
        socketio.start_background_task(run_simulation, max_time)
    else:
        
        socketio.start_background_task(accel_simulation, current_time, max_time)
    
    
@socketio.on('stopSimulation')
def stop_simulation():
    global amhs
    amhs.stop_simulation_event.set()  # Signal all running processes to stop
    socketio.emit('simulationStopped')  # Notify the frontend that the simulation has stopped

@socketio.on('modiRail')
def handle_rail_update(data):    
    global amhs

    removed_rail_key = data['removedRailKey']
    oht_positions = data['ohtPositions']
    is_removed = data['isRemoved']
    current_time = data['currentTime']  # currentTime 추가
    edge_data = data['edges']
    
    amhs.simulation_running = False
    amhs.stop_simulation_event.set()
    
    
    if amhs.simulation_running:
        amhs.stop_simulation_event.set()
        while amhs.simulation_running:
            socketio.sleep(0.01)  # Wait for the current simulation to stop
    
    socketio.emit('simulationStopped') 

    # Remove the specified rail from the simulation graph
    source, dest = removed_rail_key.split('-')
    amhs.modi_edge(source, dest, oht_positions, is_removed)
    amhs.reinitialize_simul(oht_positions, edge_data)
    
    socketio.start_background_task(restart_simulation, amhs, current_time)

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
    socketio.run(app, port=5001, debug=True)
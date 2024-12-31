from flask import Flask, jsonify
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

# Initialize AMHS

global amhs
simulation_running = False
stop_simulation_event = threading.Event()


def run_simulation():
    
    global amhs
    amhs = AMHS(nodes=nodes, edges=edges, ports=ports, num_OHTs=500, max_jobs=1000)

    """Runs the simulation using AMHS."""
    global simulation_running
    simulation_running = True
    stop_simulation_event.clear()

    time_step = 0.01
    max_time = 4000
    current_time = 0
    count = 0
    # pdb.set_trace()

    while current_time < max_time:
        if stop_simulation_event.is_set():
            break

        amhs.generate_job()

        # 작업 할당
        amhs.assign_jobs()

        # Move all OHTs
        oht_positions = []
        for oht in amhs.OHTs:
            oht.move(time_step)
        
        if count%10==0:
            for oht in amhs.OHTs:
                oht_positions.append({
                    'id': oht.id,  # Unique identifier
                    'x': oht.pos[0],
                    'y': oht.pos[1],
                    
                    'source': oht.edge.source.id if oht.edge else None,  # Source node of the current edge
                    'dest': oht.edge.dest.id if oht.edge else None,
                    
                    'speed': oht.speed,                                 # Current speed of the OHT
                    'status': oht.status,                               # Current status (e.g., IDLE, TO_START, TO_END)
                    'startPort': oht.start_port.name if oht.start_port else None,  # Start port name
                    'endPort': oht.end_port.name if oht.end_port else None,
                    
                    'from_node' : oht.from_node.id if oht.from_node else None,# End port name
                    'from_dist': oht.from_dist,
                    'wait_time': oht.wait_time
                })
                    
                    # print(oht_positions)

            # Emit the current time and OHT positions
            socketio.emit('updateOHT', {
                'time': current_time,
                'oht_positions': oht_positions
            })

        # Increment time
        current_time += time_step
        count += 1

        # Sleep for a real-time effect
        socketio.sleep(0.0001)

    simulation_running = False
    
    print('simulation ended')


@socketio.on('startSimulation')
def start_simulation():
    socketio.start_background_task(run_simulation)
    
    
@socketio.on('stopSimulation')
def stop_simulation():
    global simulation_running
    simulation_running = False  # Set the simulation as stopped
    stop_simulation_event.set()  # Signal all running processes to stop
    socketio.emit('simulationStopped')  # Notify the frontend that the simulation has stopped

@socketio.on('removeRail')
def handle_rail_update(data):    
    global amhs
    global simulation_running

    removed_rail_key = data['removedRailKey']
    oht_positions = data['ohtPositions']
    is_removed = data['isRemoved']
    current_time = data['currentTime']  # currentTime 추가
    
    # print(oht_positions)
    
    
    print('removeRail')

    
    simulation_running = False
    stop_simulation_event.set()
    
    
    if simulation_running:
        print("Stopping current simulation before removing rail...")
        stop_simulation_event.set()
        while simulation_running:
            socketio.sleep(0.01)  # Wait for the current simulation to stop
    
    socketio.emit('simulationStopped') 

    # Remove the specified rail from the simulation graph
    source, dest = removed_rail_key.split('-')
    amhs.modi_edge(source, dest, oht_positions, is_removed)
    amhs.reinitialize_simul(oht_positions)
    
    socketio.start_background_task(restart_simulation, amhs, current_time)

    
    # print(removed_rail_key)
    # print(oht_positions)
    
def restart_simulation(amhs, current_time, max_time=4000, time_step=0.01):
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
    global simulation_running
    
    if simulation_running:
        print("Simulation is already running. Stopping the current simulation...")
        stop_simulation_event.set()
        while simulation_running:
            socketio.sleep(0.01)  # Wait for the current simulation to stop
        return
            
    simulation_running = True
    stop_simulation_event.clear()

    count = 0  # Counter to manage the frequency of OHT position updates

    while current_time < max_time:
        if stop_simulation_event.is_set():
            break

        amhs.generate_job()

        # Assign jobs to OHTs
        amhs.assign_jobs()

        # Move all OHTs
        oht_positions = []
        for oht in amhs.OHTs:
            oht.move(time_step)

        if count % 10 == 0:  # Emit OHT positions at every 10th iteration
            for oht in amhs.OHTs:
                oht_positions.append({
                    'id': oht.id,
                    'x': oht.pos[0],
                    'y': oht.pos[1],
                    'source': oht.edge.source.id if oht.edge else None,
                    'dest': oht.edge.dest.id if oht.edge else None,
                    'speed': oht.speed,
                    'status': oht.status,
                    'startPort': oht.start_port.name if oht.start_port else None,
                    'endPort': oht.end_port.name if oht.end_port else None,
                    'from_node': oht.from_node.id if oht.from_node else None,
                    'from_dist': oht.from_dist,
                    'wait_time': oht.wait_time
                })

            # Emit the current time and OHT positions
            socketio.emit('updateOHT', {
                'time': current_time,
                'oht_positions': oht_positions
            })

        # Increment time
        current_time += time_step
        count += 1

        # Sleep for a real-time effect
        socketio.sleep(0.0001)

    simulation_running = False

    print('Simulation restarted and ended.')
    
    
    
if __name__ == '__main__':
    socketio.run(app, port=5001, debug=True)
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

simulation_running = False
stop_simulation_event = threading.Event()


def run_simulation():
    
    amhs = AMHS(nodes=nodes, edges=edges, ports=ports, num_OHTs=500, max_jobs=1000)

    """Runs the simulation using AMHS."""
    global simulation_running
    simulation_running = True
    stop_simulation_event.clear()

    time_step = 0.1
    max_time = 4000
    current_time = 0

    while current_time < max_time:
        if stop_simulation_event.is_set():
            break

        amhs.generate_job()

        # 작업 할당
        amhs.assign_jobs()
        
        mismatched_oht = []

        # Move all OHTs
        oht_positions = []
        for oht in amhs.OHTs:
            oht.move(time_step)
        
        for oht in amhs.OHTs:
            oht_positions.append({
                'id': oht.id,  # Unique identifier
                'x': oht.pos[0],
                'y': oht.pos[1],
                'source': oht.edge.source.id if oht.edge else None,  # Source node of the current edge
                'dest': oht.edge.dest.id if oht.edge else None   
            })
                
                # print(oht_positions)

        # Emit the current time and OHT positions
        socketio.emit('updateOHT', {
            'time': current_time,
            'oht_positions': oht_positions
        })

        # Increment time
        current_time += time_step

        # Sleep for a real-time effect
        socketio.sleep(0.001)

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
    
    
    
if __name__ == '__main__':
    socketio.run(app, port=5001, debug=True)
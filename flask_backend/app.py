from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import networkx as nx
import simpy
import random
import json
import time
import threading


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")  # Enable CORS for SocketIO

# Load the layout data
with open('fab_oht_layout.json') as f:
    layout_data = json.load(f)

# Create the network graph
network = nx.DiGraph()
for node in layout_data['nodes']:
    network.add_node(node['id'], pos=(node['x'], node['y']))

for rail in layout_data['rails']:
    network.add_edge(rail['from'], rail['to'], length=1, speed=0.01, count=0)  # Simplified length and speed
    
simulation_running = False
simulation_thread = None
stop_simulation_event = threading.Event()
simulation_env = None


@app.route('/layout')
def layout():
    return jsonify(layout_data)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    

def oht_process(env, name, network, start, end, link_resources):
    try:
        path = nx.dijkstra_path(network, start, end, weight='length')
    except nx.NetworkXNoPath:
        return

    for i in range(len(path) - 1):
        if stop_simulation_event.is_set():
            return  # Stop the process if the simulation is not running
        source, dest = path[i], path[i + 1]
        edge_data = network.get_edge_data(source, dest)
        link_resource = link_resources[(source, dest)]['resource']

        with link_resource.request() as req:
            yield req
            network[source][dest]['count'] += 1
            travel_time = edge_data['length'] / edge_data['speed']

            # Send updates during the travel
            for progress in range(1, 11):
                if stop_simulation_event.is_set():
                    return  # Stop the process if the simulation is stopped
                position = (
                    network.nodes[source]['pos'][0] + (network.nodes[dest]['pos'][0] - network.nodes[source]['pos'][0]) * (progress / 10),
                    network.nodes[source]['pos'][1] + (network.nodes[dest]['pos'][1] - network.nodes[source]['pos'][1]) * (progress / 10)
                )
                socketio.emit('updateOHT', {'id': name, 'x': position[0], 'y': position[1], 'time': env.now})
                # time.sleep(0.01)
                yield env.timeout(travel_time / 10)

        position = network.nodes[dest]['pos']
        socketio.emit('updateOHT', {'id': name, 'x': position[0], 'y': position[1], 'time': env.now})
        

# def run_simulation_test():
#         # Send updates during the travel
#     name='test_oht'
#     for progress in range(1, 10):
#         sim_time = progress
#         position = (
#             500*progress,
#             1000*progress
#         )
#         socketio.emit('updateOHT', {'id': name, 'x': position[0], 'y': position[1], 'time': sim_time})
#         time.sleep(1)
    
def run_simulation():
    global simulation_running, stop_simulation_event, simulation_env
    simulation_running = True  # Set the simulation as running
    stop_simulation_event.clear()  # Clear the stop event
    env = simpy.Environment()
    link_resources = {}
    for rail in layout_data['rails']:
        link_resources[(rail['from'], rail['to'])] = {'resource': simpy.Resource(env, capacity=1)}
    
    nodes = list(network.nodes)
    num_ohts = 500  # Number of OHTs
    oht_names = [f'oht_{i+1}' for i in range(num_ohts)]

    for name in oht_names:
        start = random.choice(nodes)
        end = random.choice(nodes)
        while end == start:
            end = random.choice(nodes)
        env.process(oht_process(env, name, network, start, end, link_resources))

    env.run(until=1000)  # Run the simulation for 100 time units
    if simulation_running and not stop_simulation_event.is_set():
        socketio.emit('simulationComplete')  # Emit event when simulation is complete
    simulation_running = False  # Reset the simulation status

@socketio.on('startSimulation')
def start_simulation():
    socketio.start_background_task(run_simulation)
    
@socketio.on('stopSimulation')
def stop_simulation():
    global simulation_running, simulation_env
    simulation_running = False  # Set the simulation as stopped
    stop_simulation_event.set()  # Signal all running processes to stop
    if simulation_env:
        simulation_env.exit()  # Exit the simulation environment
    socketio.emit('simulationStopped')  # Notify the frontend that the simulation has stopped
    
if __name__ == '__main__':
    socketio.run(app, port=5001, debug=True)
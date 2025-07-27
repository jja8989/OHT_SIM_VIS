from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import random
import json
import time
import threading
import math
from simul import *
import psycopg2
from datetime import datetime
import time
import pandas as pd
import io
from config import DATABASE_URL

user_sessions = {}
client_id_to_sid = {}


def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

def initialize_database():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS simulation_metadata (
            simulation_id SERIAL PRIMARY KEY,
            start_time TIMESTAMP DEFAULT NOW(),
            description TEXT
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
    
initialize_database()


def create_simulation_table(simulation_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS simulation_{simulation_id} (
            time TEXT,
            edge_id TEXT,
            avg_speed FLOAT,
            PRIMARY KEY (time, edge_id)  -- ✅ 시간 + 엣지 ID를 복합 키로 설정

        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    


def clear_future_edge_data(sid, simulation_id, simulation_time):
    formatted_time = format_simulation_time(simulation_time) 
    
    last_saved_time = user_sessions[sid].get('last_saved_time', 0)

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(f"""
        DELETE FROM simulation_{simulation_id} WHERE time >= %s;
    """, (formatted_time,))
    
    cur.execute(f"""
        SELECT time FROM simulation_{simulation_id} ORDER BY time DESC LIMIT 1;
    """)
    last_row = cur.fetchone()


    if last_row:
        last_saved_time = parse_simulation_time(last_row[0]) 
    else:
        last_saved_time = simulation_time 
    
    user_sessions[sid]['last_saved_time'] = last_saved_time

    conn.commit()
    cur.close()
    conn.close()
    
def parse_simulation_time(time_str):

    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

    
def format_simulation_time(sim_time):

    hours, remainder = divmod(sim_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"




app = Flask(__name__)
CORS(app) 
socketio = SocketIO(app, cors_allowed_origins="*")  

with open('fab_oht_layout.json') as f:
    layout_data = json.load(f)

@socketio.on('layout')
def layout():
    sid = request.sid

    socketio.emit('layoutData', layout_data, to=sid)



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
        max_speed=1500
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

@socketio.on('get_simulation_tables')
def handle_get_simulation_tables():
    sid = request.sid

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_name LIKE 'simulation_%'
    """)

    tables = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()

    emit('simulation_tables', {"tables": tables}, to=sid)

@socketio.on('get_simulation_data')
def handle_get_simulation_data(data):

    sid = request.sid

    table_name = data.get('table_name')

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(f"SELECT * FROM {table_name} ORDER BY time ASC;") 
    rows = cur.fetchall()

    columns = [desc[0] for desc in cur.description]
    result = [dict(zip(columns, row)) for row in rows]

    cur.close()
    conn.close()

    emit('simulation_data', {"data": result}, to=sid)


@socketio.on('delete_simulation_table')
def handle_delete_simulation_table(data):

    sid = request.sid

    table_name = data.get('table_name')

    if not table_name.startswith("simulation_"):
        emit('table_deleted', {"success": False, "message": "Invalid table name"}, to=sid)
        return

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute(f"DROP TABLE IF EXISTS {table_name};")
        conn.commit()
        emit('table_deleted', {"success": True, "message": f"Table {table_name} deleted successfully"}, to=sid)
    except Exception as e:
        emit('table_deleted', {"success": False, "message": str(e)}, to=sid)
    finally:
        cur.close()
        conn.close()

@socketio.on('uploadFiles')
def handle_file_upload(data):
    sid = request.sid
    
    user_sessions[sid] = {}
    
    user_sessions[sid]['stop_saving_event'] = threading.Event()    
    user_sessions[sid]['stop_saving_back_event'] = threading.Event()    


    job_list = []
    oht_list = []

    job_file = data.get('job_file')
    oht_file = data.get('oht_file')
    
    user_sessions[sid]['job_list'] = []
    user_sessions[sid]['oht_list'] = []
    if job_file:
        df = pd.read_csv(io.BytesIO(job_file))
        user_sessions[sid]['job_list'] = [[row['start_port'], row['end_port']] for _, row in df.iterrows()]
    if oht_file:
        df = pd.read_csv(io.BytesIO(oht_file))
        user_sessions[sid]['oht_list'] = [row['start_node'] for _, row in df.iterrows()]
    emit('filesProcessed', {"message": "Files successfully uploaded"}, to=sid)
    

def run_simulation(sid, max_time):    
    
    job_list = user_sessions[sid].get('job_list', [])
    oht_list = user_sessions[sid].get('oht_list', [])
    num_oht = user_sessions[sid].get('num_OHTs', 500)
    
    print(num_oht)
    
    print(user_sessions[sid]['num_OHTs'])

    
    amhs = AMHS(nodes=nodes, edges=edges, ports=ports, num_OHTs=num_oht, max_jobs=1000, job_list=job_list, oht_list=oht_list)
    user_sessions[sid]['amhs'] = amhs
    amhs.start_simulation(socketio, sid, 0, max_time)

    
    
    

def save_edge_data(sid):    
    current_simulation_id = user_sessions[sid].get('current_simulation_id', None)
    last_saved_time = user_sessions[sid].get('last_saved_time', -10)
    stop_saving_event = user_sessions[sid].get('stop_saving_event', None)

    amhs = user_sessions[sid].get('amhs', None)

    
    max_wait_time = 10
    waited_time = 0
    while amhs is None and waited_time < max_wait_time:
        time.sleep(1)
        waited_time += 1

    if amhs is None:
        return
    
    stop_saving_event.clear()


    while current_simulation_id:
        while amhs is not None and amhs.simulation_running and not stop_saving_event.is_set(): 
            if not amhs.queue.empty():
                edge_data = amhs.queue.get()
                if edge_data:
                    conn = get_db_connection()
                    cur = conn.cursor()


                    formatted_data = [
                        (format_simulation_time(row[0]), row[1], row[2]) for row in edge_data
                    ]

                    cur.executemany(f"""
                        INSERT INTO simulation_{current_simulation_id} (time, edge_id, avg_speed) 
                        VALUES (%s, %s, %s)
                        ON CONFLICT (time, edge_id) DO UPDATE SET avg_speed = EXCLUDED.avg_speed;
                    """, formatted_data)

                    conn.commit()
                    cur.close()
                    conn.close()
        
        break 

def save_edge_data_back(sid):    
        
    back_simulation_id = user_sessions[sid].get('back_simulation_id', None)
    last_saved_time_back = user_sessions[sid].get('last_saved_time_back', -10)
    stop_saving_back_event = user_sessions[sid].get('stop_saving_back_event', None)

    
    back_amhs = user_sessions[sid].get('back_amhs', None)    

    if back_simulation_id is None:
        return

    max_wait_time = 10
    waited_time = 0
    while back_amhs is None and waited_time < max_wait_time:
        time.sleep(1)
        waited_time += 1

    if back_amhs is None:
        return
    
    stop_saving_back_event.clear()

    while back_simulation_id:
        while back_amhs is not None and back_amhs.back_simulation_running and not stop_saving_back_event.is_set():            
            if not back_amhs.back_queue.empty():
                edge_data = back_amhs.back_queue.get()
                if edge_data:
                    conn = get_db_connection()
                    cur = conn.cursor()

                    formatted_data = [
                        (format_simulation_time(row[0]), row[1], row[2]) for row in edge_data
                    ]

                    cur.executemany(f"""
                        INSERT INTO simulation_{back_simulation_id} (time, edge_id, avg_speed) 
                        VALUES (%s, %s, %s)
                        ON CONFLICT (time, edge_id) DO UPDATE SET avg_speed = EXCLUDED.avg_speed;
                    """, formatted_data)

                    conn.commit()
                    cur.close()
                    conn.close()
                
            time.sleep(0.5)    
        break


    
def accel_simulation(sid, current_time, max_time):    

    job_list = user_sessions[sid].get('job_list', None)
    oht_list = user_sessions[sid].get('oht_list', None)
    num_oht = user_sessions[sid].get('num_OHTs', 500)

    amhs = AMHS(nodes=nodes, edges=edges, ports=ports, num_OHTs=num_oht, max_jobs=1000, job_list = job_list, oht_list = oht_list)
    user_sessions[sid]['amhs'] = amhs

    amhs.accelerate_simul(socketio, sid, current_time, max_time)
    
def only_simulation(sid, max_time):    
    
    job_list = user_sessions[sid].get('job_list', None)
    oht_list = user_sessions[sid].get('oht_list', None)
    num_oht = user_sessions[sid].get('num_OHTs', 500)
    back_amhs = AMHS(nodes=nodes, edges=edges, ports=ports, num_OHTs=num_oht, max_jobs=1000, job_list = job_list, oht_list = oht_list)
    user_sessions[sid]['back_amhs'] = back_amhs

    back_amhs.only_simulation(socketio, sid, 0, max_time)
    
    
@socketio.on('startSimulation')
def start_simulation(data):
    sid = request.sid 

    max_time = data.get('max_time', 4000)
    current_time = data.get('current_time', None)
    num_oht = data.get('num_OHTs', 500)
    
    print(num_oht)

    
    user_sessions[sid]['max_time'] = max_time
    user_sessions[sid]['current_time'] = current_time
    user_sessions[sid]['num_OHTs'] = num_oht
    
    user_sessions[sid]['amhs'] = None
    
    print(user_sessions[sid]['num_OHTs'])

    if not current_time:
        socketio.start_background_task(run_simulation, sid, max_time)
    else:    
        socketio.start_background_task(accel_simulation, sid, current_time, max_time)
        
    last_saved_time = -10
    user_sessions[sid]['last_saved_time'] = last_saved_time
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("INSERT INTO simulation_metadata (description, start_time) VALUES (%s, %s) RETURNING simulation_id;", 
                ("New simulation started", datetime.now()))
    current_simulation_id = cur.fetchone()[0]
    user_sessions[sid]['current_simulation_id'] = current_simulation_id

    conn.commit()

    create_simulation_table(current_simulation_id)

    cur.close()
    conn.close()

    socketio.start_background_task(save_edge_data, sid)
    

@socketio.on('onlySimulation')
def start_only_simulation(data):
    
    sid = request.sid 

    back_amhs = None

    max_time = data.get('max_time', 4000)
    current_time = data.get('current_time', None)
    num_oht = data.get('num_OHTs', 500)
    
    user_sessions[sid]['max_time'] = max_time
    user_sessions[sid]['current_time'] = current_time
    user_sessions[sid]['num_OHTs'] = num_oht


    socketio.start_background_task(only_simulation, sid, max_time)
    
    last_saved_time_back = -10
    
    user_sessions[sid]['last_saved_time_back'] = last_saved_time_back

    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("INSERT INTO simulation_metadata (description, start_time) VALUES (%s, %s) RETURNING simulation_id;", 
                ("New simulation started", datetime.now()))
    back_simulation_id = cur.fetchone()[0]
    conn.commit()
    
    user_sessions[sid]['back_simulation_id'] = back_simulation_id

    create_simulation_table(back_simulation_id)

    cur.close()
    conn.close()

    socketio.start_background_task(save_edge_data_back, sid) 
    
    
@socketio.on('stopSimulation')
def stop_simulation():
    sid = request.sid
    amhs = user_sessions[sid]['amhs']
    stop_saving_event = user_sessions[sid].get('stop_saving_event', None)
    amhs.stop_simulation_event.set() 
    stop_saving_event.set()
    socketio.emit('simulationStopped', to=sid)
    
@socketio.on('stopBackSimulation')
def stop_Back_simulation():
    sid = request.sid
    back_amhs = user_sessions[sid]['back_amhs']
    stop_saving_back_event = user_sessions[sid].get('stop_saving_back_event', None)


    back_amhs.back_stop_simulation_event.set()
    stop_saving_back_event.set()  
    socketio.emit('simulationBackStopped', to=sid) 

@socketio.on('modiRail')
def handle_rail_update(data):        
    sid = request.sid
    
    amhs = user_sessions[sid]['amhs']
    stop_saving_event = user_sessions[sid].get('stop_saving_event', None)

    current_simulation_id = user_sessions[sid]['current_simulation_id']
    last_saved_time = user_sessions[sid]['last_saved_time']


    removed_rail_key = data['removedRailKey']
    oht_positions = data['ohtPositions']
    is_removed = data['isRemoved']
    current_time = data['currentTime'] 
    edge_data = data['edges']
    
    amhs.simulation_running = False
    amhs.stop_simulation_event.set()
    stop_saving_event.set()
    
    if amhs.simulation_running:
        amhs.stop_simulation_event.set()
        while amhs.simulation_running:
            socketio.sleep(0.01) 
            
    socketio.emit('simulationStopped', to=sid) 
    
    amhs.current_time = current_time

    source, dest = removed_rail_key.split('-')
    amhs.modi_edge(source, dest, oht_positions, is_removed)
    amhs.reinitialize_simul(oht_positions, edge_data)
    
    if current_simulation_id:
        clear_future_edge_data(sid, current_simulation_id, current_time)
    
    
    socketio.start_background_task(restart_simulation, sid, amhs, current_time)
    socketio.start_background_task(save_edge_data, sid) 


def restart_simulation(sid, amhs, current_time):
    max_time = user_sessions[sid]['max_time']
    
    if amhs.simulation_running:
        amhs.stop_simulation_event.set()
        while amhs.simulation_running:
            socketio.sleep(0.01) 
        return
    
    amhs.start_simulation(socketio, sid, current_time, max_time)    
    

@socketio.on('connect')
def on_connect():
    sid = request.sid
    client_id = request.args.get('client_id') 
    
    if client_id in client_id_to_sid:
        user_sessions[sid] = client_id_to_sid[client_id]
        client_id_to_sid[client_id] = user_sessions[sid]

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid 
    client_id = request.args.get('client_id') 
    client_id_to_sid[client_id] = user_sessions[sid]

    if sid in user_sessions:
        del user_sessions[sid]
    
if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)

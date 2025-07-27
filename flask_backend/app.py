from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import random
import json
import time
import threading
import math
from simul import *
# from simul_parallel import *
import psycopg2
from datetime import datetime
import time


import pandas as pd
import io

from config import DATABASE_URL

user_sessions = {}


current_simulation_id = None  # 현재 실행 중인 시뮬레이션 ID

# 백엔드 전용 시뮬레이션 ID
back_simulation_id = None  


def get_db_connection():
    """PostgreSQL 데이터베이스 연결"""
    return psycopg2.connect(DATABASE_URL)

def initialize_database():
    """필요한 모든 테이블을 자동 생성하는 함수"""
    conn = get_db_connection()
    cur = conn.cursor()

    # ✅ simulation_metadata 테이블 생성
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
    """각 시뮬레이션별 데이터를 저장할 테이블 생성"""
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
    # print('table created')
    conn.commit()
    cur.close()
    conn.close()
    


def clear_future_edge_data(simulation_id, simulation_time):
    """현재 simulation time (float, 초 단위)을 HH:MM:SS 형식으로 변환 후 이후 데이터 삭제"""
    formatted_time = format_simulation_time(simulation_time)  # ✅ HH:MM:SS 변환
    
    global last_saved_time

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(f"""
        DELETE FROM simulation_{simulation_id} WHERE time >= %s;
    """, (formatted_time,))  # Simulation Time 이후 데이터 삭제
    
    cur.execute(f"""
        SELECT time FROM simulation_{simulation_id} ORDER BY time DESC LIMIT 1;
    """)
    last_row = cur.fetchone()

    # ✅ last_saved_time 업데이트
    if last_row:
        last_saved_time = parse_simulation_time(last_row[0])  # HH:MM:SS → 초 변환
    else:
        last_saved_time = simulation_time  # 모든 데이터가 삭제되었다면 현재 시간 유지

    conn.commit()
    cur.close()
    conn.close()
    
def parse_simulation_time(time_str):
    """HH:MM:SS 형식의 문자열을 초 단위 float으로 변환"""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

    
def format_simulation_time(sim_time):
    """✅ 시뮬레이션 시간을 HH:MM:SS 형태로 변환"""
    hours, remainder = divmod(sim_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

stop_saving_event = threading.Event()
stop_saving_back_event = threading.Event()




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

@socketio.on('get_simulation_tables')
def handle_get_simulation_tables():
    """저장된 시뮬레이션 테이블 목록을 소켓을 통해 조회"""
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_name LIKE 'simulation_%'
    """)

    tables = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()

    emit('simulation_tables', {"tables": tables})

@socketio.on('get_simulation_data')
def handle_get_simulation_data(data):
    """특정 시뮬레이션 결과를 소켓을 통해 조회"""
    table_name = data.get('table_name')

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(f"SELECT * FROM {table_name} ORDER BY time ASC;")  # ✅ 수정된 부분
    rows = cur.fetchall()

    columns = [desc[0] for desc in cur.description]  # 컬럼명 가져오기
    result = [dict(zip(columns, row)) for row in rows]

    cur.close()
    conn.close()

    emit('simulation_data', {"data": result})


@socketio.on('delete_simulation_table')
def handle_delete_simulation_table(data):
    """특정 시뮬레이션 테이블을 삭제"""
    table_name = data.get('table_name')

    if not table_name.startswith("simulation_"):
        emit('table_deleted', {"success": False, "message": "Invalid table name"})
        return

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute(f"DROP TABLE IF EXISTS {table_name};")
        conn.commit()
        emit('table_deleted', {"success": True, "message": f"Table {table_name} deleted successfully"})
    except Exception as e:
        emit('table_deleted', {"success": False, "message": str(e)})
    finally:
        cur.close()
        conn.close()

@socketio.on('uploadFiles')
def handle_file_upload(data):
    # global job_list
    # global oht_list
    
    sid = request.sid  # ✅ 반드시 필요


    job_list = []
    oht_list = []

    # job_file과 oht_file을 받아옴
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
    emit('filesProcessed', {"message": "Files successfully uploaded"})
    
    # # 파일이 있을 경우 파일을 처리하여 job_list와 oht_list 업데이트
    # if job_file:
    #     job_file_io = io.BytesIO(job_file)  # 받은 job_file을 BytesIO로 읽어옴
    #     df = pd.read_csv(job_file_io)
    #     for _, row in df.iterrows():
    #         start_port_name = row['start_port']
    #         end_port_name = row['end_port']
    #         job_list.append([start_port_name, end_port_name])

    # if oht_file:
    #     oht_file_io = io.BytesIO(oht_file)  # 받은 oht_file을 BytesIO로 읽어옴
    #     df = pd.read_csv(oht_file_io)
    #     for _, row in df.iterrows():
    #         start_node = row['start_node']
    #         oht_list.append(start_node)

    # # 파일 처리 완료 후, 클라이언트에 처리된 결과를 전송
    # socketio.emit('filesProcessed', {"message": "Files successfully uploaded"})

amhs = None
back_amhs = None

def run_simulation(max_time):
    # global amhs
    # global job_list
    # global oht_list
    
    # amhs = AMHS(nodes=nodes, edges=edges, ports=ports, num_OHTs=500, max_jobs=1000, job_list = job_list, oht_list = oht_list)
    # amhs.start_simulation(socketio, 0, max_time)
    
    job_list = user_sessions[sid].get('job_list', [])
    oht_list = user_sessions[sid].get('oht_list', [])
    global num_oht
    print('run', num_oht)
    amhs = AMHS(nodes=nodes, edges=edges, ports=ports, num_OHTs=num_oht, max_jobs=1000, job_list=job_list, oht_list=oht_list)
    user_sessions[sid]['amhs'] = amhs
    amhs.start_simulation(socketio, 0, max_time)

    
    
    

def save_edge_data():
    """시뮬레이션 current_time 기준으로 60초마다 Edge 데이터를 DB에 저장"""
    global current_simulation_id
    global last_saved_time
    global amhs  # ✅ 전역 변수 선언
    
    max_wait_time = 10  # 최대 대기 시간 (초)
    waited_time = 0
    while amhs is None and waited_time < max_wait_time:
        # print("⏳ Waiting for amhs to be initialized...")
        time.sleep(1)  # 1초 대기
        waited_time += 1

    if amhs is None:
        return
    
    stop_saving_event.clear()  # 🔥 중지 이벤트 초기화


    while current_simulation_id:
        while amhs is not None and amhs.simulation_running and not stop_saving_event.is_set():  # ✅ amhs가 None인지 체크
            if not amhs.queue.empty():
                edge_data = amhs.queue.get()
                if edge_data:
                    conn = get_db_connection()
                    cur = conn.cursor()

                    # ✅ 시간 포맷팅 (초 → HH:MM:SS 변환)
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
        
        break  # ✅ 루프 탈출

def save_edge_data_back():
    """백엔드 전용 시뮬레이션 데이터를 DB에 저장"""
    global back_simulation_id, last_saved_time_back, back_amhs

    if back_simulation_id is None:
        return

    max_wait_time = 10  # 최대 대기 시간 (초)
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

                    # ✅ 시간 포맷팅 (초 → HH:MM:SS 변환)
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
        break  # ✅ 루프 탈출


    
def accel_simulation(current_time, max_time):
    global amhs
    global job_list
    global oht_list
    global num_oht
    
    amhs = AMHS(nodes=nodes, edges=edges, ports=ports, num_OHTs=num_oht, max_jobs=1000, job_list = job_list, oht_list = oht_list)
    amhs.accelerate_simul(socketio, current_time, max_time)
    
def only_simulation(max_time):
    global back_amhs
    global job_list
    global oht_list
    global num_oht
    
    back_amhs = AMHS(nodes=nodes, edges=edges, ports=ports, num_OHTs=num_oht, max_jobs=1000, job_list = job_list, oht_list = oht_list)
    back_amhs.only_simulation(socketio, 0, max_time)
    
    
@socketio.on('startSimulation')
def start_simulation(data):
    global max_time
    global num_oht
    global amhs
    max_time = data.get('max_time', 4000)
    current_time = data.get('current_time', None)
    num_oht = data.get('num_OHTs', 500)
    amhs = None
    print(num_oht)
    if not current_time:
        socketio.start_background_task(run_simulation, max_time)
    else:    
        socketio.start_background_task(accel_simulation, current_time, max_time)
        
    global current_simulation_id
    global last_saved_time
    last_saved_time = -10
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("INSERT INTO simulation_metadata (description, start_time) VALUES (%s, %s) RETURNING simulation_id;", 
                ("New simulation started", datetime.now()))
    current_simulation_id = cur.fetchone()[0]
    conn.commit()

    # 새로운 시뮬레이션을 위한 테이블 생성
    create_simulation_table(current_simulation_id)

    cur.close()
    conn.close()

    socketio.start_background_task(save_edge_data)
    

@socketio.on('onlySimulation')
def start_only_simulation(data):
    global max_time
    global num_oht
    global back_amhs
    
    back_amhs = None

    max_time = data.get('max_time', 4000)
    current_time = data.get('current_time', None)
    num_oht = data.get('num_OHTs', 500)

    socketio.start_background_task(only_simulation, max_time)
        
    global back_simulation_id
    global last_saved_time_back
    last_saved_time_back = -10
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("INSERT INTO simulation_metadata (description, start_time) VALUES (%s, %s) RETURNING simulation_id;", 
                ("New simulation started", datetime.now()))
    back_simulation_id = cur.fetchone()[0]
    conn.commit()

    # 새로운 시뮬레이션을 위한 테이블 생성
    create_simulation_table(back_simulation_id)

    cur.close()
    conn.close()

    socketio.start_background_task(save_edge_data_back) 
    
    
@socketio.on('stopSimulation')
def stop_simulation():
    global amhs
    amhs.stop_simulation_event.set()  # Signal all running processes to stop
    stop_saving_event.set()  # 🔥 DB 저장 중단
    socketio.emit('simulationStopped')  # Notify the frontend that the simulation has stopped
    
@socketio.on('stopBackSimulation')
def stop_Back_simulation():
    global back_amhs
    back_amhs.back_stop_simulation_event.set()  # Signal all running processes to stop
    stop_saving_back_event.set()  # 🔥 DB 저장 중단
    socketio.emit('simulationBackStopped')  # Notify the frontend that the simulation has stopped

@socketio.on('modiRail')
def handle_rail_update(data):    
    global amhs
    global current_simulation_id
    global last_saved_time

    removed_rail_key = data['removedRailKey']
    oht_positions = data['ohtPositions']
    is_removed = data['isRemoved']
    current_time = data['currentTime']  # currentTime 추가
    edge_data = data['edges']
    
    amhs.simulation_running = False
    amhs.stop_simulation_event.set()
    stop_saving_event.set()  # 🔥 DB 저장 중단
    
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
    if current_simulation_id:
        clear_future_edge_data(current_simulation_id, current_time)
    
    
    socketio.start_background_task(restart_simulation, amhs, current_time)
    socketio.start_background_task(save_edge_data) 


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


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in user_sessions:
        del user_sessions[sid]
    
if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)

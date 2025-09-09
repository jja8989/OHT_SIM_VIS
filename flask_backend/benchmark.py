import json
import math
import time
import pandas as pd
from simul import node, edge, port, AMHS

# -------------------------------
# 레이아웃 데이터 로드
# -------------------------------
with open("fab_oht_layout_updated.json") as f:
    layout_data = json.load(f)

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
        max_speed=1000 if rail.get('curve', 0) == 1 else 5000
    )
    for rail in layout_data['rails']
]
ports = [
    port(
        name=p['name'],
        from_node=p['from_node'],
        to_node=p['to_node'],
        from_dist=p['distance']
    )
    for p in layout_data['ports']
]

# -------------------------------
# 시뮬레이션 실행 (소켓 없이)
# -------------------------------
def run_benchmark(num_ohts=200, max_time=2000, time_step=0.1):
    amhs = AMHS(nodes=nodes, edges=edges, ports=ports, num_OHTs=num_ohts, max_jobs=1000)

    start = time.time()
    current_time, count = 0, 0
    while current_time <= max_time:
        if count % 100 == 0:
            amhs.generate_job()
        amhs.assign_jobs()

        for oht in amhs.OHTs:
            oht.move(time_step, current_time)
        amhs.update_edge_metrics(current_time, time_window=60)
        for oht in amhs.OHTs:
            oht.cal_pos(time_step)

        current_time += time_step
        count += 1
    end = time.time()
    return end - start


print('start')
num_ohts_list = [100, 200, 300, 400, 500]
max_time_list = [1000, 2000, 3000, 4000, 5000]
time_step = 0.1
repeats = 3 

results = []

for num_ohts in num_ohts_list:
    for max_time in max_time_list:
        elapsed_times = []
        for r in range(repeats):
            elapsed = run_benchmark(num_ohts=num_ohts, max_time=max_time, time_step=time_step)
            elapsed_times.append(elapsed)
            print(f"Run {r+1}/{repeats} → OHT={num_ohts}, Time={max_time}, Elapsed={elapsed:.2f}s")

        avg_elapsed = sum(elapsed_times) / len(elapsed_times)
        results.append({
            "num_ohts": num_ohts,
            "max_time": max_time,
            "time_step": time_step,
            "repeats": repeats,
            "avg_elapsed_sec": round(avg_elapsed, 2)
        })
        print(f"✅ AVG → OHT={num_ohts}, Time={max_time}, Avg={avg_elapsed:.2f}s")


df = pd.DataFrame(results)
df.to_csv("benchmark_results.csv", index=False)
print("✅ Saved to benchmark_results.csv")
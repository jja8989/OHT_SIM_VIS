import os
import random
import math
import json
import pandas as pd
import networkx as nx
from simul import node, edge, port, AMHS
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

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
# networkx로 edge betweenness 계산
# -------------------------------
def compute_edge_betweenness(edges):
    G = nx.DiGraph()
    for e in edges:
        G.add_edge(e.source, e.dest, weight=e.length)

    edge_bw = nx.edge_betweenness_centrality(G, weight="weight")
    ranked = sorted(edge_bw.items(), key=lambda x: -x[1])
    return [(src, dst) for (src, dst), _ in ranked]

# -------------------------------
# dynamic cut/restore
# -------------------------------
def dynamic_cut_restore(amhs, central_edges, all_edges,
                        central_cut_prob=0.1, random_cut_prob=0.02, restore_prob=0.05):
    # central edges 조작
    for (src, dst) in central_edges:
        if amhs.graph.hasEdge(amhs.node_id_map[src], amhs.node_id_map[dst]):
            if random.random() < central_cut_prob:
                amhs.modi_edge(src, dst, [], is_removed=True)
        else:
            if random.random() < restore_prob:
                amhs.modi_edge(src, dst, [], is_removed=False)

    # random edges 조작
    for (src, dst) in random.sample(all_edges, k=max(1, len(all_edges)//20)):
        if amhs.graph.hasEdge(amhs.node_id_map[src], amhs.node_id_map[dst]):
            if random.random() < random_cut_prob:
                amhs.modi_edge(src, dst, [], is_removed=True)
        else:
            if random.random() < restore_prob:
                amhs.modi_edge(src, dst, [], is_removed=False)

# -------------------------------
# 시뮬레이션 (dynamic events 포함)
# -------------------------------
def simulate_with_dynamic_events(amhs, central_edges, all_edges,
                                 max_time=2000, time_step=0.1,
                                 interval=10, save_path=None):
    logs, current_time, count, last_logged = [], 0, 0, -interval

    while current_time <= max_time:
        if count % 100 == 0:
            amhs.generate_job()
        amhs.assign_jobs()

        for oht in amhs.OHTs:
            oht.move(time_step, current_time)
        amhs.update_edge_metrics(current_time, time_window=60)
        for oht in amhs.OHTs:
            oht.cal_pos(time_step)

        if count % 50 == 0:
            dynamic_cut_restore(amhs, central_edges, all_edges)

        if current_time - last_logged >= interval:
            for e in amhs.edges:
                logs.append({
                    "time": int(current_time),
                    "edge_id": e.id,
                    "avg_speed": e.avg_speed
                })
            last_logged = current_time

        current_time += time_step
        count += 1

    df = pd.DataFrame(logs)
    pivot_df = df.pivot(index="time", columns="edge_id", values="avg_speed").reset_index()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pivot_df.to_csv(save_path, index=False)
        print(f"✅ Saved {save_path} with shape {pivot_df.shape}")

    return pivot_df

# -------------------------------
# 개별 작업 함수 (병렬 실행용)
# -------------------------------
def run_single_experiment(args):
    num_ohts, max_time, run_id, save_dir, central_edges, all_edges = args
    fname = f"dynamic_oht{num_ohts}_time{max_time}_run{run_id}.csv"
    fpath = os.path.join(save_dir, fname)

    amhs = AMHS(nodes=nodes, edges=edges, ports=ports,
                num_OHTs=num_ohts, max_jobs=1000)

    simulate_with_dynamic_events(amhs,
                                 central_edges=central_edges[:20],  # top 20 edge만 사용
                                 all_edges=all_edges,
                                 max_time=max_time,
                                 time_step=0.1,
                                 interval=10,
                                 save_path=fpath)

# -------------------------------
# 메인 실행
# -------------------------------
if __name__ == "__main__":
    save_dir = "datasets_dynamic"
    os.makedirs(save_dir, exist_ok=True)

    # central edges 한 번만 계산
    edge_list = [(e.source, e.dest) for e in edges]
    central_edges = compute_edge_betweenness(edges)
    all_edges = edge_list

    num_ohts_list = [200, 300, 400, 500]
    max_time_list = [1000, 2000]
    repeats = 10

    tasks = []
    for num_ohts in num_ohts_list:
        for max_time in max_time_list:
            for run_id in range(1, repeats + 1):
                tasks.append((num_ohts, max_time, run_id, save_dir, central_edges, all_edges))

    # 병렬 실행
    with Pool(processes=min(cpu_count(), 8)) as pool:
        for _ in tqdm(pool.imap_unordered(run_single_experiment, tasks), total=len(tasks)):
            pass

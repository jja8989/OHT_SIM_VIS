# # viz.py
# import os, glob, json, argparse, math, random, datetime
# import numpy as np
# import pandas as pd
# import torch
# import matplotlib.pyplot as plt

# # 학습 때 만든 함수/모델 재사용
# from model import STGNN, build_edge_adjacency

# def pick_csv(data_dirs):
#     if isinstance(data_dirs, str): data_dirs=[data_dirs]
#     files=[]
#     for d in data_dirs:
#         files += glob.glob(os.path.join(d, "*.csv"))
#     if not files:
#         raise FileNotFoundError("No CSV files in given dirs.")
#     return random.choice(sorted(files))

# def inverse_transform(arr, mean, scale):
#     return arr * scale + mean

# def infer_hparams_from_state(state_dict):
#     # hidden 크기 & blocks 수 추론
#     hidden = state_dict["head.weight"].shape[1]            # (H, hidden)
#     block_ids = []
#     for k in state_dict.keys():
#         if k.startswith("blocks."):
#             try: block_ids.append(int(k.split(".")[1]))
#             except: pass
#     blocks = (max(block_ids) + 1) if block_ids else 2
#     return hidden, blocks

# @torch.no_grad()
# def predict_window(model, A_hat, data_norm, t, seq_len, horizons, device):
#     X_hist_norm = data_norm[t-seq_len:t]             # (L,N)
#     x_in = X_hist_norm.unsqueeze(0).unsqueeze(-1).to(device)  # (1,L,N,1)
#     y_pred_norm = model(x_in, A_hat).cpu().squeeze(0)         # (H,N)
#     return y_pred_norm  # normalized space

# def eval_csv_full(model, A_hat, df, edge_order, mean, scale, seq_len, horizons, device):
#     if "time" in df.columns: df = df.drop(columns=["time"])
#     df = df.reindex(columns=edge_order, fill_value=0.0)
#     data = torch.tensor(df.values, dtype=torch.float32)  # (T,N)
#     T, N = data.shape
#     H = max(horizons)

#     data_norm = (data - mean) / scale

#     # 전체 슬라이딩 평가
#     ts = list(range(seq_len, T - H))
#     if len(ts) == 0:
#         raise RuntimeError(f"CSV too short: T={T}, need >= {seq_len+H+1}")

#     preds = []
#     trues = []
#     for t in ts:
#         y_true_norm = torch.stack([data_norm[t+h] for h in horizons], 0)  # (H,N)
#         y_pred_norm = predict_window(model, A_hat, data_norm, t, seq_len, horizons, device)
#         preds.append(y_pred_norm)
#         trues.append(y_true_norm)

#     YP = torch.stack(preds, 0)  # (T',H,N)
#     YT = torch.stack(trues, 0)  # (T',H,N)

#     # 역정규화
#     YP_den = YP * scale + mean
#     YT_den = YT * scale + mean

#     # 지표
#     mae_global = torch.mean(torch.abs(YP_den - YT_den)).item()
#     rmse_global = torch.sqrt(torch.mean((YP_den - YT_den) ** 2)).item()

#     # 엣지별 MAE (호라이즌별 평균)
#     mae_edge = torch.mean(torch.abs(YP_den - YT_den), dim=(0,1)).numpy()  # (N,)

#     return {
#         "YP_den": YP_den.numpy(),
#         "YT_den": YT_den.numpy(),
#         "mae_global": mae_global,
#         "rmse_global": rmse_global,
#         "mae_edge": mae_edge,
#         "ts": ts,
#     }

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--ckpt", required=True)
#     ap.add_argument("--layout", default="/workspace/fab_oht_layout_updated.json")
#     ap.add_argument("--data_dirs", nargs="+",
#                     default=["/workspace/datasets_pivot","/workspace/datasets_dynamic"])
#     ap.add_argument("--csv", default="", help="특정 CSV 지정(옵션)")
#     ap.add_argument("--seq_len", type=int, default=-1, help="없으면 ckpt 값 사용")
#     ap.add_argument("--t_index", type=int, default=-1, help="피벗 시점(없으면 랜덤)")
#     ap.add_argument("--num_edges", type=int, default=4)
#     ap.add_argument("--outdir", default="/workspace/figs")
#     ap.add_argument("--mode", choices=["grid","worst","all"], default="grid")
#     ap.add_argument("--k", type=int, default=12, help="플롯에 쓸 엣지 개수")
#     ap.add_argument("--h_sel", type=int, default=-1, help="특정 호라이즌만(예: 60). -1이면 전체")
#     args = ap.parse_args()

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Device: {device}")

#     # 1) 체크포인트 로드
#     ckpt = torch.load(args.ckpt, map_location="cpu")
#     edge_order = ckpt["edge_order"]
#     horizons   = ckpt["horizons"]        # e.g. [6,30,60]
#     seq_len    = args.seq_len if args.seq_len > 0 else ckpt["seq_len"]
#     mean = torch.tensor(ckpt["scaler_mean"]).float()
#     scale= torch.tensor(ckpt["scaler_scale"]).float().clamp(min=1e-12)

#     # 2) A_hat 만들고 ckpt edge_order 순서로 재정렬
#     A_hat_layout, edges_layout = build_edge_adjacency(args.layout)
#     idx_map = {e: i for i, e in enumerate(edges_layout)}


#     missing = [e for e in edge_order if e not in idx_map]
#     extra   = [e for e in idx_map if e not in edge_order]

#     print(f"❌ layout에 없는 edge_order: {len(missing)}")
#     if missing:
#         print(missing[:10])

#     print(f"⚠️ edge_order에 없는 layout edge: {len(extra)}")
#     if extra:
#         print(extra[:10])

#     idxs = [idx_map[e] for e in edge_order] 
#     A_hat = A_hat_layout[idxs][:, idxs].to(device)


#     # 3) 모델 구성/로드
#     hidden, blocks = infer_hparams_from_state(ckpt["model_state"])
#     model = STGNN(num_nodes=len(edge_order),
#                   in_channels=1, hidden=hidden,
#                   horizons=len(horizons), blocks=blocks, k=3).to(device)
#     model.load_state_dict(ckpt["model_state"])
#     model.eval()
#     # === 4) CSV 선택 ===
#     csv_path = args.csv or pick_csv(args.data_dirs)
#     print(f"Using CSV: {csv_path}")
#     df = pd.read_csv(csv_path)

#     # === 5) 전체 슬라이딩 평가 & 메트릭 ===
#     res = eval_csv_full(model, A_hat, df, edge_order, mean, scale, seq_len, horizons, device)
#     YP = res["YP_den"]; YT = res["YT_den"]; mae_edge = res["mae_edge"]
#     print(f"🔎 Global  MAE={res['mae_global']:.4f}  RMSE={res['rmse_global']:.4f}")

#     # === 6) 플롯 아웃폴더 ===
#     os.makedirs(args.outdir, exist_ok=True)

#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt

#     def plot_grid_random():
#         N = YP.shape[-1]
#         sel = random.sample(range(N), k=min(args.k, N))
#         cols = min(3, len(sel))
#         rows = int(math.ceil(len(sel)/cols))
#         fig, axes = plt.subplots(rows, cols, figsize=(5.5*cols, 3.8*rows), squeeze=False)
#         t0 = res["ts"][len(res["ts"])//2]  # 중간 시점 하나 픽
#         hist = ( ( (torch.tensor(df.reindex(columns=edge_order).values, dtype=torch.float32)[t0-seq_len:t0] ).numpy() ) )
#         for ax, eidx in zip(axes.ravel(), sel):
#             ax.plot(range(-seq_len,0), hist[:, eidx], label="history")
#             ax.plot(horizons, YT[len(res["ts"])//2, :, eidx], "o", label="true@H")
#             ax.plot(horizons, YP[len(res["ts"])//2, :, eidx], "x", label="pred@H")
#             ax.set_title(f"{edge_order[eidx]} (edge {eidx})"); ax.grid(True); ax.legend()
#             ax.set_xlabel("steps (10s/step)")
#         for k in range(len(sel), rows*cols):
#             fig.delaxes(axes.ravel()[k])
#         plt.tight_layout()
#         out = os.path.join(args.outdir, "grid_random.png")
#         plt.savefig(out, dpi=140); print(f"✅ {out}")

#     def plot_worst_k():
#         N = YP.shape[-1]
#         order = np.argsort(-mae_edge)[:min(args.k, N)]  # 큰 MAE 상위 K
#         cols = min(3, len(order))
#         rows = int(math.ceil(len(order)/cols))
#         fig, axes = plt.subplots(rows, cols, figsize=(5.5*cols, 3.8*rows), squeeze=False)
#         mid = len(res["ts"])//2
#         hist = ( ( (torch.tensor(df.reindex(columns=edge_order).values, dtype=torch.float32)[res["ts"][mid]-seq_len:res["ts"][mid]] ).numpy() ) )
#         for ax, eidx in zip(axes.ravel(), order):
#             ax.plot(range(-seq_len,0), hist[:, eidx], label="history")
#             ax.plot(horizons, YT[mid, :, eidx], "o", label="true@H")
#             ax.plot(horizons, YP[mid, :, eidx], "x", label="pred@H")
#             ax.set_title(f"worst #{np.where(order==eidx)[0][0]+1} | {edge_order[eidx]}")
#             ax.grid(True); ax.legend(); ax.set_xlabel("steps (10s/step)")
#         for k in range(len(order), rows*cols):
#             fig.delaxes(axes.ravel()[k])
#         plt.tight_layout()
#         out = os.path.join(args.outdir, "grid_worstK.png")
#         plt.savefig(out, dpi=140); print(f"✅ {out}")

#     def plot_hist_and_scatter():
#         Haxis = list(horizons)
#         if args.h_sel > 0 and args.h_sel in Haxis:
#             h_idx = Haxis.index(args.h_sel)
#             err = (YP[:, h_idx, :] - YT[:, h_idx, :]).reshape(-1)
#             y_true = YT[:, h_idx, :].reshape(-1)
#             y_pred = YP[:, h_idx, :].reshape(-1)
#             tag = f"h{args.h_sel}"
#         else:
#             err = (YP - YT).reshape(-1)
#             y_true = YT.reshape(-1)
#             y_pred = YP.reshape(-1)
#             tag = "allH"

#         plt.figure(figsize=(6,4))
#         plt.hist(err, bins=60, alpha=0.9)
#         plt.title(f"Error histogram ({tag})"); plt.tight_layout()
#         out = os.path.join(args.outdir, f"hist_{tag}.png")
#         plt.savefig(out, dpi=140); print(f"✅ {out}")

#         plt.figure(figsize=(5,5))
#         plt.scatter(y_true, y_pred, s=4, alpha=0.3)
#         lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
#         plt.plot(lims, lims, 'k--', linewidth=1)
#         plt.title(f"True vs Pred ({tag})"); plt.tight_layout()
#         out = os.path.join(args.outdir, f"scatter_{tag}.png")
#         plt.savefig(out, dpi=140); print(f"✅ {out}")

#     if args.mode in ("grid","all"):
#         plot_grid_random()
#     if args.mode in ("worst","all"):
#         plot_worst_k()
#     plot_hist_and_scatter()


# if __name__ == "__main__":
#     main()

# viz.py
import os, glob, json, argparse, math, random
import numpy as np
import pandas as pd
import torch

# 학습 때 만든 함수/모델 재사용
from model import STGNN, build_edge_adjacency

def pick_csv(data_dirs):
    if isinstance(data_dirs, str): data_dirs=[data_dirs]
    files=[]
    for d in data_dirs:
        files += glob.glob(os.path.join(d, "*.csv"))
    if not files:
        raise FileNotFoundError("No CSV files in given dirs.")
    return random.choice(sorted(files))

def inverse_transform(arr, mean, scale):
    return arr * scale + mean

def infer_hparams_from_state(state_dict):
    hidden = state_dict["head.weight"].shape[1]
    block_ids = []
    for k in state_dict.keys():
        if k.startswith("blocks."):
            try: block_ids.append(int(k.split(".")[1]))
            except: pass
    blocks = (max(block_ids) + 1) if block_ids else 2
    return hidden, blocks

@torch.no_grad()
def predict_window(model, A_hat, data_norm, t, seq_len, horizons, device, predict_delta=False):
    X_hist_norm = data_norm[t-seq_len:t]                        # (L,N)
    x_in = X_hist_norm.unsqueeze(0).unsqueeze(-1).to(device)    # (1,L,N,1)
    y_pred_norm = model(x_in, A_hat).cpu().squeeze(0)           # (H,N)
    if predict_delta:
        last_x_norm = data_norm[t-1]                            # (N,)
        y_pred_norm = y_pred_norm + last_x_norm.unsqueeze(0)    # Δ → 절대값
    return y_pred_norm

def eval_csv_full(model, A_hat, df, edge_order, mean, scale, seq_len, horizons, device, predict_delta=False):
    if "time" in df.columns: df = df.drop(columns=["time"])
    df = df.reindex(columns=edge_order, fill_value=0.0)
    data = torch.tensor(df.values, dtype=torch.float32)  # (T,N)
    T, N = data.shape
    H = max(horizons)

    data_norm = (data - mean) / scale

    ts = list(range(seq_len, T - H))
    if len(ts) == 0:
        raise RuntimeError(f"CSV too short: T={T}, need >= {seq_len+H+1}")

    preds, trues = [], []
    for t in ts:
        y_true_norm = torch.stack([data_norm[t+h] for h in horizons], 0)  # (H,N)
        y_pred_norm = predict_window(model, A_hat, data_norm, t, seq_len, horizons, device, predict_delta)
        preds.append(y_pred_norm)
        trues.append(y_true_norm)

    YP = torch.stack(preds, 0)  # (T',H,N)
    YT = torch.stack(trues, 0)  # (T',H,N)

    # 역정규화
    YP_den = YP * scale + mean
    YT_den = YT * scale + mean

    mae_global = torch.mean(torch.abs(YP_den - YT_den)).item()
    rmse_global = torch.sqrt(torch.mean((YP_den - YT_den) ** 2)).item()
    mae_edge = torch.mean(torch.abs(YP_den - YT_den), dim=(0,1)).numpy()  # (N,)

    return {
        "YP_den": YP_den.numpy(),
        "YT_den": YT_den.numpy(),
        "mae_global": mae_global,
        "rmse_global": rmse_global,
        "mae_edge": mae_edge,
        "ts": ts,
        "horizons": horizons
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layout", default="/workspace/fab_oht_layout_updated.json")
    ap.add_argument("--data_dirs", nargs="+",
                    default=["/workspace/datasets_pivot","/workspace/datasets_dynamic"])
    ap.add_argument("--csv", default="", help="특정 CSV 지정(옵션)")
    ap.add_argument("--seq_len", type=int, default=-1)
    ap.add_argument("--num_edges", type=int, default=4)
    ap.add_argument("--outdir", default="/workspace/figs")
    ap.add_argument("--mode", choices=["grid","worst","all"], default="grid")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--h_sel", type=int, default=-1, help="특정 호라이즌만(예: 60). -1이면 전체")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1) 체크포인트 로드
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model_state"]

    # === 키 이름 매핑 (t1→temp1, s.→spat., t2→temp2) ===
    new_state = {}
    for k, v in state.items():
        k_new = k
        k_new = k_new.replace("t1", "temp1")
        k_new = k_new.replace("s.", "spat.")
        k_new = k_new.replace("t2", "temp2")
        new_state[k_new] = v

    edge_order = ckpt["edge_order"]
    horizons   = ckpt["horizons"]        # e.g. [6,30,60]
    seq_len    = args.seq_len if args.seq_len > 0 else ckpt["seq_len"]
    mean = torch.tensor(ckpt["scaler_mean"]).float()
    scale= torch.tensor(ckpt["scaler_scale"]).float().clamp(min=1e-12)

    # 2) A_hat 만들고 ckpt edge_order 순서로 재정렬
    A_hat_layout, edges_layout = build_edge_adjacency(args.layout)
    idx_map = {e: i for i, e in enumerate(edges_layout)}

    idxs = [idx_map[e] for e in edge_order] 
    A_hat = A_hat_layout[idxs][:, idxs].to(device)

    # 3) 모델 구성/로드
    hidden, blocks = infer_hparams_from_state(new_state)
    model = STGNN(num_nodes=len(edge_order),
                  in_channels=1, hidden=hidden,
                  horizons=len(horizons), blocks=blocks, k=3).to(device)
    model.load_state_dict(new_state, strict=False)
    model.eval()


    # === CSV ===
    csv_path = args.csv or pick_csv(args.data_dirs)
    print(f"Using CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # === 평가 ===
    res = eval_csv_full(model, A_hat, df, edge_order, mean, scale, seq_len, horizons, device, predict_delta=True)
    print(f"🔎 Global  MAE={res['mae_global']:.4f}  RMSE={res['rmse_global']:.4f}")
    
    YP = res["YP_den"]
    YT = res["YT_den"]
    mae_edge = res["mae_edge"]
    horizons = res["horizons"]   # 여기 덮어쓰기

    # === 플롯 ===
    os.makedirs(args.outdir, exist_ok=True)
    
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def plot_grid_random():
        N = YP.shape[-1]
        sel = random.sample(range(N), k=min(args.k, N))
        cols = min(3, len(sel))
        rows = int(math.ceil(len(sel)/cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5.5*cols, 3.8*rows), squeeze=False)
        t0 = res["ts"][len(res["ts"])//2]  # 중간 시점 하나 픽
        hist = ( ( (torch.tensor(df.reindex(columns=edge_order).values, dtype=torch.float32)[t0-seq_len:t0] ).numpy() ) )
        for ax, eidx in zip(axes.ravel(), sel):
            ax.plot(range(-seq_len,0), hist[:, eidx], label="history")
            ax.plot(horizons, YT[len(res["ts"])//2, :, eidx], "o", label="true@H")
            ax.plot(horizons, YP[len(res["ts"])//2, :, eidx], "x", label="pred@H")
            ax.set_title(f"{edge_order[eidx]} (edge {eidx})"); ax.grid(True); ax.legend()
            ax.set_xlabel("steps (10s/step)")
        for k in range(len(sel), rows*cols):
            fig.delaxes(axes.ravel()[k])
        plt.tight_layout()
        out = os.path.join(args.outdir, "grid_random.png")
        plt.savefig(out, dpi=140); print(f"✅ {out}")

    def plot_worst_k():
        N = YP.shape[-1]
        order = np.argsort(-mae_edge)[:min(args.k, N)]  # 큰 MAE 상위 K
        cols = min(3, len(order))
        rows = int(math.ceil(len(order)/cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5.5*cols, 3.8*rows), squeeze=False)
        mid = len(res["ts"])//2
        hist = ( ( (torch.tensor(df.reindex(columns=edge_order).values, dtype=torch.float32)[res["ts"][mid]-seq_len:res["ts"][mid]] ).numpy() ) )
        for ax, eidx in zip(axes.ravel(), order):
            ax.plot(range(-seq_len,0), hist[:, eidx], label="history")
            ax.plot(horizons, YT[mid, :, eidx], "o", label="true@H")
            ax.plot(horizons, YP[mid, :, eidx], "x", label="pred@H")
            ax.set_title(f"worst #{np.where(order==eidx)[0][0]+1} | {edge_order[eidx]}")
            ax.grid(True); ax.legend(); ax.set_xlabel("steps (10s/step)")
        for k in range(len(order), rows*cols):
            fig.delaxes(axes.ravel()[k])
        plt.tight_layout()
        out = os.path.join(args.outdir, "grid_worstK.png")
        plt.savefig(out, dpi=140); print(f"✅ {out}")

    def plot_hist_and_scatter():
        Haxis = list(horizons)
        if args.h_sel > 0 and args.h_sel in Haxis:
            h_idx = Haxis.index(args.h_sel)
            err = (YP[:, h_idx, :] - YT[:, h_idx, :]).reshape(-1)
            y_true = YT[:, h_idx, :].reshape(-1)
            y_pred = YP[:, h_idx, :].reshape(-1)
            tag = f"h{args.h_sel}"
        else:
            err = (YP - YT).reshape(-1)
            y_true = YT.reshape(-1)
            y_pred = YP.reshape(-1)
            tag = "allH"

        plt.figure(figsize=(6,4))
        plt.hist(err, bins=60, alpha=0.9)
        plt.title(f"Error histogram ({tag})"); plt.tight_layout()
        out = os.path.join(args.outdir, f"hist_{tag}.png")
        plt.savefig(out, dpi=140); print(f"✅ {out}")

        plt.figure(figsize=(5,5))
        plt.scatter(y_true, y_pred, s=4, alpha=0.3)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        plt.plot(lims, lims, 'k--', linewidth=1)
        plt.title(f"True vs Pred ({tag})"); plt.tight_layout()
        out = os.path.join(args.outdir, f"scatter_{tag}.png")
        plt.savefig(out, dpi=140); print(f"✅ {out}")

    if args.mode in ("grid","all"):
        plot_grid_random()
    if args.mode in ("worst","all"):
        plot_worst_k()
    plot_hist_and_scatter()
    # (여기서 grid, worst 플롯 함수 넣으면 됨 — 지금 구조 그대로 유지 가능)
    # ...

if __name__ == "__main__":
    main()

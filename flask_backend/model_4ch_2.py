import os
import json
import math
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Union

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, random_split

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def edges_from_layout(layout_json_path: str) -> List[str]:
    with open(layout_json_path, "r", encoding="utf-8") as f:
        layout = json.load(f)
    return [f"{r['from']}-{r['to']}" for r in layout["rails"]]

def build_edge_adjacency(layout_json_path: str) -> Tuple[torch.Tensor, List[str]]:
    with open(layout_json_path, "r", encoding="utf-8") as f:
        layout = json.load(f)
    edges = [f"{r['from']}-{r['to']}" for r in layout["rails"]]
    N = len(edges)
    A = np.zeros((N, N), dtype=np.float32)
    from_to = [(r['from'], r['to']) for r in layout['rails']]
    for i, (u1, v1) in enumerate(from_to):
        for j, (u2, v2) in enumerate(from_to):
            if v1 == u2:
                A[i, j] = 1.0
    A = A + np.eye(N, dtype=np.float32)
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.power(D, -0.5, where=D > 0)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
    A_hat = (D_inv_sqrt[:, None] * A) * D_inv_sqrt[None, :]
    return torch.tensor(A_hat, dtype=torch.float32), edges

def load_vmax_per_edge(layout_json_path: str, edges_order: List[str]) -> np.ndarray:
    with open(layout_json_path, "r", encoding="utf-8") as f:
        layout = json.load(f)
    vmax_map = {}
    for r in layout["rails"]:
        rid = f"{r['from']}-{r['to']}"
        vmax_map[rid] = 1000.0 if r.get("curve", 0) == 1 else 5000.0
    vmax = np.array([vmax_map.get(e, 5000.0) for e in edges_order], dtype=np.float32)
    return np.maximum(vmax, 1.0)

def load_length_per_edge(layout_json_path: str, edges_order: List[str]) -> np.ndarray:
    with open(layout_json_path, "r", encoding="utf-8") as f:
        layout = json.load(f)
    id2xy = {n["id"]: (float(n["x"]), float(n["y"])) for n in layout["nodes"]}
    L = []
    for e in edges_order:
        u, v = e.split("-")
        L.append(math.dist(id2xy[u], id2xy[v]))
    return np.maximum(np.array(L, dtype=np.float32), 1e-6)

class Standardizer:
    def __init__(self, mode: str = "per_edge"):
        assert mode in {"global", "per_edge"}
        self.mode = mode
        self._sum = None; self._sumsq = None
        self._count = 0; self._count_T = 0
        self.mean_ = None; self.std_ = None

    def update(self, X_used: np.ndarray):
        T, E, C = X_used.shape
        if self.mode == "global":
            flat = X_used.reshape(-1, C)
            s = flat.sum(axis=0); ss = (flat**2).sum(axis=0)
            if self._sum is None: self._sum, self._sumsq = s, ss
            else: self._sum += s; self._sumsq += ss
            self._count += flat.shape[0]
        else:
            s = X_used.sum(axis=0); ss = (X_used**2).sum(axis=0)
            if self._sum is None: self._sum, self._sumsq = s, ss
            else: self._sum += s; self._sumsq += ss
            self._count_T += T

    def finalize(self, num_edges: int):
        if self.mode == "global":
            mean = self._sum / max(1, self._count)
            var  = self._sumsq / max(1, self._count) - mean**2
            std  = np.sqrt(np.clip(var, 1e-8, None))
            self.mean_ = mean.astype(np.float32)     
            self.std_  = std.astype(np.float32)
        else:
            mean = self._sum / max(1, self._count_T) 
            var  = self._sumsq / max(1, self._count_T) - mean**2
            std  = np.sqrt(np.clip(var, 1e-8, None))
            
            Ecur = mean.shape[0]
            if Ecur < num_edges:
                pad_m = np.zeros((num_edges - Ecur, 2), dtype=np.float32)
                pad_s = np.ones((num_edges - Ecur, 2), dtype=np.float32)
                mean = np.vstack([mean, pad_m]); std = np.vstack([std, pad_s])
            self.mean_ = mean.astype(np.float32)     
            self.std_  = std.astype(np.float32)

    def transform(self, X_used: np.ndarray) -> np.ndarray:
        if self.mode == "global":
            return (X_used - self.mean_) / self.std_
        else:
            return (X_used - self.mean_[None, ...]) / self.std_[None, ...]

# -------- CSV -> (T,E,4) --------
def _extract_dyn_from_csv(df: pd.DataFrame, edges_order: List[str]) -> np.ndarray:
    if "time" in df.columns:
        df = df.drop(columns=["time"])
    avg_cols, cnt_cols = [], []
    for e in edges_order:
        c_avg = f"{e}_avg_speed"
        if c_avg not in df.columns:
            alt = f"{e}_avg"
            c_avg = alt if alt in df.columns else None
        c_cnt = f"{e}_count" if f"{e}_count" in df.columns else None
        avg_cols.append(c_avg); cnt_cols.append(c_cnt)
    df_avg = pd.DataFrame({e:(df[c] if c in df else 0.0) for e,c in zip(edges_order, avg_cols)})
    df_cnt = pd.DataFrame({e:(df[c] if c in df else 0.0) for e,c in zip(edges_order, cnt_cols)})
    X0 = df_avg.values.astype(np.float32)
    X1 = np.sqrt(df_cnt.values.astype(np.float32))
    return np.stack([X0, X1], axis=-1)

class CSVWindowDataset(Dataset):
    def __init__(self,
                 dataset_dirs: Union[str, List[str]],
                 edges_order: List[str],
                 vmax_raw: np.ndarray,
                 length_raw: np.ndarray,
                 seq_len: int,
                 horizons: Tuple[int, ...],
                 scaler: Standardizer,
                 scaler_fit: bool,
                 normalize_by_vmax: bool = True,
                 max_csv: int = 0,
                 max_windows: int = 0):
        if isinstance(dataset_dirs, str):
            dataset_dirs = [dataset_dirs]
        self.edges = list(edges_order)
        self.seq_len = seq_len
        self.horizons = list(horizons)
        self.scaler = scaler
        self.scaler_fit = scaler_fit
        self.normalize_by_vmax = normalize_by_vmax

        self.vmax_raw = vmax_raw.astype(np.float32)
        self.length_raw = length_raw.astype(np.float32)
        self.vmax_norm = (self.vmax_raw / float(self.vmax_raw.max())).astype(np.float32)
        self.length_norm = (self.length_raw / float(self.length_raw.max())).astype(np.float32)

        csvs = []
        for d in dataset_dirs:
            p = Path(d)
            csvs += [str(f) for f in p.rglob("*.csv")]
        csvs = sorted(set(csvs))
        if max_csv > 0:
            csvs = csvs[:max_csv]
        self.csvs = csvs

        self.samples: List[Tuple[int, int]] = []  
        self.runs = []                            

        for path in self.csvs:
            try:
                df = pd.read_csv(path)
                X_dyn_raw = _extract_dyn_from_csv(df, self.edges)   
                X_used = X_dyn_raw.copy()
                if self.normalize_by_vmax:
                    vmax_b = self.vmax_raw.reshape(1, -1, 1)
                    X_used[..., 0:1] = X_used[..., 0:1] / vmax_b

                T = X_dyn_raw.shape[0]
                if scaler_fit:
                    self.scaler.update(X_used)
                else:
                    X_dyn_z = self.scaler.transform(X_used)
                    static_E2 = np.stack([self.length_norm, self.vmax_norm], axis=-1)
                    static_TE2 = np.broadcast_to(static_E2[None, ...], (T, static_E2.shape[0], 2))
                    Xz = np.concatenate([X_dyn_z, static_TE2.astype(np.float32)], axis=-1).astype(np.float32)

                    max_h = max(self.horizons)
                    if T < self.seq_len + max_h:
                        continue
                    self.runs.append({"Xz": torch.from_numpy(Xz).float()})
                    rid = len(self.runs) - 1
                    for t in range(self.seq_len, T - max_h):
                        self.samples.append((rid, t))
            except Exception as e:
                print(f"❌ Read error {path}: {e}")

        if scaler_fit:
            self.scaler.finalize(num_edges=len(self.edges))
            self.samples = []; self.runs = []

        if (not scaler_fit) and max_windows > 0 and len(self.samples) > max_windows:
            self.samples = random.sample(self.samples, max_windows)

        print(f"✅ Dataset: csvs={len(self.csvs)} | runs={len(self.runs)} | windows={len(self.samples)} | L={self.seq_len}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        rid, t = self.samples[idx]
        Xz = self.runs[rid]["Xz"]            # (T,E,4)
        L = self.seq_len
        x = Xz[t-L:t]                         # (L,E,4)
        y = torch.stack([Xz[t+h, :, 0] for h in self.horizons], dim=0)  # (H,E)  (z-space)
        return x, y

# -------- Model --------
class DilatedTemporalConv(nn.Module):
    def __init__(self, in_c, out_c, k=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (k - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=k, padding=pad, dilation=dilation)
        self.act = nn.ReLU(); self.drop = nn.Dropout(dropout)
    def forward(self, x):  # x:(B,T,N,C)
        B,T,N,C = x.shape
        h = x.permute(0,2,3,1).contiguous().view(B*N, C, T)    # (B*N,C,T)
        h = self.drop(self.act(self.conv(h)))
        C2 = h.size(1)
        h = h.view(B, N, C2, T).permute(0,3,1,2).contiguous()  # (B,T,N,C2)
        return h

class DiffusionGCN(nn.Module):
    def __init__(self, in_c, out_c, K=2, dropout=0.1):
        super().__init__()
        self.theta = nn.Linear(in_c*K, out_c, bias=True)
        self.act = nn.ReLU(); self.drop = nn.Dropout(dropout)
        self.K = K
    def forward(self, x, A_hat_powers: List[torch.Tensor]):  # x:(B,T,N,C)
        B,T,N,C = x.shape
        hops = []
        h_last = x 
        for A in A_hat_powers[:self.K]:
            hops.append(torch.einsum("ij,btjc->btic", A, h_last))
        H = torch.cat(hops, dim=-1)                     # (B,T,N,C*K)
        H = self.drop(self.act(self.theta(H)))          # (B,T,N,out_c)
        return H

def build_diffusion_powers(A_hat: torch.Tensor, K: int) -> List[torch.Tensor]:
    outs = []
    cur = A_hat
    for _ in range(K):
        outs.append(cur)
        cur = cur @ A_hat
    return outs

class STBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, dilations=(1,2), K_spatial=2, dropout=0.1):
        super().__init__()
        self.temp1 = DilatedTemporalConv(in_c, out_c, k, dilations[0], dropout)
        self.spat  = DiffusionGCN(out_c, out_c, K=K_spatial, dropout=dropout)
        self.temp2 = DilatedTemporalConv(out_c, out_c, k, dilations[1], dropout)
        self.res = (nn.Identity() if in_c==out_c else nn.Linear(in_c, out_c, bias=False))
        self.norm = nn.LayerNorm(out_c)
    def forward(self, x, A_powers):
        h = self.temp1(x)
        h = self.spat(h, A_powers)
        h = self.temp2(h)
        h = h + self.res(x)
        B,T,N,C = h.shape
        h = self.norm(h.view(B*T*N, C)).view(B,T,N,C)
        return h

class STGNN(nn.Module):
    def __init__(self, num_nodes, in_channels=4, hidden=96, horizons=3,
                 blocks=3, k=3, dilations=(1,2,4,8), K_spatial=2, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        c_in = in_channels
        for b in range(blocks):
            d_pair = (dilations[(2*b)%len(dilations)], dilations[(2*b+1)%len(dilations)])
            self.blocks.append(STBlock(c_in, hidden, k, d_pair, K_spatial, dropout))
            c_in = hidden
        self.head = nn.Linear(hidden, horizons)
    def forward(self, x, A_powers):
        h = x
        for blk in self.blocks:
            h = blk(h, A_powers)
        h_last = h[:, -1]            
        out = self.head(h_last)      
        return out.permute(0,2,1)    

def mae_rmse(a: torch.Tensor, b: torch.Tensor):
    mae = torch.mean(torch.abs(a - b)).item()
    rmse = torch.sqrt(torch.mean((a - b) ** 2)).item()
    return mae, rmse

def train_model(
    dataset_dirs: List[str],
    layout_path: str,
    epochs: int = 20,
    batch_size: int = 16,
    seq_len: int = 18,
    horizons: Tuple[int, ...] = (1,3,6),
    hidden: int = 96,
    blocks: int = 3,
    k_t: int = 3,
    dilations: Tuple[int, ...] = (1,2,4,8),
    K_spatial: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    seed: int = 42,
    scaler_mode: str = "per_edge",
    normalize_by_vmax: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    amp: bool = True,
    alpha_event: float = 2.0,
    max_csv: int = 0,
    max_windows: int = 0,
):
    print("🚀 Starting training...")
    set_seed(seed); cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Device: {device}")

    A_hat_all, edges_layout = build_edge_adjacency(layout_path)
    edges_order = edges_from_layout(layout_path)   

    idx_map = {e:i for i,e in enumerate(edges_layout)}
    idxs = [idx_map[e] for e in edges_order]
    A_hat = A_hat_all[idxs][:, idxs].to(device)
    A_powers = build_diffusion_powers(A_hat, K=K_spatial)
    N = len(edges_order)
    vmax_raw = load_vmax_per_edge(layout_path, edges_order)
    length_raw = load_length_per_edge(layout_path, edges_order)
    print(f"🔗 Graph nodes(E)={N} | vmax[min={vmax_raw.min():.0f}, max={vmax_raw.max():.0f}]")

    scaler = Standardizer(mode=scaler_mode)
    _ = CSVWindowDataset(dataset_dirs, edges_order, vmax_raw, length_raw,
                         seq_len, horizons, scaler, scaler_fit=True,
                         normalize_by_vmax=normalize_by_vmax, max_csv=max_csv)

    full_ds = CSVWindowDataset(dataset_dirs, edges_order, vmax_raw, length_raw,
                               seq_len, horizons, scaler, scaler_fit=False,
                               normalize_by_vmax=normalize_by_vmax,
                               max_csv=max_csv, max_windows=max_windows)

    if len(full_ds) == 0:
        print("❌ No samples.")
        return None

    train_size = int(len(full_ds)*0.8)
    val_size   = len(full_ds)-train_size
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [train_size,val_size], generator=g)

    loader_args = dict(batch_size=batch_size, drop_last=True, pin_memory=pin_memory, num_workers=num_workers)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_args)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_args)
    print(f"🧰 windows: train={len(train_ds)} val={len(val_ds)} | batch={batch_size}")

    # Model
    model = STGNN(N, in_channels=4, hidden=hidden, horizons=len(horizons),
                  blocks=blocks, k=k_t, dilations=dilations,
                  K_spatial=K_spatial, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler_t = torch.cuda.amp.GradScaler(enabled=amp)
    mse = nn.MSELoss(reduction="none")

    # inverse z → physical helpers
    if scaler_mode == "per_edge":
        mean_dyn = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)  
        std_dyn  = torch.tensor(scaler.std_,  dtype=torch.float32, device=device).clamp(min=1e-12)
        mean_avg = mean_dyn[:,0].view(1,1,-1); std_avg = std_dyn[:,0].view(1,1,-1)
    else:
        mean_dyn = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)  
        std_dyn  = torch.tensor(scaler.std_,  dtype=torch.float32, device=device).clamp(min=1e-12)
        mean_avg = mean_dyn[0].view(1,1,1);   std_avg = std_dyn[0].view(1,1,1)
    vmax_t = torch.tensor(vmax_raw, dtype=torch.float32, device=device).view(1,1,-1)

    # ckpt
    best_rmse = float("inf")
    best_mae = float("inf")
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_sd  = ckpt_dir/f"stgnn_simple_{stamp}.pt"
    ckpt_full= ckpt_dir/f"stgnn_simple_{stamp}.full.pt"

    # Training loop
    for epoch in range(1, epochs+1):
        model.train(); total=0.0
        for X, y_z in train_loader:
            X = X.to(device); y_z = y_z.to(device)
            with torch.cuda.amp.autocast(enabled=amp):
                y_pred_z = model(X, A_powers)             
                
                y_pred_p = y_pred_z*std_avg + mean_avg
                y_true_p = y_z      *std_avg + mean_avg
                if normalize_by_vmax:
                    y_pred_p = y_pred_p * vmax_t
                    y_true_p = y_true_p * vmax_t

                x_last_z = X[:, -1, :, 0]; x_prev_z = X[:, -2, :, 0]   
                x_last_p = x_last_z*std_avg.squeeze(1)+mean_avg.squeeze(1)
                x_prev_p = x_prev_z*std_avg.squeeze(1)+mean_avg.squeeze(1)
                if normalize_by_vmax:
                    x_last_p *= vmax_t.squeeze(1)
                    x_prev_p *= vmax_t.squeeze(1)

                rel = torch.abs(x_last_p - x_prev_p) / (vmax_t.squeeze(1) + 1e-6)
                w_ev = 1.0 + alpha_event * rel.unsqueeze(1)  

                base_mae = torch.abs(y_pred_p - y_true_p)
                base_mse = (y_pred_p - y_true_p) ** 2
                loss = (0.9 * base_mae + 0.1 * base_mse) * w_ev
                loss = loss.mean()


            opt.zero_grad(set_to_none=True)
            scaler_t.scale(loss).backward()
            if grad_clip>0:
                scaler_t.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler_t.step(opt); scaler_t.update()
            total += loss.item()

        model.eval()
        mae_m, rmse_m, mae_last, rmse_last, mae_mov, rmse_mov = 0,0,0,0,0,0
        nb = 0
        with torch.no_grad():
            for X, y_z in val_loader:
                X = X.to(device); y_z = y_z.to(device)
                y_pred_z = model(X, A_powers)
                yp = y_pred_z*std_avg + mean_avg
                yt = y_z      *std_avg + mean_avg
                if normalize_by_vmax:
                    yp*=vmax_t; yt*=vmax_t
                yp = torch.clamp(yp, torch.tensor(0.0, device=yp.device), vmax_t)

                m_mae, m_rmse = mae_rmse(yp, yt)
                mae_m += m_mae; rmse_m += (m_rmse**2)

                last_z = X[:,-1,:,0].unsqueeze(1).repeat(1,y_z.size(1),1)
                bl = last_z*std_avg + mean_avg
                if normalize_by_vmax: bl*=vmax_t
                bl = torch.clamp(bl, torch.tensor(0.0, device=bl.device), vmax_t)
                b_mae, b_rmse = mae_rmse(bl, yt)
                mae_last += b_mae; rmse_last += (b_rmse**2)

                mov_z = X[:,:,:,0].mean(dim=1,keepdim=True).repeat(1,y_z.size(1),1)
                mv = mov_z*std_avg + mean_avg
                if normalize_by_vmax: mv*=vmax_t
                mv = torch.clamp(mv, torch.tensor(0.0, device=mv.device), vmax_t)
                m_mae2, m_rmse2 = mae_rmse(mv, yt)
                mae_mov += m_mae2; rmse_mov += (m_rmse2**2)
                nb += 1

        val_mae = mae_m/max(1,nb); val_rmse = math.sqrt(rmse_m/max(1,nb))
        last_mae = mae_last/max(1,nb); last_rmse = math.sqrt(rmse_last/max(1,nb))
        mov_mae  = mae_mov/max(1,nb);  mov_rmse  = math.sqrt(rmse_mov/max(1,nb))

        print(f"📊 Epoch {epoch:03d}/{epochs} | "
              f"TrainLoss={total/max(1,len(train_loader)):.4f} | "
              f"Model MAE/RMSE={val_mae:.4f}/{val_rmse:.4f} | "
              f"Last {last_mae:.4f}/{last_rmse:.4f} | MovAvg {mov_mae:.4f}/{mov_rmse:.4f}")



        if val_mae < best_mae:
            best_mae = val_mae
            payload = {
                "model_state": model.state_dict(),
                "edges_order": edges_order,
                "horizons": list(horizons),
                "seq_len": seq_len,
                "in_channels": 4,
                "hidden": hidden,
                "blocks": blocks,
                "k_t": k_t,
                "dilations": list(dilations),
                "K_spatial": K_spatial,
                "dropout": dropout,
                "scaler_mode": scaler_mode,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_std":  scaler.std_.tolist(),
                "normalize_by_vmax": normalize_by_vmax,
                "vmax": vmax_raw.tolist(),
                "length": length_raw.tolist(),
                "alpha_event": alpha_event,
            }
            torch.save(payload, ckpt_sd)
            torch.save(model, ckpt_full)
            print(f"✅ Saved state_dict: {ckpt_sd}")
            print(f"✅ Saved FULL model: {ckpt_full}")

    return str(ckpt_full)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train","viz"], default="train")

    ap.add_argument("--layout", type=str, default="fab_oht_layout_updated.json")

    ap.add_argument("--dataset_dirs", type=str, nargs="+", default=["./datasets/dynamic"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=18)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1,3,6])
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--k_t", type=int, default=3)
    ap.add_argument("--dilations", type=int, nargs="+", default=[1,2,4,8])
    ap.add_argument("--K_spatial", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scaler_mode", choices=["global","per_edge"], default="per_edge")
    ap.add_argument("--normalize_by_vmax", action="store_true")
    ap.add_argument("--no_normalize_by_vmax", dest="normalize_by_vmax", action="store_false")
    ap.set_defaults(normalize_by_vmax=True)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    ap.set_defaults(pin_memory=True)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.set_defaults(amp=True)
    ap.add_argument("--alpha_event", type=float, default=5.0)
    ap.add_argument("--max_csv", type=int, default=0)     
    ap.add_argument("--max_windows", type=int, default=0) 

    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--csv", type=str, default="")
    ap.add_argument("--outdir", type=str, default="./figs")

    args = ap.parse_args()

    if args.mode == "train":
        ckpt_full = train_model(
            dataset_dirs=args.dataset_dirs,
            layout_path=args.layout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            horizons=tuple(args.horizons),
            hidden=args.hidden,
            blocks=args.blocks,
            k_t=args.k_t,
            dilations=tuple(args.dilations),
            K_spatial=args.K_spatial,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            seed=args.seed,
            scaler_mode=args.scaler_mode,
            normalize_by_vmax=args.normalize_by_vmax,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            amp=args.amp,
            alpha_event=args.alpha_event,
            max_csv=args.max_csv,
            max_windows=args.max_windows,
        )
        print(f"🏁 Done. Full checkpoint: {ckpt_full}")

if __name__ == "__main__":
    main()

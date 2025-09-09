# model.py
import os
import glob
import json
import time
import math
import argparse
import random
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# 그래프(엣지) 인접행렬 생성 + 정규화
# -----------------------------
def build_edge_adjacency(layout_json_path):
    """
    레이아웃 JSON에서 엣지("u-v") 노드로 보는 Edge-graph를 만들고,
    A_hat = D^{-1/2} (A + I) D^{-1/2} 정규화 밀집 인접행렬을 반환.
    또한, edge_order 리스트(컬럼 정렬용)도 함께 반환.
    """
    with open(layout_json_path) as f:
        layout = json.load(f)

    # 모든 엣지 id
    edges = [f"{r['from']}-{r['to']}" for r in layout['rails']]
    N = len(edges)

    # edge i=(u1->v1)에서 j=(u2->v2)로: v1 == u2 이면 연결
    A = np.zeros((N, N), dtype=np.float32)
    from_to = [(r['from'], r['to']) for r in layout['rails']]
    for i, (u1, v1) in enumerate(from_to):
        for j, (u2, v2) in enumerate(from_to):
            if v1 == u2:
                A[i, j] = 1.0

    # self-loop
    A = A + np.eye(N, dtype=np.float32)

    # 정규화 A_hat
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.power(D, -0.5, where=D > 0)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
    A_hat = (D_inv_sqrt[:, None] * A) * D_inv_sqrt[None, :]

    A_hat = torch.tensor(A_hat, dtype=torch.float32)
    return A_hat, edges  # edges == edge_order


# -----------------------------
# Dataset (CSV → (X,y))
# -----------------------------
class MultiOHTDataset(Dataset):
    """
    여러 폴더(피벗/다이내믹) 내 CSV들을 합쳐서
    X: [seq_len, N_edges], y: [len(horizons), N_edges] 샘플을 생성.
    edge_order로 컬럼을 강제 정렬/누락엣지는 0 채움.
    scaler가 있으면 표준화 적용.
    """
    def __init__(
        self,
        folder_paths,
        edge_order,
        seq_len=12,
        horizons=(6, 30, 60),
        scaler=None,
        fit_scaler=False,
    ):
        if isinstance(folder_paths, str):
            folder_paths = [folder_paths]

        self.edge_order = list(edge_order)
        self.seq_len = seq_len
        self.horizons = list(horizons)

        csv_files = []
        for folder in folder_paths:
            files = glob.glob(os.path.join(folder, "*.csv"))
            print(f"📂 {folder}: {len(files)} CSV")
            csv_files.extend(files)
        if len(csv_files) == 0:
            print("⚠️ No CSV files found.")

        X_list, y_list = [], []

        for csv in csv_files:
            try:
                df = pd.read_csv(csv)
                if "time" in df.columns:
                    df = df.drop(columns=["time"])
                # edge_order로 정렬(없으면 0 채움)
                df = df.reindex(columns=self.edge_order, fill_value=0.0)

                data = df.values.astype(np.float32)  # [T, N_edges]
                if data.shape[0] < seq_len + max(self.horizons):
                    # 너무 짧으면 스킵
                    continue

                if scaler is not None:
                    if fit_scaler:
                        scaler.partial_fit(data)
                    else:
                        data = scaler.transform(data).astype(np.float32)

                T = data.shape[0]
                for t in range(seq_len, T - max(self.horizons)):
                    x = data[t - seq_len : t]  # [L, N]
                    y = np.stack([data[t + h] for h in self.horizons], axis=0)  # [H, N]
                    X_list.append(x)
                    y_list.append(y)
            except Exception as e:
                print(f"❌ Read error {csv}: {e}")

        if scaler is not None and fit_scaler:
            # 첫 pass(fit) 모드에서는 텐서 만들 필요 X
            self.X = np.empty(
                (0, seq_len, len(self.edge_order)), dtype=np.float32
            )
            self.y = np.empty(
                (0, len(self.horizons), len(self.edge_order)), dtype=np.float32
            )
        else:
            self.X = np.array(X_list, dtype=np.float32)
            self.y = np.array(y_list, dtype=np.float32)

        print(
            f"✅ Dataset built: {len(self)} samples | "
            f"seq_len={seq_len}, N={len(self.edge_order)}"
        )

    def __len__(self):
        return 0 if self.X is None else len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


# -----------------------------
# ST-Block (Temporal→Spatial→Temporal)
#   - Temporal: 1xK Conv (time 축)
#   - Spatial:  A_hat @ X (nodes 축, 밀집 matmul)
# -----------------------------
class TemporalConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad)

    def forward(self, x):
        """
        x: (B, T, N, C_in)
        conv는 시간축(T)에 대해 수행
        """
        B, T, N, C = x.shape
        # (B*N, C_in, T)로 바꿔서 1D-Conv
        x = x.permute(0, 2, 3, 1).contiguous().view(B * N, C, T)
        x = self.conv(x)  # (B*N, C_out, T)
        # 다시 (B, T, N, C_out)로 복원
        C_out = x.size(1)
        x = x.view(B, N, C_out, T).permute(0, 3, 1, 2).contiguous()
        return x


class SpatialGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 채널 축에만 선형변환 (N, T는 배치처럼 취급)
        self.theta = torch.nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, A_hat):
        """
        x: (B, T, N, C_in)
        A_hat: (N, N)
        return: (B, T, N, C_out)
        """
        assert x.dim() == 4, f"x dim must be 4D (B,T,N,C), got {x.shape}"
        # 그래프 라플라시안 적용: (N,N) x (B,T,N,C_in) -> (B,T,N,C_in)
        #  j축을 합치며 i축을 남긴다 -> 'ij,btjc->btic'
        x = torch.einsum("ij,btjc->btic", A_hat, x)
        # 채널 변환
        x = self.theta(x)  # (B, T, N, C_out)
        return x


class STBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k=3):
        super().__init__()
        self.temp1 = TemporalConv(in_c, out_c, kernel_size=k)
        self.spat = SpatialGCN(out_c, out_c)
        self.temp2 = TemporalConv(out_c, out_c, kernel_size=k)
        self.act = torch.nn.ReLU()

    def forward(self, x, A_hat):
        # x: (B, T, N, C)
        x = self.temp1(x)
        x = self.act(x)
        x = self.spat(x, A_hat)
        x = self.act(x)
        x = self.temp2(x)
        x = self.act(x)
        return x


class STGNN(nn.Module):
    """
    ST 블록 여러 개 쌓고, 마지막 시점의 hidden에서
    Linear(hidden -> horizons)로 다중호라이즌 출력.
    """
    def __init__(self, num_nodes, in_channels=1, hidden=64, horizons=3, blocks=2, k=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizons = horizons  # int (예: 3)

        self.blocks = nn.ModuleList()
        c_in = in_channels
        for _ in range(blocks):
            self.blocks.append(STBlock(c_in, hidden, k=k))
            c_in = hidden

        self.head = nn.Linear(hidden, horizons)

    def forward(self, x, A_hat):
        """
        x: (B, T, N, C_in)
        A_hat: (N, N)
        return: (B, H, N)
        """
        h = x
        for blk in self.blocks:
            h = blk(h, A_hat)  # (B, T, N, hidden)

        # 마지막 시점만 사용
        h_last = h[:, -1]  # (B, N, hidden)
        out = self.head(h_last)  # (B, N, H)
        return out.permute(0, 2, 1)  # (B, H, N)


# -----------------------------
# 평가 지표
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, A_hat, device):
    model.eval()
    mae_sum, mse_sum, n = 0.0, 0.0, 0
    l1 = nn.L1Loss(reduction="sum")
    l2 = nn.MSELoss(reduction="sum")

    for X, y in loader:
        # X: (B, L, N), y: (B, H, N)
        X = X.to(device).unsqueeze(-1)     # (B, L, N, 1)
        y = y.to(device)                   # (B, H, N)
        y_pred = model(X, A_hat)           # (B, H, N)
        mae_sum += l1(y_pred, y).item()
        mse_sum += l2(y_pred, y).item()
        n += y.numel()

    mae = mae_sum / n
    rmse = math.sqrt(mse_sum / n)
    return mae, rmse


# -----------------------------
# 학습 루프
# -----------------------------
def train_model(
    data_folders,
    layout_path,
    epochs=20,
    batch_size=32,
    seq_len=12,
    horizons=(6, 30, 60),
    hidden_channels=64,
    num_blocks=2,
    k_t=3,
    lr=1e-3,
    seed=42,
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Device: {device}")

    # 그래프 준비 (정렬용 edge_order 포함)
    A_hat, edge_order = build_edge_adjacency(layout_path)
    N = len(edge_order)
    A_hat = A_hat.to(device)
    print(f"🔗 Edge-graph built: N={N}")

    # 스케일러 2-pass (fit → transform)
    scaler = StandardScaler()
    _ = MultiOHTDataset(
        data_folders, edge_order, seq_len=seq_len, horizons=horizons,
        scaler=scaler, fit_scaler=True
    )
    dataset = MultiOHTDataset(
        data_folders, edge_order, seq_len=seq_len, horizons=horizons,
        scaler=scaler, fit_scaler=False
    )

    if len(dataset) == 0:
        print("❌ No samples. Check CSV length/columns.")
        return None

    # split
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 모델
    model = STGNN(
        num_nodes=N,
        in_channels=1,
        hidden=hidden_channels,
        horizons=len(horizons),
        blocks=num_blocks,
        k=k_t,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"🧠 Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    best_val = float("inf")
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(
        ckpt_dir,
        f"stgnn_dense_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for X, y in train_loader:
            # X: (B, L, N) → (B, L, N, 1)
            X = X.to(device).unsqueeze(-1)
            y = y.to(device)  # (B, H, N)

            y_pred = model(X, A_hat)  # (B, H, N)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

        val_mae, val_rmse = evaluate(model, val_loader, A_hat, device)
        print(
            f"📊 Epoch {epoch:03d}/{epochs} | "
            f"TrainLoss: {running/len(train_loader):.4f} | "
            f"ValMAE: {val_mae:.5f} | ValRMSE: {val_rmse:.5f}"
        )

        # 베스트 저장
        if val_rmse < best_val:
            best_val = val_rmse
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "edge_order": edge_order,
                    "horizons": list(horizons),
                    "seq_len": seq_len,
                },
                ckpt_path,
            )
            print(f"✅ Saved checkpoint: {ckpt_path}")

    return ckpt_path


# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    print("🔎 model.py launched")

    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", type=str, default="/workspace/fab_oht_layout_updated.json")
    parser.add_argument(
        "--data_dirs",
        type=str,
        nargs="+",
        default=["/workspace/datasets_pivot", "/workspace/datasets_dynamic"],
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=[6, 30, 60]
    )  # 10초 간격 기준: 1/5/10분
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--k_t", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpt = train_model(
        data_folders=args.data_dirs,
        layout_path=args.layout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        horizons=tuple(args.horizons),
        hidden_channels=args.hidden,
        num_blocks=args.blocks,
        k_t=args.k_t,
        lr=args.lr,
        seed=args.seed,
    )

    print(f"🏁 Done. Best checkpoint: {ckpt}")

# model.py
# -*- coding: utf-8 -*-

# import os
# import glob
# import json
# import math
# import argparse
# import random
# from datetime import datetime

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, random_split


# # -----------------------------
# # Utils
# # -----------------------------
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# def inverse_transform(x_norm: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
#     """x = x_norm * scale + mean  (broadcast-safe)"""
#     return x_norm * scale + mean


# # -----------------------------
# # 그래프(엣지) 인접행렬 생성 + 정규화
# # -----------------------------
# def build_edge_adjacency(layout_json_path):
#     """
#     레이아웃 JSON에서 엣지("u-v")를 노드로 보는 Edge-graph를 만들고,
#     A_hat = D^{-1/2} (A + I) D^{-1/2} (dense) 반환.
#     또한, edge_order 리스트(컬럼 정렬용)도 반환.
#     """
#     with open(layout_json_path) as f:
#         layout = json.load(f)

#     edges = [f"{r['from']}-{r['to']}" for r in layout['rails']]
#     N = len(edges)

#     A = np.zeros((N, N), dtype=np.float32)
#     rails = [(r['from'], r['to']) for r in layout['rails']]
#     for i, (u1, v1) in enumerate(rails):
#         for j, (u2, _) in enumerate(rails):
#             if v1 == u2:
#                 A[i, j] = 1.0

#     # self-loop
#     A = A + np.eye(N, dtype=np.float32)

#     # 정규화
#     D = np.sum(A, axis=1)
#     D_inv_sqrt = np.power(D, -0.5, where=D > 0)
#     D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
#     A_hat = (D_inv_sqrt[:, None] * A) * D_inv_sqrt[None, :]

#     return torch.tensor(A_hat, dtype=torch.float32), edges


# def make_multi_hop(A_hat: torch.Tensor, hop_order: int = 2, alpha: float = 0.6) -> torch.Tensor:
#     """
#     간단 멀티홉 혼합: A_eff = α*A + (1-α)*A^2  (hop_order=2)
#     hop_order>=3면 A^3까지 섞음 (비율은 균등 가중)
#     """
#     A_eff = A_hat
#     if hop_order >= 2:
#         A2 = A_hat @ A_hat
#         A_eff = alpha * A_hat + (1.0 - alpha) * A2
#     if hop_order >= 3:
#         A3 = A_hat @ A_hat @ A_hat
#         A_eff = (A_eff + A3) / 2.0
#     return A_eff


# # -----------------------------
# # Dataset (CSV → (X,y))
# # -----------------------------
# class MultiOHTDataset(Dataset):
#     """
#     여러 폴더(피벗/다이내믹) 내 CSV들을 합쳐서
#     X: [seq_len, N_edges], y: [len(horizons), N_edges] 샘플을 생성.
#     edge_order로 컬럼 강제 정렬/누락엣지는 0 채움.
#     scaler가 있으면 표준화 적용.
#     """
#     def __init__(
#         self,
#         folder_paths,
#         edge_order,
#         seq_len=12,
#         horizons=(6, 30, 60),
#         scaler=None,
#         fit_scaler=False,
#     ):
#         if isinstance(folder_paths, str):
#             folder_paths = [folder_paths]

#         self.edge_order = list(edge_order)
#         self.seq_len = seq_len
#         self.horizons = list(horizons)

#         csv_files = []
#         for folder in folder_paths:
#             files = glob.glob(os.path.join(folder, "*.csv"))
#             print(f"📂 {folder}: {len(files)} CSV")
#             csv_files.extend(files)
#         if len(csv_files) == 0:
#             print("⚠️ No CSV files found.")

#         X_list, y_list = [], []

#         for csv in csv_files:
#             try:
#                 df = pd.read_csv(csv)
#                 if "time" in df.columns:
#                     df = df.drop(columns=["time"])
#                 # edge_order로 정렬(없으면 0 채움)
#                 df = df.reindex(columns=self.edge_order, fill_value=0.0)

#                 data = df.values.astype(np.float32)  # [T, N]
#                 if data.shape[0] < seq_len + max(self.horizons):
#                     # 너무 짧으면 스킵
#                     continue

#                 if scaler is not None:
#                     if fit_scaler:
#                         scaler.partial_fit(data)
#                     else:
#                         data = scaler.transform(data).astype(np.float32)

#                 T = data.shape[0]
#                 for t in range(seq_len, T - max(self.horizons)):
#                     x = data[t - seq_len: t]                                        # (L,N)
#                     y = np.stack([data[t + h] for h in self.horizons], axis=0)      # (H,N)
#                     X_list.append(x)
#                     y_list.append(y)
#             except Exception as e:
#                 print(f"❌ Read error {csv}: {e}")

#         if scaler is not None and fit_scaler:
#             # fit pass는 텐서 만들 필요 없음
#             self.X = np.empty((0, seq_len, len(self.edge_order)), dtype=np.float32)
#             self.y = np.empty((0, len(self.horizons), len(self.edge_order)), dtype=np.float32)
#         else:
#             self.X = np.array(X_list, dtype=np.float32)
#             self.y = np.array(y_list, dtype=np.float32)

#         print(f"✅ Dataset built: {len(self)} samples | seq_len={seq_len}, N={len(self.edge_order)}")

#     def __len__(self):
#         return 0 if self.X is None else len(self.X)

#     def __getitem__(self, idx):
#         return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


# # -----------------------------
# # ST-Block (Temporal→Spatial→Temporal)
# # -----------------------------
# class TemporalConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3):
#         super().__init__()
#         pad = kernel_size // 2
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad)

#     def forward(self, x):
#         """
#         x: (B, T, N, C_in)  -> Conv over T
#         """
#         B, T, N, C = x.shape
#         x = x.permute(0, 2, 3, 1).contiguous().view(B * N, C, T)  # (B*N, C, T)
#         x = self.conv(x)                                          # (B*N, C_out, T)
#         C_out = x.size(1)
#         x = x.view(B, N, C_out, T).permute(0, 3, 1, 2).contiguous()  # (B, T, N, C_out)
#         return x


# class SpatialGCN(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.theta = nn.Linear(in_channels, out_channels, bias=True)

#     def forward(self, x, A_hat):
#         """
#         x: (B, T, N, C_in),  A_hat: (N, N)
#         """
#         x = torch.einsum("ij,btjc->btic", A_hat, x)  # 그래프 전파
#         x = self.theta(x)                            # 채널 변환
#         return x


# class STBlock(nn.Module):
#     def __init__(self, in_c, out_c, k=3, dropout=0.1):
#         super().__init__()
#         self.t1 = TemporalConv(in_c, out_c, kernel_size=k)
#         self.s  = SpatialGCN(out_c, out_c)
#         self.t2 = TemporalConv(out_c, out_c, kernel_size=k)
#         self.act = nn.ReLU()
#         self.drop = nn.Dropout(dropout)

#     def forward(self, x, A_hat):
#         x = self.act(self.t1(x))
#         x = self.drop(x)
#         x = self.act(self.s(x, A_hat))
#         x = self.drop(x)
#         x = self.act(self.t2(x))
#         return x


# class STGNN(nn.Module):
#     """
#     입력 : (B, T, N, 1)
#     출력 : (B, H, N)  ← Δ(증분) 예측
#     """
#     def __init__(self, num_nodes, in_channels=1, hidden=64, horizons=3, blocks=2, k=3, dropout=0.1):
#         super().__init__()
#         self.blocks = nn.ModuleList()
#         c_in = in_channels
#         for _ in range(blocks):
#             self.blocks.append(STBlock(c_in, hidden, k=k, dropout=dropout))
#             c_in = hidden
#         self.head = nn.Linear(hidden, horizons)

#     def forward(self, x, A_hat):
#         # x: (B, T, N, C_in)
#         h = x
#         for blk in self.blocks:
#             h = blk(h, A_hat)           # (B, T, N, hidden)
#         h_last = h[:, -1]               # (B, N, hidden)
#         out_delta = self.head(h_last)   # (B, N, H)  ← Δ
#         return out_delta.permute(0, 2, 1)  # (B, H, N)


# # -----------------------------
# # 평가 지표 (원 스케일로 MAE/RMSE)
# # -----------------------------
# @torch.no_grad()
# def evaluate(model, loader, A_eff, device, mean_vec, scale_vec):
#     model.eval()
#     l1 = nn.L1Loss(reduction="sum")
#     l2 = nn.MSELoss(reduction="sum")

#     total_mae, total_mse, total_count = 0.0, 0.0, 0

#     for X, y_abs_norm in loader:
#         # X,y는 "정규화된" 값으로 들어옴
#         X = X.to(device)                    # (B,L,N)
#         y_abs_norm = y_abs_norm.to(device)  # (B,H,N)
#         last_x_norm = X[:, -1, :]           # (B,N)

#         # Δ 예측
#         x_in = X.unsqueeze(-1)              # (B,L,N,1)
#         y_pred_delta = model(x_in, A_eff)   # (B,H,N)

#         # 절대값으로 복원 (정규화 공간에서)
#         y_pred_abs_norm = y_pred_delta + last_x_norm.unsqueeze(1)  # (B,H,N)

#         # 역정규화하여 지표 계산 (원 스케일)
#         y_pred_abs = inverse_transform(y_pred_abs_norm, mean_vec, scale_vec)
#         y_abs      = inverse_transform(y_abs_norm,      mean_vec, scale_vec)

#         total_mae += l1(y_pred_abs, y_abs).item()
#         total_mse += l2(y_pred_abs, y_abs).item()
#         total_count += y_abs.numel()

#     mae = total_mae / total_count
#     rmse = math.sqrt(total_mse / total_count)
#     return mae, rmse


# # -----------------------------
# # 학습 루프 (Δ 학습 + 멀티홉)
# # -----------------------------
# def train_model(
#     data_folders,
#     layout_path,
#     epochs=20,
#     batch_size=32,
#     seq_len=12,
#     horizons=(6, 30, 60),
#     hidden_channels=64,
#     num_blocks=2,
#     k_t=3,
#     lr=1e-3,
#     seed=42,
#     hop_order=2,
#     alpha=0.6,
#     dropout=0.1,
#     horizon_weights=None,   # 예: [1.0, 1.0, 1.2]
# ):
#     set_seed(seed)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"🚀 Device: {device}")

#     # 그래프
#     A_hat, edge_order = build_edge_adjacency(layout_path)
#     N = len(edge_order)
#     A_eff = make_multi_hop(A_hat, hop_order=hop_order, alpha=alpha).to(device)
#     print(f"🔗 Edge-graph built: N={N} | hop_order={hop_order}, alpha={alpha}")

#     # 스케일러 2-pass (fit → transform)
#     scaler = StandardScaler()
#     _ = MultiOHTDataset(
#         data_folders, edge_order, seq_len=seq_len, horizons=horizons,
#         scaler=scaler, fit_scaler=True
#     )
#     dataset = MultiOHTDataset(
#         data_folders, edge_order, seq_len=seq_len, horizons=horizons,
#         scaler=scaler, fit_scaler=False
#     )

#     if len(dataset) == 0:
#         print("❌ No samples. Check CSV length/columns.")
#         return None

#     # split (랜덤)
#     train_size = int(len(dataset) * 0.8)
#     val_size = len(dataset) - train_size
#     train_ds, val_ds = random_split(dataset, [train_size, val_size])
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
#     val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,    num_workers=0)

#     # 모델 (Δ 예측기)
#     model = STGNN(
#         num_nodes=N,
#         in_channels=1,
#         hidden=hidden_channels,
#         horizons=len(horizons),
#         blocks=num_blocks,
#         k=k_t,
#         dropout=dropout,
#     ).to(device)

#     # 손실 (호라이즌 가중치 옵션)
#     if horizon_weights is not None:
#         w = torch.tensor(horizon_weights, dtype=torch.float32, device=device).view(1, -1, 1)  # (1,H,1)
#         def weighted_mse(pred, target):
#             return torch.mean(((pred - target) ** 2) * w)
#         criterion = weighted_mse
#         print(f"⚖️ Horizon weights: {horizon_weights}")
#     else:
#         criterion = nn.MSELoss()

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     mean_vec = torch.tensor(scaler.mean_,  dtype=torch.float32).to(device)   # (N,)
#     scale_vec= torch.tensor(scaler.scale_, dtype=torch.float32).clamp_min(1e-12).to(device)

#     print(f"🧠 Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

#     # 체크포인트
#     best_val = float("inf")
#     ckpt_dir = "checkpoints"
#     os.makedirs(ckpt_dir, exist_ok=True)
#     ckpt_path = os.path.join(ckpt_dir, f"stgnn_dense_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")

#     for epoch in range(1, epochs + 1):
#         model.train()
#         running = 0.0

#         for X, y_abs_norm in train_loader:
#             # X,y는 "정규화된" 값
#             X = X.to(device)                    # (B,L,N)
#             y_abs_norm = y_abs_norm.to(device)  # (B,H,N)
#             last_x_norm = X[:, -1, :]           # (B,N)

#             # Δ 타겟 & Δ 예측
#             y_tgt_delta = y_abs_norm - last_x_norm.unsqueeze(1)   # (B,H,N)
#             x_in = X.unsqueeze(-1)                                # (B,L,N,1)
#             y_pred_delta = model(x_in, A_eff)                     # (B,H,N)

#             loss = criterion(y_pred_delta, y_tgt_delta)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             running += loss.item()

#         # 검증: 원 스케일 MAE/RMSE
#         val_mae, val_rmse = evaluate(model, val_loader, A_eff, device, mean_vec, scale_vec)

#         print(f"📊 Epoch {epoch:03d}/{epochs} | TrainLoss(Δ, norm): {running/len(train_loader):.4f} | "
#               f"ValMAE: {val_mae:.5f} | ValRMSE: {val_rmse:.5f}")

#         if val_rmse < best_val:
#             best_val = val_rmse
#             torch.save({
#                 "model_state": model.state_dict(),
#                 "scaler_mean": scaler.mean_.tolist(),
#                 "scaler_scale": scaler.scale_.tolist(),
#                 "edge_order": edge_order,
#                 "horizons": list(horizons),
#                 "seq_len": seq_len,
#                 "predict_delta": True,
#                 "hop_order": hop_order,
#                 "alpha": alpha,
#                 "dropout": float(dropout),
#             }, ckpt_path)
#             print(f"✅ Saved checkpoint: {ckpt_path}")

#     return ckpt_path


# # -----------------------------
# # main
# # -----------------------------
# if __name__ == "__main__":
#     print("🔎 model.py launched")

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--layout", type=str, default="/workspace/fab_oht_layout_updated.json")
#     parser.add_argument("--data_dirs", type=str, nargs="+",
#                         default=["/workspace/datasets_pivot", "/workspace/datasets_dynamic"])
#     parser.add_argument("--epochs", type=int, default=20)
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--seq_len", type=int, default=12)
#     parser.add_argument("--horizons", type=int, nargs="+", default=[6, 30, 60])
#     parser.add_argument("--hidden", type=int, default=64)
#     parser.add_argument("--blocks", type=int, default=2)
#     parser.add_argument("--k_t", type=int, default=3)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--hop_order", type=int, default=2)
#     parser.add_argument("--alpha", type=float, default=0.6)
#     parser.add_argument("--dropout", type=float, default=0.1)
#     parser.add_argument("--horizon_weights", type=float, nargs="*", default=None,
#                         help="예: --horizon_weights 1 1 1.2")
#     args = parser.parse_args()

#     ckpt = train_model(
#         data_folders=args.data_dirs,
#         layout_path=args.layout,
#         epochs=args.epochs,
#         batch_size=args.batch_size,
#         seq_len=args.seq_len,
#         horizons=tuple(args.horizons),
#         hidden_channels=args.hidden,
#         num_blocks=args.blocks,
#         k_t=args.k_t,
#         lr=args.lr,
#         seed=args.seed,
#         hop_order=args.hop_order,
#         alpha=args.alpha,
#         dropout=args.dropout,
#         horizon_weights=args.horizon_weights,
#     )

#     print(f"🏁 Done. Best checkpoint: {ckpt}")


# # -*- coding: utf-8 -*-
# import os, glob, json, math, argparse, random
# from datetime import datetime

# import numpy as np
# import pandas as pd

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, random_split


# # -----------------------------
# # Utils
# # -----------------------------
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# # -----------------------------
# # 그래프(엣지→엣지) + max_speed 불러오기
# #   - 연속 엣지 + 공유 노드 기반 인접성 반영
# #   - A_hat = D^{-1/2} (A + I) D^{-1/2}
# # -----------------------------
# def build_edge_adjacency(layout_json_path):
#     with open(layout_json_path) as f:
#         layout = json.load(f)

#     rails = [(r['from'], r['to']) for r in layout['rails']]
#     edges = [f"{u}-{v}" for u, v in rails]
#     N = len(edges)

#     A = np.zeros((N, N), dtype=np.float32)
#     for i, (u1, v1) in enumerate(rails):
#         for j, (u2, v2) in enumerate(rails):
#             # (1) 연속 edge 연결
#             if v1 == u2:
#                 A[i, j] = 1.0
#             # (2) 공유 노드 기반 연결 (교차/merge/split 반영)
#             if u1 == u2 or v1 == v2 or u1 == v2 or v1 == u2:
#                 A[i, j] = 1.0

#     # self-loop 추가
#     A = A + np.eye(N, dtype=np.float32)

#     # 정규화
#     D = np.sum(A, axis=1)
#     D_inv_sqrt = np.power(D, -0.5, where=D > 0)
#     D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
#     A_hat = (D_inv_sqrt[:, None] * A) * D_inv_sqrt[None, :]

#     # edge별 max_speed
#     edge2max = {f"{r['from']}-{r['to']}": r['max_speed'] for r in layout['rails']}
#     max_speeds = np.array([edge2max[e] for e in edges], dtype=np.float32)

#     return torch.tensor(A_hat, dtype=torch.float32), edges, torch.tensor(max_speeds, dtype=torch.float32)


# # -----------------------------
# # Dataset
# # -----------------------------
# class MultiOHTDataset(Dataset):
#     def __init__(self, folder_paths, edge_order, max_speeds, seq_len=12, horizons=(6, 30, 60), stride=1):
#         if isinstance(folder_paths, str):
#             folder_paths = [folder_paths]

#         self.edge_order = list(edge_order)
#         self.seq_len = seq_len
#         self.horizons = list(horizons)
#         self.max_speeds = max_speeds  # (N,)
#         self.stride = max(1, int(stride))

#         csv_files = []
#         for folder in folder_paths:
#             files = glob.glob(os.path.join(folder, "*.csv"))
#             print(f"📂 {folder}: {len(files)} CSV")
#             csv_files.extend(files)

#         X_list, y_list = [], []

#         for csv in csv_files:
#             try:
#                 df = pd.read_csv(csv)
#                 if "time" in df.columns:
#                     df = df.drop(columns=["time"])

#                 # edge_order에 맞춰 정렬 (없는 컬럼은 0으로 채움)
#                 df = df.reindex(columns=self.edge_order, fill_value=0.0)

#                 # (T, N) 실속도 → 정규화(0~1)
#                 data = df.values.astype(np.float32)
#                 data = data / (self.max_speeds.numpy() + 1e-8)  # 나눗셈 안정성

#                 # 길이 체크
#                 if data.shape[0] < seq_len + max(self.horizons) + 1:
#                     continue

#                 T = data.shape[0]
#                 # stride 적용한 슬라이딩 윈도우
#                 for t in range(seq_len, T - max(self.horizons), self.stride):
#                     x = data[t - seq_len:t]                             # (L, N)  정규화 입력
#                     y = np.stack([data[t + h] for h in self.horizons],  # (H, N) 정규화 타겟
#                                  axis=0)
#                     X_list.append(x)
#                     y_list.append(y)
#             except Exception as e:
#                 print(f"❌ Read error {csv}: {e}")

#         self.X = np.array(X_list, dtype=np.float32)
#         self.y = np.array(y_list, dtype=np.float32)

#         print(f"✅ Dataset built: {len(self)} samples | seq_len={seq_len}, N={len(self.edge_order)}, stride={self.stride}")

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


# # -----------------------------
# # ST-Block
# # -----------------------------
# class TemporalConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3):
#         super().__init__()
#         pad = kernel_size // 2
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad)

#     def forward(self, x):
#         # x: (B, T, N, C)
#         B, T, N, C = x.shape
#         x = x.permute(0, 2, 3, 1).contiguous().view(B * N, C, T)  # (B*N, C, T)
#         x = self.conv(x)                                          # (B*N, Cout, T)
#         C_out = x.size(1)
#         x = x.view(B, N, C_out, T).permute(0, 3, 1, 2).contiguous()  # (B, T, N, C_out)
#         return x


# class SpatialGCN(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.theta = nn.Linear(in_channels, out_channels, bias=True)

#     def forward(self, x, A_hat):
#         # x: (B, T, N, C)
#         # A_hat: (N, N)
#         x = torch.einsum("ij,btjc->btic", A_hat, x)  # (B, T, N, C)
#         return self.theta(x)


# class STBlock(nn.Module):
#     def __init__(self, in_c, out_c, k=3, dropout=0.1):
#         super().__init__()
#         self.t1 = TemporalConv(in_c, out_c, k)
#         self.s = SpatialGCN(out_c, out_c)
#         self.t2 = TemporalConv(out_c, out_c, k)
#         self.act = nn.ReLU()
#         self.drop = nn.Dropout(dropout)

#     def forward(self, x, A_hat):
#         x = self.drop(self.act(self.t1(x)))
#         x = self.drop(self.act(self.s(x, A_hat)))
#         x = self.drop(self.act(self.t2(x)))
#         return x


# # -----------------------------
# # STGNN (멀티-호라이즌 헤드 + Sigmoid로 [0,1] 제한)
# # -----------------------------
# class STGNN(nn.Module):
#     def __init__(self, num_nodes, in_channels=1, hidden=64, horizons=(6, 30, 60),
#                  blocks=2, k=3, dropout=0.1):
#         super().__init__()
#         self.blocks = nn.ModuleList()
#         c_in = in_channels
#         for _ in range(blocks):
#             self.blocks.append(STBlock(c_in, hidden, k, dropout))
#             c_in = hidden

#         self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in horizons])
#         self.horizons = horizons
#         self.out_act = nn.Sigmoid()  # [0,1]로 매끈하게 제한 -> max_speed 초과 방지

#     def forward(self, x, A_hat):
#         # x: (B, T, N, C=1)
#         h = x
#         for blk in self.blocks:
#             h = blk(h, A_hat)
#         h_last = h[:, -1]  # (B, N, C)

#         outs = []
#         for head in self.heads:
#             outs.append(head(h_last))  # (B, N, 1)

#         out = torch.cat(outs, dim=-1)         # (B, N, H)
#         out = self.out_act(out)               # (B, N, H) in [0,1]
#         return out.permute(0, 2, 1)           # (B, H, N)


# # -----------------------------
# # 평가 지표 (real scale)
# # -----------------------------
# @torch.no_grad()
# def evaluate(model, loader, A_hat, device, max_speeds):
#     model.eval()
#     total_mae, total_mse, total_count = 0.0, 0.0, 0
#     for X, y_norm in loader:
#         X, y_norm = X.to(device), y_norm.to(device)

#         # 예측/정답: 정규화 → 실제 스케일 복원
#         y_pred_norm = model(X.unsqueeze(-1), A_hat)   # (B,H,N) in [0,1]
#         y_true = y_norm * max_speeds.to(device)       # (B,H,N)
#         y_pred = y_pred_norm * max_speeds.to(device)  # (B,H,N)

#         total_mae += torch.sum(torch.abs(y_pred - y_true)).item()
#         total_mse += torch.sum((y_pred - y_true) ** 2).item()
#         total_count += y_true.numel()

#     mae = total_mae / total_count
#     rmse = math.sqrt(total_mse / total_count)
#     return mae, rmse


# # -----------------------------
# # 학습 루프
# #   - 학습: 정규화 스케일(MSE)
# #   - 검증: 실제 스케일(MAE/RMSE)
# #   - LR 스케줄러, 그래드 클리핑, 얼리 스톱 포함
# # -----------------------------
# def train_model(data_folders, layout_path, epochs=20, batch_size=32, seq_len=12,
#                 horizons=(6, 30, 60), hidden_channels=64, num_blocks=2, k_t=3,
#                 lr=1e-3, seed=42, dropout=0.1, stride=1, num_workers=0, patience=10):
#     set_seed(seed)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"🚀 Device: {device}")

#     A_hat, edge_order, max_speeds = build_edge_adjacency(layout_path)
#     N = len(edge_order)
#     A_hat = A_hat.to(device)
#     max_speeds = max_speeds.to(device)
#     print(f"🔗 Edge-graph built: N={N}")

#     dataset = MultiOHTDataset(data_folders, edge_order, max_speeds.detach().cpu(), seq_len, horizons, stride=stride)
#     if len(dataset) == 0:
#         print("❌ No samples.")
#         return None

#     train_size = int(len(dataset) * 0.8)
#     val_size = len(dataset) - train_size
#     train_ds, val_ds = random_split(dataset, [train_size, val_size])

#     train_loader = DataLoader(
#         train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
#         num_workers=num_workers, pin_memory=(device == "cuda")
#     )
#     val_loader = DataLoader(
#         val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
#         num_workers=num_workers, pin_memory=(device == "cuda")
#     )

#     model = STGNN(
#         num_nodes=N, in_channels=1, hidden=hidden_channels,
#         horizons=tuple(horizons), blocks=num_blocks, k=k_t, dropout=dropout
#     ).to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
#     criterion = nn.MSELoss()  # 정규화 스케일에서 학습

#     print(f"🧠 Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

#     best_val = float("inf")
#     best_epoch = -1
#     ckpt_dir = "checkpoints"
#     os.makedirs(ckpt_dir, exist_ok=True)
#     ckpt_path = os.path.join(ckpt_dir, f"stgnn_dense_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")

#     for epoch in range(1, epochs + 1):
#         model.train()
#         running = 0.0

#         for X, y_norm in train_loader:
#             X, y_norm = X.to(device), y_norm.to(device)

#             # Forward (정규화 스케일)
#             y_pred_norm = model(X.unsqueeze(-1), A_hat)
#             loss = criterion(y_pred_norm, y_norm)

#             optimizer.zero_grad()
#             loss.backward()
#             # 그래드 클리핑
#             nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()

#             running += loss.item()

#         # 검증 (실제 스케일)
#         val_mae, val_rmse = evaluate(model, val_loader, A_hat, device, max_speeds)
#         train_loss_norm = running / max(1, len(train_loader))
#         print(f"📊 Epoch {epoch:03d}/{epochs} | TrainLoss(norm): {train_loss_norm:.6f} | ValMAE: {val_mae:.5f} | ValRMSE: {val_rmse:.5f}")

#         # 스케줄러(plateau 기반)
#         scheduler.step(val_rmse)

#         # 체크포인트 & 얼리 스톱
#         if val_rmse < best_val - 1e-6:
#             best_val = val_rmse
#             best_epoch = epoch
#             torch.save({
#                 "model_state": model.state_dict(),
#                 "edge_order": edge_order,
#                 "horizons": list(horizons),
#                 "seq_len": seq_len,
#                 "max_speeds": max_speeds.detach().cpu().tolist(),
#                 "hidden": hidden_channels,
#                 "blocks": num_blocks,
#                 "k_t": k_t,
#                 "dropout": dropout,
#             }, ckpt_path)
#             print(f"✅ Saved checkpoint: {ckpt_path}")
#         elif epoch - best_epoch >= patience:
#             print(f"⏹ Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
#             break

#     return ckpt_path


# # -----------------------------
# # main
# # -----------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--layout", type=str, default="/workspace/fab_oht_layout_updated.json")
#     parser.add_argument("--data_dirs", type=str, nargs="+", default=["/workspace/datasets_pivot", "/workspace/datasets_dynamic"])
#     parser.add_argument("--epochs", type=int, default=50)
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--seq_len", type=int, default=12)
#     parser.add_argument("--horizons", type=int, nargs="+", default=[6, 30, 60])
#     parser.add_argument("--hidden", type=int, default=64)
#     parser.add_argument("--blocks", type=int, default=2)
#     parser.add_argument("--k_t", type=int, default=3)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--dropout", type=float, default=0.1)
#     parser.add_argument("--stride", type=int, default=1)
#     parser.add_argument("--num_workers", type=int, default=0)
#     parser.add_argument("--patience", type=int, default=10)
#     args = parser.parse_args()

#     set_seed(args.seed)
#     ckpt = train_model(
#         data_folders=args.data_dirs,
#         layout_path=args.layout,
#         epochs=args.epochs,
#         batch_size=args.batch_size,
#         seq_len=args.seq_len,
#         horizons=tuple(args.horizons),
#         hidden_channels=args.hidden,
#         num_blocks=args.blocks,
#         k_t=args.k_t,
#         lr=args.lr,
#         seed=args.seed,
#         dropout=args.dropout,
#         stride=args.stride,
#         num_workers=args.num_workers,
#         patience=args.patience
#     )
#     print(f"🏁 Done. Best checkpoint: {ckpt}")

import torch
import numpy as np
import model_4ch_2
from model_4ch_2 import STGNN, build_edge_adjacency, build_diffusion_powers
import time


class Predictor:
    def __init__(self, ckpt_full_path: str, meta_pt_path: str, layout_json_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        meta = torch.load(meta_pt_path, map_location="cpu")
        state_dict = meta.pop("model_state")   

        self.edges_order = meta["edges_order"]
        self.horizons = meta["horizons"]
        self.seq_len = meta["seq_len"]
        self.scaler_mode = meta["scaler_mode"]
        self.normalize_by_vmax = meta["normalize_by_vmax"]

        self.mean_dyn = torch.tensor(meta["scaler_mean"], dtype=torch.float32, device=self.device)
        self.std_dyn  = torch.tensor(meta["scaler_std"],  dtype=torch.float32, device=self.device).clamp(min=1e-12)

        self.vmax = torch.tensor(meta["vmax"], dtype=torch.float32, device=self.device).view(1, 1, -1)
        self.length = torch.tensor(meta["length"], dtype=torch.float32, device=self.device).view(1, 1, -1)

        self.vmax_flat = self.vmax.view(-1)      
        self.length_flat = self.length.view(-1)  

        E = len(self.edges_order)
        self.model = STGNN(
            num_nodes=E,
            in_channels=meta.get("in_channels", 4),
            hidden=meta.get("hidden", 96),
            horizons=len(self.horizons),
            blocks=meta.get("blocks", 3),
            k=meta.get("k_t", 3),
            dilations=tuple(meta.get("dilations", [1,2,4,8])),
            K_spatial=meta.get("K_spatial", 2),
            dropout=meta.get("dropout", 0.1)
        ).to(self.device)

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        A_hat_all, edges_layout = build_edge_adjacency(layout_json_path)
        idx_map = {e: i for i, e in enumerate(edges_layout)}
        idxs = [idx_map[e] for e in self.edges_order]
        A_hat = A_hat_all[idxs][:, idxs].to(self.device)
        self.A_powers = build_diffusion_powers(A_hat, meta["K_spatial"])

        ln = (self.length_flat / self.length_flat.max()).to(torch.float32)
        vn = (self.vmax_flat / self.vmax_flat.max()).to(torch.float32)
        self.Lnorm = ln.to(self.device)
        self.Vnorm = vn.to(self.device)

        x0z = torch.zeros(E, dtype=torch.float32, device=self.device)
        x1z = torch.zeros(E, dtype=torch.float32, device=self.device)
        Xz = torch.stack([x0z, x1z, self.Lnorm, self.Vnorm], dim=-1)  # (E,4)
        self.buffer = [Xz.clone() for _ in range(self.seq_len)]

    def step_and_predict_bin(self, avg_speed_arr, count_bin_arr):
        
        x0 = torch.as_tensor(avg_speed_arr, dtype=torch.float32, device=self.device)
        x1 = torch.as_tensor(np.sqrt(count_bin_arr), dtype=torch.float32, device=self.device)

        if self.normalize_by_vmax:
            x0 = x0 / self.vmax_flat

        if self.scaler_mode == "per_edge":
            x0z = (x0 - self.mean_dyn[:, 0]) / self.std_dyn[:, 0]
            x1z = (x1 - self.mean_dyn[:, 1]) / self.std_dyn[:, 1]
        else:
            x0z = (x0 - self.mean_dyn[0]) / self.std_dyn[0]
            x1z = (x1 - self.mean_dyn[1]) / self.std_dyn[1]

        Xz = torch.stack([x0z, x1z, self.Lnorm, self.Vnorm], dim=-1)

        self.buffer.append(Xz)
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

        if len(self.buffer) < self.seq_len:
            return None

        Xseq = torch.stack(self.buffer, dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            yp_z = self.model(Xseq, self.A_powers).squeeze(0)  
        if self.scaler_mode == "per_edge":
            mean_avg = self.mean_dyn[:, 0].view(1, -1)
            std_avg = self.std_dyn[:, 0].view(1, -1)
        else:
            mean_avg = self.mean_dyn[0].view(1, 1)
            std_avg = self.std_dyn[0].view(1, 1)

        yp_norm = yp_z * std_avg + mean_avg
        yp = yp_norm * self.vmax_flat if self.normalize_by_vmax else yp_norm

        yp = torch.clamp(yp, torch.tensor(0.0, device=yp.device), self.vmax_flat).cpu().numpy()

        out = {}
        for i, h in enumerate(self.horizons):
            if h in [3, 6]:
                out[str(h*10)] = {
                    edge_id: float(yp[i, j].round(2))
                    for j, edge_id in enumerate(self.edges_order)
                }
                

        return out

    def reset_buffer_with_current(self, avg_speed_arr, count_bin_arr):
        x0 = torch.as_tensor(avg_speed_arr, dtype=torch.float32, device=self.device)
        x1 = torch.as_tensor(np.sqrt(count_bin_arr), dtype=torch.float32, device=self.device)

        if self.normalize_by_vmax:
            x0 = x0 / self.vmax_flat

        if self.scaler_mode == "per_edge":
            x0z = (x0 - self.mean_dyn[:, 0]) / self.std_dyn[:, 0]
            x1z = (x1 - self.mean_dyn[:, 1]) / self.std_dyn[:, 1]
        else:
            x0z = (x0 - self.mean_dyn[0]) / self.std_dyn[0]
            x1z = (x1 - self.mean_dyn[1]) / self.std_dyn[1]

        Xz = torch.stack([x0z, x1z, self.Lnorm, self.Vnorm], dim=-1)


        self.buffer = [Xz.clone() for _ in range(self.seq_len)]

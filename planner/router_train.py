# router_train_vllm.py
import json
import math
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from openai import OpenAI  # vLLM provides OpenAI-compatible API


# ==========================================================
# CONFIGURATION
# ==========================================================
@dataclass
class RouterConfig:
    vllm_base_url: str = "http://localhost:8001/v1"  # your vLLM endpoint
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    dim_embed: int = 1024       # dimension of Qwen3-Embedding-0.6B output
    dim_meta: int = 8
    hidden: int = 256
    depth: int = 2
    dropout: float = 0.1

    # Budget control
    delta0: float = 0.0
    alpha_B: float = 1.0
    alpha_L: float = 1.0
    tau: float = 0.5
    B_max: float = 5000.0
    L_max: float = 5.0
    lambda_B: float = 1.0
    lambda_L: float = 1.0
    huber_delta: float = 1.0

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================================
# DATA LOADING
# ==========================================================
def load_profile_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# ==========================================================
# EMBEDDING EXTRACTOR (via vLLM)
# ==========================================================
class VLLMEmbedder:
    """Fetch embeddings from local vLLM embedding server."""
    def __init__(self, cfg: RouterConfig):
        self.client = OpenAI(base_url=cfg.vllm_base_url, api_key="dummy")
        self.model = cfg.embed_model

    def embed(self, texts: List[str]) -> torch.Tensor:
        """Return mean embeddings for all texts (batched if needed)."""
        embeddings = []
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            batch_emb = [d.embedding for d in resp.data]
            embeddings.extend(batch_emb)
        return torch.tensor(embeddings, dtype=torch.float32)


# ==========================================================
# DATASET CREATION
# ==========================================================
def make_router_dataset(samples: List[Dict[str, Any]], embedder: VLLMEmbedder) -> Dict[str, torch.Tensor]:
    texts, meta, rB, rL, deltaU, kC, L_edge, L_cloud = [], [], [], [], [], [], [], []
    λ_q, λ_t, λ_k = 1.0, 0.5, 0.001

    for s in samples:
        edge = s["edge"]
        cloud = s["cloud"]

        # latencies in seconds
        L_e = edge["latency_ms"] / 1000.0
        L_c = cloud["latency_ms"] / 1000.0

        # token usage
        k_e = edge["usage"]["total_tokens"]
        k_c = cloud["usage"]["total_tokens"]

        # approximate quality gain by completion length diff
        dq = (cloud["usage"]["completion_tokens"] - edge["usage"]["completion_tokens"]) / 100.0
        dt = L_c - L_e
        dU = λ_q * dq - λ_t * dt - λ_k * k_c

        task_text = s["task"]
        subtask_text = s["subtask"]["description"]
        texts.append(task_text + " " + subtask_text)

        dep_depth = len(s["subtask"].get("depends_on", []))
        meta.append([dep_depth, k_e, k_c, L_e, L_c, dq, dt, dU])

        rB.append(np.random.uniform(0.5, 1.5))
        rL.append(np.random.uniform(0.5, 1.5))
        deltaU.append(dU)
        kC.append(k_c)
        L_edge.append(L_e)
        L_cloud.append(L_c)

    print(f"Requesting {len(texts)} embeddings from vLLM ...")
    E = embedder.embed(texts)
    M = torch.tensor(meta, dtype=torch.float32)
    rB = torch.tensor(rB, dtype=torch.float32)
    rL = torch.tensor(rL, dtype=torch.float32)
    deltaU = torch.tensor(deltaU, dtype=torch.float32)
    kC = torch.tensor(kC, dtype=torch.float32)
    L_edge = torch.tensor(L_edge, dtype=torch.float32)
    L_cloud = torch.tensor(L_cloud, dtype=torch.float32)

    return dict(E=E, M=M, rB=rB, rL=rL, deltaU=deltaU, kC=kC, L_edge=L_edge, L_cloud=L_cloud)


# ==========================================================
# ROUTER MODEL AND LOSS
# ==========================================================
class RouterNet(nn.Module):
    def __init__(self, cfg: RouterConfig):
        super().__init__()
        d_in = cfg.dim_embed + cfg.dim_meta + 2
        layers = []
        d = d_in
        for _ in range(cfg.depth):
            layers += [nn.Linear(d, cfg.hidden), nn.ReLU(), nn.Dropout(cfg.dropout)]
            d = cfg.hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, E, M, rB, rL):
        z = torch.cat([E, M, rB.unsqueeze(-1), rL.unsqueeze(-1)], dim=1)
        return self.net(z).squeeze(-1)


class RouterLoss(nn.Module):
    def __init__(self, cfg: RouterConfig):
        super().__init__()
        self.cfg = cfg
        self.huber = nn.HuberLoss(delta=cfg.huber_delta)

    def forward(self, s, deltaU, kC, L_edge, L_cloud, rB, rL):
        cfg = self.cfg
        reg = self.huber(s, deltaU)
        δt = cfg.delta0 + cfg.alpha_B * (rB - 1.0) + cfg.alpha_L * (rL - 1.0)
        p_cloud = torch.sigmoid((s - δt) / cfg.tau)
        exp_C = (p_cloud * kC).mean()
        exp_L = (p_cloud * L_cloud + (1 - p_cloud) * L_edge).mean()
        penB = torch.relu(exp_C - cfg.B_max) ** 2
        penL = torch.relu(exp_L - cfg.L_max) ** 2
        return reg + cfg.lambda_B * penB + cfg.lambda_L * penL


# ==========================================================
# TRAINING
# ==========================================================
class RouterDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["E"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def train_router(cfg: RouterConfig, data: Dict[str, torch.Tensor]):
    ds = RouterDataset(data)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    model = RouterNet(cfg).to(cfg.device)
    loss_fn = RouterLoss(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(cfg.device)
            s = model(batch["E"], batch["M"], batch["rB"], batch["rL"])
            loss = loss_fn(s, batch["deltaU"], batch["kC"], batch["L_edge"], batch["L_cloud"], batch["rB"], batch["rL"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(loader):.4f}")
    return model


# ==========================================================
# MAIN PIPELINE
# ==========================================================
def main():
    cfg = RouterConfig()
    embedder = VLLMEmbedder(cfg)

    # Path to your profiling file (JSONL)
    samples = load_profile_jsonl("/home/jwdong/demo2/router/profile/profiled_data.jsonl")
    data = make_router_dataset(samples, embedder)
    model = train_router(cfg, data)
    torch.save(model.state_dict(), "router_trained_vllm.pt")
    print("✅ Router training completed and saved as router_trained_vllm.pt")


if __name__ == "__main__":
    main()

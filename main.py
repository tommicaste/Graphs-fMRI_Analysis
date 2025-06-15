from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.nn import (
    GCNConv,          # Graph Convolutional Network
    SAGEConv,         # GraphSAGE
    GATConv,          # Graph Attention Network
    GATv2Conv,        # GAT v2
    GINConv,          # Graph Isomorphism Network
    ChebConv,         # Chebyshev Spectral GCN
    TransformerConv   # Graph Transformer-style convolution
)
from data import *
from static_models.gnn import LightningGNN
import torch
import random 
import numpy as np

SEED = 23

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# ───────────────────── data ─────────────────────
path = "/project2/cdonnat/sleepstages/data/pt/non_overlapping_data.pt"
train_loader, val_loader, test_loader = load_data(
    path, augment_strategy="interpolate", augment_proportion=0.5
)

# ───────────────────── model ────────────────────
model = LightningGNN(
    input_dim=347,
    hidden_channels=64,
    num_layers=2,
    GNNLayer=SAGEConv ,
    dropout=0.5,
    num_classes=4,
    mlp_hidden=[64, 32],
    lr=1e-3,
    edge_tsh=0,
)

# ──────────── NEW: run folder and checkpoint ────────────
save_dir = Path("/home/tcastellani/sleepstages/upsample_run")  
save_dir.mkdir(parents=True, exist_ok=True)

ckpt_cb = ModelCheckpoint(
    dirpath=save_dir / "checkpoints",
    filename="epoch={epoch:02d}_val_bacc={val_bacc:.4f}",
    monitor="val_bacc",
    mode="max",
    save_top_k=1,
    save_last=True,
)

# ─────────────────── trainer ────────────────────
trainer = pl.Trainer(
    max_epochs=50,
    default_root_dir=str(save_dir),
    callbacks=[ckpt_cb],
    accelerator="auto",
    devices=1,
    log_every_n_steps=10,
    enable_progress_bar=True,
)

trainer.fit(model, train_loader, val_loader)

# ──────────── test best on val_bacc ─────────────
trainer.test(ckpt_path="best", dataloaders=test_loader)

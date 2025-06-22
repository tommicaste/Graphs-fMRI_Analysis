from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.nn import (
    GCNConv,          
    SAGEConv,         
    GATConv,          
    GATv2Conv,        
    GINConv,                  
    TransformerConv   
)
from data import *
from static_models.gnn import LightningGNN
import torch
import random 
import numpy as np

SEED = 12
pl.seed_everything(SEED, workers=True)

# ───────────────────── data ─────────────────────
augment_strategy   = None
augment_proportion = None

path = "/home/tcastellani/projects/sleepstages/neurograph/HCPGender/data.pt"
train_loader, val_loader, test_loader, weights = load_data(
    path,
    augment_strategy     = augment_strategy,
    augment_proportion   = augment_proportion,
    batch_size           = 64,
    workers          = 3,
    train_ratio= 0.70,
    val_ratio = 0.15,
    test_ratio = 0.15,
)

run_name = f"gnn_{augment_strategy}_{augment_proportion}_overlapping_no_edges"

# ───────────────────── model ────────────────────
model = LightningGNN(
    input_dim=1000,
    hidden_channels=64,
    num_layers=3,
    GNNLayer=SAGEConv, 
    dropout=0.5,
    edge_dropout =  1,
    num_classes = 2,
    mlp_hidden=[64, 32],
    lr = 0.0005,
    wd = 0.0001,
    edge_tsh = None,
    edge_top = None, 
    loss_type = 'cross_entropy',
    class_weights = weights['class_weights'],
    residual_connections = True, 
    pooling_fn = 'mean'
)

# ──────────── checkpoint ────────────
save_dir = Path(f"/home/tcastellani/sleepstages/run/{run_name}")  
save_dir.mkdir(parents=True, exist_ok=True)

ckpt_cb = ModelCheckpoint(
    dirpath=save_dir / "checkpoints",
    filename="epoch={epoch:02d}_val_bacc={val_acc:.4f}",
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
    #precision="bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true",
)

trainer.fit(model, train_loader, val_loader)

# ──────────── test best on val_acc ─────────────
trainer.test(ckpt_path="best", dataloaders=test_loader)

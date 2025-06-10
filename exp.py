# ───────────────────────────────
# Imports
# ───────────────────────────────
import os
import shutil
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from data import (
    split_patient,
    augment_interpolate,
    add_edge_list,
    feature_corr,
)
from static_models.gnn import LightningGNN

# ───────────────────────────────
# Data loading & preprocessing
# ───────────────────────────────
DATA_PATH = "/project2/cdonnat/sleepstages/data/pt/non_overlapping_data.pt"
data_list = torch.load(DATA_PATH, weights_only=False)
print(f"✅ Loaded {len(data_list)} data files\n")

data = split_patient(data_list, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

# Print split composition
splits = defaultdict(int)
for d in data:
    splits[d.metadata["split"]] += 1
print("📊 Split composition (by segment count):")
for split, count in splits.items():
    print(f"   {split:5s}: {count}")

# Check patient overlap
patients = defaultdict(set)
for d in data:
    patients[d.metadata["split"]].add(d.metadata["sample"])
overlaps = {(a, b): patients[a] & patients[b] for a in patients for b in patients if a < b}
print("\n🔍 Patient overlaps:")
for (a, b), shared in overlaps.items():
    print(f"   {a} ∩ {b}: {len(shared)} patients")

# Augmentation, edges, feature transform
data_aug = augment_interpolate(data, proportion=0.30)
data_aug = add_edge_list(data_aug, tsh = 0)
data_feat = feature_corr(data_aug)

# Inspect attributes
print("\n🔧 Post-feature attributes:", set(data_feat[0].keys()))
print("🗂️  Metadata keys:", set(data_feat[0].metadata.keys()))

# Split loaders
train_data = [d for d in data_feat if d.metadata["split"] == "train"]
val_data   = [d for d in data_feat if d.metadata["split"] == "val"]
test_data  = [d for d in data_feat if d.metadata["split"] == "test"]

# after splitting:
splits = {
    'train': train_data,
    'val':   val_data,
    'test':  test_data
}

# count per split → per class → real vs synth
for split_name, lst in splits.items():
    counts = defaultdict(lambda: {'real': 0, 'synth': 0})
    for d in lst:
        cls = int(d.y.item())
        if d.metadata.get('synthetic', False):
            counts[cls]['synth'] += 1
        else:
            counts[cls]['real'] += 1

    print(f"\n📦 {split_name.upper()} split distribution:")
    for cls in sorted(counts):
        real = counts[cls]['real']
        synth = counts[cls]['synth']
        total = real + synth
        print(f"  class {cls}:  real={real:3d}  synth={synth:3d}  total={total:3d}")

BATCH_SIZE = 8
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, persistent_workers=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True)

# ───────────────────────────────
# Run directory cleanup
# ───────────────────────────────
RUN_ROOT = "FirstGNN"
if os.path.exists(RUN_ROOT):
    shutil.rmtree(RUN_ROOT)
os.makedirs(RUN_ROOT)

# ───────────────────────────────
# Logger & callbacks
# ───────────────────────────────
logger = CSVLogger(save_dir=RUN_ROOT, name="logs", version="")

checkpoint_cb = ModelCheckpoint(
    monitor="val_BAcc",
    mode="max",
    save_top_k=1,
    dirpath=os.path.join(RUN_ROOT, "checkpoints"),
    filename="best.ckpt",
)

earlystop_cb = EarlyStopping(
    monitor="val_BAcc",
    mode="max",
    patience=15,
    verbose=False,
)

# ───────────────────────────────
# Model
# ───────────────────────────────
model = LightningGNN(
    input_dim=347,
    hidden_channels=64,
    num_layers=2,
    GNNLayer=SAGEConv,
    dropout=0.5,
    num_classes=4,
    mlp_hidden=[32],
    lr=1e-3,
)

# ───────────────────────────────
# Trainer (GPU enabled)
# ───────────────────────────────
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=100,
    logger=logger,
    callbacks=[checkpoint_cb, earlystop_cb],
    log_every_n_steps=50,
    enable_progress_bar=True,
)

# ───────────────────────────────
# Fit
# ───────────────────────────────
trainer.fit(model, train_loader, val_loader)

# ───────────────────────────────
# Test on best checkpoint
# ───────────────────────────────
best_ckpt   = checkpoint_cb.best_model_path
results_dir = os.path.join(RUN_ROOT, "results")
model.test_model(best_ckpt, test_loader, results_dir)

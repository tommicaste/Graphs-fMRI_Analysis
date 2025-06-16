from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import random
import numpy as np

from data import load_data
from static_models.logistic import LightningLogisticRegression 

# reproducibility ─────────────────────────────────────────────
SEED = 12
pl.seed_everything(SEED, workers=True)

# data ────────────────────────────────────────────────────────
augment_strategy   = None
augment_proportion = None

path = "/project2/cdonnat/sleepstages/data/pt/non_overlapping_data.pt"
train_loader, val_loader, test_loader, weights = load_data(
    path,
    augment_strategy     = augment_strategy,
    augment_proportion   = augment_proportion,
    batch_size           = 48,
    workers          = 3,
    train_ratio= 0.65,
    val_ratio = 0.20,
    test_ratio = 0.15,
)

run_name = f"logistic_{augment_strategy}_{augment_proportion}_nonoverlapping"

# model ───────────────────────────────────────────────────────
model = LightningLogisticRegression(
    input_dim      = 347,
    num_classes    = 4,
    lr             = 1e-3,
    loss_type      = "cross_entropy",
    class_weights  = weights["class_weights"],
)

# checkpoint ─────────────────────────────────────────────────
save_dir = Path(f"/home/tcastellani/sleepstages/run/{run_name}")
save_dir.mkdir(parents=True, exist_ok=True)

ckpt_cb = ModelCheckpoint(
    dirpath   = save_dir / "checkpoints",
    filename  = "epoch={epoch:02d}_val_acc={val_acc:.4f}",
    monitor   = "val_acc",
    mode      = "max",
    save_top_k = 1,
    save_last  = True,
)

# trainer ────────────────────────────────────────────────────
trainer = pl.Trainer(
    max_epochs         = 70,
    default_root_dir   = str(save_dir),
    callbacks          = [ckpt_cb],
    accelerator        = "auto",
    devices            = 1,
    log_every_n_steps  = 10,
    enable_progress_bar = True,
)

# run ────────────────────────────────────────────────────────
trainer.fit(model, train_loader, val_loader)

# evaluate best checkpoint ──────────────────────────────────
trainer.test(ckpt_path="best", dataloaders=test_loader)

# main_mlp.py  ── two-layer MLP baseline
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch, random, numpy as np

from data import load_data
from static_models.mlp import LightningMLP          # << using the new MLP module

# reproducibility ─────────────────────────────────────────────
SEED = 12
pl.seed_everything(SEED, workers=True)

# data ────────────────────────────────────────────────────────
path = "/project2/cdonnat/sleepstages/data/pt/overlapping_data.pt"
train_loader, val_loader, test_loader, weights = load_data(
    path,
    augment_strategy=None,
    batch_size=48,
)

run_name = "mlp_non_overlapping"        

# model ───────────────────────────────────────────────────────
model = LightningMLP(
    input_dim    = 294,
    hidden_dims  = [128, 64],   
    num_classes  = 4,
    lr           = 1e-3,
    dropout      = 0.2,
    loss_type    = "cross_entropy",
    class_weights= weights["class_weights"],
)

# checkpoint ─────────────────────────────────────────────────
save_dir = Path(f"/home/tcastellani/sleepstages/run/{run_name}")
save_dir.mkdir(parents=True, exist_ok=True)

ckpt_cb = ModelCheckpoint(
    dirpath    = save_dir / "checkpoints",
    filename   = "epoch={epoch:02d}_val_acc={val_acc:.4f}",
    monitor    = "val_acc",
    mode       = "max",
    save_top_k = 1,
    save_last  = True,
)

# trainer ────────────────────────────────────────────────────
trainer = pl.Trainer(
    max_epochs          = 50,
    default_root_dir    = str(save_dir),
    callbacks           = [ckpt_cb],
    accelerator         = "auto",
    devices             = 1,
    log_every_n_steps   = 10,
    enable_progress_bar = True,
)

trainer.fit(model, train_loader, val_loader)
trainer.test(ckpt_path="best", dataloaders=test_loader)

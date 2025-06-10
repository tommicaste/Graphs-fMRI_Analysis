import os
import shutil

import pytorch_lightning as pl
from torch_geometric.nn import SAGEConv
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from static_models.gnn import LightningGNN
from data.datamodule import StaticDataModule

# ───────────────────────────────
# 1. DataModule setup (self-contained)
# ───────────────────────────────
DATA_PATH = "/project2/cdonnat/sleepstages/data/pt/overlapping_data.pt"

import random
import numpy as np
import torch

# 0) Fix all seeds so both pipelines draw the same synthetic samples
random.seed(23)
np.random.seed(23)
torch.manual_seed(23)

# 1) Instantiate the DataModule with the same split ratios and interpolate augment
dm = StaticDataModule(
    raw_path     = DATA_PATH,
    batch_size   = 32,
    num_workers  = 4,
    split_ratios = (0.7, 0.2, 0.1),
    augment_cfg  = {
        "interpolate": {"proportion": 0.30, "random_state": 23}
    },
    edge_tsh     = 0.0,
)

# 2) Run the same setup sequence
dm.prepare_data()

print(f"✅ Loaded")

dm.setup(stage="fit")
dm.setup(stage="test")

# 3) Pull out loaders to feed into trainer.fit / test
train_loader = dm.train_dataloader()
val_loader   = dm.val_dataloader()
test_loader  = dm.test_dataloader()


# ───────────────────────────────
# 2. run directory cleanup
# ───────────────────────────────
RUN_ROOT = "FirstGNN"
if os.path.exists(RUN_ROOT):
    shutil.rmtree(RUN_ROOT)
os.makedirs(RUN_ROOT)

# ───────────────────────────────
# 3. logger & callbacks
# ───────────────────────────────
logger = CSVLogger(
    save_dir = RUN_ROOT,
    name     = "logs",
    version  = ""
)

checkpoint_cb = ModelCheckpoint(
    monitor    = "val_BAcc",
    mode       = "max",
    save_top_k = 1,
    dirpath    = os.path.join(RUN_ROOT, "checkpoints"),
    filename   = "best.ckpt"
)

earlystop_cb = EarlyStopping(
    monitor  = "val_BAcc",
    mode     = "max",
    patience = 15,
    verbose  = False
)

# ───────────────────────────────
# 4. model
# ───────────────────────────────
model = LightningGNN(
    input_dim       = 294,
    hidden_channels = 64,
    num_layers      = 2,
    GNNLayer        = SAGEConv,
    dropout         = 0.5,
    num_classes     = 4,
    mlp_hidden      = [32],
    lr              = 1e-3
)

# ───────────────────────────────
# 5. trainer (GPU enabled)
# ───────────────────────────────
trainer = pl.Trainer(
    accelerator           = "gpu",
    devices               = 1,
    max_epochs            = 100,
    logger                = logger,
    callbacks             = [checkpoint_cb, earlystop_cb],
    log_every_n_steps     = 50,
    enable_progress_bar   = True,
)

# ───────────────────────────────
# 6. fit
# ───────────────────────────────
trainer.fit(model, train_loader, val_loader)

# ───────────────────────────────
# 7. test on best checkpoint
# ───────────────────────────────
best_ckpt   = checkpoint_cb.best_model_path
results_dir = os.path.join(RUN_ROOT, "results")
model.test_model(best_ckpt, test_loader, results_dir)

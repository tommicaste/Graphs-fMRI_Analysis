# run_optuna.py
#
# End-to-end Optuna sweep for the LightningGNN sleep-stage task.
# ─────────────────────────────────────────────────────────────

from pathlib import Path
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from torch_geometric.nn import SAGEConv, GATConv, GCNConv
from data import load_data
from static_models.gnn import LightningGNN

# ╭───────────────────── 0.  GLOBAL CONFIG ─────────────────────╮
SEED = 12
pl.seed_everything(SEED, workers=True)

ROOT = Path("optuna_search")         # everything (DB + ckpts) lives here
ROOT.mkdir(exist_ok=True)
STUDY_DB = ROOT / "study.db"

DATA_PATH = Path(
    "/project2/cdonnat/sleepstages/data/pt/non_overlapping_data.pt"
)
BATCH_SIZE = 48

# ╭───────────────────── 1.  FIXED HYPER-PARAMS ─────────────────╮
FIXED = dict(
    input_dim=347,
    hidden_channels=64,
    dropout=0.5,                # ← global dropout restored
    num_classes=4,
    edge_tsh=0,
    edge_top=None,
    residual_connections=True,
    pooling_fn="mean",
)

BACKBONES = {"SAGEConv": SAGEConv, "GATConv": GATConv, "GCNConv": GCNConv}

# ╭───────────────────── 2.  TRIAL SAMPLER ─────────────────────╮
def sample_params(trial: optuna.Trial) -> dict:
    return dict(
        GNNLayer=BACKBONES[trial.suggest_categorical("backbone", list(BACKBONES))],
        num_layers=trial.suggest_int("num_layers", 1, 3),
        augment_strategy=trial.suggest_categorical(
            "augment_strategy", ["interpolate", "geodesic"]
        ),
        augment_proportion=trial.suggest_float("augment_proportion", 0.2, 0.8),
        lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        wd=trial.suggest_float("wd", 1e-6, 1e-2, log=True),
        mlp_hidden=trial.suggest_categorical(
            "mlp_hidden", [[64, 32], [64]]
        ),
        loss_type=trial.suggest_categorical(
            "loss_type", ["cross_entropy", "weighted_cross_entropy"]
        ),
    )

# ╭───────────────────── 3.  OBJECTIVE ─────────────────────────╮
def objective(trial: optuna.Trial) -> float:
    p = sample_params(trial)

    train_loader, val_loader, _, weights = load_data(
        DATA_PATH,
        augment_strategy=p["augment_strategy"],
        augment_proportion=p["augment_proportion"],
        batch_size=BATCH_SIZE,
    )

    model = LightningGNN(
        **FIXED,
        num_layers=p["num_layers"],
        GNNLayer=p["GNNLayer"],
        mlp_hidden=p["mlp_hidden"],
        lr=p["lr"],
        wd=p["wd"],
        loss_type=p["loss_type"],
        class_weights=weights["class_weights"],
    )

    trial_dir = ROOT / f"trial_{trial.number}"
    ckpt_dir = trial_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch={epoch:03d}_val_bacc={val_bacc:.4f}",
        monitor="val_bacc",
        mode="max",
        save_last=True,
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=125,                      # pruner truncates early if needed
        default_root_dir=str(trial_dir),
        callbacks=[ckpt_cb],
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
    )

    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=ckpt_cb.last_model_path if ckpt_cb.last_model_path else None,
    )

    return trainer.callback_metrics["val_bacc"].item()

# ╭───────────────────── 4.  STUDY CONFIG ──────────────────────╮
storage = optuna.storages.RDBStorage(
    url=f"sqlite:///{STUDY_DB}",
    engine_kwargs={"connect_args": {"timeout": 60}},
)

study = optuna.create_study(
    study_name="gnn_sleep",
    direction="maximize",
    sampler=TPESampler(multivariate=True, seed=SEED),
    pruner=SuccessiveHalvingPruner(min_resource=5, reduction_factor=5),
    storage=storage,
    load_if_exists=True,        # ← auto-resume if interrupted
)

# ╭───────────────────── 5.  RUN ───────────────────────────────╮
if __name__ == "__main__":
    study.optimize(objective, n_trials=200, gc_after_trial=True)

    print("Best balanced-accuracy:", study.best_value)
    print("Best parameters:", study.best_params)

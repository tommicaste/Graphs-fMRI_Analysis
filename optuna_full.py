from pathlib import Path
import json, torch
import optuna, pytorch_lightning as pl
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from tqdm.auto import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.nn import SAGEConv, GATConv, GCNConv
from data import load_data
from static_models.gnn import LightningGNN

# ───────────────────── custom Optuna progress bar ───
class TqdmCallback:
    def __init__(self, total):
        self.pbar = tqdm(total=total, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")

    def __call__(self, study, trial):
        self.pbar.update(1)

# ───────────────────── 0. SETUP ─────────────────────
ROOT = Path("non_overlapping_top_connections_optuna")
ROOT.mkdir(parents=True, exist_ok=True)
DB_FILE = ROOT / "optuna_pipeline.db"

SEED, BATCH_SIZE, WORKERS = 12, 48, 3
DATA_PATH = Path("/project2/cdonnat/sleepstages/data/pt/non_overlapping_data.pt")
pl.seed_everything(SEED, workers=True)

FIXED_DATA = dict(batch_size=BATCH_SIZE, workers=WORKERS,
                  train_ratio=0.65, val_ratio=0.20, test_ratio=0.15)

FIXED_MODEL = dict(
    input_dim=347,
    dropout=0.5,
    num_classes=4,
    edge_tsh=None,
    residual_connections=True,
)

BACKBONES = {"SAGEConv": SAGEConv, "GATConv": GATConv, "GCNConv": GCNConv}

# ───────────────────── pruning callback ─────────────
class LightningPruningCallback(pl.callbacks.Callback):
    def __init__(self, trial, monitor):
        self.trial = trial
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        if isinstance(current, torch.Tensor):
            current = current.item()
        self.trial.report(float(current), step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()

# ───────────────────── 1. STAGE-1 SPACE ────────────
def sample_stage1(trial):
    return dict(
        edge_top        = trial.suggest_categorical("edge_top", [0.10, 0.15, 0.20, 0.25, 0.30]),
        GNNLayer        = BACKBONES[trial.suggest_categorical("backbone", list(BACKBONES))],
        num_layers      = trial.suggest_int("num_layers", 1, 3),
        augment_strategy   = trial.suggest_categorical("augment_strategy", ["interpolate", "geodesic"]),
        augment_proportion = trial.suggest_categorical("augment_proportion",
                                [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        hidden_channels = 64,
        pooling_fn      = "mean",
        mlp_hidden      = [64, 32],
        loss_type       = trial.suggest_categorical("loss_type",
                                                   ["cross_entropy", "weighted_cross_entropy"]),
        lr              = trial.suggest_categorical("lr", [1e-4, 3e-4, 1e-3, 3e-3]),
        wd              = trial.suggest_categorical("wd", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
        edge_dropout    = 0.0,
    )

# ───────────────────── 2. STAGE-2 SPACE (CORRECTED) ────────────
def sample_stage2(trial, arch):
    override = {"lr", "wd", "hidden_channels", "edge_dropout", "pooling_fn", "mlp_hidden"}
    
    base = {k: v for k, v in arch.items() if k not in override and k != 'backbone'}
    
    return dict(
        **base,
        GNNLayer        = BACKBONES[arch["backbone"]],
        hidden_channels = trial.suggest_categorical("hidden_channels", [64, 96, 128]),
        edge_dropout    = trial.suggest_float("edge_dropout", 0.1, 0.5),
        pooling_fn      = trial.suggest_categorical("pooling_fn", ["mean", "sum", "max"]),
        mlp_hidden      = trial.suggest_categorical("mlp_hidden", [[64, 32], [64]]),
        lr              = trial.suggest_float("lr", arch["lr"] / 2, arch["lr"] * 2, log=True),
        wd              = trial.suggest_float("wd", arch["wd"] / 5, arch["wd"] * 5, log=True),
    )

# ───────────────────── 3. OBJECTIVE ────────────────
def make_objective(space_fn):
    def objective(trial):
        hp = space_fn(trial)
        
        train_loader, val_loader, _, weights = load_data(
            DATA_PATH,
            augment_strategy   = hp["augment_strategy"],
            augment_proportion = hp["augment_proportion"],
            **FIXED_DATA)
        
        model = LightningGNN(
            **FIXED_MODEL,
            edge_top        = hp["edge_top"],
            hidden_channels = hp["hidden_channels"],
            num_layers      = hp["num_layers"],
            GNNLayer        = hp["GNNLayer"],
            pooling_fn      = hp["pooling_fn"],
            mlp_hidden      = hp["mlp_hidden"],
            lr              = hp["lr"],
            wd              = hp["wd"],
            loss_type       = hp["loss_type"],
            class_weights   = weights["class_weights"],
            edge_dropout    = hp["edge_dropout"],
        )
            
        run_dir = ROOT / "runs" / f"{trial.study.study_name}_{trial.number}"
        ckpt_cb = ModelCheckpoint(dirpath=run_dir, monitor="val_bacc",
                                  mode="max", save_last=True, save_top_k=1)
        pruning_cb = LightningPruningCallback(trial, monitor="val_bacc")
        trainer = pl.Trainer(max_epochs=40, callbacks=[ckpt_cb, pruning_cb],
                             accelerator="auto", devices="auto", enable_progress_bar=False)
        
        trainer.fit(model, train_loader, val_loader)
        
        return trainer.callback_metrics["val_bacc"].item()
    return objective

# ───────────────────── 4. RUNNER UTIL ───────────────
def run_stage(name, sampler_fn, n_trials, pruner):
    study = optuna.create_study(
        study_name=name, direction="maximize",
        storage=f"sqlite:///{DB_FILE}",
        sampler=TPESampler(multivariate=True, seed=SEED),
        pruner=pruner, load_if_exists=True)
    finished = sum(t.state.is_finished() for t in study.trials)
    if finished < n_trials:
        study.optimize(
            make_objective(sampler_fn),
            n_trials=n_trials - finished,
            gc_after_trial=True,
            callbacks=[TqdmCallback(total=n_trials - finished)]
        )
    return study

# ───────────────────── 5. STAGE-1 ───────────────────
stage1 = run_stage("stage1_arch", sample_stage1, 160,
                   SuccessiveHalvingPruner(min_resource=5, reduction_factor=5))
best_arch = stage1.best_trial.params
print("\nStage-1 best:", best_arch, "val_bacc:", stage1.best_value)

# ───────────────────── 6. STAGE-2 ───────────────────
stage2 = run_stage("stage2_reg",
                   lambda tr: sample_stage2(tr, best_arch), 40,
                   SuccessiveHalvingPruner(min_resource=5, reduction_factor=4))
print("\nStage-2 best add-ons:", stage2.best_trial.params)
print("Stage-2 best val_bacc:", stage2.best_value)

# ───────────────────── 7. SAVE FINAL CONFIG ─────────
combined = {**best_arch, **stage2.best_trial.params}
(Path(ROOT) / "best_model_config.json").write_text(json.dumps(combined, indent=2))
print("\nFinal best config saved to", ROOT / "best_model_config.json")
import argparse
import copy
import yaml
from pathlib import Path
import datetime

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from static_models.pipeline import run_from_config

# ---- Helper to set nested value by dotted key ----
def set_by_dotted_path(d, dotted, value):
    keys = dotted.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value

# ---- Search Space Definition ----
def suggest_params(trial: optuna.Trial) -> dict:
    """Return a dictionary mapping dotted keys to suggested values."""
    suggestions = {}
    
    # Data Augmentation Strategy - choose either upsample OR interpolate
    augment_strategy = trial.suggest_categorical("augment_strategy", [
        "upsample",  # simple duplication
        "interpolate"  # correlation‐matrix interpolation
    ])
    
    if augment_strategy == "upsample":
        suggestions["data.augment_cfg.upsample.proportion"] = trial.suggest_categorical("upsample_prop", [0.3, 0.6, 0.9])
    
    if augment_strategy == "interpolate":
        suggestions["data.augment_cfg.interpolate.proportion"] = trial.suggest_categorical("interpolate_prop", [0.3, 0.6, 0.9])
    
     # ── Edge-list construction: choose ONE strategy ────────────────────────────
    edge_mode = trial.suggest_categorical("edge_mode", ["tsh", "top"])

    if edge_mode == "tsh":
        # Hard correlation threshold
        suggestions["data.edge_tsh"] = trial.suggest_categorical(
            "edge_tsh", [0.15, 0.20]
        )
        suggestions["data.edge_top"] = None          # ignored downstream
    else:
        # Top-k percentile selection
        suggestions["data.edge_tsh"] = None          # disables thresholding
        suggestions["data.edge_top"] = trial.suggest_categorical(
            "edge_top", [0.15, 0.20]
        )

    # GNN Architecture
    suggestions["model.params.gnn_layer"] = trial.suggest_categorical("gnn_layer", ["SAGEConv", "GCNConv", "GATConv"])
    suggestions["model.params.hidden_channels"] = trial.suggest_categorical("hidden_channels", [64, 128])
    suggestions["model.params.num_layers"] = trial.suggest_int("num_layers", 1, 3)
    suggestions["model.params.dropout"] = trial.suggest_categorical("dropout", [0.0, 0.5])

    # MLP Head
    mlp_hidden_options = [[64, 32], [32, 32], [64, 64]]
    suggestions["model.params.mlp_hidden"] = trial.suggest_categorical("mlp_hidden", [str(opt) for opt in mlp_hidden_options])

    # Optimizer
    suggestions["model.params.lr"] = trial.suggest_float("lr", 1e-3, 0.1, log=True)
    
    return suggestions

# ---- Objective Function ----
def objective(trial: optuna.Trial, base_cfg: dict) -> float:
    """The Optuna objective function."""
    
    # 1. Create a trial-specific config from suggested parameters
    cfg = copy.deepcopy(base_cfg)
    trial_params = suggest_params(trial)
    for key, value in trial_params.items():
        # Handle MLP hidden which is a stringified list
        if key == "model.params.mlp_hidden":
            value = eval(value)
        set_by_dotted_path(cfg, key, value)
        
    # 2. Create the pruning callback to be passed to the pipeline
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_BAcc")
    
    # 3. Run the experiment, passing the config and callback separately
    score = run_from_config(cfg, extra_callbacks=[pruning_callback])
    
    return score

# -------------------------
# Utility: default SQLite file in working dir
# -------------------------
def _default_storage_path(study_name: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d")
    db_name = f"{study_name}.db"
    return f"sqlite:///{db_name}"

# ---- Custom Callback to delay pruning ----
class DelayedPruningCallback(PyTorchLightningPruningCallback):
    """Wrapper that skips pruning decisions for the first `delay_epochs` validation epochs."""

    def __init__(self, trial, monitor: str, *, delay_epochs: int = 3, **kwargs):
        super().__init__(trial, monitor, **kwargs)
        self.delay_epochs = delay_epochs

    def on_validation_end(self, trainer, pl_module):  # type: ignore[override]
        # Skip pruning during the warm-up epochs
        if pl_module.current_epoch < self.delay_epochs:
            return
        super().on_validation_end(trainer, pl_module)

# ---- Main Execution ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an Optuna hyperparameter search.")
    parser.add_argument("--config", default="config.yaml", help="Path to the base YAML configuration file.")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials to run.")
    parser.add_argument("--study_name", default="gnn-hyperparameter-search", help="Name for the Optuna study.")
    parser.add_argument("--storage", default=None, help="Optuna storage URL (e.g., sqlite:///my.db). Defaults to <study_name>.db in current dir.")
    args = parser.parse_args()

    # Resolve storage
    storage_url = args.storage or _default_storage_path(args.study_name)

    # Load base config
    base_config = yaml.safe_load(Path(args.config).read_text())

    # Create and run the study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        load_if_exists=True, # Allows resuming studies
    )
    
    study.optimize(lambda trial: objective(trial, base_config), n_trials=args.trials, gc_after_trial=True)

    # Print results
    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    
    print("Best trial:")
    best = study.best_trial
    print(f"  Value: {best.value:.4f}")
    
    print("  Params: ")
    for key, value in best.params.items():
        print(f"    {key}: {value}")

    # Persist a CSV summary of all trials for convenience
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    csv_path = Path(f"{args.study_name}_trials.csv")
    df.to_csv(csv_path, index=False)
    print(f"Trials CSV saved to {csv_path}") 
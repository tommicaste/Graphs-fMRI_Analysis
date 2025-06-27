from __future__ import annotations

import argparse
import copy
import importlib
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import optuna
import pytorch_lightning as pl
import torch
import yaml
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

def _resolve_gnn_layer(layer_name: str):
    """Convert string like 'SAGEConv' -> torch_geometric.nn.SAGEConv class."""
    tg_nn = importlib.import_module("torch_geometric.nn")
    if not hasattr(tg_nn, layer_name):
        raise ValueError(f"Unknown GNN layer '{layer_name}' in torch_geometric.nn")
    return getattr(tg_nn, layer_name)


def _inject_model_specific(
    arch: str,
    params: Dict[str, Any],
    trial: optuna.Trial,
) -> Dict[str, Any]:
    """Sample model-specific hyper-parameters and write them into param dict."""
    p = params.copy()

    if arch == "gnn":
        # GNN layer type
        layer_name = trial.suggest_categorical("gnn_layer", ["SAGEConv", "GCNConv", "GATConv"])
        p["GNNLayer"] = _resolve_gnn_layer(layer_name)

        # Edge rule: choose exactly one of top-k or threshold or none
        edge_rule = trial.suggest_categorical("edge_rule", ["top", "threshold"])
        if edge_rule == "top":
            p["edge_top"] = trial.suggest_float("edge_top", 0.05, 0.20)
            p.pop("edge_tsh", None)
        elif edge_rule == "threshold":
            p["edge_tsh"] = trial.suggest_float("edge_tsh", 0.0, 0.3)
            p.pop("edge_top", None)

        # Pooling rule
        pooling = trial.suggest_categorical("pooling", ["mean", "sort"])
        p["pooling_fn"] = pooling
        if pooling == "sort":
            p["sort_pool_k"] = trial.suggest_int("sort_k", 10, 30)

        # Message-passing depth and width
        p["hidden_channels"] = trial.suggest_categorical("hidden_channels", [64, 128])
        p["num_layers"] = trial.suggest_int("num_layers", 1, 3)

    elif arch == "mlp":
        # Hidden layer width & depth; mimic parameter budget of the GNN
        width = trial.suggest_categorical("hidden_units", [64, 128])
        depth = trial.suggest_int("num_layers", 2, 3)
        # lightning MLP expects a list under the key `hidden_dims`
        p["hidden_dims"] = [width] * depth

    elif arch == "logistic":
        # No additional architecture-specific HPs at the moment
        pass

    return p


def _sample_global(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample hyper-parameters shared across architectures."""
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "wd": trial.suggest_float("wd", 1e-6, 1e-2, log=True),
        # imbalance / augmentation knobs
        "imbalance": trial.suggest_categorical("imbalance", ["none", "weighted"]),
        "augment_strategy": trial.suggest_categorical("aug_strategy", ["none", "geodesic", "interpolate"]),
        "augment_proportion": trial.suggest_float("aug_prop", 0.3, 1),
    }


# ────────────────────────────────────────────────────────────────────────────────
# Objective function
# ────────────────────────────────────────────────────────────────────────────────

def make_objective(base_cfg: Dict[str, Any], model_idx: int = 0):
    """Return a closure usable as Optuna objective."""

    # Heavy import inside so that optuna workers without GPU memory can load quickly
    from static_models.pipeline import train_model

    def _objective(trial: optuna.Trial):
        cfg = copy.deepcopy(base_cfg)

        # Global seeds for reproducibility
        seed = cfg.get("seed", 123)
        seed = seed + trial.number  # perturb per trial
        pl.seed_everything(seed, workers=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Extract model dict we will mutate
        model_dict = cfg["models"][model_idx]
        arch = model_dict["architecture"]
        params = model_dict.get("params", {})

        # ─── inject shared hyper-parameters ───
        shared = _sample_global(trial)

        # Select only the hyper-parameters that are accepted by the current
        # architecture to avoid passing unexpected keywords to the model
        if arch == "gnn":
            param_keys = ("lr", "wd")
        elif arch == "mlp":
            param_keys = ("lr", "wd")
        else:  # logistic or others that only take learning-rate
            param_keys = ("lr",)

        params.update({k: shared[k] for k in param_keys})

        # Always set dropout to 0.5 for GNN and MLP if not present
        if arch in {"gnn", "mlp"} and "dropout" not in params:
            params["dropout"] = 0.5

        # imbalance and aug settings are stored under data section
        data_cfg = cfg["data"]
        if shared["imbalance"] == "weighted":
            params["loss_type"] = "weighted_cross_entropy"
        else:
            params["loss_type"] = "cross_entropy"
        data_cfg.setdefault("augment", {})
        data_cfg["augment"]["strategy"] = (
            None if shared["augment_strategy"] == "none" else shared["augment_strategy"]
        )
        data_cfg["augment"]["proportion"] = shared["augment_proportion"]

        # ─── model-specific sampling ───
        params = _inject_model_specific(arch, params, trial)
        # Remove gnn_layer key if present (only GNNLayer should be passed)
        if arch == "gnn" and "gnn_layer" in params:
            params.pop("gnn_layer")
        model_dict["params"] = params

        # Set a moderate training duration for HPO
        model_dict.setdefault("trainer", {})
        model_dict["trainer"]["max_epochs"] =70

        # Build save directory per trial under the study name
        root = Path(cfg.get("save_dir", "runs"))
        trial_dir = root / f"optuna_{trial.study.study_name}" / f"trial_{trial.number}"
        cfg["save_dir"] = str(trial_dir)

        # Call train_model directly (mirrors logic in main.py)
        trainer, _ = train_model(
            model=arch,
            run_name=f"trial_{trial.number}",
            data_path=cfg["data"]["path"],
            save_dir=trial_dir,
            model_kwargs=params,
            augment_strategy=data_cfg["augment"].get("strategy"),
            augment_proportion=data_cfg["augment"].get("proportion"),
            batch_size=data_cfg.get("batch_size", 32),
            workers=data_cfg.get("num_workers", 0),
            train_ratio=cfg["data"]["split"][0],
            val_ratio=cfg["data"]["split"][1],
            test_ratio=cfg["data"]["split"][2],
            max_epochs=model_dict["trainer"]["max_epochs"],
            accelerator="auto",
            devices=1,
            log_every_n_steps=10,
            trainer_kwargs={},
            seed=seed,
        )

        # Use validation macro-balanced accuracy as objective (higher is better).
        # The values are stored in `trainer.callback_metrics` after training.
        val_bacc = trainer.callback_metrics.get("val_bacc")
        val_acc = trainer.callback_metrics.get("val_acc")
        test_bacc = trainer.callback_metrics.get("test_bacc")
        test_acc = trainer.callback_metrics.get("test_acc")
        val_loss = trainer.callback_metrics.get("val_loss")

        # Fallback order for objective: val_bacc, val_acc, test_bacc, test_acc
        metric = val_bacc if val_bacc is not None else (
            val_acc if val_acc is not None else (
                test_bacc if test_bacc is not None else test_acc
            )
        )
        if metric is None:
            raise RuntimeError("Could not find any suitable accuracy metric in callback metrics.")

        # Retrieve loss for pruning – prefer validation loss, else test_loss, else skip
        if val_loss is None:
            val_loss = trainer.callback_metrics.get("test_loss")

        # Convert possible tensors to a plain float then report to Optuna
        if isinstance(metric, torch.Tensor):
            metric = metric.item()
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()

        # If we have a loss value, use it for pruning
        if val_loss is not None:
            trial.report(val_loss, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned at loss={val_loss}")

        # Optuna minimises → negate to maximise validation accuracy
        return 1.0 - float(metric)

    return _objective


# ────────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ────────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Universal Optuna sweep launcher")
    parser.add_argument("config", type=Path, help="Path to base YAML config")
    parser.add_argument("--study", type=str, default="optuna_study", help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (optional)")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed")
    args = parser.parse_args()

    base_cfg = yaml.safe_load(args.config.read_text())
    base_cfg["seed"] = args.seed

    sampler = TPESampler(seed=args.seed)
    pruner = SuccessiveHalvingPruner()

    study = optuna.create_study(
        study_name=args.study,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
    )

    objective = make_objective(base_cfg)
    study.optimize(objective, n_trials=args.trials, timeout=None, n_jobs=1, gc_after_trial=True)

    print("Best value:", study.best_value)
    print("Best params:", study.best_trial.params)


if __name__ == "__main__":
    main()

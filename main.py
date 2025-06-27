from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from static_models.pipeline import train_model

if len(sys.argv) > 1:
    CONFIG_PATH = Path(sys.argv[1]).expanduser().resolve()
else:
    CONFIG_PATH = Path(__file__).with_name("config.yaml")

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")


def _resolve_gnn_layer(layer_name: str):
    """Convert string like 'SAGEConv' -> torch_geometric.nn.SAGEConv class."""
    tg_nn = importlib.import_module("torch_geometric.nn")
    if not hasattr(tg_nn, layer_name):
        raise ValueError(f"Unknown GNN layer '{layer_name}' in torch_geometric.nn")
    return getattr(tg_nn, layer_name)


def _prepare_model_kwargs(arch: str, params: Dict[str, Any], data_cfg: Dict[str, Any]):
    """Process params dict: resolve layer strings, inject edge config."""
    params = params.copy() 

    # Harmonise key name for GNN layer
    if arch == "gnn":
        if "gnn_layer" in params:
            layer_val = params.pop("gnn_layer")
            if isinstance(layer_val, str):
                layer_val = _resolve_gnn_layer(layer_val)
            params["GNNLayer"] = layer_val
        # Inject edge sparsification parameters if provided globally
        edge_cfg = data_cfg.get("edge", {})
        if edge_cfg.get("top") is not None:
            params.setdefault("edge_top", edge_cfg["top"])
        if edge_cfg.get("tsh") is not None:
            params.setdefault("edge_tsh", edge_cfg["tsh"])
    return params


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())

    # ─── global seed ───
    seed = cfg.get("seed", 123)
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ─── shared data settings ───
    data_cfg = cfg["data"]
    data_path = data_cfg["path"]
    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 0)
    train_ratio, val_ratio, test_ratio = data_cfg.get("split", [0.7, 0.2, 0.1])

    augment_cfg = data_cfg.get("augment", {}) or {}
    augment_strategy = augment_cfg.get("strategy") or None
    augment_proportion = augment_cfg.get("proportion")

    save_root = Path(cfg.get("save_dir", "runs"))

    # ─── iterate over models ───
    for idx, m in enumerate(cfg.get("models", []), start=1):
        name = m.get("name", f"model_{idx}")
        arch = m["architecture"]
        params = _prepare_model_kwargs(arch, m.get("params", {}), data_cfg)

        # Ensure weight decay is present for models that support it
        if arch in {"gnn", "mlp"} and "wd" not in params:
            params["wd"] = 1e-3  # sensible default

        trainer_cfg = m.get("trainer", {})
        max_epochs = trainer_cfg.pop("max_epochs", 50)
        accelerator = trainer_cfg.pop("accelerator", "auto")
        devices = trainer_cfg.pop("devices", 1)
        log_every_n_steps = trainer_cfg.pop("log_every_n_steps", 10)

        run_dir = save_root / name

        print(f"\n======== Training '{name}' ({arch}) ========")
        trainer, metrics = train_model(
            model=arch,
            run_name=name,
            data_path=data_path,
            save_dir=run_dir,
            model_kwargs=params,
            # data args
            augment_strategy=augment_strategy,
            augment_proportion=augment_proportion,
            batch_size=batch_size,
            workers=num_workers,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            # trainer args
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            log_every_n_steps=log_every_n_steps,
            trainer_kwargs=trainer_cfg, 
            seed=seed,
        )
        print("Test metrics:", metrics)


if __name__ == "__main__":
    main()

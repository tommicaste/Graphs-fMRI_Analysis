from __future__ import annotations
import importlib, os, random, shutil, json, hashlib, copy
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers   import CSVLogger
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

from data.datamodule import StaticDataModule
from static_models.gnn import LightningGNN   # current only supported model

# ────────────────────────────────────────────────
# 1. Model factory  (extendable)
# ────────────────────────────────────────────────
def _build_gnn(cfg: Dict) -> LightningGNN:
    """Build a LightningGNN from a config dict."""
    
    # Convert GNN layer string to class
    gnn_layer_str = cfg["model"]["params"]["gnn_layer"]
    gnn_layer_class = getattr(importlib.import_module("torch_geometric.nn"), gnn_layer_str)
    
    model_params = cfg["model"]["params"].copy()
    model_params["gnn_layer"] = gnn_layer_class # Pass the class to the model
    
    return LightningGNN(
        input_dim=model_params["input_dim"],
        hidden_channels=model_params["hidden_channels"],
        num_layers=model_params["num_layers"],
        GNNLayer=model_params["gnn_layer"],
        dropout=model_params["dropout"],
        num_classes=model_params["num_classes"],
        mlp_hidden=model_params["mlp_hidden"],
        lr=model_params["lr"],
    )

MODEL_REGISTRY = {"gnn": _build_gnn}

# ────────────────────────────────────────────────
# 2. Main runner class
# ────────────────────────────────────────────────
class ExperimentPipeline:
    """
    End-to-end experiment handler: data → model → trainer → results.
    """

    def __init__(self, cfg: Dict, extra_callbacks: Optional[List] = None):
        self.cfg = cfg
        self.seed          = cfg["seed"]
        self.arch          = cfg["model"]["architecture"]

        # Resolve GNN layer string to class before hashing
        run_cfg = copy.deepcopy(cfg)
        if run_cfg["model"]["architecture"] == "gnn":
            layer_name = run_cfg["model"]["params"]["gnn_layer"]
            # using a placeholder string for the class itself for hashing
            run_cfg["model"]["params"]["gnn_layer"] = str(globals().get(layer_name))

        # Build unique run folder name from configuration (stable short hash)
        hash_source = {k: v for k, v in run_cfg.items() if k != "save_dir"}
        param_hash = hashlib.sha1(json.dumps(hash_source, sort_keys=True).encode()).hexdigest()[:8]
        self.run_root = Path(cfg["save_dir"]) / self.arch / param_hash

        # (Re)create fresh folder every run → guarantees "delete & retrain" semantics
        if self.run_root.exists():
            shutil.rmtree(self.run_root)
        (self.run_root / "checkpoints").mkdir(parents=True, exist_ok=True)

        # 2.1 set seeds
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        pl.seed_everything(self.seed, workers=True)

        # 2.2 datamodule
        datamodule_kwargs = {
            "raw_path":     cfg["data"]["path"],
            "batch_size":   cfg["data"]["batch_size"],
            "num_workers":  cfg["data"]["num_workers"],
            "split_ratios": tuple(cfg["data"]["split"]),
            "augment_cfg":  cfg["data"].get("augment_cfg", {}),
            "edge_tsh":     cfg["data"].get("edge_tsh", 0.0),
            "seed":         self.seed,
        }
        if "cache_path" in cfg["data"]:
            datamodule_kwargs["cache_path"] = cfg["data"]["cache_path"]

        self.dm = StaticDataModule(**datamodule_kwargs)
        self.dm.prepare_data(); self.dm.setup("fit"); self.dm.setup("test")

        # 2.3 model
        if self.arch not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported architecture: {self.arch}")
        self.model = MODEL_REGISTRY[self.arch](self.cfg)

        # 2.4 logger / callbacks / trainer
        self._build_trainer(extra_callbacks)

    # --------------------------------------------------
    def _build_trainer(self, extra_callbacks: Optional[List] = None):
        logger = CSVLogger(save_dir=str(self.run_root), name="logs", version="")
        ckpt_cb = ModelCheckpoint(
            monitor    = "val_BAcc",
            mode       = "max",
            dirpath    = self.run_root / "checkpoints",
            filename   = "best.ckpt",
            save_top_k = 1,
        )
        es_cb = EarlyStopping(
            monitor  = "val_BAcc",
            mode     = "max",
            patience = self.cfg["trainer"].pop("early_stop_patience"),
        )

        callbacks_basic = [ckpt_cb, es_cb]

        # Build the trainer first with only Lightning-native callbacks
        self.trainer = pl.Trainer(
            logger    = logger,
            callbacks = callbacks_basic,
            **self.cfg["trainer"],
        )
        self._ckpt_cb = ckpt_cb           # stash for later

        # Safely extend with any user-provided callbacks *after* instantiation
        if extra_callbacks:
            # Attempt to append all provided callbacks; warn if class mismatch
            for cb in extra_callbacks:
                if isinstance(cb, Callback):
                    self.trainer.callbacks.append(cb)
                else:
                    print(f"[WARN] Provided extra_callback is not a Lightning Callback (type={type(cb)}). Adding it anyway may fail.")
                    self.trainer.callbacks.append(cb)

    # --------------------------------------------------
    def run(self):
        """Fit, test on last epoch, then test with best checkpoint."""
        self.trainer.fit(self.model, self.dm)
        self.trainer.test(self.model, self.dm)

        best_score = float(self._ckpt_cb.best_model_score) if self._ckpt_cb.best_model_score is not None else float('nan')

        best_ckpt = self._ckpt_cb.best_model_path
        
        # Load the best model using the same dynamic build process
        model_class = self.model.__class__
        best_model = model_class.load_from_checkpoint(best_ckpt)
        
        best_model.test_model(
            best_ckpt,
            self.dm.test_dataloader(),
            self.run_root,
        )

        print(f"✅  Training + testing complete. Artefacts in: {self.run_root}  (val_BAcc={best_score:.4f})")

        return best_score

# ────────────────────────────────────────────────
# 3. Helper entry-point (optional)
# ────────────────────────────────────────────────
def run_from_config(cfg: Dict, extra_callbacks: Optional[List] = None):
    """Convenience one-liner:  run_from_config(yaml.safe_load(...))"""
    exp = ExperimentPipeline(cfg, extra_callbacks=extra_callbacks)
    return exp.run()

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn

# Data loading
from data import load_data

# Model
from static_models.gnn import LightningGNN

# ---------------------------------------------------------------------------
# Public API ----------------------------------------------------------------
# ---------------------------------------------------------------------------

# ONLY ONE entry-point is exposed – easy to maintain.
__all__ = ["train_model"]

# ----------------------------------------------------------------------------
# Generic training utility ----------------------------------------------------
# ----------------------------------------------------------------------------

from static_models.mlp import LightningMLP  # noqa: E402
from static_models.logistic import LightningLogisticRegression  # noqa: E402

MODEL_REGISTRY: dict[str, type[pl.LightningModule]] = {
    "gnn": LightningGNN,
    "mlp": LightningMLP,
    "logistic": LightningLogisticRegression,
}


def _default_save_dir(run_name: str) -> Path:
    """Utility that returns a run directory under <project_root>/run/<run_name>."""
    root = Path(__file__).resolve().parent.parent  # .. / sleepstages
    return root / "run" / run_name


def train_model(
    *,
    model: str | type[pl.LightningModule],
    run_name: str,
    data_path: str,
    model_kwargs: Optional[dict] = None,
    # ---------- data args (see data.load_data) ----------
    augment_strategy: Optional[str] = None,
    augment_proportion: Optional[float] = None,
    batch_size: int = 64,
    workers: int = 3,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    # ---------- trainer / bookkeeping ----------
    save_dir: Optional[Union[str, Path]] = None,
    seed: int = 123,
    max_epochs: int = 50,
    accelerator: str = "auto",
    devices: Union[int, str, List[int]] = 1,
    log_every_n_steps: int = 10,
    trainer_kwargs: Optional[dict] = None,
):
    """Train & test any registered Lightning model in *one* call.

    Parameters
    ----------
    model : str | pl.LightningModule subclass
        Either a key in {"gnn", "mlp", "logistic"} or a custom subclass.
    model_kwargs : dict, optional
        Passed directly to the model constructor (after some smart defaults).
    All other arguments are identical to `train_gnn`, but apply to every model.
    """

    # 1. resolve model class ---------------------------------------------------
    if isinstance(model, str):
        if model not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key '{model}'. Available: {list(MODEL_REGISTRY)}")
        model_cls = MODEL_REGISTRY[model]
    else:
        model_cls = model  # type: ignore[assignment]

    if model_kwargs is None:
        model_kwargs = {}

    # Provide sensible defaults for GNN specific arguments if missing ---------
    if model_cls is LightningGNN:
        # default GNNLayer
        model_kwargs.setdefault("GNNLayer", __import__("torch_geometric.nn").torch_geometric.nn.SAGEConv)  # type: ignore[index]
        model_kwargs.setdefault("mlp_hidden", [64, 32])
        model_kwargs.setdefault("input_dim", 347)
        model_kwargs.setdefault("hidden_channels", 32)
        model_kwargs.setdefault("num_layers", 3)
        model_kwargs.setdefault("num_classes", 4)
    elif model_cls is LightningMLP:
        model_kwargs.setdefault("hidden_dims", [128, 64])
        model_kwargs.setdefault("input_dim", 347)
        model_kwargs.setdefault("num_classes", 4)
    elif model_cls is LightningLogisticRegression:
        model_kwargs.setdefault("input_dim", 347)
        model_kwargs.setdefault("num_classes", 4)

    # 2. seed everything -------------------------------------------------------
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 3. data -----------------------------------------------------------------
    train_loader, val_loader, test_loader, weight_dict = load_data(
        data_path,
        batch_size=batch_size,
        workers=workers,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        augment_strategy=augment_strategy,
        augment_proportion=augment_proportion,
    )

    # 4. auto-inject class_weights if needed ----------------------------------
    if model_kwargs.get("loss_type") == "weighted_cross_entropy" and "class_weights" not in model_kwargs:
        model_kwargs["class_weights"] = weight_dict["class_weights"]

    # 5. instantiate model ----------------------------------------------------
    model_instance = model_cls(**model_kwargs)  # type: ignore[arg-type]

    # 6. trainer / callbacks ---------------------------------------------------
    if save_dir is None:
        save_dir = _default_save_dir(run_name)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=save_dir / "checkpoints",
        filename="epoch={epoch:02d}_val_bacc={val_bacc:.4f}",
        monitor="val_bacc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    if trainer_kwargs is None:
        trainer_kwargs = {}

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        default_root_dir=str(save_dir),
        callbacks=[ckpt_cb],
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=True,
        **trainer_kwargs,
    )

    # 7. fit & test -----------------------------------------------------------
    trainer.fit(model_instance, train_loader, val_loader)
    test_metrics = trainer.test(ckpt_path="best", dataloaders=test_loader)

    return trainer, test_metrics

# ----------------------------------------------------------------------------
# Remove obsolete wrappers to enforce a *single* API. -------------------------
# ----------------------------------------------------------------------------

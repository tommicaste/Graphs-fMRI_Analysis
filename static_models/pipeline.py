from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn

from data import load_data

from static_models.gnn import LightningGNN
from static_models.neurograph import NeurographGNN
from static_models.bnt import BNT_Lightning
from static_models.weighted_gnn import WeightedGNN

__all__ = ["train_model"]


from static_models.mlp import LightningMLP
from static_models.logistic import LightningLogisticRegression

MODEL_REGISTRY: dict[str, type[pl.LightningModule]] = {
    "gnn": LightningGNN,
    "weighted_gnn": WeightedGNN,
    "mlp": LightningMLP,
    "logistic": LightningLogisticRegression,
    "neurograph": NeurographGNN,
    "bnt": BNT_Lightning,
}


def _default_save_dir(run_name: str) -> Path:
    """Utility that returns a run directory under <project_root>/run/<run_name>."""
    root = Path(__file__).resolve().parent.parent 
    return root / "run" / run_name


def train_model(
    *,
    model: str | type[pl.LightningModule],
    run_name: str,
    data_path: str,
    model_kwargs: Optional[dict] = None,

    augment_strategy: Optional[str] = None,
    augment_proportion: Optional[float] = None,
    batch_size: int = 64,
    workers: int = 3,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,

    save_dir: Optional[Union[str, Path]] = None,
    seed: int = 123,
    max_epochs: int = 50,
    accelerator: str = "auto",
    devices: Union[int, str, List[int]] = 1,
    log_every_n_steps: int = 10,
    trainer_kwargs: Optional[dict] = None,
):

    
    if isinstance(model, str):
        if model not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key '{model}'. Available: {list(MODEL_REGISTRY)}")
        model_cls = MODEL_REGISTRY[model]
    else:
        model_cls = model  

    if model_kwargs is None:
        model_kwargs = {}

    
    if model_cls is LightningGNN:
        
        model_kwargs.setdefault("GNNLayer", __import__("torch_geometric.nn").torch_geometric.nn.SAGEConv)
        model_kwargs.setdefault("mlp_hidden", [64, 32])
        model_kwargs.setdefault("input_dim", 347)
        model_kwargs.setdefault("hidden_channels", 32)
        model_kwargs.setdefault("num_layers", 3)
        model_kwargs.setdefault("num_classes", 4)
    elif model_cls is WeightedGNN:
        model_kwargs.setdefault("GNNLayer", __import__("torch_geometric.nn").torch_geometric.nn.GCNConv)
        model_kwargs.setdefault("mlp_hidden", [64, 32])
        model_kwargs.setdefault("input_dim", 347)
        model_kwargs.setdefault("hidden_channels", 32)
        model_kwargs.setdefault("num_layers", 3)
        model_kwargs.setdefault("num_classes", 4)
        model_kwargs.setdefault("feature_type", "corr")
    elif model_cls is NeurographGNN:
        model_kwargs.setdefault("GNNLayer", __import__("torch_geometric.nn").torch_geometric.nn.SAGEConv)
        model_kwargs.setdefault("hidden_channels", 64)
        model_kwargs.setdefault("hidden", 128)
        model_kwargs.setdefault("input_dim", 347)
        model_kwargs.setdefault("num_layers", 2)
        model_kwargs.setdefault("num_classes", 4)
        model_kwargs.setdefault("loss_type", "cross_entropy")
    elif model_cls is BNT_Lightning:
        model_kwargs.setdefault("input_dim", 26)
        model_kwargs.setdefault("num_classes", 4)
        model_kwargs.setdefault("n_layers", 2)
        model_kwargs.setdefault("n_heads", 4)
        model_kwargs.setdefault("n_clusters", 10)
    elif model_cls is LightningMLP:
        model_kwargs.setdefault("hidden_dims", [128, 64])
        model_kwargs.setdefault("input_dim", 347)
        model_kwargs.setdefault("num_classes", 4)
        model_kwargs.setdefault("self_conv", False)
    elif model_cls is LightningLogisticRegression:
        model_kwargs.setdefault("input_dim", 347)
        model_kwargs.setdefault("num_classes", 4)
        model_kwargs.setdefault("self_conv", False)
        model_kwargs.setdefault("self_conv_adj", False)
        model_kwargs.setdefault("edge_top", None)
        model_kwargs.setdefault("edge_tsh", None)
        model_kwargs.setdefault("edge_weighted", False)
        model_kwargs.setdefault("wd", 1e-3)

    
    pl.seed_everything(seed, workers=True)

    
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

    
    if model_kwargs.get("loss_type") == "weighted_cross_entropy" and "class_weights" not in model_kwargs:
        model_kwargs["class_weights"] = weight_dict["class_weights"]

    
    model_instance = model_cls(**model_kwargs)

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

    
    trainer.fit(model_instance, train_loader, val_loader)

    best_path = ckpt_cb.best_model_path or "best"
    test_metrics = trainer.test(ckpt_path=best_path, dataloaders=test_loader)

    return trainer, test_metrics

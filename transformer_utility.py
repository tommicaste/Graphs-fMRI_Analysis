import torch
import numpy as np
from typing import List, Optional, Any
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datasets import Dataset, DatasetDict
import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy 

def project_to_psd(C: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    C = 0.5 * (C + C.T)

    vals, vecs = torch.linalg.eigh(C)
    C = vecs @ torch.diag(torch.clamp(vals, min=eps)) @ vecs.T

    std = torch.sqrt(torch.diag(C))
    C = C / std[:, None] / std[None, :]
    C.fill_diagonal_(1.0)

    λ_min = torch.linalg.eigvalsh(C).min()
    if λ_min < eps:
        C += torch.eye(C.size(0), device=C.device) * (eps - λ_min + eps)
        std = torch.sqrt(torch.diag(C))
        C = C / std[:, None] / std[None, :]
        C.fill_diagonal_(1.0)

    return C.clamp_(-1.0, 1.0)

def evaluate_classification(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
    *,
    labels: Optional[list[int]] = None,
    metrics: bool = False,
    plot: bool = False,
    display: bool = False,
    save_confusion_path: Optional[str] = None,
    save_report_path: Optional[str] = None,
):
    """Compute accuracy / balanced accuracy and optionally plot / save."""
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_share = np.nan_to_num(cm_raw / cm_raw.sum(axis=1, keepdims=True))

    if plot or save_confusion_path:
        disp = ConfusionMatrixDisplay(cm_share, display_labels=labels)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", values_format=".2f")
        ax.set_title("Confusion Matrix")
        if save_confusion_path:
            Path(save_confusion_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_confusion_path, bbox_inches="tight")
        if plot and display:
            plt.show()
        plt.close(fig)

    rep_dict: Any = classification_report(
        y_true,
        y_pred,
        output_dict=True,
    )

    if metrics:
        print(f"Accuracy         : {acc:.4f}")
        print(f"Balanced accuracy: {bacc:.4f}")
        report_str: Any = classification_report(y_true, y_pred)
        print(report_str)

    if save_report_path:
        Path(save_report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_report_path, "w") as f:
            json.dump(
                {
                    "accuracy": acc,
                    "balanced_accuracy": bacc,
                    "confusion_matrix_share": cm_share.tolist(),
                    "classification_report": rep_dict,
                },
                f,
                indent=2,
            )

    return {"accuracy": acc, "balanced_accuracy": bacc}


def simulate_ar1(C: torch.Tensor, phi: float, T: int, snr: float | None = None):
    """Return a (T+1) × d AR(1) sequence from correlation matrix C."""
    Sigma_tilde = (1 - phi ** 2) * C
    L_C     = torch.linalg.cholesky(C)
    L_tilde = torch.linalg.cholesky(Sigma_tilde)

    d   = C.size(0)
    x   = torch.empty(T + 1, d)
    x[0] = torch.randn(d) @ L_C.T
    eps  = torch.randn(T, d) @ L_tilde.T

    for t in range(1, T + 1):
        x[t] = phi * x[t - 1] + eps[t - 1]

    if snr is None:
        return x
    noise_std = (1 / snr) ** 0.5
    return x + torch.randn_like(x) * noise_std


def build_hf_datasets(records):
    split_alias = {"train": "train", "val": "validation", "test": "test"}

    buckets = {"train": [], "validation": [], "test": []}
    for rec in records:
        buckets[split_alias[rec["split"]]].append(rec)

    def to_hf_ds(items):
        data = {
            "inputs": [r["seq"] for r in items],
            "labels": [r["y"].item() for r in items],
        }
        ds = Dataset.from_dict(data)
        ds.set_format(type="torch", columns=["inputs", "labels"])
        return ds

    return DatasetDict(
        {
            "train": to_hf_ds(buckets["train"]),
            "validation": to_hf_ds(buckets["validation"]),
            "test": to_hf_ds(buckets["test"]),
        }
    )

class LightningTSClassifier(pl.LightningModule):
    """
    Transformer-based time-series classifier
    ------------------------------------------------
    • linear projection  → N encoder blocks
    • temporal mean pool → logits
    • CrossEntropy loss
    • micro & macro accuracy for train / val / test
    """

    def __init__(
        self,
        T: int = 100,              
        N: int = 347,              
        C: int = 4,               
        d_model: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.5,
        lr: float = 1e-4,
        wd: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ─── model backbone ─────────────────────────────────────────────
        self.input_proj = nn.Linear(N, d_model)
        self.dropout = nn.Dropout(dropout)
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=n_layers)
        self.head    = nn.Linear(d_model, C)

        # custom weight initialization (less aggressive)
        self.apply(self._init_weights)

        # ─── loss & metrics ────────────────────────────────────────────
        self.criterion  = nn.CrossEntropyLoss()
        self.train_acc  = MulticlassAccuracy(num_classes=C, average="micro")
        self.val_acc    = MulticlassAccuracy(num_classes=C, average="micro")
        self.test_acc   = MulticlassAccuracy(num_classes=C, average="micro")
        self.train_bacc = MulticlassAccuracy(num_classes=C, average="macro")
        self.val_bacc   = MulticlassAccuracy(num_classes=C, average="macro")
        self.test_bacc  = MulticlassAccuracy(num_classes=C, average="macro")
        # ─── containers for test predictions ───────────────────────────
        self.test_logits: list[torch.Tensor] = []
        self.test_labels: list[torch.Tensor] = []

    # ────────────────────────────────────────────────────────────────────
    # forward (inference)
    # ────────────────────────────────────────────────────────────────────
    def forward(self, x):                      # x: (B, T, N)
        h   = self.dropout(self.input_proj(x))
        enc = self.encoder(h)
        pooled = enc.mean(dim=1)               # temporal mean pool
        return self.head(pooled)               # logits

    # ─── training -------------------------------------------------------
    def training_step(self, batch, _):
        logits = self(batch["inputs"])
        loss   = self.criterion(logits, batch["labels"])
        self.train_acc.update(logits, batch["labels"])
        self.train_bacc.update(logits, batch["labels"])
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch["labels"].size(0),
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc",  self.train_acc.compute(),  prog_bar=True, sync_dist=True)
        self.log("train_bacc", self.train_bacc.compute(), prog_bar=True, sync_dist=True)
        self.train_acc.reset()
        self.train_bacc.reset()

    # ─── validation -----------------------------------------------------
    def validation_step(self, batch, _):
        logits = self(batch["inputs"])
        loss   = self.criterion(logits, batch["labels"])
        self.val_acc.update(logits, batch["labels"])
        self.val_bacc.update(logits, batch["labels"])
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch["labels"].size(0),
            sync_dist=True,
        )

    def on_validation_epoch_end(self):
        self.log("val_acc",  self.val_acc.compute(), prog_bar=True, sync_dist=True)
        self.log("val_bacc", self.val_bacc.compute(), prog_bar=True, sync_dist=True)
        self.val_acc.reset()
        self.val_bacc.reset()

    # ─── testing --------------------------------------------------------
    def on_test_epoch_start(self):
        self.test_logits.clear()
        self.test_labels.clear()

    def test_step(self, batch, _):
        logits = self(batch["inputs"])
        loss   = self.criterion(logits, batch["labels"])
        self.test_acc.update(logits, batch["labels"])
        self.test_bacc.update(logits, batch["labels"])
        self.test_logits.append(logits.cpu())
        self.test_labels.append(batch["labels"].cpu())
        self.log(
            "test_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["labels"].size(0),
            sync_dist=True,
        )
        return loss

    def on_test_epoch_end(self):
        self.log("test_acc",  self.test_acc.compute(),  prog_bar=False, sync_dist=True)
        self.log("test_bacc", self.test_bacc.compute(), prog_bar=True,  sync_dist=True)
        self.test_acc.reset()
        self.test_bacc.reset()

        # ─── evaluate & save confusion matrix / report ────────────────
        if self.test_logits and self.test_labels:
            preds  = torch.cat(self.test_logits).argmax(1).numpy()
            labels = torch.cat(self.test_labels).numpy()
            results_dir = Path(self.trainer.default_root_dir) / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            evaluate_classification(
                labels,
                preds,
                metrics=False,
                plot=True,
                display=True,
                save_confusion_path=str(results_dir / "confusion_matrix.png"),
                save_report_path=str(results_dir / "classification_report.json"),
            )

    # ─── optimiser ------------------------------------------------------
    def configure_optimizers(self):
        lr: float = float(self.hparams["lr"])
        wd: float = float(self.hparams["wd"])
        return torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=wd,
        )

    # ─── weight init helper ───────────────────────────────────────────
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

class LightningCorrClassifier(pl.LightningModule):
    """
    Linear classifier for vectorized correlation features
    -----------------------------------------------------
    • optional dropout → linear layer → logits
    • CrossEntropy loss with the same metrics as LightningTSClassifier
    """

    def __init__(
        self,
        input_dim: int,  # length of correlation vector, e.g., N*(N-1)//2
        C: int = 4,      # number of sleep stages
        dropout: float = 0.5,
        lr: float = 1e-4,
        wd: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ─── model ────────────────────────────────────────────────────
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, C),
        )

        # ─── loss & metrics ───────────────────────────────────────────
        self.criterion  = nn.CrossEntropyLoss()
        self.train_acc  = MulticlassAccuracy(num_classes=C, average="micro")
        self.val_acc    = MulticlassAccuracy(num_classes=C, average="micro")
        self.test_acc   = MulticlassAccuracy(num_classes=C, average="micro")
        self.train_bacc = MulticlassAccuracy(num_classes=C, average="macro")
        self.val_bacc   = MulticlassAccuracy(num_classes=C, average="macro")
        self.test_bacc  = MulticlassAccuracy(num_classes=C, average="macro")
        # ─── containers for test predictions ─────────────────────────
        self.test_logits: list[torch.Tensor] = []
        self.test_labels: list[torch.Tensor] = []

    # ──────────────────────────────────────────────────────────────────
    # forward (inference)
    # ──────────────────────────────────────────────────────────────────
    def forward(self, x):  # x: (B, D)
        return self.net(x)

    # ─── training -----------------------------------------------------
    def training_step(self, batch, _):
        logits = self(batch["inputs"])
        loss   = self.criterion(logits, batch["labels"])
        self.train_acc.update(logits, batch["labels"])
        self.train_bacc.update(logits, batch["labels"])
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch["labels"].size(0),
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc",  self.train_acc.compute(),  prog_bar=True, sync_dist=True)
        self.log("train_bacc", self.train_bacc.compute(), prog_bar=True, sync_dist=True)
        self.train_acc.reset()
        self.train_bacc.reset()

    # ─── validation ---------------------------------------------------
    def validation_step(self, batch, _):
        logits = self(batch["inputs"])
        loss   = self.criterion(logits, batch["labels"])
        self.val_acc.update(logits, batch["labels"])
        self.val_bacc.update(logits, batch["labels"])
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch["labels"].size(0),
            sync_dist=True,
        )

    def on_validation_epoch_end(self):
        self.log("val_acc",  self.val_acc.compute(), prog_bar=True, sync_dist=True)
        self.log("val_bacc", self.val_bacc.compute(), prog_bar=True, sync_dist=True)
        self.val_acc.reset()
        self.val_bacc.reset()

    # ─── testing ------------------------------------------------------
    def test_step(self, batch, _):
        logits = self(batch["inputs"])
        loss   = self.criterion(logits, batch["labels"])
        self.test_acc.update(logits, batch["labels"])
        self.test_bacc.update(logits, batch["labels"])
        self.test_logits.append(logits.detach().cpu())
        self.test_labels.append(batch["labels"].detach().cpu())
        self.log(
            "test_loss",
            loss,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch["labels"].size(0),
            sync_dist=True,
        )

    def on_test_epoch_end(self):
        self.log("test_acc",  self.test_acc.compute(), prog_bar=True, sync_dist=True)
        self.log("test_bacc", self.test_bacc.compute(), prog_bar=True, sync_dist=True)
        self.test_acc.reset()
        self.test_bacc.reset()

        # gather predictions across devices
        all_logits = torch.cat(self.test_logits)
        all_labels = torch.cat(self.test_labels)
        preds = all_logits.argmax(dim=1).numpy()
        true  = all_labels.numpy()
        self.test_logits.clear()
        self.test_labels.clear()

        results_dir = Path(self.trainer.default_root_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        evaluate_classification(
            true,
            preds,
            metrics=False,
            plot=True,
            display=True,
            save_confusion_path=str(results_dir / "confusion_matrix.png"),
            save_report_path=str(results_dir / "classification_report.json"),
        )

    # ─── optimizer config --------------------------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,  # type: ignore[attr-defined]
            weight_decay=self.hparams.wd,  # type: ignore[attr-defined]
        )

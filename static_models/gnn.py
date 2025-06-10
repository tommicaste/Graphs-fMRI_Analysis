import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from static_models.utils import evaluate_classification
import os
import numpy as np

class LightningGNN(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        num_layers: int,
        GNNLayer: nn.Module,
        dropout: float,
        num_classes: int,
        mlp_hidden: list[int],
        lr: float = 1e-3,
    ):
        super().__init__()
        # save all init args to self.hparams
        self.save_hyperparameters()

        # build GNN backbone
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GNNLayer(input_dim, hidden_channels))
        self.norms.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(1, num_layers):
            self.convs.append(GNNLayer(hidden_channels, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        # dropout for conv layers
        self.dropout = dropout

        # build MLP head
        mlp_layers: list[nn.Module] = []
        in_dim = hidden_channels
        for h in mlp_hidden:
            mlp_layers += [
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        mlp_layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*mlp_layers)

        # buffers for epoch‐end metrics
        self.train_logits: list[torch.Tensor] = []
        self.train_labels: list[torch.Tensor] = []
        self.val_logits: list[torch.Tensor] = []
        self.val_labels: list[torch.Tensor] = []

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.mlp(x)

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = F.cross_entropy(logits, batch.y)
        # accumulate for epoch‐end
        self.train_logits.append(logits.cpu())
        self.train_labels.append(batch.y.cpu())
        # log once per epoch
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.y.size(0),
        )
        return loss

    def on_train_epoch_end(self):
        all_logits = torch.cat(self.train_logits, dim=0)
        all_y      = torch.cat(self.train_labels, dim=0)
        metrics    = evaluate_classification(
            all_y.numpy(),
            all_logits.argmax(1).numpy(),
            metrics=False,
            plot=False
        )

        self.print(
            f"Epoch {self.current_epoch:2d} ▶ Train   "
            f"Acc: {metrics['accuracy']:.4f}  "
            f"BAcc: {metrics['balanced_accuracy']:.4f}"
        )

        total = all_y.size(0)  # ← compute total
        self.log(
            "train_acc",
            metrics["accuracy"],
            on_epoch=True,
            prog_bar=True,
            batch_size=total,
        )
        self.log(
            "train_BAcc",
            metrics["balanced_accuracy"],
            on_epoch=True,
            prog_bar=True,
            batch_size=total,
        )

        self.train_logits.clear()
        self.train_labels.clear()

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = F.cross_entropy(logits, batch.y)
        self.val_logits.append(logits.cpu())
        self.val_labels.append(batch.y.cpu())
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.y.size(0),
        )

    def on_validation_epoch_end(self):
        all_logits = torch.cat(self.val_logits, dim=0)
        all_y      = torch.cat(self.val_labels, dim=0)
        metrics    = evaluate_classification(
            all_y.numpy(),
            all_logits.argmax(1).numpy(),
            metrics=False,
            plot=False
        )

        self.print(
            f"Epoch {self.current_epoch:2d} ▶ Val     "
            f"Acc: {metrics['accuracy']:.4f}  "
            f"BAcc: {metrics['balanced_accuracy']:.4f}"
        )

        total = all_y.size(0)  # ← compute total
        self.log(
            "val_acc",
            metrics["accuracy"],
            on_epoch=True,
            prog_bar=False,
            batch_size=total,
        )
        self.log(
            "val_BAcc",
            metrics["balanced_accuracy"],
            on_epoch=True,
            prog_bar=False,
            batch_size=total,
        )

        self.val_logits.clear()
        self.val_labels.clear()

    def test_model(self, checkpoint_path: str, test_loader, results_dir: str):
        import os, numpy as np, torch

        # ensure results folder exists
        os.makedirs(results_dir, exist_ok=True)
        cm_path     = os.path.join(results_dir, "confusion_matrix.png")
        report_path = os.path.join(results_dir, "classification_report.json")

        # ── load best checkpoint as a *new* model instance
        best_model = self.__class__.load_from_checkpoint(checkpoint_path)
        best_model.to(self.device)
        best_model.eval()

        # ── collect predictions
        y_preds, y_trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                logits = best_model(batch)
                y_preds.append(logits.argmax(1).cpu().numpy())
                y_trues.append(batch.y.cpu().numpy())

        y_pred = np.concatenate(y_preds)
        y_true = np.concatenate(y_trues)

        # ── evaluate & save artifacts
        evaluate_classification(
            y_true,
            y_pred,
            metrics=True,
            plot=True,
            display=True,
            save_confusion_path=cm_path,
            save_report_path=report_path,
        )
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

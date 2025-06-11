import os
from pathlib import Path                # ← added
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from static_models.utils import evaluate_classification


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
        self.save_hyperparameters()

        # GNN backbone
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GNNLayer(input_dim, hidden_channels))
        self.norms.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(1, num_layers):
            self.convs.append(GNNLayer(hidden_channels, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        # MLP head
        mlp_layers: list[nn.Module] = []
        in_dim = hidden_channels
        for h in mlp_hidden:
            mlp_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        mlp_layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*mlp_layers)

        # epoch-buffers (original)
        self.train_logits: list[torch.Tensor] = []
        self.train_labels: list[torch.Tensor] = []
        self.val_logits:   list[torch.Tensor] = []
        self.val_labels:   list[torch.Tensor] = []

    # 1. Forward Pass
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.hparams.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.mlp(x)

    # 2. Train loop
    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = F.cross_entropy(logits, batch.y)
        self.train_logits.append(logits.cpu())
        self.train_labels.append(batch.y.cpu())
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch.y.size(0))
        return loss

    def on_train_epoch_end(self):
        all_logits = torch.cat(self.train_logits, 0)
        all_y      = torch.cat(self.train_labels, 0)
        metrics = evaluate_classification(all_y.numpy(), all_logits.argmax(1).numpy(),
                                          metrics=False, plot=False)
        self.print(f"Epoch {self.current_epoch:2d} ▶ Train   "
                   f"Acc: {metrics['accuracy']:.4f}  BAcc: {metrics['balanced_accuracy']:.4f}")
        total = all_y.size(0)
        self.log("train_acc",  metrics["accuracy"],          on_epoch=True, prog_bar=True, batch_size=total)
        self.log("train_BAcc", metrics["balanced_accuracy"], on_epoch=True, prog_bar=True, batch_size=total)
        self.train_logits.clear(); self.train_labels.clear()

    # 3. Val loop  
    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = F.cross_entropy(logits, batch.y)
        self.val_logits.append(logits.cpu())
        self.val_labels.append(batch.y.cpu())
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=batch.y.size(0))

    def on_validation_epoch_end(self):
        all_logits = torch.cat(self.val_logits, 0)
        all_y      = torch.cat(self.val_labels, 0)
        metrics = evaluate_classification(all_y.numpy(), all_logits.argmax(1).numpy(),
                                          metrics=False, plot=False)
        self.print(f"Epoch {self.current_epoch:2d} ▶ Val     "
                   f"Acc: {metrics['accuracy']:.4f}  BAcc: {metrics['balanced_accuracy']:.4f}")
        total = all_y.size(0)
        self.log("val_acc",  metrics["accuracy"],          on_epoch=True, batch_size=total)
        self.log("val_BAcc", metrics["balanced_accuracy"], on_epoch=True, batch_size=total)
        self.val_logits.clear(); self.val_labels.clear()

    # 4. Test loop
    def on_test_epoch_start(self):
        self.test_logits: list[torch.Tensor] = []
        self.test_labels: list[torch.Tensor] = []

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        loss   = F.cross_entropy(logits, batch.y)
        self.test_logits.append(logits.cpu())
        self.test_labels.append(batch.y.cpu())
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=batch.y.size(0))
        return loss

    def on_test_epoch_end(self):
        all_logits = torch.cat(self.test_logits, 0)
        all_y      = torch.cat(self.test_labels, 0)
        metrics = evaluate_classification(all_y.numpy(), all_logits.argmax(1).numpy(),
                                          metrics=False, plot=False)
        self.print(f"Epoch {self.current_epoch:2d} ▶ Test    "
                   f"Acc: {metrics['accuracy']:.4f}  BAcc: {metrics['balanced_accuracy']:.4f}")
        total = all_y.size(0)
        self.log("test_acc",  metrics["accuracy"],          on_epoch=True, prog_bar=True, batch_size=total)
        self.log("test_BAcc", metrics["balanced_accuracy"], on_epoch=True, prog_bar=True, batch_size=total)
        self.test_logits.clear(); self.test_labels.clear()

    def test_model(self, checkpoint_path: str, test_loader, results_dir: str | None = None):
        """
        Load the best checkpoint, run inference on `test_loader`, and
        save confusion-matrix PNG + classification-report JSON to
        `results_dir` (defaults to <run_root>/results).
        """
        # auto-infer results folder
        if results_dir is None:
            run_root    = Path(checkpoint_path).parents[1] 
            results_dir = run_root / "results"
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        cm_path     = results_dir / "confusion_matrix.png"
        report_path = results_dir / "classification_report.json"

        # fresh model instance on same device
        best_model = self.__class__.load_from_checkpoint(checkpoint_path)
        best_model.to(self.device)
        best_model.eval()

        y_preds, y_trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                logits = best_model(batch)
                y_preds.append(logits.argmax(1).cpu().numpy())
                y_trues.append(batch.y.cpu().numpy())

        y_pred = np.concatenate(y_preds)
        y_true = np.concatenate(y_trues)

        # DO NOT print duplicate metrics; only save & plot
        evaluate_classification(
            y_true,
            y_pred,
            metrics=False,  # avoid printing duplicate accuracy lines
            plot=True,
            display=True,
            save_confusion_path=cm_path,
            save_report_path=report_path,
        )

    # 6. Optimiser
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=float(self.hparams.lr), weight_decay=5e-4)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import global_mean_pool
from torchmetrics.classification import MulticlassAccuracy
try:
    from torchmetrics.classification import MulticlassBalancedAccuracy
except ImportError:
    # Fallback for older torchmetrics versions
    from torchmetrics import Accuracy as MulticlassBalancedAccuracy
from pathlib import Path
from static_models.utils import evaluate_classification

from static_models.transforms import BatchEdgeListTransform


class LightningGNN(pl.LightningModule):
    """Generic GNN → global pooling → MLP classifier."""

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
        edge_top=0.10, 
        edge_tsh=None
    ):
        super().__init__()
        self.save_hyperparameters()

        #self.node_tf = BatchNodeFeatureTransform()                  
        self.edge_tf = BatchEdgeListTransform(top=edge_top,
                                              tsh=edge_tsh)  

        # ─────────── GNN backbone ───────────
        self.convs = nn.ModuleList(
            [GNNLayer(input_dim, hidden_channels)]
            + [GNNLayer(hidden_channels, hidden_channels) for _ in range(num_layers - 1)]
        )
        self.norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)]
        )

        # ─────────── MLP head ───────────
        layers: list[nn.Module] = []
        in_dim = hidden_channels
        for h in mlp_hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

        # ─────────── Metrics ───────────
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc   = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc  = MulticlassAccuracy(num_classes=num_classes)

        # Handle different torchmetrics versions
        try:
            self.train_bacc = MulticlassBalancedAccuracy(num_classes=num_classes)
            self.val_bacc   = MulticlassBalancedAccuracy(num_classes=num_classes)
            self.test_bacc  = MulticlassBalancedAccuracy(num_classes=num_classes)
        except TypeError:
            # Fallback for older versions
            self.train_bacc = MulticlassBalancedAccuracy(num_classes=num_classes, task="multiclass")
            self.val_bacc   = MulticlassBalancedAccuracy(num_classes=num_classes, task="multiclass")
            self.test_bacc  = MulticlassBalancedAccuracy(num_classes=num_classes, task="multiclass")

        # for confusion-matrix at test time
        self.test_logits: list[torch.Tensor] = []
        self.test_labels: list[torch.Tensor] = []

    # ───────────────────────────────── forward ─────────────────────────────────
    def forward(self, data):  
        data = self.edge_tf(data) 
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x).relu()
            x = F.dropout(x, p=self.hparams.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.mlp(x)

    # ───────────────────────────────── train ──────────────────────────────────
    def training_step(self, batch, _):
        logits = self(batch)
        loss   = F.cross_entropy(logits, batch.y)
        self.train_acc.update(logits, batch.y)
        self.train_bacc.update(logits, batch.y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch.y.size(0))
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc",  self.train_acc.compute(),  prog_bar=True)
        self.log("train_bacc", self.train_bacc.compute(), prog_bar=False)
        self.train_acc.reset(); self.train_bacc.reset()

    # ──────────────────────────────── validate ────────────────────────────────
    def validation_step(self, batch, _):
        logits = self(batch)
        loss   = F.cross_entropy(logits, batch.y)
        self.val_acc.update(logits, batch.y)
        self.val_bacc.update(logits, batch.y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch.y.size(0))

    def on_validation_epoch_end(self):
        self.log("val_acc",  self.val_acc.compute(),
                 prog_bar=True)
        self.log("val_bacc", self.val_bacc.compute(),
                 prog_bar=True)         
        self.val_acc.reset()
        self.val_bacc.reset()

    # ───────────────────────────────── test ───────────────────────────────────
    def on_test_epoch_start(self):
        self.test_logits.clear()
        self.test_labels.clear()

    def test_step(self, batch, _):
        logits = self(batch)
        loss   = F.cross_entropy(logits, batch.y)
        self.test_acc.update(logits, batch.y)
        self.test_bacc.update(logits, batch.y)
        self.test_logits.append(logits.cpu())
        self.test_labels.append(batch.y.cpu())
        self.log("test_loss", loss, on_epoch=True, prog_bar=False, batch_size=batch.y.size(0))
        return loss

    def on_test_epoch_end(self):
        # 1) Log aggregated metrics
        self.log("test_acc",  self.test_acc.compute(),  prog_bar=True)
        self.log("test_bacc", self.test_bacc.compute(), prog_bar=True)
        self.test_acc.reset(); self.test_bacc.reset()

        # 2) Confusion-matrix + report via helper
        logits = torch.cat(self.test_logits).argmax(1).numpy()
        labels = torch.cat(self.test_labels).numpy()

        results_dir = Path(self.trainer.default_root_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        evaluate_classification(
            labels,
            logits,
            metrics=False,
            plot=True,
            display=True,
            save_confusion_path=results_dir / "confusion_matrix.png",
            save_report_path=results_dir / "classification_report.json",
        )

    # ─────────────────────────── optimiser / scheduler ─────────────────────────
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=float(self.hparams.lr), weight_decay=5e-4
        )



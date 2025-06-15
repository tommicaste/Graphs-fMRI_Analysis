import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from pathlib import Path
from static_models.utils import evaluate_classification


class LogisticRegression(pl.LightningModule):
    """
    A PyTorch Lightning module for a multiclass logistic regression baseline.
    This module's structure mirrors the GNN for consistent evaluation.
    """
    def __init__(self, input_dim: int, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        # The entire model is just one linear layer
        self.linear = nn.Linear(self.hparams.input_dim, self.hparams.num_classes)
        
        # The loss function is standard CrossEntropyLoss
        self.criterion = nn.CrossEntropyLoss()
        
        # --- Set up metrics, now including Balanced Accuracy ---
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.test_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        
        self.train_bacc = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_bacc = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.test_bacc = MulticlassAccuracy(num_classes=num_classes, average="macro")

        # --- Lists for aggregating test results ---
        self.test_logits: list[torch.Tensor] = []
        self.test_labels: list[torch.Tensor] = []

    def forward(self, x):
        # The input 'x' is the flattened vector of features
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        # Assumes the dataloader yields a flat vector 'x' and label 'y'
        x, y = batch 
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Update metrics
        self.train_acc.update(logits, y)
        self.train_bacc.update(logits, y)
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=y.size(0))
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc",  self.train_acc.compute(),  prog_bar=True)
        self.log("train_bacc", self.train_bacc.compute(), prog_bar=False)
        # Reset metrics
        self.train_acc.reset()
        self.train_bacc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Update metrics
        self.val_acc.update(logits, y)
        self.val_bacc.update(logits, y)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=y.size(0))

    def on_validation_epoch_end(self):
        self.log("val_acc",  self.val_acc.compute(), prog_bar=True)
        self.log("val_bacc", self.val_bacc.compute(), prog_bar=True)
        # Reset metrics
        self.val_acc.reset()
        self.val_bacc.reset()

    # --- NEW: Full testing routine mirroring the GNN ---
    def on_test_epoch_start(self):
        self.test_logits.clear()
        self.test_labels.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        # Update metrics
        self.test_acc.update(logits, y)
        self.test_bacc.update(logits, y)
        
        # Append results for final evaluation
        self.test_logits.append(logits.cpu())
        self.test_labels.append(y.cpu())
        
        return self.criterion(logits, y)

    def on_test_epoch_end(self):
        # 1) Log aggregated metrics
        self.log("test_acc",  self.test_acc.compute(),  prog_bar=True)
        self.log("test_bacc", self.test_bacc.compute(), prog_bar=True)
        self.test_acc.reset()
        self.test_bacc.reset()

        # 2) Generate confusion matrix and classification report
        logits = torch.cat(self.test_logits).argmax(1).numpy()
        labels = torch.cat(self.test_labels).numpy()

        results_dir = Path(self.trainer.default_root_dir) / "results_baseline"
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
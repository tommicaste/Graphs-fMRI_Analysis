from static_models.transforms import LowerTriFlattenBatch
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from pathlib import Path
from static_models.utils import evaluate_classification


class LightningLogisticRegression(pl.LightningModule):
    """
    Logistic regression model for classification tasks, with support for weighted loss and PyTorch Lightning integration.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        lr: float = 1e-3,
        loss_type: str = "cross_entropy",
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Feature flattening transform
        self.flatten = LowerTriFlattenBatch(input_dim)
        in_features = input_dim * (input_dim - 1) // 2

        # Linear classifier head
        self.clf = nn.Linear(in_features, num_classes)

        # Loss function
        if loss_type == "weighted_cross_entropy":
            if class_weights is None:
                raise ValueError("class_weights must be provided for weighted_cross_entropy")
            self.register_buffer("class_weights", class_weights)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        elif loss_type == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        # Performance metrics
        self.train_wacc = MulticlassAccuracy(num_classes=num_classes, average="weighted")
        self.val_wacc   = MulticlassAccuracy(num_classes=num_classes, average="weighted")
        self.test_wacc  = MulticlassAccuracy(num_classes=num_classes, average="weighted")

        self.train_bacc = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_bacc   = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.test_bacc  = MulticlassAccuracy(num_classes=num_classes, average="macro")

        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_acc   = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.test_acc  = MulticlassAccuracy(num_classes=num_classes, average="micro")

        self.test_logits: list[torch.Tensor] = []
        self.test_labels: list[torch.Tensor] = []

    # Forward pass
    def forward(self, data):
        z = self.flatten(data)         # (B, input_dim(input_dim−1)//2)
        return self.clf(z)

    # Training methods
    def training_step(self, batch, _):
        logits = self(batch)
        loss = self.criterion(logits, batch.y)
        self.train_acc.update(logits, batch.y)
        self.train_bacc.update(logits, batch.y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch.y.size(0))
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc",  self.train_acc.compute(),  prog_bar=True)
        self.log("train_bacc", self.train_bacc.compute(), prog_bar=False)
        self.train_acc.reset()
        self.train_bacc.reset()

    # Validation methods
    def validation_step(self, batch, _):
        logits = self(batch)
        loss = self.criterion(logits, batch.y)
        self.val_acc.update(logits, batch.y)
        self.val_bacc.update(logits, batch.y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch.y.size(0))

    def on_validation_epoch_end(self):
        self.log("val_acc",  self.val_acc.compute(),  prog_bar=True)
        self.log("val_bacc", self.val_bacc.compute(), prog_bar=True)
        self.val_acc.reset()
        self.val_bacc.reset()

    # Test methods
    def on_test_epoch_start(self):
        self.test_logits.clear()
        self.test_labels.clear()

    def test_step(self, batch, _):
        logits = self(batch)
        loss = self.criterion(logits, batch.y)
        self.test_acc.update(logits, batch.y)
        self.test_bacc.update(logits, batch.y)
        self.test_logits.append(logits.cpu())
        self.test_labels.append(batch.y.cpu())
        self.log("test_loss", loss, on_epoch=True, prog_bar=False, batch_size=batch.y.size(0))
        return loss

    def on_test_epoch_end(self):
        self.log("test_acc",  self.test_acc.compute(),  prog_bar=True)
        self.log("test_bacc", self.test_bacc.compute(), prog_bar=True)
        self.test_acc.reset()
        self.test_bacc.reset()
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
            save_confusion_path=str(results_dir / "confusion_matrix.png"),
            save_report_path=str(results_dir / "classification_report.json"),
        )

    # Optimizer configuration
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=float(getattr(self.hparams, 'lr', 1e-3)), weight_decay=5e-3
        )
        return {"optimizer": optimizer}

from static_models.transforms import LowerTriFlattenBatch
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from pathlib import Path
from static_models.utils import evaluate_classification


class LightningMLP(pl.LightningModule):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        lr: float = 1e-3,
        wd: float = 1e-3,
        loss_type: str = "cross_entropy",
        class_weights: torch.Tensor | None = None,
        self_conv: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.flatten = LowerTriFlattenBatch(input_dim, self_conv=self_conv)
        in_features = input_dim * (input_dim - 1) // 2

        layers: list[nn.Module] = []
        for h in hidden_dims:
            layers += [nn.Linear(in_features, h), nn.ReLU(), nn.Dropout(0.5)]
            in_features = h
        layers.append(nn.Linear(in_features, num_classes))
        self.mlp = nn.Sequential(*layers)

        if loss_type == "weighted_cross_entropy":
            if class_weights is None:
                raise ValueError("class_weights must be provided for weighted_cross_entropy")
            self.register_buffer("class_weights", class_weights)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif loss_type == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

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

    def forward(self, data):
        z = self.flatten(data)
        return self.mlp(z)

    def training_step(self, batch, _):
        logits = self(batch)
        loss = self.criterion(logits, batch.y)
        self.train_acc.update(logits, batch.y)
        self.train_bacc.update(logits, batch.y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=False, batch_size=batch.y.size(0), sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc",  self.train_acc.compute(),  prog_bar=True, sync_dist=True)
        self.log("train_bacc", self.train_bacc.compute(), prog_bar=True, sync_dist=True)
        self.train_acc.reset()
        self.train_bacc.reset()

    def validation_step(self, batch, _):
        logits = self(batch)
        loss = self.criterion(logits, batch.y)
        self.val_acc.update(logits, batch.y)
        self.val_bacc.update(logits, batch.y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, batch_size=batch.y.size(0), sync_dist=True)

    def on_validation_epoch_end(self):
        self.log("val_acc",  self.val_acc.compute(), prog_bar=True, sync_dist=True)
        self.log("val_bacc", self.val_bacc.compute(), prog_bar=True, sync_dist=True)
        self.val_acc.reset()
        self.val_bacc.reset()

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
        self.log(
            "test_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.y.size(0),
            sync_dist=True,
        )
        return loss

    def on_test_epoch_end(self):
        self.log("test_acc",  self.test_acc.compute(),  prog_bar=False, sync_dist=True)
        self.log("test_bacc", self.test_bacc.compute(), prog_bar=True, sync_dist=True)
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

    def configure_optimizers(self):
        lr = float(getattr(self.hparams, 'lr', 1e-3))
        wd = float(getattr(self.hparams, 'wd', 1e-3))
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=lr, 
            weight_decay=wd
        )
        return {"optimizer": optimizer}

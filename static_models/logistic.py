import torch
import torch.nn as nn
import pytorch_lightning as pl

class LogisticRegression(pl.LightningModule):
    def __init__(self, input_dim: int, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.linear = nn.Linear(self.hparams.input_dim, self.hparams.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        
        # --- SUGGESTION 1: Add train_acc metric ---
        self.train_acc = pl.metrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_acc = pl.metrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_acc = pl.metrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch 
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        
        # --- SUGGESTION 2: Update train_acc in the training step ---
        self.train_acc.update(logits.softmax(dim=-1), y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_acc.update(logits.softmax(dim=-1), y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.test_acc.update(logits.softmax(dim=-1), y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
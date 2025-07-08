import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn import Parameter
from typing import Optional, List
from torchmetrics.classification import MulticlassAccuracy
from pathlib import Path
from static_models.utils import evaluate_classification


class InterpretableTransformerEncoder(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype
        )
        self.attention_weights: Optional[torch.Tensor] = None

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Self-attention block with weight capture.

        Accepts `*args` and `**kwargs` to stay compatible with newer PyTorch
        versions that may pass extra arguments such as `is_causal`.
        """
        x, weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        self.attention_weights = weights
        return self.dropout1(x)

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        return self.attention_weights


class ClusterAssignment(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        embedding_dimension: int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None,
        orthogonal=True,
        freeze_center=True,
    ) -> None:
        """
        Module to handle the soft assignment for OCREAD.
        """
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers

        if orthogonal:
            orthogonal_cluster_centers = torch.zeros_like(initial_cluster_centers)
            for i in range(cluster_number):
                project = 0
                for j in range(i):
                    project += self.project(
                        orthogonal_cluster_centers[j], initial_cluster_centers[i]
                    )
                u_i = initial_cluster_centers[i] - project
                orthogonal_cluster_centers[i] = u_i / torch.norm(u_i, p=2)
            initial_cluster_centers = orthogonal_cluster_centers

        self.cluster_centers = Parameter(
            initial_cluster_centers, requires_grad=(not freeze_center)
        )

    @staticmethod
    def project(u, v):
        return (torch.dot(u, v) / torch.dot(u, u)) * u

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors.
        """
        assignment = batch @ self.cluster_centers.T
        assignment = torch.pow(assignment, 2)
        norm = torch.norm(self.cluster_centers, p=2, dim=-1)
        soft_assign = assignment / norm
        return F.softmax(soft_assign, dim=-1)


class BNT_Lightning(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        n_layers: int = 2,
        n_heads: int = 4,
        n_clusters: int = 10,
        d_ff: int = 2048,
        dropout: float = 0.1,
        lr: float = 1e-4,
        wd: float = 1e-4,
        loss_type: str = 'cross_entropy',
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.transformer_encoders = nn.ModuleList([
            InterpretableTransformerEncoder(
                d_model=input_dim,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True
            ) for _ in range(n_layers)
        ])

        self.pooling = ClusterAssignment(
            cluster_number=n_clusters,
            embedding_dimension=input_dim
        )
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(n_clusters * input_dim),
            nn.Linear(n_clusters * input_dim, num_classes)
        )
        
        self.criterion = self._configure_loss()
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

    def _configure_loss(self):
        loss_type = getattr(self.hparams, 'loss_type', 'cross_entropy')
        if loss_type == 'weighted_cross_entropy':
            weights = getattr(self.hparams, 'class_weights', None)
            if weights is None:
                raise ValueError("class_weights must be provided for weighted_cross_entropy")
            return nn.CrossEntropyLoss(weight=weights)
        elif loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def forward(self, data):
        x = data.x

        if x.dim() == 2:
            B = getattr(data, "num_graphs", None)
            if B is None:
                raise ValueError("Cannot infer batch size from data. Expected PyG Batch with 'num_graphs'.")
            N = x.size(0) // B
            x = x.view(B, N, -1)

        for encoder in self.transformer_encoders:
            x = encoder(x)

        assignment_probs = self.pooling(x)

        assignment_probs_t = assignment_probs.transpose(1, 2)

        graph_embedding = assignment_probs_t @ x

        flattened_embedding = graph_embedding.flatten(start_dim=1)
        logits = self.mlp(flattened_embedding)
        return logits

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
        self.log("test_acc", self.test_acc.compute(), prog_bar=False, sync_dist=True)
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
        lr = getattr(self.hparams, 'lr', 1e-4)
        wd = getattr(self.hparams, 'wd', 1e-4)
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=lr, 
            weight_decay=wd
        )
        return {"optimizer": optimizer}
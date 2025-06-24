import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, global_sort_pool
from torchmetrics.classification import MulticlassAccuracy
from pathlib import Path
from static_models.utils import evaluate_classification
from static_models.transforms import BatchEdgeListTransform
from torch_geometric.utils import dropout_edge


class LightningGNN(pl.LightningModule):
    """
    General Graph Neural Network (GNN) LightningModule supporting configurable backbone, pooling, and MLP head.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        num_layers: int,
        GNNLayer: nn.Module,
        num_classes: int,
        mlp_hidden: list[int],
        dropout: float = 0.0,
        edge_dropout: float = 0.0,
        lr: float = 1e-3,
        wd: float = 1e-3,
        edge_top: float | None = None,
        edge_tsh: float | None = None,
        loss_type: str = 'cross_entropy',
        class_weights: torch.Tensor | None = None,
        residual_connections: bool = False,
        pooling_fn: str = 'mean',
        sort_pool_k: int = 10,
        conv1d_out: int = 128,
        conv1d_kernel_size: int = 5
    ):
        super().__init__()
        self.save_hyperparameters()
        self.edge_tf = BatchEdgeListTransform(top=edge_top, tsh=edge_tsh)
        self.convs = nn.ModuleList(
            [GNNLayer(input_dim, hidden_channels)]
            + [GNNLayer(hidden_channels, hidden_channels) for _ in range(num_layers - 1)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_channels) for _ in range(num_layers)]
        )
        self.pool = self._configure_pooling()
        if pooling_fn == 'sort':
            # Store k so we can reshape in forward
            self.sort_pool_k = sort_pool_k

            # 1-D convolution hyper-parameters -------------
            self.conv1d_out = conv1d_out
            self.conv1d = nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=self.conv1d_out,
                kernel_size=conv1d_kernel_size,
                padding=conv1d_kernel_size // 2,   # same-padding keeps length → k
            )
            self.global_pool_1d = nn.AdaptiveMaxPool1d(1)

            mlp_input_dim = self.conv1d_out
        else:
            self.sort_pool_k = None
            mlp_input_dim = hidden_channels
        layers: list[nn.Module] = []
        in_dim = mlp_input_dim
        for h in mlp_hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)
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
        """Set up the loss based on hyperparameters."""
        loss_type = getattr(self.hparams, 'loss_type', 'cross_entropy')
        if loss_type == 'weighted_cross_entropy':
            weights = getattr(self.hparams, 'class_weights', None)
            if weights is None:
                raise ValueError("class_weights must be provided for weighted_cross_entropy")
            self.register_buffer("class_weights", weights)
            return nn.CrossEntropyLoss(weight=self.class_weights)
        elif loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def _configure_pooling(self):
        """Set up the pooling function based on hyperparameters."""
        pooling_fn_str = getattr(self.hparams, 'pooling_fn', 'mean')
        if pooling_fn_str == 'mean':
            return global_mean_pool
        elif pooling_fn_str == 'sum':
            return global_add_pool
        elif pooling_fn_str == 'max':
            return global_max_pool
        elif pooling_fn_str == 'sort':
            sort_pool_k = getattr(self.hparams, 'sort_pool_k', 10)
            return lambda x, batch: global_sort_pool(x, batch, k=sort_pool_k)
        else:
            raise ValueError(f"Unsupported pooling_fn: {pooling_fn_str}")

    def forward(self, data):
        edge_top = getattr(self.hparams, 'edge_top', None)
        edge_tsh = getattr(self.hparams, 'edge_tsh', None)
        if edge_top is not None or edge_tsh is not None:
            data = self.edge_tf(data)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_dropout = getattr(self.hparams, 'edge_dropout', 0.0)
        if edge_index is not None:
            edge_index, _ = dropout_edge(
                edge_index,
                p=edge_dropout,
                force_undirected=True,
                training=self.training
            )
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_residual = x
            x = conv(x, edge_index)
            x = norm(x).relu()
            residual_connections = getattr(self.hparams, 'residual_connections', False)
            if residual_connections and i > 0:
                x = x + x_residual
            dropout = getattr(self.hparams, 'dropout', 0.0)
            x = F.dropout(x, p=dropout, training=self.training)
        x = self.pool(x, batch)

    
        if self.sort_pool_k is not None:
            k = self.sort_pool_k
            x = x.view(x.size(0), k, -1)  
            x = x.transpose(1, 2)         
            x = F.relu(self.conv1d(x))    
            x = self.global_pool_1d(x)    
            x = x.squeeze(-1)             

        return self.mlp(x)

    def training_step(self, batch, _):
        logits = self(batch)
        loss = self.criterion(logits, batch.y)
        self.train_acc.update(logits, batch.y)
        self.train_bacc.update(logits, batch.y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=False, batch_size=batch.y.size(0))
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc",  self.train_acc.compute(),  prog_bar=True)
        self.log("train_bacc", self.train_bacc.compute(), prog_bar=True)
        self.train_acc.reset()
        self.train_bacc.reset()

    def validation_step(self, batch, _):
        logits = self(batch)
        loss = self.criterion(logits, batch.y)
        self.val_acc.update(logits, batch.y)
        self.val_bacc.update(logits, batch.y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, batch_size=batch.y.size(0))

    def on_validation_epoch_end(self):
        self.log("val_acc",  self.val_acc.compute(), prog_bar=True)
        self.log("val_bacc", self.val_bacc.compute(), prog_bar=True)
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
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch.y.size(0))
        return loss

    def on_test_epoch_end(self):
        self.log("test_acc",  self.test_acc.compute(),  prog_bar=False)
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

    def configure_optimizers(self):
        lr = float(getattr(self.hparams, 'lr', 1e-3))
        wd = float(getattr(self.hparams, 'wd', 1e-3))
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=lr, 
            weight_decay=wd
        )
        return {"optimizer": optimizer}
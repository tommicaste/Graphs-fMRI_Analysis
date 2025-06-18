import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv

class SnapshotClassifier(pl.LightningModule):
    """
    A baseline model using a GCN layer followed by a GRU to classify each
    snapshot in a temporal graph sequence.
    """
    def __init__(self, node_features: int, hidden_dim: int, num_classes: int, learning_rate: float = 0.01):
        """
        Args:
            node_features (int): The number of features on each node.
            hidden_dim (int): The dimensionality of the hidden state in the GCN and GRU.
            num_classes (int): The number of classes for the node-level classification.
            learning_rate (float): The learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()

        # 1. GCN layer to extract spatial features at each time step
        self.gcn_layer = GCNConv(self.hparams.node_features, self.hparams.hidden_dim)

        # 2. Standard GRU layer to process the sequence of node embeddings
        self.gru_layer = torch.nn.GRU(input_size=self.hparams.hidden_dim,
                                      hidden_size=self.hparams.hidden_dim)

        # 3. A classifier to map the final hidden states to class logits
        self.classifier = torch.nn.Linear(self.hparams.hidden_dim, self.hparams.num_classes)

    def forward(self, data_signal):
        """
        Processes a single temporal data signal to produce a prediction for each snapshot.
        """
        # --- Spatial Processing ---
        gcn_outputs = []
        for snapshot in data_signal:
            embedding = self.gcn_layer(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            embedding = F.relu(embedding)
            gcn_outputs.append(embedding)
        
        gcn_sequence = torch.stack(gcn_outputs, dim=0)

        # --- Temporal Processing ---
        gru_outputs, _ = self.gru_layer(gcn_sequence)

        # --- Classification ---
        num_timesteps, num_nodes, hidden_dim = gru_outputs.shape
        predictions = self.classifier(gru_outputs.view(-1, hidden_dim))
        
        return predictions.view(num_timesteps, num_nodes, self.hparams.num_classes)

    def _calculate_loss(self, batch):
        """Helper function to calculate loss for a batch."""
        total_loss = 0
        for data_signal in batch:
            targets = torch.tensor(data_signal.targets, dtype=torch.long, device=self.device)
            predictions = self(data_signal)
            
            reshaped_predictions = predictions.view(-1, self.hparams.num_classes)
            reshaped_targets = targets.view(-1)
            
            total_loss += F.cross_entropy(reshaped_predictions, reshaped_targets)
        
        return total_loss / len(batch) if batch else 0

    def training_step(self, batch, batch_idx):
        """Defines the training logic for one batch."""
        loss = self._calculate_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Defines the validation logic for one batch."""
        loss = self._calculate_loss(batch)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Performs a test step on a batch of data."""
        loss = self._calculate_loss(batch)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        """Sets up the optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
import torch
import pytorch_lightning as pl
from pathlib import Path

# Assuming your functions are in these files
from data.loader import load_temporal_data
from temporal_models.temporal_gnn import SnapshotClassifier 

# --- Configuration ---
DATA_DIR = "/project2/cdonnat/sleepstages/data/pt/dynamic/"
BATCH_SIZE = 1
NUM_WORKERS = 0 

# --- Model Hyperparameters ---
NODE_FEATURES = 294 
NUM_CLASSES = 4     
LEARNING_RATE = 0.001
# Add the required hidden dimension for the GCN and GRU layers
HIDDEN_DIM = 64     

# --- 1. Data Loading ---
print(f"Loading and splitting data from {DATA_DIR}...")
train_loader, val_loader, test_loader = load_temporal_data(
    data_directory=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)
print("DataLoaders created successfully.")


# --- 2. Model Initialization ---
print("\nInitializing model...")
# FIX: Pass all required hyperparameters to the model, including hidden_dim and learning_rate
model = SnapshotClassifier(
    node_features=NODE_FEATURES,
    hidden_dim=HIDDEN_DIM,
    num_classes=NUM_CLASSES,
    learning_rate=LEARNING_RATE
)
print("Model initialized.")


# --- 3. Trainer Setup and Execution ---
# Initialize the PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=50,
    accelerator='auto' # Automatically selects GPU if available
)

# Start the training process
print("\nStarting training...")
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
print("Training complete.")

# Run the test phase
print("\nStarting testing...")
trainer.test(model, dataloaders=test_loader)
print("Testing complete.")
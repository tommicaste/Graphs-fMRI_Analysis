from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
)
from data import *
from static_models.gnn import LightningGNN
import torch
import random 
import numpy as np
import itertools
from tqdm.auto import tqdm

# --- CONFIGURATION ---
SEED = 23
BASE_PATH = "/project2/cdonnat/sleepstages/data/pt/non_overlapping_data.pt"
RUNS_DIR = Path("/home/tcastellani/sleepstages/run_augment_search")

# --- Set seeds for reproducibility ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def run_experiment(config: dict):
    """
    Runs a single training and testing experiment based on a configuration dictionary.
    """
    # This function's internal print statements can be commented out for a cleaner tqdm bar
    # print("─" * 80)
    # print(f"🚀 STARTING RUN: {config['name']}")
    # print(f"   Configuration: {config}")
    # print("─" * 80)

    # 1. Load data with specified augmentation
    train_loader, val_loader, test_loader, data_info = load_data(
        path=BASE_PATH,
        augment_strategy=config['augment_strategy'],
        augment_proportion=config['augment_proportion']
    )

    # 2. Determine class weights if needed
    class_weights = None
    if config['loss_type'] == 'weighted_cross_entropy':
        class_weights = data_info['class_weights']

    # 3. Instantiate the model with the correct parameters
    model = LightningGNN(
        input_dim=347,
        hidden_channels=64,
        num_layers=3,
        GNNLayer=config['GNNLayer'],
        dropout=0.5,
        num_classes=4,
        mlp_hidden=[64, 32],
        lr=1e-3,
        edge_tsh=0,
        loss_type=config['loss_type'],
        class_weights=class_weights,
        residual_connections=True
    )

    # 4. Set up run-specific directory and checkpointing
    save_dir = RUNS_DIR / config['name']
    save_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_cb = ModelCheckpoint(
        dirpath=save_dir / "checkpoints",
        filename="best_model-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True
    )

    # 5. Configure and run the trainer
    trainer = pl.Trainer(
        max_epochs=50,
        default_root_dir=str(save_dir),
        callbacks=[ckpt_cb],
        accelerator="auto",
        devices=1,
        log_every_n_steps=20,
        enable_progress_bar=False, # Disable internal PL bar for a cleaner look
        enable_model_summary=False
    )
    
    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(ckpt_path="best", dataloaders=test_loader)
    except Exception as e:
        print(f"🚨 RUN FAILED: {config['name']} with error: {e}")

    # print(f"✅ FINISHED RUN: {config['name']}\n\n")

def main():
    """
    Defines the search space and launches the grid search.
    """
    param_grid = {
        'GNNLayer': [SAGEConv, GATConv],
        'augment_strategy': ['geodesic', 'interpolate', 'upsample'],
        'augment_proportion': [round(p, 2) for p in np.arange(0.3, 1.1, 0.1)],
        'loss_type': ['weighted_cross_entropy', 'cross_entropy']
    }

    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Add baseline experiments (No augmentation, standard CE) for each model type
    for gnn_layer in param_grid['GNNLayer']:
        baseline_config = {
            'GNNLayer': gnn_layer,
            'augment_strategy': None,
            'augment_proportion': None,
            'loss_type': 'cross_entropy'
        }
        experiments.insert(0, baseline_config)

    print(f"Generated {len(experiments)} unique experiments to run.")

    # Run all experiments with a tqdm progress bar
    for i, config in enumerate(tqdm(experiments, desc="Executing Grid Search")):
        model_name = config['GNNLayer'].__name__.replace("Conv","")
        strategy_name = config['augment_strategy'] or "baseline"
        prop_name = config['augment_proportion'] or 0.0
        loss_name = "w" if config['loss_type'] == 'weighted_cross_entropy' else "std"
        
        config['name'] = f"run{i+1:03d}_{model_name}_{strategy_name}_p{prop_name:.1f}_{loss_name}"
        
        run_experiment(config)

if __name__ == "__main__":
    main()
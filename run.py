import torch
import random
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from static_models.pipeline import train_model

def run_single_model():
    """
    A simplified runner for training a single model without a YAML config.
    Modify the parameters in this file to experiment.
    """
    # ─── Global Seed ───
    seed = 23
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ─── Data Settings ───
    data_path = "/project2/cdonnat/sleepstages/data/pt/non_overlapping_data.pt"
    batch_size = 48
    num_workers = 4
    train_ratio, val_ratio, test_ratio = [0.6, 0.2, 0.2]

    # ─── Model Configuration ───
    model_name = "bnt_test_run"
    model_arch = "bnt"  # 'bnt', 'gnn', 'neurograph', 'mlp', 'logistic'
    
    # Model-specific parameters
    model_params = {
        "input_dim": 347,      
        "num_classes": 4,     
        "n_layers": 1,
        "n_heads": 1,
        "n_clusters": 8,
        "lr": 1e-4,
        "wd": 1e-4,
        "loss_type": "weighted_cross_entropy",
    }

    # ─── Trainer Settings ───
    save_root = Path("/home/tcastellani/projects/run")
    run_dir = save_root / model_name
    max_epochs = 40
    accelerator = "auto"
    devices = 1
    
    print(f"\n======== Training '{model_name}' ({model_arch}) ========")
    
    trainer, metrics = train_model(
        model=model_arch,
        run_name=model_name,
        data_path=data_path,
        save_dir=run_dir,
        model_kwargs=model_params,
        # Data args
        batch_size=batch_size,
        workers=num_workers,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        # Trainer args
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        seed=seed,
    )
    
    print(f"\n======== Training Finished ========")
    print(f"Best model and logs saved in: {run_dir}")
    print("Test metrics:", metrics)

if __name__ == "__main__":
    run_single_model() 
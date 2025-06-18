from typing import List, Tuple, Optional, Dict
import os, hashlib, torch
#from torch_geometric.loader import DataLoader
from data.split import split_patient, temporal_splits
from data.augment import augment_upsample, augment_interpolate, augment_geodesic
import random
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

def _cache_file(
    raw_path: str,
    cache_root: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    augment_strategy: Optional[str],
    augment_proportion: Optional[float],
) -> str:
    """
    Build a unique cache filename based on the key hyper-parameters so that
    each (split, augmentation) setup gets its own .pt file.
    """
    os.makedirs(cache_root, exist_ok=True)
    key = f"{os.path.basename(raw_path)}_{train_ratio}_{val_ratio}_{test_ratio}_" \
          f"{augment_strategy}_{augment_proportion}"
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    return os.path.join(cache_root, f"cached_{h}.pt")

def load_data(
    path: str,
    batch_size: int = 32,
    workers: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    augment_strategy: Optional[str] = None,
    augment_proportion: Optional[float] = None,
    cache_root: str = "/scratch/midway3/tcastellani/sleepstages",
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, torch.Tensor]]:
    """
    Load → (split, augment) → cache → return PyG DataLoaders and weights.
    """
    # 1. validate split ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train + val + test must sum to 1.0")

    cache_path = _cache_file(path, cache_root, train_ratio, val_ratio, test_ratio,
                             augment_strategy, augment_proportion)

    if os.path.exists(cache_path):
        print(f"Loading cached dataset: {cache_path}")
        train_data, val_data, test_data = torch.load(cache_path, weights_only=False)
    else:
        # 2. load raw list[Data]
        dataset: List = torch.load(path, weights_only=False)
        print("Data loaded")

        # 3. split by patient
        print("Splitting dataset")
        split_data = split_patient(
            dataset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        train_data = [d for d in split_data if d.metadata["split"] == "train"]
        val_data   = [d for d in split_data if d.metadata["split"] == "val"]
        test_data  = [d for d in split_data if d.metadata["split"] == "test"]

        # 4. augment if requested
        if augment_strategy:
            if augment_proportion is None:
                raise ValueError("augment_proportion must be provided with a strategy.")
            if augment_strategy == "upsample":
                train_data = augment_upsample(train_data, proportion=augment_proportion)
            elif augment_strategy == "interpolate":
                train_data = augment_interpolate(train_data, proportion=augment_proportion)
            elif augment_strategy == "geodesic":
                train_data = augment_geodesic(train_data, proportion=augment_proportion)
            else:
                raise ValueError(f"Unknown augment_strategy: {augment_strategy}")
        print("Data augmented" if augment_strategy else "No augmentation")

        # 5. cache the processed split
        torch.save((train_data, val_data, test_data), cache_path)
        print(f"Cached dataset to {cache_path}")

    # 6. compute class weights from training set
    train_labels = torch.tensor([d.y.item() for d in train_data])
    val_labels  = torch.tensor([d.y.item() for d in val_data])
    test_labels = torch.tensor([d.y.item() for d in test_data])

    print("Train class distribution:", torch.bincount(train_labels).tolist())
    print("Val class distribution:", torch.bincount(val_labels).tolist()) 
    print("Test class distribution:", torch.bincount(test_labels).tolist())

    class_counts = torch.bincount(train_labels)
    class_weights = (1.0 / class_counts.float()) / (1.0 / class_counts.float()).sum()
    
    results = {'class_weights': class_weights}

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # 7. build loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=workers)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, num_workers=workers)

    return train_loader, val_loader, test_loader, results

def load_temporal_data(data_directory, batch_size, num_workers, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Calls the splitting function and creates train, validation, and test DataLoaders
    with parallel worker processes.
    """
    # 1. Get the lists of dictionaries for each split
    train_data_dicts, val_data_dicts, test_data_dicts = temporal_splits(
        data_directory, train_ratio, val_ratio, test_ratio
    )

    # 2. Extract only the 'loader' object from each dictionary
    train_dataset = [item['loader'] for item in train_data_dicts]
    val_dataset =   [item['loader'] for item in val_data_dicts]
    test_dataset =  [item['loader'] for item in test_data_dicts]

    # 3. Create the DataLoader for each dataset with workers
    # persistent_workers=True avoids restarting workers between epochs, speeding things up.
    # Note: Using num_workers > 0 on Windows can require extra setup (e.g., if __name__ == '__main__').
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader, test_loader
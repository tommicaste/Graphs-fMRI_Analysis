from typing import List, Tuple, Optional, Dict
import os, hashlib, torch
from torch_geometric.loader import DataLoader
from data.split import split_patient
from data.augment import augment_upsample, augment_interpolate, augment_geodesic
import random
import numpy as np
from pathlib import Path

def _cache_file(raw_path: str,cache_root: str,train_ratio: float,val_ratio: float,test_ratio: float,augment_strategy: Optional[str],augment_proportion: Optional[float]) -> str:
    """
    Build a unique cache filename based on key hyper-parameters for each (split, augmentation) setup
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
    cache_root: str = ".cache",
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, torch.Tensor]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train + val + test must sum to 1.0")

    cache_path = _cache_file(path, cache_root, train_ratio, val_ratio, test_ratio,
                             augment_strategy, augment_proportion)

    if os.path.exists(cache_path):
        print(f"Loading cached dataset: {cache_path}")
        train_data, val_data, test_data = torch.load(cache_path, weights_only=False)
    else:
        dataset: List = torch.load(path, weights_only=False)
        print("Data loaded")

        print("Splitting dataset")
        split_data = split_patient(
            dataset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        train_data = [d for d in split_data if d.metadata["split"] == "train"]
        val_data = [d for d in split_data if d.metadata["split"] == "val"]
        test_data = [d for d in split_data if d.metadata["split"] == "test"]

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

        torch.save((train_data, val_data, test_data), cache_path)
        print(f"Cached dataset to {cache_path}")

    train_labels = torch.tensor([d.y.item() for d in train_data])
    val_labels = torch.tensor([d.y.item() for d in val_data])
    test_labels = torch.tensor([d.y.item() for d in test_data])

    print("Train class distribution:", torch.bincount(train_labels).tolist())
    print("Val class distribution:", torch.bincount(val_labels).tolist())
    print("Test class distribution:", torch.bincount(test_labels).tolist())

    class_counts = torch.bincount(train_labels)
    class_weights = (1.0 / class_counts.float()) / (1.0 / class_counts.float()).sum()
    results = {'class_weights': class_weights}

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers)

    return train_loader, val_loader, test_loader, results
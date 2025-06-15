from typing import List, Tuple, Optional
import os, hashlib, torch
from torch_geometric.loader import DataLoader
from data.split import split_patient
from data.augment import augment_upsample, augment_interpolate

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
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    augment_strategy: Optional[str] = None,
    augment_proportion: Optional[float] = None,
    cache_root: str = "/scratch/midway3/tcastellani/sleepstages",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load → (split, augment) → cache → return PyG DataLoaders.
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
                pass
            if augment_strategy == "upsample":
                train_data = augment_upsample(train_data, proportion=augment_proportion)
            elif augment_strategy == "interpolate":
                train_data = augment_interpolate(train_data, proportion=augment_proportion)
            else:
                raise ValueError("augment_strategy must be 'upsample' or 'interpolate'")
        print("Data augmented" if augment_strategy else "No augmentation")

        # 5. cache the processed split
        torch.save((train_data, val_data, test_data), cache_path)
        print(f"Cached dataset to {cache_path}")

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # 6. build loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=workers)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, num_workers=workers)

    return train_loader, val_loader, test_loader


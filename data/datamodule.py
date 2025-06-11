import os, json, hashlib, random, numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data import (
    split_patient,
    augment_upsample,
    augment_interpolate,
    add_edge_list,
    feature_corr,
)


class StaticDataModule(pl.LightningDataModule):
    """
    Static-graph DataModule with on-disk graph cache.
    """

    def __init__(
        self,
        raw_path: str | Path,
        batch_size: int = 8,
        num_workers: int = 4,
        *,
        seed: int | None = None,
        cache_path: str | Path = "/scratch/midway3/tcastellani/sleep_cache",
        augment_cfg: Dict[str, Dict[str, Any]] | None = None,
        edge_tsh: float | None = 0.0,
        edge_top: float = 0.10,
        feature_cfg: Dict[str, Any] | None = None,
        split_ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.raw_path     = Path(raw_path)
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.pin_memory   = pin_memory
        self.seed         = seed
        self.cache_path   = Path(cache_path) if cache_path else None
        self.augment_cfg  = augment_cfg or {}
        self.edge_tsh     = edge_tsh
        self.edge_top     = edge_top
        self.feature_cfg  = feature_cfg or {}
        self.split_ratios = split_ratios

        self.train_data: List[Data] | None = None
        self.val_data:   List[Data] | None = None
        self.test_data:  List[Data] | None = None

    # Build cache key to save datasets
    def _build_cache_key(self) -> str:
        cfg = dict(
            raw_path     = str(self.raw_path.resolve()),
            seed         = self.seed,
            augment_cfg  = self.augment_cfg,
            edge_tsh     = self.edge_tsh,
            edge_top     = self.edge_top,
            feature_cfg  = self.feature_cfg,
            split_ratios = self.split_ratios,
        )
        return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]

    def _cache_file(self) -> Path:
        cache_dir = self.cache_path or self.raw_path.parent / ".cache_graphs"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{self._build_cache_key()}.pt"

    # Load raw data
    def prepare_data(self) -> None:
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {self.raw_path}")

    # Data setups depending on stage
    def setup(self, stage: str | None = None) -> None:
        if stage == "predict":
            return
        if stage in (None, "fit"):
            if self.train_data is None:
                self._setup_fit()
        if stage == "test":
            if self.test_data is None:
                self._setup_test()

    # Data setup for fit stage (train and val)
    def _setup_fit(self) -> None:
        cache_file = self._cache_file()
        if cache_file.exists():
            print(f"[FIT] Loading cached graphs from {cache_file}")
            data_list: List[Data] = torch.load(cache_file, weights_only=False)
        else:
            print("[FIT] Loading raw data …")
            data_list: List[Data] = torch.load(self.raw_path, weights_only=False)

            if self.seed is not None:
                random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)

            print("[FIT] Splitting patients …")
            tr, va, te = self.split_ratios
            data_list = split_patient(data_list, train_ratio=tr, val_ratio=va, test_ratio=te)

            if "upsample" in self.augment_cfg:
                print("[FIT] Applying upsample augmentation …")
                data_list = augment_upsample(data_list, **self.augment_cfg["upsample"])
            if "interpolate" in self.augment_cfg:
                print("[FIT] Applying interpolate augmentation …")
                data_list = augment_interpolate(data_list, **self.augment_cfg["interpolate"])

            print("[FIT] Building edge_index for all graphs …")
            data_list = add_edge_list(data_list, tsh=self.edge_tsh, top=self.edge_top)

            print("[FIT] Preparing features for all graphs …")
            data_list = feature_corr(data_list, **self.feature_cfg)

            torch.save(data_list, cache_file)
            print(f"[FIT] Cached graphs to {cache_file}")

        # split references
        self.train_data = [d for d in data_list if d.metadata["split"] == "train"]
        self.val_data   = [d for d in data_list if d.metadata["split"] == "val"]
        self.test_data  = [d for d in data_list if d.metadata["split"] == "test"]

        # drop the big list container to free RAM
        del data_list
        print(f"[FIT] Done.  train={len(self.train_data)}, val={len(self.val_data)}, test={len(self.test_data)}")

    # Data setup for fit test stage
    def _setup_test(self) -> None:
        cache_file = self._cache_file()
        if not cache_file.exists():
            # first ever run for this config – _setup_fit already populated self.test_data
            self._setup_fit()
            return

        print(f"[TEST] Loading cached graphs from {cache_file}")
        data_list: List[Data] = torch.load(cache_file, weights_only=False)
        self.test_data = [d for d in data_list if d.metadata["split"] == "test"]
        del data_list
        print(f"[TEST] Done.  test={len(self.test_data)}")

    # Build dataloader
    def _build_loader(self, data: List[Data], shuffle: bool) -> DataLoader:
        return DataLoader(
            data, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader: return self._build_loader(self.train_data, True)
    def val_dataloader(self)   -> DataLoader: return self._build_loader(self.val_data,   False)
    def test_dataloader(self)  -> DataLoader: return self._build_loader(self.test_data,  False)

    # Repr
    def __repr__(self) -> str:
        n_train = len(self.train_data) if self.train_data is not None else "?"
        n_val   = len(self.val_data)   if self.val_data   is not None else "?"
        n_test  = len(self.test_data)  if self.test_data  is not None else "?"
        return f"StaticDataModule(batch={self.batch_size}, train={n_train}, val={n_val}, test={n_test})"

import os
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
    Static-graph DataModule.

    Workflow for 'fit':
      1. load raw .pt file containing a list[Data]
      2. split patients into train / val / test
      3. optional class-balancing augmentations on train split only
      4. build edge_index from connectivity matrix for all splits
      5. move connectivity matrix into node-feature tensor via feature_corr
    Workflow for 'test':
      - If already ran fit, reuse processed test_data
      - Otherwise load raw, split, build edges, feature_corr, and keep only test split
    """

    def __init__(
        self,
        raw_path: str | Path,
        batch_size: int = 8,
        num_workers: int = 4,
        *,
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
        self.augment_cfg  = augment_cfg or {}
        self.edge_tsh     = edge_tsh
        self.edge_top     = edge_top
        self.feature_cfg  = feature_cfg or {}
        self.split_ratios = split_ratios

        # placeholders for processed splits
        self.train_data: List[Data] | None = None
        self.val_data:   List[Data] | None = None
        self.test_data:  List[Data] | None = None

    def prepare_data(self) -> None:
        """Validate existence of raw file (no heavy I/O)."""
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {self.raw_path}")

    def setup(self, stage: str | None = None) -> None:
        """Process data splits and features according to stage."""
        # skip predict
        if stage == "predict":
            return

        # Fit stage: only run once
        if stage in (None, "fit"):
            if self.train_data is not None:
                return
            self._setup_fit()

        # Test stage: if fit already ran, test_data is ready
        if stage == "test":
            if self.test_data is not None:
                return
            self._setup_test()

    def _setup_fit(self) -> None:
        print("🔄  [FIT] Loading raw data …")
        data_list: List[Data] = torch.load(self.raw_path, weights_only=False)

        print("🔄  [FIT] Splitting patients into train / val / test …")
        train_ratio, val_ratio, test_ratio = self.split_ratios
        data_list = split_patient(
            data_list,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        # augment train only
        if "upsample" in self.augment_cfg:
            print("🔄  [FIT] Applying upsample augmentation …")
            data_list = augment_upsample(data_list, **self.augment_cfg["upsample"])
        if "interpolate" in self.augment_cfg:
            print("🔄  [FIT] Applying interpolate augmentation …")
            data_list = augment_interpolate(data_list, **self.augment_cfg["interpolate"])

        print("🔄  [FIT] Building edge_index for all graphs …")
        data_list = add_edge_list(data_list, tsh=self.edge_tsh, top=self.edge_top)

        print("🔄  [FIT] Moving connectivity matrices to node features …")
        data_list = feature_corr(data_list, **self.feature_cfg)

        # split into lists
        self.train_data = [d for d in data_list if d.metadata["split"] == "train"]
        self.val_data   = [d for d in data_list if d.metadata["split"] == "val"]
        self.test_data  = [d for d in data_list if d.metadata["split"] == "test"]

        print(f"✅  [FIT] Done.  train={len(self.train_data)}, val={len(self.val_data)}, test={len(self.test_data)}")

    def _setup_test(self) -> None:
        print("🔄  [TEST] Loading raw data …")
        data_list: List[Data] = torch.load(self.raw_path, weights_only=False)

        print("🔄  [TEST] Splitting patients …")
        train_ratio, val_ratio, test_ratio = self.split_ratios
        data_list = split_patient(
            data_list,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        print("🔄  [TEST] Building edge_index for test graphs …")
        data_list = add_edge_list(data_list, tsh=self.edge_tsh, top=self.edge_top)

        print("🔄  [TEST] Moving connectivity matrices to node features …")
        data_list = feature_corr(data_list, **self.feature_cfg)

        self.test_data = [d for d in data_list if d.metadata["split"] == "test"]

        print(f"✅  [TEST] Done.  test={len(self.test_data)}")

    def _build_loader(self, data: List[Data], shuffle: bool) -> DataLoader:
        return DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        return self._build_loader(self.train_data, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._build_loader(self.val_data, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._build_loader(self.test_data, shuffle=False)

    def __repr__(self) -> str:
        n_train = len(self.train_data) if self.train_data is not None else "?"
        n_val   = len(self.val_data)   if self.val_data   is not None else "?"
        n_test  = len(self.test_data)  if self.test_data  is not None else "?"
        return (
            f"StaticDataModule(batch={self.batch_size}, "
            f"train={n_train}, val={n_val}, test={n_test})"
        )

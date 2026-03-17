from __future__ import annotations

from typing import List

import torch
from torch_geometric.data import Data, InMemoryDataset


class FMRIGraphDataset(InMemoryDataset):
    """In-memory PyG dataset wrapping a pre-built list of Data objects.

    Rather than owning raw files that need downloading/processing, this class
    accepts an already-constructed list of ``torch_geometric.data.Data`` objects
    (e.g. the output of the split/augment pipeline) and exposes the standard
    PyG dataset interface so that it can be used directly with
    ``torch_geometric.loader.DataLoader``.

    Parameters
    ----------
    data_list:
        List of ``Data`` objects to wrap.
    """

    def __init__(self, data_list: List[Data]) -> None:
        super().__init__(root=None, transform=None, pre_transform=None)
        self.data, self.slices = self.collate(data_list)

    def _download(self) -> None:  # pragma: no cover
        pass  # nothing to download

    def _process(self) -> None:  # pragma: no cover
        pass  # already in memory

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"

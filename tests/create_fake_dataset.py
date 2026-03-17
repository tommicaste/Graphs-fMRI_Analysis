"""Generate a synthetic fMRI graph dataset for smoke-testing the pipeline.

Produces ``tests/fake_dataset.pt``: a list of 80 ``torch_geometric.data.Data``
objects with N=26 nodes (matching the BNT default ``input_dim=26``), 4 classes,
and 8 fake patients.

Usage
-----
    python tests/create_fake_dataset.py
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
from torch_geometric.data import Data

# ── Parameters ──────────────────────────────────────────────────────────────
N_NODES     = 26    # number of ROIs; matches BNT default input_dim
N_SAMPLES   = 80    # total graph samples
N_CLASSES   = 4     # sleep stages
N_PATIENTS  = 8     # unique patient IDs
SEED        = 42

OUT_PATH = Path(__file__).parent / "fake_dataset.pt"
# ─────────────────────────────────────────────────────────────────────────────


def _make_corr_matrix(n: int, rng: torch.Generator) -> torch.Tensor:
    """Return a random symmetric correlation matrix with diagonal = 1."""
    A = torch.empty(n, n).uniform_(-1, 1, generator=rng)
    A = (A + A.T) / 2
    A.fill_diagonal_(1.0)
    return A


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    rng = torch.Generator()
    rng.manual_seed(SEED)

    dataset = []
    for i in range(N_SAMPLES):
        patient_id = f"pat_{i % N_PATIENTS}"
        segment    = i // N_PATIENTS

        x = _make_corr_matrix(N_NODES, rng)
        y = torch.tensor(i % N_CLASSES, dtype=torch.long)

        d = Data(
            x=x,
            y=y,
            metadata={"sample": patient_id, "segment": segment},
        )
        dataset.append(d)

    torch.save(dataset, OUT_PATH)
    print(f"Saved {len(dataset)} samples to {OUT_PATH}")
    print(f"  Node feature shape : {dataset[0].x.shape}")
    print(f"  Classes            : {sorted({int(d.y) for d in dataset})}")
    print(f"  Patients           : {sorted({d.metadata['sample'] for d in dataset})}")


if __name__ == "__main__":
    main()

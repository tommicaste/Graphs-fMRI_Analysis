from __future__ import annotations

import copy
import random
from collections import defaultdict
from typing import List

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _project_to_psd(C: torch.Tensor) -> torch.Tensor:
    """Project a symmetric matrix to the nearest symmetric positive semi-definite
    correlation matrix (diagonal = 1, off-diagonal in [-1, 1]).

    Parameters
    ----------
    C:
        Square symmetric tensor of shape ``(N, N)``.

    Returns
    -------
    torch.Tensor
        PSD correlation matrix of the same shape.
    """
    C = (C + C.T) / 2
    eigvals, eigvecs = torch.linalg.eigh(C)
    eigvals.clamp_(min=0)
    C_psd = eigvecs @ torch.diag(eigvals) @ eigvecs.T
    D = torch.sqrt(torch.diag(C_psd) + 1e-8)
    C_psd = C_psd / D[:, None] / D[None, :]
    C_psd.fill_diagonal_(1.0)
    return C_psd.clamp_(-1, 1)


# ---------------------------------------------------------------------------
# Public augmentation strategies
# ---------------------------------------------------------------------------

def augment_upsample(data_list: List[Data], proportion: float = 0.5) -> List[Data]:
    """Upsample minority classes by random duplication.

    For each class that has fewer samples than ``proportion * max_class_count``,
    randomly duplicate existing samples until the target is reached.

    Parameters
    ----------
    data_list:
        Training split (all samples should have ``metadata["split"] == "train"``).
    proportion:
        Target proportion of the largest class in ``(0, 1]``.

    Returns
    -------
    List[Data]
        Original list with synthetic duplicates appended.
    """
    assert 0 < proportion <= 1

    for d in data_list:
        d.metadata["synthetic"] = False

    train_by_class: dict[int, list] = defaultdict(list)
    for d in data_list:
        if d.metadata.get("split") == "train":
            train_by_class[int(d.y.item())].append(d)

    counts   = {cls: len(items) for cls, items in train_by_class.items()}
    max_count = max(counts.values(), default=0)
    target   = int(max_count * proportion)

    augmented: List[Data] = []
    for cls, examples in train_by_class.items():
        n_needed = target - len(examples)
        for _ in range(max(0, n_needed)):
            new_d = copy.deepcopy(random.choice(examples))
            new_d.metadata["synthetic"] = True
            new_d.metadata["split"]     = "train"
            augmented.append(new_d)

    return data_list + augmented


def augment_interpolate(
    data_list: List[Data],
    proportion: float = 1.0,
    verbose: bool = True,
) -> List[Data]:
    """Balance classes by linear interpolation between consecutive temporal samples.

    Interpolated matrices are projected onto the PSD correlation manifold.

    Parameters
    ----------
    data_list:
        Training split.
    proportion:
        Target proportion of the largest class in ``(0, 1]``.
    verbose:
        Show tqdm progress bars.

    Returns
    -------
    List[Data]
        Original list with synthetic interpolated samples appended.
    """
    assert 0 < proportion <= 1, "proportion must be in (0, 1]"

    for d in data_list:
        assert torch.is_tensor(d.x) and torch.is_tensor(d.y)
        assert torch.isfinite(d.x).all(), f"NaNs in x for {d.metadata}"
        assert (d.x != 0).all(), f"Zeros in x for {d.metadata}"
        d.metadata["synthetic"] = False

    train_by_class: dict[int, list] = defaultdict(list)
    for d in data_list:
        if d.metadata.get("split") == "train":
            train_by_class[int(d.y.item())].append(d)

    real_counts  = {cls: len(lst) for cls, lst in train_by_class.items()}
    max_count    = max(real_counts.values(), default=0)
    targets      = {cls: int(np.ceil(proportion * max_count)) for cls in real_counts}
    synth_needs  = {cls: max(0, targets[cls] - real_counts[cls]) for cls in real_counts}

    gaps_per_class: dict[int, list] = {}
    for cls, examples in train_by_class.items():
        by_patient: dict = defaultdict(list)
        for d in examples:
            by_patient[d.metadata["sample"]].append(d)
        pairs: list = []
        for seq in by_patient.values():
            seq.sort(key=lambda d: d.metadata["segment"])
            pairs.extend(zip(seq, seq[1:]))
        gaps_per_class[cls] = pairs

    synthetic_data: List[Data] = []
    for cls, need in tqdm(synth_needs.items(), desc="Augmenting classes", disable=not verbose, leave=False):
        pairs = gaps_per_class.get(cls, [])
        if need == 0 or not pairs:
            continue

        G = len(pairs)
        base, rem = divmod(need, G)
        quotas = [base + (i < rem) for i in range(G)]

        for (d1, d2), n_interp in tqdm(
            zip(pairs, quotas),
            total=G,
            desc=f"  class {cls}",
            disable=not verbose,
            leave=False,
        ):
            if n_interp == 0:
                continue
            s1, s2   = d1.metadata["segment"], d2.metadata["segment"]
            alphas   = torch.linspace(1, n_interp, n_interp, device=d1.x.device) / (n_interp + 1)
            Cs_interp = torch.stack([(1 - a) * d1.x + a * d2.x for a in alphas])

            for alpha, C_interp in zip(alphas, Cs_interp):
                new_d = Data(
                    y=d1.y.clone(),
                    x=_project_to_psd(C_interp),
                    metadata={
                        **d1.metadata,
                        "segment":   (1 - alpha.item()) * s1 + alpha.item() * s2,
                        "synthetic": True,
                    },
                )
                synthetic_data.append(new_d)

    return data_list + synthetic_data


def augment_geodesic(
    data_list: List[Data],
    proportion: float = 1.0,
    verbose: bool = True,
) -> List[Data]:
    """Balance classes by geodesic interpolation on the PSD matrix manifold.

    Parameters
    ----------
    data_list:
        Training split.
    proportion:
        Target proportion of the largest class in ``(0, 1]``.
    verbose:
        Show tqdm progress bars.

    Returns
    -------
    List[Data]
        Original list with synthetic geodesic-interpolated samples appended.
    """
    assert 0 < proportion <= 1, "proportion must be in (0, 1]"

    for d in data_list:
        assert torch.is_tensor(d.x) and torch.is_tensor(d.y)
        assert torch.isfinite(d.x).all(), f"NaNs in x for {d.metadata}"
        d.metadata["synthetic"] = False

    train_by_class: dict[int, list] = defaultdict(list)
    for d in data_list:
        if d.metadata.get("split") == "train":
            train_by_class[int(d.y.item())].append(d)

    real_counts = {cls: len(lst) for cls, lst in train_by_class.items()}
    max_count   = max(real_counts.values(), default=0)
    targets     = {cls: int(np.ceil(proportion * max_count)) for cls in real_counts}
    synth_needs = {cls: max(0, targets[cls] - real_counts[cls]) for cls in real_counts}

    gaps_per_class: dict[int, list] = {}
    for cls, examples in train_by_class.items():
        by_patient: dict = defaultdict(list)
        for d in examples:
            by_patient[d.metadata["sample"]].append(d)
        pairs: list = []
        for seq in by_patient.values():
            seq.sort(key=lambda d: d.metadata["segment"])
            pairs.extend(zip(seq, seq[1:]))
        gaps_per_class[cls] = pairs

    def _geodesic(C1: torch.Tensor, C2: torch.Tensor, t: float) -> torch.Tensor:
        """Geodesic interpolation C(t) on the manifold of PSD matrices."""
        eigvals1, eigvecs1 = torch.linalg.eigh(C1)
        eigvals1.clamp_(min=1e-8)
        C1_sqrt     = eigvecs1 @ torch.diag(eigvals1 ** 0.5)  @ eigvecs1.T
        C1_inv_sqrt = eigvecs1 @ torch.diag(eigvals1 ** -0.5) @ eigvecs1.T
        M           = C1_inv_sqrt @ C2 @ C1_inv_sqrt
        eigvalsM, eigvecsM = torch.linalg.eigh(M)
        eigvalsM.clamp_(min=0)
        M_t = eigvecsM @ torch.diag(eigvalsM ** t) @ eigvecsM.T
        return C1_sqrt @ M_t @ C1_sqrt

    synthetic_data: List[Data] = []
    for cls, need in tqdm(synth_needs.items(), desc="Geodesic augmentation", disable=not verbose, leave=False):
        pairs = gaps_per_class.get(cls, [])
        if need == 0 or not pairs:
            continue

        G = len(pairs)
        base, rem = divmod(need, G)
        quotas = [base + (i < rem) for i in range(G)]

        for (d1, d2), n_interp in tqdm(
            zip(pairs, quotas),
            total=G,
            desc=f"  class {cls}",
            disable=not verbose,
            leave=False,
        ):
            if n_interp == 0:
                continue
            s1, s2 = d1.metadata["segment"], d2.metadata["segment"]
            alphas = torch.linspace(1, n_interp, n_interp, device=d1.x.device) / (n_interp + 1)

            for alpha in alphas:
                C_interp = _geodesic(d1.x, d2.x, alpha.item())
                new_d = Data(
                    y=d1.y.clone(),
                    x=_project_to_psd(C_interp),
                    metadata={
                        **d1.metadata,
                        "segment":   (1 - alpha.item()) * s1 + alpha.item() * s2,
                        "synthetic": True,
                    },
                )
                synthetic_data.append(new_d)

    return data_list + synthetic_data

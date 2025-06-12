import random
import copy
from collections import defaultdict
import random
import torch
import numpy as np
from collections import defaultdict
from torch_geometric.data import Data
from tqdm import tqdm


def augment_upsample(data_list, proportion=0.5):
    """
    Ensure each class in the training split has at least proportion * (size of the largest class) samples by upsampling underrepresented classes
    """
    assert 0 < proportion <= 1

    # Mark all originals as non-synthetic
    for d in data_list:
        d.metadata['synthetic'] = False

    # Group training samples by class label
    train_by_class = defaultdict(list)
    for d in data_list:
        if d.metadata.get('split') == 'train':
            cls = int(d.y.item())
            train_by_class[cls].append(d)

    # Determine target count based on the largest class
    counts    = {cls: len(items) for cls, items in train_by_class.items()}
    max_count = max(counts.values(), default=0)
    target    = int(max_count * proportion)

    augmented = []
    for cls, examples in train_by_class.items():
        n_current = len(examples)
        if n_current < target:
            n_needed = target - n_current
            for _ in range(n_needed):
                # Deep-copy a random example and mark it synthetic
                original = random.choice(examples)
                new_example = copy.deepcopy(original)
                new_example.metadata['synthetic'] = True
                new_example.metadata['split']     = 'train'
                augmented.append(new_example)

    # Return combined list (originals + synthetic upsamples)
    return data_list + augmented

def augment_interpolate(
    data_list,
    proportion: float = 1.0,
    verbose: bool = True,
):
    """
    Interpolate between consecutive training samples to balance classes
    by `proportion` × (size of largest class).
    """
    assert 0 < proportion <= 1, "proportion must be in (0,1]"

   
    # Mark originals as non-synthetic 

    for d in data_list:
        assert torch.is_tensor(d.c) and torch.is_tensor(d.y)
        assert torch.isfinite(d.c).all(), f"NaNs in c for {d.metadata}"
        assert (d.c != 0).all(),          f"Zeros in c for {d.metadata}"
        d.metadata["synthetic"] = False


    # Group training samples by class and by patient

    train_by_class = defaultdict(list)
    for d in data_list:
        if d.metadata.get("split") == "train":
            train_by_class[int(d.y.item())].append(d)


    real_counts = {cls: len(lst) for cls, lst in train_by_class.items()}
    max_count   = max(real_counts.values(), default=0)
    targets     = {cls: int(np.ceil(proportion * max_count)) for cls in real_counts}
    synth_needs = {cls: max(0, targets[cls] - real_counts[cls]) for cls in real_counts}


    # Pre-compute patient-wise ordered gaps once, per class

    gaps_per_class = {}
    for cls, examples in train_by_class.items():
        by_patient = defaultdict(list)
        for d in examples:
            by_patient[d.metadata["sample"]].append(d)
        pairs = []
        for seq in by_patient.values():
            seq.sort(key=lambda d: d.metadata["segment"])
            pairs.extend(zip(seq, seq[1:]))  
        gaps_per_class[cls] = pairs


    # Helper: project matrix to symmetric, PSD, unit-diag correlation

    def project_to_psd(C: torch.Tensor) -> torch.Tensor:
        C = (C + C.T) / 2
        eigvals, eigvecs = torch.linalg.eigh(C)
        eigvals.clamp_(min=0)
        C_psd  = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        D      = torch.sqrt(torch.diag(C_psd) + 1e-8)
        C_psd  = C_psd / D[:, None] / D[None, :]
        C_psd.fill_diagonal_(1.0)
        return C_psd.clamp_(-1, 1)


    # Main interpolation loop (now progress-tracked)

    synthetic_data = []

    outer_iter = tqdm(
        synth_needs.items(),
        desc="⏩ classes",
        disable=not verbose,
        leave=False,
    )

    for cls, need in outer_iter:
        pairs = gaps_per_class.get(cls, [])
        if need == 0 or not pairs:
            continue

        G = len(pairs)
        base, rem = divmod(need, G)
        quotas = [base + (i < rem) for i in range(G)]

        inner_iter = (
            tqdm(
                zip(pairs, quotas),
                total=len(pairs),
                desc=f"  ↳ class {cls}",
                disable=not verbose,
                leave=False,
            )
            if verbose
            else zip(pairs, quotas)
        )

        for (d1, d2), n_interp in inner_iter:
            if n_interp == 0:
                continue
            s1, s2 = d1.metadata["segment"], d2.metadata["segment"]

            # vectorized alpha values → interpolate in one go
            alphas = torch.linspace(1, n_interp, n_interp, device=d1.c.device) / (n_interp + 1)
            Cs_interp = torch.stack([(1 - a) * d1.c + a * d2.c for a in alphas])

            for alpha, C_interp in zip(alphas, Cs_interp):
                new_d = Data(
                    y=d1.y.clone(),
                    c=project_to_psd(C_interp),
                    metadata={
                        **d1.metadata,
                        "segment": (1 - alpha.item()) * s1 + alpha.item() * s2,
                        "synthetic": True,
                    },
                )
                synthetic_data.append(new_d)

    return data_list + synthetic_data

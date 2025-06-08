import random
import copy
from collections import defaultdict
import random
import torch
import numpy as np
from collections import defaultdict
from torch_geometric.data import Data

def augment_upsample(data_list, proportion=0.5, random_state=23):
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

    # Upsample each underrepresented class
    random.seed(random_state)
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

def augment_interpolate(data_list, proportion=1.0, random_state=42):
    """
    Linearly interpolate between consecutive training samples to balance classes by a given proportion, ensuring all matrices remain valid.
    """
    assert 0 < proportion <= 1, "proportion must be in (0,1]"

    # mark originals as non-synthetic and validate input tensors
    for d in data_list:
        assert torch.is_tensor(d.c), "c must be a tensor"
        assert torch.is_tensor(d.y), "y must be a tensor"
        assert torch.isfinite(d.c).all(), f"Non-finite values in c for sample {d.metadata}"
        assert (d.c != 0).all(), f"Zero values found in c for sample {d.metadata}"
        d.metadata['synthetic'] = False

    # collect all training samples grouped by class
    train_by_class = defaultdict(list)
    for d in data_list:
        if d.metadata.get('split') == 'train':
            train_by_class[int(d.y.item())].append(d)

    # determine how many synthetic samples each class requires
    real_counts = {cls: len(lst) for cls, lst in train_by_class.items()}
    max_count   = max(real_counts.values(), default=0)
    targets     = {cls: int(np.ceil(proportion * max_count)) for cls in real_counts}
    synth_needs = {cls: max(0, targets[cls] - real_counts[cls]) for cls in real_counts}

    # gather adjacent sample pairs (gaps) for each class
    gaps_per_class = {}
    for cls, examples in train_by_class.items():
        by_patient = defaultdict(list)
        for d in examples:
            by_patient[d.metadata['sample']].append(d)
        pairs = []
        for seq in by_patient.values():
            seq.sort(key=lambda d: d.metadata['segment'])
            pairs.extend(zip(seq, seq[1:]))
        gaps_per_class[cls] = pairs

    # ensure a symmetric, PSD connectivity matrix after interpolation
    def project_to_psd(C):
        C = (C + C.T) / 2
        eigvals, eigvecs = torch.linalg.eigh(C)
        eigvals = torch.clamp(eigvals, min=0)
        C_psd  = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        D      = torch.sqrt(torch.diag(C_psd) + 1e-8)
        C_psd  = C_psd / D[:, None] / D[None, :]
        C_psd.fill_diagonal_(1.0)
        C_psd  = C_psd.clamp(-1, 1)
        assert torch.isfinite(C_psd).all(), "Interpolated C contains non-finite values"
        assert (C_psd != 0).all(), "Interpolated C contains zero values"
        return C_psd

    synthetic_data = []
    random.seed(random_state)

    # interpolate within each gap according to calculated quotas
    for cls, need in synth_needs.items():
        pairs = gaps_per_class.get(cls, [])
        G = len(pairs)
        if need <= 0 or G == 0:
            continue
        base, rem = divmod(need, G)
        quotas = [base + (1 if i < rem else 0) for i in range(G)]

        for (d1, d2), n_interp in zip(pairs, quotas):
            if n_interp <= 0:
                continue
            s1, s2 = d1.metadata['segment'], d2.metadata['segment']
            for i in range(1, n_interp + 1):
                alpha = i / (n_interp + 1)
                C_interp = (1 - alpha) * d1.c + alpha * d2.c
                C_psd    = project_to_psd(C_interp)
                # carry over metadata and flag as synthetic
                new = Data(
                    y=d1.y.clone(),
                    c=C_psd,
                    metadata={**d1.metadata, 'segment': (1 - alpha) * s1 + alpha * s2, 'synthetic': True}
                )
                synthetic_data.append(new)

    # return dataset augmented with interpolated samples
    return data_list + synthetic_data

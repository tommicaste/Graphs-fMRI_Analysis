import numpy as np
import random
from collections import defaultdict

def split_patient(data_list,
                  train_ratio=0.6,
                  val_ratio=0.2,
                  test_ratio=0.2):
    """
    Greedy stratified group split by patient, preserving class proportions and avoiding patient overlap.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Count per-class segment totals for each sample
    sample_counts = {}
    all_labels = [int(d.y.item()) for d in data_list if not getattr(d, "synthetic", False)]
    num_classes = max(all_labels) + 1

    for d in data_list:
        if getattr(d, "synthetic", False):
            continue
        pid = d.metadata['sample']
        if pid not in sample_counts:
            sample_counts[pid] = np.zeros(num_classes, dtype=int)
        sample_counts[pid][int(d.y.item())] += 1

    patients = list(sample_counts.keys())

    # Compute target segment totals per split
    total_per_class = sum(sample_counts.values())
    targets = {
        'train': total_per_class * train_ratio,
        'val':   total_per_class * val_ratio,
        'test':  total_per_class * test_ratio,
    }
    running = {k: np.zeros(num_classes, dtype=float) for k in targets}
    random.shuffle(patients)

    # Greedy assignment: place each patient where they reduce class imbalance most
    assignment = {}
    for pid in patients:
        deficits = {k: targets[k] - running[k] for k in running}
        scores = {k: np.dot(deficits[k], sample_counts[pid]) for k in deficits}
        best = max(scores, key=scores.get)
        assignment[pid] = best
        running[best] += sample_counts[pid]

    # Tag each data point with its assigned split
    for d in data_list:
        if getattr(d, "synthetic", False):
            d.metadata['split'] = 'train'
        else:
            d.metadata['split'] = assignment[d.metadata['sample']]

    return data_list





import numpy as np
import random
from collections import defaultdict
from pathlib import Path
import torch
from typing import List, Dict, Tuple, Optional
from torch_geometric.data import Data, DataLoader


def split_patient(data_list,
                  train_ratio=0.6,
                  val_ratio=0.2,
                  test_ratio=0.2):
    """
    Greedy stratified group split by patient, preserving class proportions and avoiding patient overlap.
    Falls back to stratified random split if patient identifiers are missing.

    Args:
        data_list (list): List of Data objects.
        train_ratio (float): Training split ratio.
        val_ratio (float): Validation split ratio.
        test_ratio (float): Test split ratio.

    Returns:
        list: Data objects with 'split' field in metadata.
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    for d in data_list:
        if not hasattr(d, "metadata"):
            d.metadata = {}

    has_patient_ids = all(
        getattr(d, "synthetic", False) or ("sample" in d.metadata)
        for d in data_list
    )

    if not has_patient_ids:
        rng = np.random.default_rng()
        indices_by_class = {}
        for idx, d in enumerate(data_list):
            if getattr(d, "synthetic", False):
                continue
            label = int(d.y.item())
            indices_by_class.setdefault(label, []).append(idx)
        split_assignment = {}
        for cls, idxs in indices_by_class.items():
            rng.shuffle(idxs)
            n = len(idxs)
            n_train = int(round(n * train_ratio))
            n_val = int(round(n * val_ratio))
            n_test = n - n_train - n_val
            for i in idxs[:n_train]:
                split_assignment[i] = "train"
            for i in idxs[n_train:n_train + n_val]:
                split_assignment[i] = "val"
            for i in idxs[n_train + n_val:]:
                split_assignment[i] = "test"
        for idx, d in enumerate(data_list):
            if getattr(d, "synthetic", False):
                d.metadata["split"] = "train"
            else:
                d.metadata["split"] = split_assignment.get(idx, "train")
        return data_list

    sample_counts = {}
    all_labels = [int(d.y.item()) for d in data_list if not getattr(d, "synthetic", False)]
    num_classes = max(all_labels) + 1 if all_labels else 0
    for d in data_list:
        if getattr(d, "synthetic", False):
            continue
        pid = d.metadata["sample"]
        if pid not in sample_counts:
            sample_counts[pid] = np.zeros(num_classes, dtype=int)
        sample_counts[pid][int(d.y.item())] += 1
    patients = list(sample_counts.keys())
    total_per_class = np.sum(list(sample_counts.values()), axis=0)
    targets = {
        "train": total_per_class * train_ratio,
        "val": total_per_class * val_ratio,
        "test": total_per_class * test_ratio,
    }
    running = {k: np.zeros(num_classes, dtype=float) for k in targets}
    assignment = {}
    for pid in patients:
        deficits = {k: targets[k] - running[k] for k in running}
        scores = {k: np.dot(deficits[k], sample_counts[pid]) for k in deficits}
        best = max(scores, key=lambda k: scores[k])
        assignment[pid] = best
        running[best] += sample_counts[pid]
    for d in data_list:
        if getattr(d, "synthetic", False):
            d.metadata["split"] = "train"
        else:
            d.metadata["split"] = assignment[d.metadata["sample"]]
    return data_list

def temporal_splits(data_directory, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Performs a greedy, stratified, group-by-patient split of temporal data.

    Args:
        data_directory (str): Directory containing temporal data files.
        train_ratio (float): Training split ratio.
        val_ratio (float): Validation split ratio.
        test_ratio (float): Test split ratio.

    Returns:
        tuple: Lists of dictionaries for train, val, and test splits.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    patient_data = defaultdict(list)
    patient_class_counts = defaultdict(lambda: defaultdict(int))
    all_class_indices = set()
    for file_path in Path(data_directory).glob('*.pt'):
        for item in torch.load(file_path):
            patient_id = item['id'].split('_')[0]
            patient_data[patient_id].append(item)
            for cls, count in item['class_counts'].items():
                patient_class_counts[patient_id][cls] += count
                all_class_indices.add(cls)
    num_classes = max(all_class_indices) + 1 if all_class_indices else 0
    patient_ids = list(patient_data.keys())
    random.shuffle(patient_ids)
    patient_counts_np = {
        pid: np.array([counts.get(i, 0) for i in range(num_classes)])
        for pid, counts in patient_class_counts.items()
    }
    total_per_class = np.sum(list(patient_counts_np.values()), axis=0)
    targets = {
        'train': total_per_class * train_ratio,
        'val': total_per_class * val_ratio,
        'test': total_per_class * test_ratio,
    }
    running_counts = {k: np.zeros(num_classes) for k in targets}
    patient_assignment = {}
    for pid in patient_ids:
        deficits = {k: targets[k] - running_counts[k] for k in running_counts}
        scores = {k: np.dot(deficits[k], patient_counts_np[pid]) for k in deficits}
        best_split = max(scores, key=lambda k: scores[k])
        patient_assignment[pid] = best_split
        running_counts[best_split] += patient_counts_np[pid]
    temp_splits = defaultdict(list)
    for pid, assigned_split in patient_assignment.items():
        temp_splits[assigned_split].extend(patient_data[pid])
    train_data = [{'id': item['id'], 'loader': item['loader']} for item in temp_splits['train']]
    val_data = [{'id': item['id'], 'loader': item['loader']} for item in temp_splits['val']]
    test_data = [{'id': item['id'], 'loader': item['loader']} for item in temp_splits['test']]
    return train_data, val_data, test_data


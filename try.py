import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import random

def temporal_splits(data_directory, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """Performs a greedy, stratified, group-by-patient split of temporal data."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Aggregate data and class counts for each patient from all .pt files
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
    random.shuffle(patient_ids) # Shuffle for random assignment

    patient_counts_np = {
        pid: np.array([counts.get(i, 0) for i in range(num_classes)])
        for pid, counts in patient_class_counts.items()
    }

    # Calculate target class distributions for each split
    total_per_class = sum(patient_counts_np.values())
    targets = {
        'train': total_per_class * train_ratio,
        'val':   total_per_class * val_ratio,
        'test':  total_per_class * test_ratio,
    }
    running_counts = {k: np.zeros(num_classes) for k in targets}

    # Greedily assign each patient to the best-fitting split
    patient_assignment = {}
    for pid in patient_ids:
        deficits = {k: targets[k] - running_counts[k] for k in running_counts}
        scores = {k: np.dot(deficits[k], patient_counts_np[pid]) for k in deficits}
        best_split = max(scores, key=scores.get)
        patient_assignment[pid] = best_split
        running_counts[best_split] += patient_counts_np[pid]

    # Create temporary lists based on assignments
    temp_splits = defaultdict(list)
    for pid, assigned_split in patient_assignment.items():
        temp_splits[assigned_split].extend(patient_data[pid])

    # Create the final lists with the desired dictionary structure
    # Each dictionary will only contain the 'id' and 'loader'
    train_data = [{'id': item['id'], 'loader': item['loader']} for item in temp_splits['train']]
    val_data =   [{'id': item['id'], 'loader': item['loader']} for item in temp_splits['val']]
    test_data =  [{'id': item['id'], 'loader': item['loader']} for item in temp_splits['test']]
        
    return train_data, val_data, test_data

# --- Example Usage ---
DATA_DIR = "/project2/cdonnat/sleepstages/data/pt/dynamic/"
print(f"Processing data in {DATA_DIR}...")

# The function now returns three separate lists
train_data, val_data, test_data = temporal_splits(DATA_DIR)

# --- Summary ---
print("\n--- Split Summary ---")
# Create a temporary dictionary to easily loop for printing the summary
all_splits_for_summary = {'train': train_data, 'val': val_data, 'test': test_data}

for name, data_list in all_splits_for_summary.items():
    if not data_list: # Handle case where a split might be empty
        num_patients = 0
    else:
        num_patients = len(set(item['id'].split('_')[0] for item in data_list))
    
    print(f"{name.upper():<6} | Patients: {num_patients:<4} | Segments: {len(data_list):<5}")
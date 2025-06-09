import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from data import split_patient, augment_interpolate, add_edge_list, feature_corr

# --- Load ---
path = "/project2/cdonnat/sleepstages/data/pt/overlapping_data.pt"
data_list = torch.load(path, weights_only=False)
print(f"✅ Loaded {len(data_list)} data files\n")

# --- Initial split and dataset composition ---
data = split_patient(data_list)
splits = defaultdict(int)
for d in data:
    splits[d.metadata['split']] += 1
print("📊 Split composition (by segment count):")
for split, cnt in splits.items():
    print(f"   {split:5s}: {cnt}")
# check patient overlap
patients = defaultdict(set)
for d in data:
    patients[d.metadata['split']].add(d.metadata['sample'])
overlaps = {
    (a, b): patients[a] & patients[b]
    for a in patients for b in patients if a < b
}
print("\n🔍 Patient overlaps:")
for (a, b), shared in overlaps.items():
    print(f"   {a} ∩ {b}: {len(shared)} patients")

# --- After interpolation augmentation ---
data_aug = augment_interpolate(data, proportion=0.5)
aug_counts = defaultdict(int)
for d in data_aug:
    if d.metadata['synthetic']:
        aug_counts['synthetic'] += 1
    else:
        aug_counts['original'] += 1
print(f"\n🛠  After interpolation: {aug_counts['synthetic']} synthetic added, {aug_counts['original']} original remain")

# --- Feature transform and final attribute check ---
data_feat = feature_corr(data_aug)

# Public Data attributes
attrs = set(data_feat[0].keys())
print(f"\n🔧 Post-feature attributes on each Data object: {attrs}")

# Metadata keys
meta_keys = set(data_feat[0].metadata.keys())
print(f"🗂️ Metadata keys on each Data object: {meta_keys}")


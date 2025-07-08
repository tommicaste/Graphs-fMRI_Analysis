
### Data Prep

import numpy as np
import torch
from torch_geometric.data import Data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import random
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import os
import json
from models import TimeSeriesTransformer
from utils import project_to_psd, simulate_timeseries, estimate_c, evaluate_classification, prepare_data, simulate_data

seed = 23
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data = torch.load("/project2/cdonnat/sleepstages/data/pt/non_overlapping_data.pt", weights_only=False)

corr_data = prepare_data(data)


from collections import Counter, defaultdict

split_counts = defaultdict(Counter)
patient_in_split = defaultdict(set)

for d in corr_data:
    split = d["split"]
    y = d["y"].item()
    split_counts[split][y] += 1
    patient_in_split[split].add(d["patient_id"])

print("Class distribution by split:")
for split in ("train", "val", "test"):
    print(f"{split}: {dict(split_counts[split])}")

print("\nPatient overlap across splits:")
splits = list(patient_in_split)
for i in range(len(splits)):
    for j in range(i + 1, len(splits)):
        a, b = splits[i], splits[j]
        common = patient_in_split[a] & patient_in_split[b]
        tag = "NONE" if not common else common
        print(f"{a} ∩ {b}: {tag}")


### Simulation
phi, T, n  = 0, 100, 400

timeseries = simulate_data(corr_data, phi, T, n)
print(f"Shape of estimated correlation matrix: {timeseries[0]['c_hat'].shape}")
print(f"Shape of time series paths: {timeseries[0]['seq'].shape}")
print(f"Total number of paths: {len(timeseries)*n}")


# ## Transformer

# Loaders

from torch.utils.data import TensorDataset, DataLoader
splits = {'train': [], 'val': [], 'test': []}
for dp in timeseries:                     
    lbl = dp['y'].item()
    for seq in dp['seq']:                
        splits[dp['split']].append((seq.squeeze(), lbl))

data_tensors, loaders = {}, {}
for name, items in splits.items():
    X = torch.stack([s[0] for s in items])      
    y = torch.tensor([s[1] for s in items])    
    data_tensors[name] = {'sequences': X, 'labels': y}
    loaders[name]     = DataLoader(TensorDataset(X, y),
                                   batch_size=64,
                                   shuffle=(name == 'train'))
    print(f'{name}: sequences {X.shape}, labels {y.shape}')




# Transfomer main model that is trained to predict one extra token for classification
model = TimeSeriesTransformer(
    input_dim=347,
    num_classes=4,
    d_model=64,
    num_layers=4,
    num_heads=4,
    dropout=0.3
)

import torch.optim as optim
from tqdm import tqdm

criterion  = nn.CrossEntropyLoss()
optimizer  = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

n_epochs = 10    
for epoch in range(1, n_epochs + 1):

    
    model.train()
    running_loss = running_correct = running_total = 0
    pbar = tqdm(loaders['train'], desc=f'Epoch {epoch} [train]', leave=False)
    for seqs, labels in pbar:
        seqs, labels = seqs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(seqs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss   += loss.item() * labels.size(0)
        running_correct+= (logits.argmax(1) == labels).sum().item()
        running_total  += labels.size(0)
        pbar.set_postfix(loss=running_loss / running_total,
                         acc =running_correct / running_total)

    
    model.eval()
    val_loss = val_correct = val_total = 0
    with torch.no_grad():
        for seqs, labels in loaders['val']:
            seqs, labels = seqs.to(device), labels.to(device)
            logits = model(seqs)
            val_loss  += criterion(logits, labels).item() * labels.size(0)
            val_correct+= (logits.argmax(1) == labels).sum().item()
            val_total  += labels.size(0)
    print(f'Epoch {epoch}: val loss {val_loss / val_total:.4f} | '
          f'val acc {val_correct / val_total:.4f}')


model.eval()
y_true_list, y_pred_list = [], []
with torch.no_grad():
    for seqs, labels in loaders['test']:
        seqs, labels = seqs.to(device), labels.to(device)
        logits = model(seqs)
        preds  = logits.argmax(1)
        y_true_list.append(labels.cpu())
        y_pred_list.append(preds.cpu())

y_true = torch.cat(y_true_list).numpy()
y_pred = torch.cat(y_pred_list).numpy()

# Evaluate and print metrics using the helper function
_ = evaluate_classification(
        y_true,
        y_pred,
        metrics=True,
        display=True 
    )







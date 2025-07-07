import random
from collections import Counter, defaultdict
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from datasets import Dataset, DatasetDict
from torchmetrics.classification import MulticlassAccuracy
from pathlib import Path
from typing import List, Optional, Any
import json
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from transformer_utility import (
    project_to_psd,
    evaluate_classification,
    build_hf_datasets,
    simulate_ar1,
    LightningTSClassifier,
    LightningCorrClassifier,
)

# ─── Configuration ───────────────────────────────────────────────────
DATA_PATH = "/project2/cdonnat/sleepstages/data/pt/non_overlapping_data.pt"

# output directories
BASE_RUN_DIR = Path("/home/tcastellani/projects/run_transformer")

# data split ratios
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.6, 0.2, 0.2
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

# AR(1) simulation parameters
AR1_PHI  = 0.3
# range of sequence lengths to evaluate
SEQ_LENS = [16, 32, 64, 128]
# number of AR(1) paths per covariance matrix
NUM_PATHS_PER_COV = 64

# dataloader parameters
BATCH_SIZE   = 64
NUM_WORKERS  = 4

# Transformer hyper-parameters
TS_D_MODEL   = 64
TS_N_LAYERS  = 4
TS_N_HEADS   = 4
TS_DROPOUT   = 0.5

# Linear classifier hyper-parameters
LINEAR_DROPOUT = 0.5

# training parameters
MAX_EPOCHS = 30

# list of noise levels (SNR); 0 means no noise
NOISE_SNRS = [0.0, 1.0]  # adjust the second value as desired

# ─── Load raw correlation matrices ────────────────────────────────────
raw_data = torch.load(DATA_PATH, weights_only=False)

# ─── Group matrices by patient & label ────────────────────────────────
grouped_matrices = defaultdict(list)
for sample in raw_data:
    patient_id  = sample.metadata["sample"]
    stage_label = sample.y[0].item()
    corr_matrix = sample.x if torch.is_tensor(sample.x) else torch.tensor(sample.x)
    grouped_matrices[(patient_id, stage_label)].append(corr_matrix)

# ─── Build PSD correlation matrix per patient ────────────────────────
patient_corr_records: list[dict] = []
for (patient_id, stage_label), mats in grouped_matrices.items():
    corr_avg = torch.stack(mats).mean(0)
    corr_psd = project_to_psd(corr_avg)
    patient_corr_records.append(
        {
            "patient_id": patient_id,
            "y": torch.tensor(stage_label),
            "c": corr_psd,
        }
    )

# ─── Compute sample counts per patient & class ───────────────────────
sample_counts = defaultdict(Counter)
for record in patient_corr_records:
    sample_counts[record["patient_id"]][record["y"]] += 1

classes = sorted({int(y) for counts in sample_counts.values() for y in counts})
num_classes = max(classes) + 1

patients = list(sample_counts)
counts_matrix = np.zeros((len(patients), num_classes), dtype=int)
for i, pid in enumerate(patients):
    for y, cnt in sample_counts[pid].items():
        counts_matrix[i, y] = cnt

total_per_class = counts_matrix.sum(axis=0).astype(float)
targets = {
    "train": total_per_class * TRAIN_RATIO,
    "val": total_per_class * VAL_RATIO,
    "test": total_per_class * TEST_RATIO,
}
running = {k: np.zeros_like(total_per_class) for k in targets}
assignment = {}

order = patients.copy()
random.shuffle(order)
for pid in order:
    counts = counts_matrix[patients.index(pid)]
    best_split, best_score = None, -np.inf
    for split in ("train", "val", "test"):
        deficit = targets[split] - running[split]
        score = np.dot(deficit, counts)
        if score > best_score:
            best_split, best_score = split, score
    chosen_split = best_split or "train"
    assignment[pid] = chosen_split
    running[chosen_split] += counts

for record in patient_corr_records:
    record["split"] = assignment[record["patient_id"]]

# ─── Helper: run one experiment ───────────────────────────────────────

def run_experiment(seq_len: int, noise_snr: float):
    """Train both models for a given sequence length and noise level."""

    # 1) simulate sequences -------------------------------------------------
    simulated_records: list[dict] = []
    snr_arg = None if noise_snr == 0 else noise_snr
    for corr_record in patient_corr_records:
        corr_matrix = corr_record["c"]
        for _ in range(NUM_PATHS_PER_COV):
            seq = simulate_ar1(corr_matrix, AR1_PHI, seq_len, snr=snr_arg)[1:]
            simulated_records.append({**corr_record, "seq": seq})

    # 2) build datasets ----------------------------------------------------
    ts_dataset_dict = build_hf_datasets(simulated_records)

    # correlation-vector dataset
    corr_feature_records: list[dict] = []
    for sim_record in simulated_records:
        C_hat = torch.corrcoef(sim_record["seq"].T)
        idx   = torch.tril_indices(C_hat.size(0), C_hat.size(1), offset=-1)
        corr_vec = C_hat[idx[0], idx[1]]
        corr_feature_records.append({**sim_record, "corr_vec": corr_vec})
    corr_dataset_dict = build_hf_datasets([{**r, "seq": r["corr_vec"]} for r in corr_feature_records])

    print(f"[INFO] seq_len={seq_len}  noise_snr={noise_snr}  ▶  dataset sizes →",
          {k: len(v) for k, v in ts_dataset_dict.items()})

    # 3) data loaders ------------------------------------------------------
    def make_loader(ds_dict, split, shuffle):
        return DataLoader(
            ds_dict[split],
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    train_loader = make_loader(ts_dataset_dict, "train", True)
    val_loader   = make_loader(ts_dataset_dict, "validation", False)
    test_loader  = make_loader(ts_dataset_dict, "test", False)

    # correlation loaders
    corr_train_loader = make_loader(corr_dataset_dict, "train", True)
    corr_val_loader   = make_loader(corr_dataset_dict, "validation", False)
    corr_test_loader  = make_loader(corr_dataset_dict, "test", False)

    # 4) train Transformer model ------------------------------------------
    ts_run_dir = BASE_RUN_DIR / "timeseries" / f"seq{seq_len}_noise{noise_snr}"
    ts_run_dir.mkdir(parents=True, exist_ok=True)

    ts_model = LightningTSClassifier(
        T=seq_len,
        d_model=TS_D_MODEL,
        n_layers=TS_N_LAYERS,
        n_heads=TS_N_HEADS,
        dropout=TS_DROPOUT,
    )

    ts_ckpt_cb = ModelCheckpoint(
        dirpath=ts_run_dir / "checkpoints",
        filename="epoch={epoch:02d}_val_bacc={val_bacc:.4f}",
        monitor="val_bacc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    ts_trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        default_root_dir=str(ts_run_dir),
        callbacks=[ts_ckpt_cb],
        accelerator="gpu",
        devices="auto",
        log_every_n_steps=10,
    )

    ts_trainer.fit(ts_model, train_loader, val_loader)
    ts_trainer.test(ckpt_path=ts_ckpt_cb.best_model_path or "best", dataloaders=test_loader)

    # 5) train correlation-vector model -----------------------------------
    input_dim = corr_feature_records[0]["corr_vec"].numel()
    corr_model = LightningCorrClassifier(input_dim=input_dim, C=num_classes, dropout=LINEAR_DROPOUT)

    corr_run_dir = BASE_RUN_DIR / "corrvec" / f"seq{seq_len}_noise{noise_snr}"
    corr_run_dir.mkdir(parents=True, exist_ok=True)

    corr_ckpt_cb = ModelCheckpoint(
        dirpath=corr_run_dir / "checkpoints",
        filename="epoch={epoch:02d}_val_bacc={val_bacc:.4f}",
        monitor="val_bacc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    corr_trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        default_root_dir=str(corr_run_dir),
        callbacks=[corr_ckpt_cb],
        accelerator="gpu",
        devices="auto",
        log_every_n_steps=10,
    )

    corr_trainer.fit(corr_model, corr_train_loader, corr_val_loader)
    corr_trainer.test(ckpt_path=corr_ckpt_cb.best_model_path or "best", dataloaders=corr_test_loader)

    # free memory ----------------------------------------------------------
    del ts_model, corr_model, ts_trainer, corr_trainer
    torch.cuda.empty_cache()

# ─── orchestrate grid of experiments ──────────────────────────────────

if __name__ == "__main__":
    for seq_len in SEQ_LENS:
        for noise_snr in NOISE_SNRS:
            run_experiment(seq_len, noise_snr)


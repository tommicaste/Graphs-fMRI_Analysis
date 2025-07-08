import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import os
import json
from collections import defaultdict, Counter
import random

def project_to_psd(C: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    C = 0.5 * (C + C.T)

    vals, vecs = torch.linalg.eigh(C)
    C = vecs @ torch.diag(torch.clamp(vals, min=eps)) @ vecs.T

    std = torch.sqrt(torch.diag(C))
    C = C / std[:, None] / std[None, :]
    C.fill_diagonal_(1.0)

    lambda_min = torch.linalg.eigvalsh(C).min()
    if lambda_min < eps:
        C += torch.eye(C.size(0), device=C.device) * (eps - lambda_min + eps)
        std = torch.sqrt(torch.diag(C))
        C = C / std[:, None] / std[None, :]
        C.fill_diagonal_(1.0)

    return C.clamp_(-1.0, 1.0)

def simulate_timeseries(C: torch.Tensor, phi: float, T: int, n: int = 1, snr: float | None = None):
    """Generate n AR(1) time series paths from covariance matrix C.
    
    Returns tensor of shape (n, T+1, d) where d is the dimension of C.
    """
    Sigma_tilde = (1 - phi ** 2) * C
    L_C     = torch.linalg.cholesky(C)
    L_tilde = torch.linalg.cholesky(Sigma_tilde)

    d = C.size(0)
    x = torch.empty(n, T + 1, d)
    
    # Generate n initial conditions
    x[:, 0] = torch.randn(n, d) @ L_C.T
    
    # Generate all noise terms at once
    eps = torch.randn(n, T, d) @ L_tilde.T

    # Generate the AR(1) process for all paths
    for t in range(1, T + 1):
        x[:, t] = phi * x[:, t - 1] + eps[:, t - 1]

    if snr is None:
        return x
    noise_std = (1 / snr) ** 0.5
    return x + torch.randn_like(x) * noise_std

def estimate_c(paths: torch.Tensor) -> torch.Tensor:
    n, T, d = paths.shape
    # demean
    centered = paths - paths.mean(dim=1, keepdim=True)  # (n, T, d)
    
    # Compute covariance matrices 
    cov_matrices = torch.bmm(
        centered.transpose(1, 2),  # (n, d, T)
        centered                   # (n, T, d)
    ) / (T - 1)  # (n, d, d)
    
    # Extract standard deviations
    std_devs = torch.sqrt(torch.diagonal(cov_matrices, dim1=1, dim2=2))  # (n, d)
    
    # Normalize to get correlation matrices
    corr_matrices = cov_matrices / (std_devs.unsqueeze(2) * std_devs.unsqueeze(1))
    for i in range(n):
        corr_matrices[i].fill_diagonal_(1.0)
    
    return corr_matrices

def evaluate_classification(
    y_true,
    y_pred,
    *,
    labels: list[int] | None = None,
    metrics: bool = False,
    plot: bool = False,
    display: bool = False,
    save_confusion_path: str | None = None,
    save_report_path: str | None = None
):
    """
    Compute standard and balanced accuracy metrics and optionally print, plot, or save them.
    """
    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_share = np.nan_to_num(cm_raw / cm_raw.sum(axis=1, keepdims=True))
    if plot or save_confusion_path:
        disp = ConfusionMatrixDisplay(cm_share, display_labels=labels)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", values_format=".2f")
        ax.set_title("Confusion Matrix")
        if save_confusion_path:
            os.makedirs(os.path.dirname(save_confusion_path), exist_ok=True)
            fig.savefig(save_confusion_path, bbox_inches="tight")
        if plot:
            plt.show()
        plt.close(fig)
    rep_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)  # type: ignore[arg-type]
    if metrics:
        print(f"Accuracy         : {acc:.4f}")
        print(f"Balanced accuracy: {bacc:.4f}")
        print("Classification report:")
        print(classification_report(y_true, y_pred, zero_division=0))  # type: ignore[arg-type]
    if save_report_path:
        os.makedirs(os.path.dirname(save_report_path), exist_ok=True)
        json.dump({
            "accuracy"              : acc,
            "balanced_accuracy"     : bacc,
            "confusion_matrix_share": cm_share.tolist(),
            "classification_report" : rep_dict
        }, open(save_report_path, "w"), indent=2)
    return {
        "accuracy"         : acc,
        "balanced_accuracy": bacc
    }

def simulate_data(corr_data, phi, T, n):
    timeseries = []
    for d in corr_data:
        paths = simulate_timeseries(d['c'], phi, T, n)[:, 1:] 
        c_hat = estimate_c(paths)
        timeseries.append({**d, "seq": paths, "c_hat": c_hat}) 
    return timeseries


def prepare_data(data):
    # Group by patient and stage to compute the average correlation matrix
    groups = defaultdict(list)
    for d in data:
        p = d.metadata["sample"]          
        y = d.y[0].item()                 
        x = d.x if torch.is_tensor(d.x) else torch.tensor(d.x)
        groups[(p, y)].append(x)


    corr_data = []
    for (p, y), mats in groups.items():
        c = torch.stack(mats).mean(0) 
        c_psd = project_to_psd(c)  
        corr_data.append({"patient_id": p, "y": torch.tensor(y), "c": c_psd})


    # Split into Train, Val and Test

    # Choose ratios for train/val/test split
    train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Count samples per patient and stage
    sample_counts = defaultdict(Counter)
    for d in corr_data:
        sample_counts[d["patient_id"]][d["y"]] += 1

    # Get unique classes
    classes = sorted({y for counts in sample_counts.values() for y in counts})
    num_classes = max(classes) + 1

    # Create counts matrix
    patients = list(sample_counts)
    counts_matrix = np.zeros((len(patients), num_classes), dtype=int)
    for i, pid in enumerate(patients):
        for y, cnt in sample_counts[pid].items():
            counts_matrix[i, y] = cnt

    # Compute targets for each split
    total_per_class = counts_matrix.sum(axis=0).astype(float)
    targets = {
        "train": total_per_class * train_ratio,
        "val": total_per_class * val_ratio,
        "test": total_per_class * test_ratio,
    }
    running = {k: np.zeros_like(total_per_class) for k in targets}
    assignment = {}

    # Shuffle patients and assign splits
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
        assignment[pid] = best_split
        running[best_split] += counts

    for d in corr_data:
        d["split"] = assignment[d["patient_id"]]
    
    return corr_data


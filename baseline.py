import time
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import numpy as np

from sklearn.model_selection    import GroupShuffleSplit
from sklearn.preprocessing     import StandardScaler
from sklearn.linear_model      import SGDClassifier
from sklearn.metrics           import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# --- 1) Load data list (no full-stack in RAM) ---
save_path = "/project2/cdonnat/sleepstages/data/overlapping/overlapping_data.pt"
data_list = torch.load(save_path, weights_only=False)
print(f"Data Loaded: {len(data_list)} samples")

def flatten_lower(corr):
    mat = corr.numpy() if hasattr(corr, 'numpy') else corr
    i_lower = np.tril_indices(mat.shape[0], k=-1)
    return mat[i_lower]

# --- 2) Build y and groups arrays, but skip X until needed ---
y      = np.array([int(d['y'].item()) for d in data_list])
groups = np.array([     d['metadata']['patient']   for d in data_list])

# --- 3) Split indices by patient ---
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X=None, y=y, groups=groups))

# --- 4) Set up online scaler + classifier with L2 penalty ---
scaler = StandardScaler()
clf    = SGDClassifier(
    loss='log_loss',
    penalty='l2',
    alpha=1e-4,
    max_iter=1,
    tol=None,
    verbose=0,
    n_jobs=-1,
    random_state=42
)
classes = np.unique(y)

# --- 5) Stream in mini‐batches for partial_fit with progress bars ---
batch_size = 5000
n_epochs   = 20
print(f"Starting streaming training: {n_epochs} epochs, batch size {batch_size}")
start_time = time.time()

for epoch in range(n_epochs):
    print(f"\n=== Epoch {epoch+1}/{n_epochs} ===")
    np.random.shuffle(train_idx)
    for start_i in tqdm(range(0, len(train_idx), batch_size), desc="Batches", unit="batch"):
        batch_idxs = train_idx[start_i:start_i+batch_size]
        Xb = np.vstack([
            flatten_lower(data_list[i]['c'])
            for i in batch_idxs
        ]).astype(np.float32)
        yb = y[batch_idxs]

        # online scale and fit
        scaler.partial_fit(Xb)
        Xb = scaler.transform(Xb)
        clf.partial_fit(Xb, yb, classes=classes)

elapsed = time.time() - start_time
print(f"\nStreaming training completed in {elapsed:.1f} seconds")

# --- 6) Prepare full test set in RAM ---
print("Loading and scaling test set...")
X_test = np.vstack([
    flatten_lower(data_list[i]['c'])
    for i in test_idx
]).astype(np.float32)
y_test = y[test_idx]
X_test = scaler.transform(X_test)

# --- 7) Evaluate ---
print("Evaluating on test set...")
y_pred = clf.predict(X_test)
print(f"Accuracy:          {accuracy_score(y_test, y_pred):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# --- 8) Confusion matrix ---
print("Plotting and saving confusion matrix...")
disp = ConfusionMatrixDisplay.from_estimator(
    clf, X_test, y_test,
    display_labels=clf.classes_
)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved confusion_matrix.png")
plt.close()

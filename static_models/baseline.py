import os
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from models.utils import evaluate_classification
import torch



def build_pca_logreg_pipeline(var_ratio: float = 0.90,
                               C: float = 0.1,
                               max_iter: int = 3000) -> Pipeline:
    """Returns a scikit-learn pipeline with standardization, PCA, and multinomial logistic regression."""
    return Pipeline([
        ('scale', StandardScaler(with_mean=True, with_std=True)),
        ('pca', PCA(n_components=var_ratio, svd_solver='full')),
        ('clf', LogisticRegression(
            penalty='l2',
            solver='saga',
            multi_class='multinomial',
            C=C,
            max_iter=max_iter,
            n_jobs=-1,
            verbose=0
        ))
    ])


def train_baseline(X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   *,
                   var_ratio: float = 0.90,
                   C: float = 0.1,
                   max_iter: int = 3000,
                   save_dir: str | None = None,
                   plot: bool = False,
                   report: bool = False):
    """
    Trains a PCA-LogReg pipeline on provided data, evaluates test performance, and optionally saves artefacts.

    Returns:
        pipeline: Fitted scikit-learn pipeline.
        metrics: Dict of evaluation results from evaluate_classification.
    """
    pipe = build_pca_logreg_pipeline(var_ratio, C, max_iter)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    metrics = evaluate_classification(
        y_test, y_pred,
        plot=plot,
        report=report,
        save_confusion_path=os.path.join(save_dir, "confusion.png") if save_dir else None,
        save_report_path=os.path.join(save_dir, "report.json") if save_dir else None,
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(pipe, os.path.join(save_dir, "pca_logreg.joblib"))

    return pipe, metrics


def prepare_baseline_data(data_list, *, include_synthetic: bool = False):
    """
    Extracts lower-triangular entries from correlation matrices in data_list and returns
    train/test splits for baseline classification tasks. Optionally includes synthetic samples.
    """

    # Infer matrix size and device from the first real (non-synthetic) sample
    for d in data_list:
        if not d.metadata.get('synthetic', False):
            n = d.metadata['c'].size(0)
            device = d.metadata['c'].device
            break
    else:
        raise ValueError("No non-synthetic samples found in data_list")

    i_tril, j_tril = torch.tril_indices(n, n, offset=-1, device=device)

    X_rows, y_rows, splits = [], [], []
    for d in data_list:
        if d.metadata.get('synthetic', False) and not include_synthetic:
            continue  # Skip synthetic samples if False

        C = d.metadata['c']
        X_rows.append(C[i_tril, j_tril].cpu().numpy())
        y_rows.append(int(d.y.item()))
        splits.append(d.metadata['split'])

    X = np.stack(X_rows, axis=0)
    y = np.array(y_rows, dtype=int)
    splits = np.array(splits)

    is_test = splits == 'test'
    return X[~is_test], y[~is_test], X[is_test], y[is_test]

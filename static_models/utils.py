import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

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
    Compute standard and balanced accuracy metrics and optionally print them.
    """
    # core scores
    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    # determine full set of labels for correct confusion‐matrix shape
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    # confusion matrices
    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_share = np.nan_to_num(cm_raw / cm_raw.sum(axis=1, keepdims=True))

    # optional plot / save confusion matrix (shares)
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

    # full sklearn report dict (for saving/printing)
    rep_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # print metrics if requested
    if metrics:
        print(f"Accuracy         : {acc:.4f}")
        print(f"Balanced accuracy: {bacc:.4f}")
        print("Classification report:")
        print(classification_report(y_true, y_pred, zero_division=0))

    # save *everything* to JSON
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



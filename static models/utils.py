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
    metrics: bool = False,
    plot: bool = False,
    display: bool = False,
    save_confusion_path: str | None = None,
    save_report_path: str | None = None
):
    """
    Compute standard metrics and optionally print them.
    """
    # core scores
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    # confusion matrices
    cm_raw = confusion_matrix(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_share = np.nan_to_num(cm_raw / cm_raw.sum(axis=1, keepdims=True))

    # optional plot / save confusion matrix (shares)
    if plot or save_confusion_path:
        disp = ConfusionMatrixDisplay(cm_share)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", values_format=".2f")
        ax.set_title("Confusion Matrix (row-normalised shares)")
        if save_confusion_path:
            os.makedirs(os.path.dirname(save_confusion_path), exist_ok=True)
            fig.savefig(save_confusion_path, bbox_inches="tight")
        if plot:
            plt.show()
        plt.close(fig)

    # full sklearn report dict
    rep_dict = classification_report(y_true, y_pred, output_dict=True)
    weighted_acc = rep_dict["weighted avg"]["recall"]

    # print metrics if requested
    if metrics:
        print(f"Accuracy: {acc:.4f}")
        print(f"Weighted accuracy: {weighted_acc:.4f}")
        print("Classification report:")
        print(classification_report(y_true, y_pred))

    # save *everything* to JSON
    if save_report_path:
        os.makedirs(os.path.dirname(save_report_path), exist_ok=True)
        json.dump({
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "weighted_accuracy": weighted_acc,
            "confusion_matrix_share": cm_share.tolist(),
            "classification_report": rep_dict
        }, open(save_report_path, "w"), indent=2)

    return {
        "accuracy": acc,
        "weighted_accuracy": weighted_acc,
        "report": rep_dict
    }

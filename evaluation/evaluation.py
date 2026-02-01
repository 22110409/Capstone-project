# Client/evaluation.py

import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


@torch.no_grad()
def evaluate_model_on_client_arrays(
    model,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: str = "cpu",
    threshold: float = 0.5,
):


    model.eval()
    model.to(device)

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    logits = model(X_test)

    probs = torch.sigmoid(logits).cpu().numpy().ravel()
    y_pred = (probs >= threshold).astype(int)

    y_true = y_test.cpu().numpy().ravel().astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    return metrics


@torch.no_grad()
def evaluate_model_on_loader(
    model,
    dataloader,
    device: str = "cpu",
    threshold: float = 0.5,
):


    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        probs = torch.sigmoid(logits)

        preds = (probs >= threshold).int()

        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    y_pred = np.vstack(all_preds).ravel()
    y_true = np.vstack(all_labels).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

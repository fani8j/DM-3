"""Validation module for HAR models.

Runs inference on a validation fold in mini-batches under torch.no_grad(),
reduces to argmax predictions per batch before accumulation to control memory,
and computes Macro F1, per-class F1/precision/recall, and confusion matrix.

Requirements: R7.6
"""

import torch

from har.data.types import FoldMetrics
from har.eval.metrics import compute_fold_metrics


def validate_model(
    model: torch.nn.Module,
    val_seq: torch.Tensor,  # [N, 300, C]
    val_labels: torch.Tensor,  # [N]
    val_static: torch.Tensor | None = None,  # [N, F] or None
    batch_size: int = 64,
    device: str = "cpu",
) -> FoldMetrics:
    """Run validation on a model and compute metrics.

    Iterates in mini-batches under torch.no_grad() to control memory.
    Reduces to argmax predictions per batch before accumulation.

    R7.6: Computes Macro F1, per-class F1/precision/recall, confusion matrix.
    """
    model.eval()
    all_preds: list[int] = []

    n = val_seq.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_seq = val_seq[start:end].to(device)
            batch_static = (
                val_static[start:end].to(device) if val_static is not None else None
            )

            logits = model(batch_seq, batch_static)  # [B, 6]
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)

    y_true = val_labels.tolist()
    return compute_fold_metrics(y_true, all_preds)

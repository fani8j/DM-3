"""Stack the tree ensemble OOF with the TCN OOF, optimize blend + thresholds
on honest OOF predictions, then produce the final submission.
"""

import sys
import logging
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def macro(y, probs, w=None):
    p = probs * w if w is not None else probs
    return f1_score(y, p.argmax(1), average="macro", labels=list(range(6)), zero_division=0)


def optimize_thresholds(y, probs, n_iter=60):
    w = np.ones(6)
    best = macro(y, probs, w)
    for _ in range(n_iter):
        improved = False
        for c in range(6):
            for m in np.linspace(0.15, 7.0, 80):
                w2 = w.copy(); w2[c] = m
                s = macro(y, probs, w2)
                if s > best + 1e-6:
                    best, w, improved = s, w2, True
        if not improved:
            break
    return w, best


def main():
    # Tree ensemble OOF/test (aligned to train_feats_v2 order)
    with open("feature_cache/train_feats_v2.pkl", "rb") as f:
        train_df = pickle.load(f)
    with open("feature_cache/test_feats_v2.pkl", "rb") as f:
        test_df = pickle.load(f)
    y = train_df["label"].values.astype(int)
    tree_oof = np.load("feature_cache/ens_oof.npy")
    tree_test = np.load("feature_cache/ens_test.npy")
    tree_train_fids = train_df["file_id"].values
    tree_test_fids = test_df["file_id"].values

    # TCN OOF/test (aligned by its own file_id arrays)
    tcn_oof = np.load("feature_cache/tcn_oof.npy")
    tcn_test = np.load("feature_cache/tcn_test.npy")
    tcn_train_fids = np.load("feature_cache/tcn_oof_fileids.npy")
    tcn_test_fids = np.load("feature_cache/tcn_test_fileids.npy")

    # Align TCN OOF to tree order by file_id
    tcn_train_map = {fid: i for i, fid in enumerate(tcn_train_fids)}
    tcn_oof_aligned = np.zeros_like(tree_oof)
    for i, fid in enumerate(tree_train_fids):
        tcn_oof_aligned[i] = tcn_oof[tcn_train_map[fid]]

    tcn_test_map = {fid: i for i, fid in enumerate(tcn_test_fids)}
    tcn_test_aligned = np.zeros_like(tree_test)
    for i, fid in enumerate(tree_test_fids):
        tcn_test_aligned[i] = tcn_test[tcn_test_map[fid]]

    logger.info(f"Tree OOF macro (raw argmax): {macro(y, tree_oof):.4f}")
    logger.info(f"TCN  OOF macro (raw argmax): {macro(y, tcn_oof_aligned):.4f}")

    # Search blend weight a*tree + (1-a)*tcn
    best_a, best_f1 = 1.0, 0.0
    for a in np.linspace(0, 1, 41):
        blend = a * tree_oof + (1 - a) * tcn_oof_aligned
        f1 = macro(y, blend)
        if f1 > best_f1:
            best_f1, best_a = f1, a
    logger.info(f"Best blend (pre-threshold): a={best_a:.3f} -> macro={best_f1:.4f}")

    blend_oof = best_a * tree_oof + (1 - best_a) * tcn_oof_aligned
    blend_test = best_a * tree_test + (1 - best_a) * tcn_test_aligned

    # Optimize per-class thresholds on the blend
    w, tuned = optimize_thresholds(y, blend_oof)
    logger.info(f"Tuned multipliers: {np.round(w,3).tolist()}")
    logger.info(f"Stacked + tuned OOF macro F1: {tuned:.4f}")

    per = f1_score(y, (blend_oof * w).argmax(1), average=None, labels=list(range(6)), zero_division=0)
    cc = Counter(y.tolist())
    for c in range(6):
        logger.info(f"  Class {c} F1: {per[c]:.4f} (n={cc[c]})")

    cm = confusion_matrix(y, (blend_oof * w).argmax(1), labels=list(range(6)))
    logger.info("Confusion matrix (rows=true, cols=pred):")
    for i in range(6):
        logger.info("  T%d: %s", i, "  ".join(f"{cm[i,j]:4d}" for j in range(6)))

    # Final test prediction
    test_labels = (blend_test * w).argmax(1)
    sample = pd.read_csv("sample_submission.csv")
    pred_map = dict(zip(tree_test_fids, test_labels))
    out = Path("submissions/submission_stacked.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="ascii") as f:
        f.write("Id,Label\n")
        for sid in sample["Id"].values:
            f.write(f"{int(sid)},{int(pred_map[int(sid)])}\n")
    logger.info(f"Submission written to: {out}")
    logger.info(f"Test label distribution: {dict(sorted(Counter(test_labels.tolist()).items()))}")


if __name__ == "__main__":
    main()

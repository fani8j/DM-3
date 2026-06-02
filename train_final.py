"""Final pipeline: multiclass ensemble + per-class one-vs-rest specialists.

Targets the class-2 bottleneck directly:
- One-vs-rest LightGBM binary specialists for each class (esp. class 2)
- Specialist probabilities blended into the multiclass ensemble probs
- Optimize blend + per-class thresholds on honest OOF
- Combine with the existing tree ensemble + TCN OOF for the final submission
"""

import sys
import logging
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("train_final.log", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def macro(y, probs, w=None):
    p = probs * w if w is not None else probs
    return f1_score(y, p.argmax(1), average="macro", labels=list(range(6)), zero_division=0)


def optimize_thresholds(y, probs, n_iter=80):
    w = np.ones(6)
    best = macro(y, probs, w)
    for _ in range(n_iter):
        improved = False
        for c in range(6):
            for m in np.linspace(0.1, 8.0, 100):
                w2 = w.copy(); w2[c] = m
                s = macro(y, probs, w2)
                if s > best + 1e-6:
                    best, w, improved = s, w2, True
        if not improved:
            break
    return w, best


def main():
    with open("feature_cache/train_feats_v2.pkl", "rb") as f:
        train_df = pickle.load(f)
    with open("feature_cache/test_feats_v2.pkl", "rb") as f:
        test_df = pickle.load(f)

    feat_cols = [c for c in train_df.columns if c not in ("file_id", "user_id", "label")]
    X = train_df[feat_cols].values.astype(np.float32)
    y = train_df["label"].values.astype(int)
    groups = train_df["user_id"].values
    X_test = test_df[feat_cols].values.astype(np.float32)
    cc = Counter(y.tolist()); total = len(y)

    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(sgkf.split(X, y, groups))

    # One-vs-rest specialist for each class
    ovr_oof = np.zeros((len(y), 6))
    ovr_test = np.zeros((len(X_test), 6))

    base_params = {
        "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
        "num_leaves": 47, "learning_rate": 0.05, "feature_fraction": 0.6,
        "bagging_fraction": 0.8, "bagging_freq": 5, "min_child_samples": 20,
        "lambda_l1": 0.3, "lambda_l2": 0.3, "verbose": -1, "seed": 42, "num_threads": 0,
    }

    for cls in range(6):
        logger.info(f"=== One-vs-rest specialist for class {cls} ===")
        y_bin = (y == cls).astype(int)
        # Strong positive weight for rare classes
        pos_weight = (total - cc[cls]) / cc[cls]
        for fold, (tr, va) in enumerate(folds):
            w_tr = np.where(y_bin[tr] == 1, pos_weight, 1.0)
            dtr = lgb.Dataset(X[tr], label=y_bin[tr], weight=w_tr)
            dva = lgb.Dataset(X[va], label=y_bin[va], reference=dtr)
            m = lgb.train(base_params, dtr, num_boost_round=800, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
            ovr_oof[va, cls] = m.predict(X[va], num_iteration=m.best_iteration)
            ovr_test[:, cls] += m.predict(X_test, num_iteration=m.best_iteration) / n_splits
        # Report this specialist's separation
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_bin, ovr_oof[:, cls])
        logger.info(f"  Class {cls} OvR AUC: {auc:.4f} (pos_weight={pos_weight:.1f})")

    # Normalize OvR rows to form a probability-like distribution
    ovr_oof_norm = ovr_oof / (ovr_oof.sum(axis=1, keepdims=True) + 1e-9)
    ovr_test_norm = ovr_test / (ovr_test.sum(axis=1, keepdims=True) + 1e-9)
    logger.info(f"OvR-only OOF macro (raw argmax): {macro(y, ovr_oof_norm):.4f}")

    # Load existing multiclass tree ensemble + TCN OOF
    tree_oof = np.load("feature_cache/ens_oof.npy")
    tree_test = np.load("feature_cache/ens_test.npy")
    tcn_oof = np.load("feature_cache/tcn_oof.npy")
    tcn_test = np.load("feature_cache/tcn_test.npy")
    tcn_train_fids = np.load("feature_cache/tcn_oof_fileids.npy")
    tcn_test_fids = np.load("feature_cache/tcn_test_fileids.npy")

    # Align TCN by file_id
    tr_fids = train_df["file_id"].values
    te_fids = test_df["file_id"].values
    tmap = {f: i for i, f in enumerate(tcn_train_fids)}
    tcn_oof_al = np.stack([tcn_oof[tmap[f]] for f in tr_fids])
    tmap2 = {f: i for i, f in enumerate(tcn_test_fids)}
    tcn_test_al = np.stack([tcn_test[tmap2[f]] for f in te_fids])

    logger.info(f"Tree OOF macro: {macro(y, tree_oof):.4f}")
    logger.info(f"TCN  OOF macro: {macro(y, tcn_oof_al):.4f}")

    # 3-way blend search: tree, tcn, ovr
    best = (0, 0, 0, 0.0)
    for a in np.linspace(0, 1, 21):
        for b in np.linspace(0, 1 - a, int((1 - a) * 20) + 1):
            cc_ = 1 - a - b
            if cc_ < 0:
                continue
            blend = a * tree_oof + b * tcn_oof_al + cc_ * ovr_oof_norm
            f1 = macro(y, blend)
            if f1 > best[3]:
                best = (a, b, cc_, f1)
    a, b, c_, f1 = best
    logger.info(f"Best 3-way blend: tree={a:.2f} tcn={b:.2f} ovr={c_:.2f} -> macro={f1:.4f}")

    blend_oof = a * tree_oof + b * tcn_oof_al + c_ * ovr_oof_norm
    blend_test = a * tree_test + b * tcn_test_al + c_ * ovr_test_norm

    w, tuned = optimize_thresholds(y, blend_oof)
    logger.info(f"Tuned multipliers: {np.round(w,3).tolist()}")
    logger.info(f"FINAL stacked + tuned OOF macro F1: {tuned:.4f}")

    per = f1_score(y, (blend_oof * w).argmax(1), average=None, labels=list(range(6)), zero_division=0)
    for cl in range(6):
        logger.info(f"  Class {cl} F1: {per[cl]:.4f} (n={cc[cl]})")

    cm = confusion_matrix(y, (blend_oof * w).argmax(1), labels=list(range(6)))
    logger.info("Confusion matrix:")
    for i in range(6):
        logger.info("  T%d: %s", i, "  ".join(f"{cm[i,j]:4d}" for j in range(6)))

    test_labels = (blend_test * w).argmax(1)
    sample = pd.read_csv("sample_submission.csv")
    pred_map = dict(zip(te_fids, test_labels))
    out = Path("submissions/submission_final.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="ascii") as f:
        f.write("Id,Label\n")
        for sid in sample["Id"].values:
            f.write(f"{int(sid)},{int(pred_map[int(sid)])}\n")
    logger.info(f"Submission written to: {out}")
    logger.info(f"Test label distribution: {dict(sorted(Counter(test_labels.tolist()).items()))}")


if __name__ == "__main__":
    main()

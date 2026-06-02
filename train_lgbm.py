"""LightGBM pipeline for HAR — feature-engineering approach.

Strategy:
1. Extract ~250 window-level features per file (gravity-intact, orientation-aware)
2. GroupKFold by user for honest CV (test users are unseen)
3. Train LightGBM per fold, average test predictions across folds
4. Class-weighted to handle imbalance
5. Cache features to disk to avoid recomputation
"""

import sys
import logging
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from features_lgbm import extract_window_features, BASE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path("feature_cache")
CACHE_DIR.mkdir(exist_ok=True)


def load_and_extract(root, expect_label, cache_name):
    """Load all CSVs under root and extract features. Caches to disk."""
    cache_path = CACHE_DIR / f"{cache_name}.pkl"
    if cache_path.exists():
        logger.info(f"Loading cached features from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    root = Path(root)
    rows = []
    user_dirs = sorted([d for d in root.iterdir() if d.is_dir()],
                       key=lambda d: int(d.name.split("_")[1]))
    for user_dir in user_dirs:
        user_id = int(user_dir.name.split("_")[1])
        csv_files = sorted(user_dir.glob("*.csv"), key=lambda p: int(p.stem))
        for csv_path in csv_files:
            file_id = int(csv_path.stem)
            df = pd.read_csv(csv_path)
            data = df[BASE_COLS].values
            feats = extract_window_features(data)
            feats["file_id"] = file_id
            feats["user_id"] = user_id
            if expect_label:
                feats["label"] = int(df["label"].iloc[0])
            rows.append(feats)
        logger.info(f"  Processed {user_dir.name}: {len(csv_files)} files")

    result = pd.DataFrame(rows)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    logger.info(f"Cached features to {cache_path}")
    return result


def main():
    logger.info("=== Feature extraction ===")
    train_df = load_and_extract("train/train", expect_label=True, cache_name="train_feats")
    test_df = load_and_extract("test/test", expect_label=False, cache_name="test_feats")

    logger.info(f"Train: {len(train_df)} windows, Test: {len(test_df)} windows")

    feature_cols = [c for c in train_df.columns if c not in ("file_id", "user_id", "label")]
    logger.info(f"Number of features: {len(feature_cols)}")

    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df["label"].values.astype(int)
    groups = train_df["user_id"].values

    X_test = test_df[feature_cols].values.astype(np.float32)

    # Class weights (inverse frequency)
    class_counts = Counter(y)
    total = len(y)
    class_weight = {c: total / (6 * class_counts[c]) for c in range(6)}
    sample_weight = np.array([class_weight[label] for label in y])

    logger.info(f"Class distribution: {dict(sorted(class_counts.items()))}")

    # GroupKFold by user (StratifiedGroupKFold keeps class balance across folds)
    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    params = {
        "objective": "multiclass",
        "num_class": 6,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.02,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "max_depth": -1,
        "verbose": -1,
        "seed": 42,
        "num_threads": 0,
    }

    oof_preds = np.zeros((len(X), 6))
    test_preds = np.zeros((len(X_test), 6))
    fold_f1s = []

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups)):
        logger.info(f"\n=== Fold {fold+1}/{n_splits} ===")
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        w_tr = sample_weight[tr_idx]

        train_set = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        val_set = lgb.Dataset(X_va, label=y_va, reference=train_set)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=2000,
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        va_pred = model.predict(X_va, num_iteration=model.best_iteration)
        oof_preds[va_idx] = va_pred
        va_labels = va_pred.argmax(axis=1)
        fold_f1 = f1_score(y_va, va_labels, average="macro", labels=list(range(6)), zero_division=0)
        fold_f1s.append(fold_f1)
        logger.info(f"  Fold {fold+1} Macro F1: {fold_f1:.4f} (best_iter={model.best_iteration})")

        # Accumulate test predictions
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

    # Overall OOF F1
    oof_labels = oof_preds.argmax(axis=1)
    oof_f1 = f1_score(y, oof_labels, average="macro", labels=list(range(6)), zero_division=0)
    logger.info(f"\n=== CV Results ===")
    logger.info(f"Fold F1s: {[f'{f:.4f}' for f in fold_f1s]}")
    logger.info(f"Mean fold F1: {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}")
    logger.info(f"OOF Macro F1: {oof_f1:.4f}")

    # Per-class OOF F1
    per_class = f1_score(y, oof_labels, average=None, labels=list(range(6)), zero_division=0)
    for c in range(6):
        logger.info(f"  Class {c} F1: {per_class[c]:.4f} (n={class_counts[c]})")

    # Save OOF and test preds for potential ensembling
    np.save("feature_cache/lgbm_oof.npy", oof_preds)
    np.save("feature_cache/lgbm_test.npy", test_preds)

    # Generate submission
    submission_ids = test_df["file_id"].values
    test_labels = test_preds.argmax(axis=1)

    # Reorder to match sample_submission
    sample = pd.read_csv("sample_submission.csv")
    pred_map = dict(zip(submission_ids, test_labels))

    output_path = Path("submissions/submission_lgbm.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="ascii") as f:
        f.write("Id,Label\n")
        for sid in sample["Id"].values:
            f.write(f"{int(sid)},{int(pred_map[int(sid)])}\n")

    logger.info(f"\nSubmission written to: {output_path}")
    logger.info(f"Label distribution: {dict(sorted(Counter(test_labels.tolist()).items()))}")
    logger.info("Done!")


if __name__ == "__main__":
    main()

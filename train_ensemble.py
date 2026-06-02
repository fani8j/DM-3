"""Strong ensemble pipeline: LightGBM + XGBoost with threshold optimization.

Strategy to reach 0.8 macro F1:
1. Rich window-level features (gravity-intact, ~330 features)
2. GroupKFold by user (honest CV, matches unseen-user test setup)
3. LightGBM + XGBoost ensemble (different tree algos, averaged probabilities)
4. Per-class probability multiplier optimization on OOF (directly boosts macro F1)
5. Apply tuned multipliers to test predictions

Results logged to train_ensemble.log for durability.
"""

import sys
import logging
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from features_lgbm import extract_window_features, BASE_COLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("train_ensemble.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

CACHE_DIR = Path("feature_cache")
CACHE_DIR.mkdir(exist_ok=True)


def load_and_extract(root, expect_label, cache_name):
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
    result = pd.DataFrame(rows)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    logger.info(f"Cached {len(result)} rows to {cache_path}")
    return result


def optimize_thresholds(y_true, oof_probs, n_iter=40):
    """Coordinate ascent on per-class probability multipliers to maximize macro F1."""
    weights = np.ones(6)

    def score(w):
        return f1_score(y_true, (oof_probs * w).argmax(1), average="macro",
                        labels=list(range(6)), zero_division=0)

    best = score(weights)
    for _ in range(n_iter):
        improved = False
        for c in range(6):
            for mult in np.linspace(0.2, 6.0, 60):
                w = weights.copy()
                w[c] = mult
                s = score(w)
                if s > best + 1e-6:
                    best = s
                    weights = w
                    improved = True
        if not improved:
            break
    return weights, best


def main():
    logger.info("=== Feature extraction ===")
    train_df = load_and_extract("train/train", expect_label=True, cache_name="train_feats_v2")
    test_df = load_and_extract("test/test", expect_label=False, cache_name="test_feats_v2")

    feature_cols = [c for c in train_df.columns if c not in ("file_id", "user_id", "label")]
    logger.info(f"Train: {len(train_df)} windows, Test: {len(test_df)} windows, Features: {len(feature_cols)}")

    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df["label"].values.astype(int)
    groups = train_df["user_id"].values
    X_test = test_df[feature_cols].values.astype(np.float32)

    class_counts = Counter(y)
    total = len(y)
    class_weight = {c: total / (6 * class_counts[c]) for c in range(6)}
    sample_weight = np.array([class_weight[label] for label in y])
    logger.info(f"Class distribution: {dict(sorted(class_counts.items()))}")

    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    lgb_params = {
        "objective": "multiclass", "num_class": 6, "metric": "multi_logloss",
        "boosting_type": "gbdt", "num_leaves": 63, "learning_rate": 0.05,
        "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 5,
        "min_child_samples": 30, "lambda_l1": 0.5, "lambda_l2": 0.5,
        "max_depth": -1, "verbose": -1, "seed": 42, "num_threads": 0,
    }
    xgb_params = {
        "objective": "multi:softprob", "num_class": 6, "eval_metric": "mlogloss",
        "eta": 0.05, "max_depth": 7, "subsample": 0.8, "colsample_bytree": 0.6,
        "min_child_weight": 5, "reg_alpha": 0.5, "reg_lambda": 1.0,
        "tree_method": "hist", "seed": 42, "nthread": 0,
    }

    oof_lgb = np.zeros((len(X), 6))
    oof_xgb = np.zeros((len(X), 6))
    test_lgb = np.zeros((len(X_test), 6))
    test_xgb = np.zeros((len(X_test), 6))

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups)):
        logger.info(f"--- Fold {fold+1}/{n_splits} ---")
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        w_tr = sample_weight[tr_idx]

        # LightGBM
        ltr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        lva = lgb.Dataset(X_va, label=y_va, reference=ltr)
        lmodel = lgb.train(
            lgb_params, ltr, num_boost_round=1200, valid_sets=[lva],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )
        oof_lgb[va_idx] = lmodel.predict(X_va, num_iteration=lmodel.best_iteration)
        test_lgb += lmodel.predict(X_test, num_iteration=lmodel.best_iteration) / n_splits

        # XGBoost
        dtr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
        dva = xgb.DMatrix(X_va, label=y_va)
        dtest = xgb.DMatrix(X_test)
        xmodel = xgb.train(
            xgb_params, dtr, num_boost_round=1200, evals=[(dva, "val")],
            early_stopping_rounds=80, verbose_eval=False,
        )
        oof_xgb[va_idx] = xmodel.predict(dva, iteration_range=(0, xmodel.best_iteration + 1))
        test_xgb += xmodel.predict(dtest, iteration_range=(0, xmodel.best_iteration + 1)) / n_splits

        lf1 = f1_score(y_va, oof_lgb[va_idx].argmax(1), average="macro", labels=list(range(6)), zero_division=0)
        xf1 = f1_score(y_va, oof_xgb[va_idx].argmax(1), average="macro", labels=list(range(6)), zero_division=0)
        logger.info(f"  LGB fold F1={lf1:.4f} (iter={lmodel.best_iteration}), XGB fold F1={xf1:.4f} (iter={xmodel.best_iteration})")

    # Evaluate individual and blended OOF
    for name, oof in [("LGB", oof_lgb), ("XGB", oof_xgb)]:
        f1 = f1_score(y, oof.argmax(1), average="macro", labels=list(range(6)), zero_division=0)
        logger.info(f"{name} OOF Macro F1: {f1:.4f}")

    # Blend (search best blend weight)
    best_blend, best_blend_f1 = 0.5, 0.0
    for a in np.linspace(0, 1, 21):
        blend = a * oof_lgb + (1 - a) * oof_xgb
        f1 = f1_score(y, blend.argmax(1), average="macro", labels=list(range(6)), zero_division=0)
        if f1 > best_blend_f1:
            best_blend_f1, best_blend = f1, a
    logger.info(f"Best blend: {best_blend:.2f}*LGB + {1-best_blend:.2f}*XGB -> OOF F1={best_blend_f1:.4f}")

    oof_blend = best_blend * oof_lgb + (1 - best_blend) * oof_xgb
    test_blend = best_blend * test_lgb + (1 - best_blend) * test_xgb

    # Threshold optimization on OOF
    weights, tuned_f1 = optimize_thresholds(y, oof_blend)
    logger.info(f"Tuned multipliers: {np.round(weights, 3).tolist()}")
    logger.info(f"Tuned OOF Macro F1: {tuned_f1:.4f}")

    per_class = f1_score(y, (oof_blend * weights).argmax(1), average=None, labels=list(range(6)), zero_division=0)
    for c in range(6):
        logger.info(f"  Class {c} F1: {per_class[c]:.4f} (n={class_counts[c]})")

    cm = confusion_matrix(y, (oof_blend * weights).argmax(1), labels=list(range(6)))
    logger.info("Confusion matrix (rows=true, cols=pred):")
    for i in range(6):
        logger.info("  T%d: %s", i, "  ".join(f"{cm[i,j]:4d}" for j in range(6)))

    # Apply tuned weights to test
    test_labels = (test_blend * weights).argmax(1)

    np.save("feature_cache/ens_oof.npy", oof_blend)
    np.save("feature_cache/ens_test.npy", test_blend)
    np.save("feature_cache/ens_weights.npy", weights)

    # Write submission
    sample = pd.read_csv("sample_submission.csv")
    pred_map = dict(zip(test_df["file_id"].values, test_labels))
    output_path = Path("submissions/submission_ensemble.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="ascii") as f:
        f.write("Id,Label\n")
        for sid in sample["Id"].values:
            f.write(f"{int(sid)},{int(pred_map[int(sid)])}\n")
    logger.info(f"Submission written to: {output_path}")
    logger.info(f"Test label distribution: {dict(sorted(Counter(test_labels.tolist()).items()))}")
    logger.info("Done!")


if __name__ == "__main__":
    main()

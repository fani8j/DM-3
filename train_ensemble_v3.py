"""3-booster ensemble (LGB + XGB + CatBoost) on v3 features with OOF for stacking.

Produces OOF + test probabilities saved to feature_cache for the final stack.
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
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from features_v3 import extract_window_features_v3, BASE_COLS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("train_ensemble_v3.log", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

CACHE = Path("feature_cache")
CACHE.mkdir(exist_ok=True)


def load_extract(root, expect_label, cache_name):
    cache_path = CACHE / f"{cache_name}.pkl"
    if cache_path.exists():
        logger.info(f"Loading cached {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    root = Path(root)
    rows = []
    user_dirs = sorted([d for d in root.iterdir() if d.is_dir()],
                       key=lambda d: int(d.name.split("_")[1]))
    for ud in user_dirs:
        uid = int(ud.name.split("_")[1])
        for csv_path in sorted(ud.glob("*.csv"), key=lambda p: int(p.stem)):
            fid = int(csv_path.stem)
            df = pd.read_csv(csv_path)
            feats = extract_window_features_v3(df[BASE_COLS].values)
            feats["file_id"] = fid; feats["user_id"] = uid
            if expect_label:
                feats["label"] = int(df["label"].iloc[0])
            rows.append(feats)
        logger.info(f"  processed {ud.name}")
    result = pd.DataFrame(rows)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    logger.info(f"Cached {len(result)} rows to {cache_path}")
    return result


def main():
    logger.info("=== v3 feature extraction ===")
    train_df = load_extract("train/train", True, "train_feats_v3")
    test_df = load_extract("test/test", False, "test_feats_v3")

    feat_cols = [c for c in train_df.columns if c not in ("file_id", "user_id", "label")]
    logger.info(f"Train {len(train_df)}, Test {len(test_df)}, Features {len(feat_cols)}")

    X = train_df[feat_cols].values.astype(np.float32)
    y = train_df["label"].values.astype(int)
    groups = train_df["user_id"].values
    X_test = test_df[feat_cols].values.astype(np.float32)

    cc = Counter(y.tolist()); total = len(y)
    cw = {c: total / (6 * cc[c]) for c in range(6)}
    sw = np.array([cw[l] for l in y])
    logger.info(f"Class dist: {dict(sorted(cc.items()))}")

    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(sgkf.split(X, y, groups))

    lgb_params = {
        "objective": "multiclass", "num_class": 6, "metric": "multi_logloss",
        "num_leaves": 63, "learning_rate": 0.03, "feature_fraction": 0.6,
        "bagging_fraction": 0.8, "bagging_freq": 5, "min_child_samples": 25,
        "lambda_l1": 0.5, "lambda_l2": 0.5, "verbose": -1, "seed": 42, "num_threads": 0,
    }
    xgb_params = {
        "objective": "multi:softprob", "num_class": 6, "eval_metric": "mlogloss",
        "eta": 0.03, "max_depth": 7, "subsample": 0.8, "colsample_bytree": 0.6,
        "min_child_weight": 5, "reg_alpha": 0.5, "reg_lambda": 1.0,
        "tree_method": "hist", "seed": 42, "nthread": 0,
    }

    oof = {m: np.zeros((len(X), 6)) for m in ["lgb", "xgb", "cat"]}
    test = {m: np.zeros((len(X_test), 6)) for m in ["lgb", "xgb", "cat"]}

    for fold, (tr, va) in enumerate(folds):
        logger.info(f"--- Fold {fold+1}/{n_splits} ---")
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        wtr = sw[tr]

        # LightGBM
        ltr = lgb.Dataset(Xtr, label=ytr, weight=wtr)
        lva = lgb.Dataset(Xva, label=yva, reference=ltr)
        lm = lgb.train(lgb_params, ltr, num_boost_round=1500, valid_sets=[lva],
                       callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        oof["lgb"][va] = lm.predict(Xva, num_iteration=lm.best_iteration)
        test["lgb"] += lm.predict(X_test, num_iteration=lm.best_iteration) / n_splits

        # XGBoost
        dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
        dva = xgb.DMatrix(Xva, label=yva)
        dte = xgb.DMatrix(X_test)
        xm = xgb.train(xgb_params, dtr, num_boost_round=1500, evals=[(dva, "v")],
                       early_stopping_rounds=80, verbose_eval=False)
        oof["xgb"][va] = xm.predict(dva, iteration_range=(0, xm.best_iteration + 1))
        test["xgb"] += xm.predict(dte, iteration_range=(0, xm.best_iteration + 1)) / n_splits

        # CatBoost
        cm = CatBoostClassifier(
            iterations=1500, learning_rate=0.03, depth=7, l2_leaf_reg=5.0,
            loss_function="MultiClass", eval_metric="TotalF1", random_seed=42,
            class_weights=[cw[c] for c in range(6)], early_stopping_rounds=80,
            verbose=0, thread_count=-1,
        )
        cm.fit(Pool(Xtr, ytr), eval_set=Pool(Xva, yva))
        oof["cat"][va] = cm.predict_proba(Xva)
        test["cat"] += cm.predict_proba(X_test) / n_splits

        for m in ["lgb", "xgb", "cat"]:
            f1 = f1_score(yva, oof[m][va].argmax(1), average="macro", labels=list(range(6)), zero_division=0)
            logger.info(f"  {m} fold F1={f1:.4f}")

    for m in ["lgb", "xgb", "cat"]:
        f1 = f1_score(y, oof[m].argmax(1), average="macro", labels=list(range(6)), zero_division=0)
        logger.info(f"{m} OOF Macro F1: {f1:.4f}")

    # Equal blend of the three boosters
    oof_blend = (oof["lgb"] + oof["xgb"] + oof["cat"]) / 3
    test_blend = (test["lgb"] + test["xgb"] + test["cat"]) / 3
    f1b = f1_score(y, oof_blend.argmax(1), average="macro", labels=list(range(6)), zero_division=0)
    logger.info(f"3-booster equal blend OOF Macro F1: {f1b:.4f}")

    np.save(CACHE / "tree3_oof.npy", oof_blend)
    np.save(CACHE / "tree3_test.npy", test_blend)
    np.save(CACHE / "tree3_train_fids.npy", train_df["file_id"].values)
    np.save(CACHE / "tree3_test_fids.npy", test_df["file_id"].values)
    logger.info("Saved 3-booster OOF/test predictions.")


if __name__ == "__main__":
    main()

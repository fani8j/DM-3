from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


CACHE = Path("feature_cache")
OUT = Path("report_figures")
OUT.mkdir(exist_ok=True)
CLASSES = list(range(6))


def align_by_file_id(probs: np.ndarray, source_file_ids: np.ndarray, target_file_ids: np.ndarray) -> np.ndarray:
    positions = {int(file_id): idx for idx, file_id in enumerate(source_file_ids)}
    return np.stack([probs[positions[int(file_id)]] for file_id in target_file_ids])


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_class_distribution(train_df: pd.DataFrame) -> None:
    counts = train_df["label"].value_counts().sort_index()
    plt.figure(figsize=(7.0, 4.0))
    bars = plt.bar(counts.index.astype(str), counts.values, color="#4C78A8")
    plt.title("Training Class Distribution")
    plt.xlabel("Activity label")
    plt.ylabel("Number of windows")
    for bar, value in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 40, str(value), ha="center", fontsize=9)
    savefig(OUT / "class_distribution.png")


def plot_feature_label_correlations(train_df: pd.DataFrame) -> None:
    feature_cols = [c for c in train_df.columns if c not in ("file_id", "user_id", "label")]
    # Spearman correlation is a descriptive ranking only; it is not used as a causal claim.
    corr = train_df[feature_cols + ["label"]].corr(method="spearman", numeric_only=True)["label"].drop("label")
    top = corr.abs().sort_values(ascending=False).head(15)
    vals = corr.loc[top.index].sort_values()

    plt.figure(figsize=(8.4, 5.2))
    colors = ["#F58518" if v < 0 else "#54A24B" for v in vals.values]
    plt.barh(range(len(vals)), vals.values, color=colors)
    plt.yticks(range(len(vals)), vals.index, fontsize=8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Top 15 Feature-Label Spearman Correlations")
    plt.xlabel("Spearman correlation with label")
    savefig(OUT / "feature_label_correlation.png")


def cached_oof_best_oof(train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_file_ids = train_df["file_id"].values.astype(int)
    y = train_df["label"].values.astype(int)

    ens_oof = np.load(CACHE / "ens_oof.npy")
    tree3_oof = align_by_file_id(
        np.load(CACHE / "tree3_oof.npy"),
        np.load(CACHE / "tree3_train_fids.npy"),
        train_file_ids,
    )
    deep_oof = align_by_file_id(
        np.load(CACHE / "deep_oof.npy"),
        np.load(CACHE / "deep_oof_fileids.npy"),
        train_file_ids,
    )
    tcn_oof = align_by_file_id(
        np.load(CACHE / "tcn_oof.npy"),
        np.load(CACHE / "tcn_oof_fileids.npy"),
        train_file_ids,
    )

    blend = 0.05 * ens_oof + 0.25 * tree3_oof + 0.30 * deep_oof + 0.40 * tcn_oof
    class_weights = np.array([1.137, 1.536, 1.536, 1.000, 0.579, 1.000])
    pred = (blend * class_weights).argmax(1)
    return y, pred, blend


def plot_confusion_matrix(y: np.ndarray, pred: np.ndarray) -> None:
    cm = confusion_matrix(y, pred, labels=CLASSES)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sums, 1)

    plt.figure(figsize=(6.2, 5.2))
    im = plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Row-normalized rate")
    plt.title("Cached OOF-Best Blend OOF Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(CLASSES)
    plt.yticks(CLASSES)
    for i in CLASSES:
        for j in CLASSES:
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    savefig(OUT / "oof_confusion_matrix.png")


def plot_submission_distributions() -> None:
    names = {
        "final\n0.7940": "submission_final.csv",
        "public_safe\n0.8274": "submission_cached_blend_public_safe.csv",
        "oof_best\n0.8305": "submission_cached_blend.csv",
        "more_tree\n0.8270": "submission_lb_probe_more_tree.csv",
        "c5_up\npending": "submission_lb_probe_oofbest_c5_up.csv",
        "c4_up\npending": "submission_lb_probe_oofbest_c4_up.csv",
        "c3_up\npending": "submission_lb_probe_oofbest_c3_up.csv",
    }
    rows = []
    for label, filename in names.items():
        df = pd.read_csv(Path("submissions") / filename)
        counts = Counter(df["Label"].astype(int).tolist())
        for cls in CLASSES:
            rows.append({"submission": label, "class": cls, "count": counts.get(cls, 0)})
    table = pd.DataFrame(rows)
    pivot = table.pivot(index="submission", columns="class", values="count")

    plt.figure(figsize=(9.2, 4.8))
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    for cls, color in zip(CLASSES, colors):
        vals = pivot[cls].values
        plt.bar(x, vals, bottom=bottom, label=f"Label {cls}", color=color)
        bottom += vals
    plt.xticks(x, pivot.index, rotation=25, ha="right")
    plt.ylabel("Number of test windows")
    plt.title("Submission Label Distributions")
    plt.legend(ncol=3, fontsize=8)
    savefig(OUT / "submission_distributions.png")


def main() -> None:
    with open(CACHE / "train_feats_v3.pkl", "rb") as f:
        train_v3 = pickle.load(f)
    with open(CACHE / "train_feats_v2.pkl", "rb") as f:
        train_v2 = pickle.load(f)

    plot_class_distribution(train_v3)
    plot_feature_label_correlations(train_v3)
    y, pred, probs = cached_oof_best_oof(train_v2)
    plot_confusion_matrix(y, pred)
    oof_best_score = f1_score(y, pred, average="macro", labels=CLASSES, zero_division=0)
    plot_submission_distributions()

    print(f"oof_best_score={oof_best_score:.4f}")
    print(f"figures_dir={OUT}")


if __name__ == "__main__":
    main()

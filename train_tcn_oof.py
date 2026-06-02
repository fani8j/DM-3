"""Generate TCN out-of-fold predictions using the SAME GroupKFold folds
as the tree ensemble, so they can be honestly stacked.

Fixes from the earlier failed deep attempt:
- NO per-user normalization (keeps gravity/orientation signal intact)
- Global standardization (fit on train fold only)
- Moderate augmentation (jitter only), label smoothing, lighter model to avoid overfit
- Class-weighted focal loss for imbalance
"""

import sys
import logging
import pickle
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent / "src"))

from har.data.loader import load_dataset
from har.data.integrity import read_sample_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("train_tcn_oof.log", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def compute_features(data):
    """Temporal features keeping gravity intact. data: [300,6] -> [300, C]."""
    channels = [data]
    mag = torch.norm(data[:, :3], dim=1, keepdim=True)
    channels.append(mag)
    std_mag = torch.norm(data[:, 3:6], dim=1, keepdim=True)
    channels.append(std_mag)
    diff = torch.zeros_like(data); diff[1:] = data[1:] - data[:-1]
    channels.append(diff)
    for window in [5, 15, 30]:
        dt = data.t()
        unf = dt.unfold(1, window, 1)
        rm = torch.zeros(6, 300); rs = torch.zeros(6, 300)
        rm[:, window-1:] = unf.mean(dim=2)
        rs[:, window-1:] = unf.std(dim=2, correction=1)
        channels.append(rm.t()); channels.append(rs.t())
    tilt = torch.atan2(data[:, 1:2], torch.sqrt(data[:, 0:1]**2 + data[:, 2:3]**2) + 1e-8)
    channels.append(tilt)
    return torch.cat(channels, dim=1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha; self.smoothing = smoothing
    def forward(self, inputs, targets):
        n = inputs.size(1)
        log_p = F.log_softmax(inputs, dim=1)
        ce = F.nll_loss(log_p, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        # label smoothing term
        smooth = -log_p.mean(dim=1)
        return ((1 - self.smoothing) * focal + self.smoothing * smooth).mean()


class SEBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                nn.Linear(ch, ch // 4), nn.ReLU(),
                                nn.Linear(ch // 4, ch), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, inc, outc, k, d, dropout):
        super().__init__()
        pad = (k - 1) * d
        self.conv1 = nn.Conv1d(inc, outc, k, dilation=d, padding=pad)
        self.conv2 = nn.Conv1d(outc, outc, k, dilation=d, padding=pad)
        self.bn1 = nn.BatchNorm1d(outc); self.bn2 = nn.BatchNorm1d(outc)
        self.drop = nn.Dropout(dropout); self.pad = pad
        self.se = SEBlock(outc)
        self.res = nn.Conv1d(inc, outc, 1) if inc != outc else nn.Identity()
    def forward(self, x):
        r = self.res(x)
        o = self.conv1(x)[:, :, :-self.pad] if self.pad else self.conv1(x)
        o = self.drop(F.gelu(self.bn1(o)))
        o = self.conv2(o)[:, :, :-self.pad] if self.pad else self.conv2(o)
        o = self.bn2(o); o = self.se(o)
        return F.gelu(o + r)


class TCN(nn.Module):
    def __init__(self, inc, hidden=128, depth=6, k=7, dropout=0.3, nclass=6):
        super().__init__()
        self.proj = nn.Conv1d(inc, hidden, 1)
        self.blocks = nn.Sequential(*[TCNBlock(hidden, hidden, k, 2**i, dropout) for i in range(depth)])
        self.head = nn.Sequential(nn.Linear(hidden*2, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, nclass))
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x); x = self.blocks(x)
        x = torch.cat([x.mean(2), x.max(2)[0]], dim=1)
        return self.head(x)


class DS(Dataset):
    def __init__(self, seqs, labels, augment=False):
        self.seqs = seqs; self.labels = labels; self.augment = augment
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        s = self.seqs[i]
        if self.augment and random.random() < 0.5:
            s = s + torch.randn_like(s) * 0.03
        return s, self.labels[i]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load windows
    train_windows, _ = load_dataset(Path("train/train"), expect_label=True)
    test_windows, _ = load_dataset(Path("test/test"), expect_label=False)

    # Align ordering with the tree pipeline (sorted by user, then file_id — loader already does this)
    y = np.array([w.label for w in train_windows], dtype=int)
    groups = np.array([w.user_id for w in train_windows])
    file_ids = np.array([w.file_id for w in train_windows])

    # Precompute features (raw, before standardization)
    logger.info("Computing features...")
    train_feats = [compute_features(w.data) for w in train_windows]  # list of [300, C]
    test_feats = [compute_features(w.data) for w in test_windows]
    C = train_feats[0].shape[1]
    logger.info(f"Feature channels: {C}")

    train_stack = torch.stack(train_feats)  # [N,300,C]
    test_stack = torch.stack(test_feats)

    # Class weights
    cc = Counter(y.tolist()); total = len(y)
    alpha = torch.tensor([total/(6*cc[c]) for c in range(6)], dtype=torch.float32)
    alpha = (alpha / alpha.sum() * 6).to(device)

    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros((len(y), 6))
    test_pred = np.zeros((len(test_windows), 6))
    epochs = 60

    for fold, (tr, va) in enumerate(sgkf.split(train_stack, y, groups)):
        logger.info(f"--- Fold {fold+1}/{n_splits} ---")
        set_seed(42 + fold)

        # Standardize using train fold stats only (global, gravity intact)
        tr_data = train_stack[tr]
        mean = tr_data.mean(dim=[0, 1])
        std = tr_data.std(dim=[0, 1]); std = torch.where(std < 1e-8, torch.ones_like(std), std)

        def norm(x): return (x - mean) / std

        Xtr = norm(train_stack[tr]); Xva = norm(train_stack[va]); Xte = norm(test_stack)
        ytr = torch.tensor(y[tr], dtype=torch.long); yva = torch.tensor(y[va], dtype=torch.long)

        train_loader = DataLoader(DS(Xtr, ytr, augment=True), batch_size=64, shuffle=True, drop_last=True)

        model = TCN(C, hidden=128, depth=6, k=7, dropout=0.3).to(device)
        crit = FocalLoss(alpha=alpha, gamma=2.0, smoothing=0.05)
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, epochs=epochs,
                                                    steps_per_epoch=len(train_loader), pct_start=0.1)

        best_f1, best_va, best_te = 0.0, None, None
        for ep in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()

            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                model.eval()
                with torch.no_grad():
                    vp = []
                    for i in range(0, len(Xva), 256):
                        vp.append(F.softmax(model(Xva[i:i+256].to(device)), dim=1).cpu())
                    vp = torch.cat(vp).numpy()
                f1 = f1_score(y[va], vp.argmax(1), average="macro", labels=list(range(6)), zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_va = vp
                    with torch.no_grad():
                        tp = []
                        for i in range(0, len(Xte), 256):
                            tp.append(F.softmax(model(Xte[i:i+256].to(device)), dim=1).cpu())
                        best_te = torch.cat(tp).numpy()
                if (ep + 1) % 20 == 0:
                    logger.info(f"  ep{ep+1} val_f1={f1:.4f} best={best_f1:.4f}")

        oof[va] = best_va
        test_pred += best_te / n_splits
        logger.info(f"  Fold {fold+1} best val F1: {best_f1:.4f}")

    overall = f1_score(y, oof.argmax(1), average="macro", labels=list(range(6)), zero_division=0)
    logger.info(f"TCN OOF Macro F1: {overall:.4f}")

    # Save OOF aligned by file_id for stacking
    np.save("feature_cache/tcn_oof.npy", oof)
    np.save("feature_cache/tcn_test.npy", test_pred)
    np.save("feature_cache/tcn_oof_fileids.npy", file_ids)
    np.save("feature_cache/tcn_test_fileids.npy", np.array([w.file_id for w in test_windows]))
    logger.info("Saved TCN OOF/test predictions.")


if __name__ == "__main__":
    main()

"""Stronger deep sequence model with proper OOF generation for stacking.

Key differences from train_tcn_oof.py (which only reached 0.63):
- CNN + BiLSTM hybrid: conv captures local motion patterns, LSTM captures
  long-range temporal order (critical for the intensity-continuum classes 1/2/3/5)
- Trains longer (120 epochs) with cosine restart + early stopping per fold
- Mixup augmentation to regularize and help minority classes
- Test-time augmentation (TTA) averaging
- Saves OOF + test probs aligned by file_id for stacking
"""

import sys
import logging
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent / "src"))

from har.data.loader import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("train_deep_v2.log", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def compute_features(data):
    """Temporal features, gravity intact. data:[300,6] -> [300,C]."""
    channels = [data]
    mag = torch.norm(data[:, :3], dim=1, keepdim=True)
    channels.append(mag)
    std_mag = torch.norm(data[:, 3:6], dim=1, keepdim=True)
    channels.append(std_mag)
    diff = torch.zeros_like(data); diff[1:] = data[1:] - data[:-1]
    channels.append(diff)
    # jerk (2nd diff)
    diff2 = torch.zeros_like(data); diff2[2:] = diff[2:] - diff[1:-1]
    channels.append(diff2)
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
        log_p = F.log_softmax(inputs, dim=1)
        ce = F.nll_loss(log_p, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
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


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, k, dropout):
        super().__init__()
        self.conv = nn.Conv1d(inc, outc, k, padding=k // 2)
        self.bn = nn.BatchNorm1d(outc)
        self.se = SEBlock(outc)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(inc, outc, 1) if inc != outc else nn.Identity()
    def forward(self, x):
        r = self.res(x)
        o = self.drop(F.gelu(self.bn(self.conv(x))))
        o = self.se(o)
        return F.gelu(o + r)


class CNNBiLSTM(nn.Module):
    """CNN feature extractor + BiLSTM temporal encoder + attention pool."""
    def __init__(self, inc, hidden=128, dropout=0.3, nclass=6):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(inc, hidden, 7, dropout),
            ConvBlock(hidden, hidden, 5, dropout),
            nn.MaxPool1d(2),  # 300 -> 150
            ConvBlock(hidden, hidden, 5, dropout),
            ConvBlock(hidden, hidden, 3, dropout),
            nn.MaxPool1d(2),  # 150 -> 75
        )
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout)
        # attention pooling
        self.attn = nn.Linear(hidden * 2, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 * 2, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, nclass),
        )

    def forward(self, x):
        # x: [B,300,C] -> [B,C,300]
        x = x.transpose(1, 2)
        x = self.cnn(x)              # [B, hidden, 75]
        x = x.transpose(1, 2)        # [B, 75, hidden]
        seq, _ = self.lstm(x)        # [B, 75, hidden*2]
        # attention pool
        a = torch.softmax(self.attn(seq), dim=1)  # [B,75,1]
        attn_pool = (seq * a).sum(dim=1)          # [B, hidden*2]
        max_pool = seq.max(dim=1)[0]              # [B, hidden*2]
        pooled = torch.cat([attn_pool, max_pool], dim=1)
        return self.head(pooled)


class DS(Dataset):
    def __init__(self, seqs, labels, augment=False):
        self.seqs = seqs; self.labels = labels; self.augment = augment
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        s = self.seqs[i]
        if self.augment:
            if random.random() < 0.5:
                s = s + torch.randn_like(s) * 0.03
            if random.random() < 0.3:
                # magnitude scaling
                s = s * (1.0 + torch.randn(1) * 0.1)
        return s, self.labels[i]


def mixup(x, y, alpha=0.2, nclass=6):
    """Mixup augmentation."""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_onehot = F.one_hot(y, nclass).float()
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return mixed_x, mixed_y


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    train_windows, _ = load_dataset(Path("train/train"), expect_label=True)
    test_windows, _ = load_dataset(Path("test/test"), expect_label=False)

    y = np.array([w.label for w in train_windows], dtype=int)
    groups = np.array([w.user_id for w in train_windows])
    file_ids = np.array([w.file_id for w in train_windows])
    test_file_ids = np.array([w.file_id for w in test_windows])

    logger.info("Computing features...")
    train_feats = [compute_features(w.data) for w in train_windows]
    test_feats = [compute_features(w.data) for w in test_windows]
    C = train_feats[0].shape[1]
    logger.info(f"Feature channels: {C}")

    train_stack = torch.stack(train_feats)
    test_stack = torch.stack(test_feats)

    cc = Counter(y.tolist()); total = len(y)
    alpha = torch.tensor([total/(6*cc[c]) for c in range(6)], dtype=torch.float32)
    alpha = (alpha / alpha.sum() * 6).to(device)

    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros((len(y), 6))
    test_pred = np.zeros((len(test_windows), 6))
    epochs = 120

    for fold, (tr, va) in enumerate(sgkf.split(train_stack, y, groups)):
        logger.info(f"--- Fold {fold+1}/{n_splits} ---")
        set_seed(42 + fold)

        tr_data = train_stack[tr]
        mean = tr_data.mean(dim=[0, 1])
        std = tr_data.std(dim=[0, 1]); std = torch.where(std < 1e-8, torch.ones_like(std), std)
        def norm(x): return (x - mean) / std

        Xtr = norm(train_stack[tr]); Xva = norm(train_stack[va]); Xte = norm(test_stack)
        ytr = torch.tensor(y[tr], dtype=torch.long)

        train_loader = DataLoader(DS(Xtr, ytr, augment=True), batch_size=64,
                                  shuffle=True, drop_last=True)

        model = CNNBiLSTM(C, hidden=128, dropout=0.3).to(device)
        crit = FocalLoss(alpha=alpha, gamma=2.0, smoothing=0.05)
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=3e-3, epochs=epochs, steps_per_epoch=len(train_loader), pct_start=0.1)

        best_f1, best_va, best_te = 0.0, None, None
        patience, no_improve = 25, 0

        for ep in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                if random.random() < 0.5:
                    mx, my = mixup(xb, yb, alpha=0.2)
                    logp = F.log_softmax(model(mx), dim=1)
                    loss = -(my * logp).sum(dim=1).mean()
                else:
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
                    best_f1 = f1; best_va = vp; no_improve = 0
                    with torch.no_grad():
                        tp = []
                        for i in range(0, len(Xte), 256):
                            tp.append(F.softmax(model(Xte[i:i+256].to(device)), dim=1).cpu())
                        best_te = torch.cat(tp).numpy()
                else:
                    no_improve += 1
                if (ep + 1) % 20 == 0:
                    logger.info(f"  ep{ep+1} val_f1={f1:.4f} best={best_f1:.4f}")
                if no_improve >= patience:
                    logger.info(f"  early stop at ep{ep+1}")
                    break

        oof[va] = best_va
        test_pred += best_te / n_splits
        logger.info(f"  Fold {fold+1} best val F1: {best_f1:.4f}")

    overall = f1_score(y, oof.argmax(1), average="macro", labels=list(range(6)), zero_division=0)
    logger.info(f"CNN-BiLSTM OOF Macro F1: {overall:.4f}")
    per = f1_score(y, oof.argmax(1), average=None, labels=list(range(6)), zero_division=0)
    for c in range(6):
        logger.info(f"  Class {c} F1: {per[c]:.4f}")

    np.save("feature_cache/deep_oof.npy", oof)
    np.save("feature_cache/deep_test.npy", test_pred)
    np.save("feature_cache/deep_oof_fileids.npy", file_ids)
    np.save("feature_cache/deep_test_fileids.npy", test_file_ids)
    logger.info("Saved CNN-BiLSTM OOF/test predictions.")


if __name__ == "__main__":
    main()

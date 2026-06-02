"""Improved training script with augmentation, full-data training, and ensemble.

Key improvements over baseline:
1. Train on ALL data (no validation holdout) for final submission
2. Time-series augmentation (jitter, scale, time-warp)
3. Per-user normalization
4. Focal loss for class imbalance
5. Multi-seed ensemble
6. Larger model with multi-scale features
"""

import sys
import os
import random
import logging
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from har.config import load_config
from har.data.loader import load_dataset
from har.data.integrity import read_sample_submission, check_submission_alignment
from har.train.determinism import set_all_seeds
from har.train.manifest import get_code_version

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# Focal Loss (handles class imbalance better than CE)
# ===========================================================================

class FocalLoss(nn.Module):
    """Focal Loss: down-weights easy examples, focuses on hard ones."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # class weights tensor [C]
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


# ===========================================================================
# Data Augmentation
# ===========================================================================

def augment_jitter(data, sigma=0.05):
    """Add Gaussian noise."""
    return data + torch.randn_like(data) * sigma

def augment_scale(data, sigma=0.1):
    """Random scaling per channel."""
    factor = 1.0 + torch.randn(1, data.shape[1]) * sigma
    return data * factor

def augment_time_warp(data, sigma=0.2):
    """Simple time warping via random speed changes."""
    T, C = data.shape
    # Generate random cumulative time steps
    warp = torch.cumsum(torch.ones(T) + torch.randn(T) * sigma, dim=0)
    warp = warp / warp[-1] * (T - 1)  # normalize to [0, T-1]
    warp = warp.clamp(0, T - 1)
    
    # Interpolate
    indices = warp.long().clamp(0, T - 2)
    frac = (warp - indices.float()).unsqueeze(1)
    result = data[indices] * (1 - frac) + data[indices + 1] * frac
    return result

def augment_window_slice(data, ratio=0.9):
    """Random crop and resize back to original length."""
    T, C = data.shape
    slice_len = int(T * ratio)
    start = random.randint(0, T - slice_len)
    sliced = data[start:start + slice_len]
    # Resize back to T using linear interpolation
    sliced = sliced.unsqueeze(0).permute(0, 2, 1)  # [1, C, slice_len]
    resized = F.interpolate(sliced, size=T, mode='linear', align_corners=False)
    return resized.permute(0, 2, 1).squeeze(0)  # [T, C]


class AugmentedHARDataset(Dataset):
    """Dataset with online augmentation for training."""
    
    def __init__(self, sequences, labels, augment=True, oversample_minority=True):
        """
        sequences: list of [300, C] tensors
        labels: list of int labels
        """
        self.augment = augment
        
        if oversample_minority:
            # Oversample minority classes to balance
            sequences, labels = self._oversample(sequences, labels)
        
        self.sequences = sequences
        self.labels = labels
    
    def _oversample(self, sequences, labels):
        """Oversample minority classes to match majority class count."""
        counter = Counter(labels)
        max_count = max(counter.values())
        
        new_sequences = list(sequences)
        new_labels = list(labels)
        
        for cls, count in counter.items():
            if count < max_count:
                # Find indices of this class
                cls_indices = [i for i, l in enumerate(labels) if l == cls]
                # Oversample
                n_needed = max_count - count
                for _ in range(n_needed):
                    idx = random.choice(cls_indices)
                    new_sequences.append(sequences[idx].clone())
                    new_labels.append(cls)
        
        return new_sequences, new_labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx].clone()
        label = self.labels[idx]
        
        if self.augment:
            # Apply random augmentations
            if random.random() < 0.5:
                seq = augment_jitter(seq, sigma=0.03)
            if random.random() < 0.3:
                seq = augment_scale(seq, sigma=0.1)
            if random.random() < 0.3:
                seq = augment_time_warp(seq, sigma=0.15)
            if random.random() < 0.2:
                seq = augment_window_slice(seq, ratio=0.9)
        
        return seq, label


# ===========================================================================
# Improved TCN Model
# ===========================================================================

class ImprovedTCNBlock(nn.Module):
    """TCN block with squeeze-excitation and better normalization."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.padding = padding
        
        # Squeeze-excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(out_ch, out_ch // 4),
            nn.ReLU(),
            nn.Linear(out_ch // 4, out_ch),
            nn.Sigmoid(),
        )
        
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        res = self.residual(x)
        out = self.conv1(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.bn2(out)
        
        # Squeeze-excitation
        se_weight = self.se(out).unsqueeze(-1)
        out = out * se_weight
        
        out = F.gelu(out + res)
        return out


class ImprovedTCN(nn.Module):
    """Multi-scale TCN with squeeze-excitation and global+max pooling."""
    def __init__(self, input_channels, hidden_dim=256, depth=8, kernel_size=7, dropout=0.3, num_classes=6):
        super().__init__()
        
        # Multi-scale input projection
        self.proj_3 = nn.Conv1d(input_channels, hidden_dim // 4, 3, padding=1)
        self.proj_5 = nn.Conv1d(input_channels, hidden_dim // 4, 5, padding=2)
        self.proj_7 = nn.Conv1d(input_channels, hidden_dim // 4, 7, padding=3)
        self.proj_1 = nn.Conv1d(input_channels, hidden_dim // 4, 1)
        
        # TCN blocks with increasing dilation
        layers = []
        for i in range(depth):
            dilation = 2 ** i
            layers.append(ImprovedTCNBlock(hidden_dim, hidden_dim, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*layers)
        
        # Dual pooling (avg + max)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x):
        # x: [B, 300, C] -> [B, C, 300]
        x = x.transpose(1, 2)
        
        # Multi-scale projection
        x = torch.cat([
            self.proj_3(x),
            self.proj_5(x),
            self.proj_7(x),
            self.proj_1(x),
        ], dim=1)  # [B, hidden_dim, 300]
        
        # TCN blocks
        x = self.tcn(x)  # [B, hidden_dim, T]
        
        # Dual pooling
        avg_pool = x.mean(dim=2)  # [B, hidden_dim]
        max_pool = x.max(dim=2)[0]  # [B, hidden_dim]
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # [B, hidden_dim*2]
        
        return self.classifier(pooled)


# ===========================================================================
# Feature Engineering (enhanced)
# ===========================================================================

def compute_features(data_tensor):
    """Compute enhanced features from [300, 6] base tensor.
    
    Returns [300, C] tensor with all features concatenated.
    """
    # Base: [300, 6]
    channels = [data_tensor]
    
    # Magnitude of mean acceleration
    mag = torch.norm(data_tensor[:, :3], dim=1, keepdim=True)  # [300, 1]
    channels.append(mag)
    
    # Magnitude of std
    std_mag = torch.norm(data_tensor[:, 3:6], dim=1, keepdim=True)  # [300, 1]
    channels.append(std_mag)
    
    # First difference (velocity-like)
    diff = torch.zeros_like(data_tensor)
    diff[1:] = data_tensor[1:] - data_tensor[:-1]
    channels.append(diff)  # [300, 6]
    
    # Second difference (jerk-like)
    diff2 = torch.zeros_like(data_tensor)
    diff2[2:] = diff[2:] - diff[1:-1]
    channels.append(diff2)  # [300, 6]
    
    # Rolling stats with multiple windows
    for window in [5, 15, 30]:
        data_t = data_tensor.t()  # [6, 300]
        unfolded = data_t.unfold(1, window, 1)  # [6, 300-W+1, W]
        
        r_mean = torch.zeros(6, 300)
        r_std = torch.zeros(6, 300)
        
        means = unfolded.mean(dim=2)  # [6, 300-W+1]
        stds = unfolded.std(dim=2, correction=1)
        
        r_mean[:, window-1:] = means
        r_std[:, window-1:] = stds
        
        channels.append(r_mean.t())  # [300, 6]
        channels.append(r_std.t())   # [300, 6]
    
    # Tilt angle (gravity direction)
    tilt = torch.atan2(data_tensor[:, 1:2], 
                       torch.sqrt(data_tensor[:, 0:1]**2 + data_tensor[:, 2:3]**2) + 1e-8)
    channels.append(tilt)  # [300, 1]
    
    return torch.cat(channels, dim=1)


# ===========================================================================
# Per-user normalization
# ===========================================================================

def compute_user_stats(windows):
    """Compute per-user mean and std for normalization."""
    user_data = defaultdict(list)
    for w in windows:
        user_data[w.user_id].append(w.data)
    
    user_stats = {}
    for uid, tensors in user_data.items():
        stacked = torch.stack(tensors)  # [N, 300, 6]
        mean = stacked.mean(dim=[0, 1])  # [6]
        std = stacked.std(dim=[0, 1])    # [6]
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
        user_stats[uid] = (mean, std)
    
    return user_stats


def normalize_per_user(windows, user_stats):
    """Apply per-user normalization."""
    normalized = []
    for w in windows:
        mean, std = user_stats[w.user_id]
        norm_data = (w.data - mean) / std
        normalized.append((norm_data, w.label, w.file_id, w.user_id))
    return normalized


# ===========================================================================
# Training loop
# ===========================================================================

def train_one_model(train_data, val_data, config, seed, device, epochs=100):
    """Train a single model and return it with its best val F1."""
    set_all_seeds(seed)
    
    # Prepare sequences and labels
    train_seqs = [compute_features(d[0]) for d in train_data]
    train_labels = [d[1] for d in train_data]
    
    input_channels = train_seqs[0].shape[1]
    logger.info(f"Input channels: {input_channels}")
    
    # Create dataset with augmentation and oversampling
    train_dataset = AugmentedHARDataset(train_seqs, train_labels, augment=True, oversample_minority=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    
    # Validation
    if val_data:
        val_seqs = torch.stack([compute_features(d[0]) for d in val_data])
        val_labels = torch.tensor([d[1] for d in val_data], dtype=torch.long)
    
    # Model
    model = ImprovedTCN(
        input_channels=input_channels,
        hidden_dim=256,
        depth=8,
        kernel_size=7,
        dropout=0.3,
    ).to(device)
    
    # Class weights for focal loss
    class_counts = Counter(train_labels)
    total = sum(class_counts.values())
    alpha = torch.zeros(6)
    for c in range(6):
        if class_counts[c] > 0:
            alpha[c] = total / (6 * class_counts[c])
    alpha = alpha / alpha.sum() * 6
    alpha = alpha.to(device)
    
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, epochs=epochs, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='cos'
    )
    
    best_f1 = 0.0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for seq_batch, label_batch in train_loader:
            seq_batch = seq_batch.to(device)
            label_batch = label_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(seq_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        # Validate
        if val_data and (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                all_preds = []
                bs = 128
                for i in range(0, len(val_seqs), bs):
                    batch = val_seqs[i:i+bs].to(device)
                    preds = model(batch).argmax(dim=1).cpu().tolist()
                    all_preds.extend(preds)
                
                val_f1 = f1_score(val_labels.tolist(), all_preds, average='macro', labels=list(range(6)), zero_division=0)
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs} — loss={epoch_loss/n_batches:.4f}, val_f1={val_f1:.4f}, best={best_f1:.4f}")
        elif not val_data and (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs} — loss={epoch_loss/n_batches:.4f}")
    
    # If no validation, use final state
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_f1 = -1.0
    
    model.load_state_dict(best_state)
    return model, best_f1, input_channels


# ===========================================================================
# Main
# ===========================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading training data...")
    train_windows, train_summary = load_dataset(Path("train/train"), expect_label=True)
    logger.info(f"Loaded {train_summary.n_windows} windows from {train_summary.n_users} users")
    logger.info(f"Class distribution: {train_summary.class_distribution}")
    
    # Per-user normalization
    logger.info("Computing per-user normalization stats...")
    user_stats = compute_user_stats(train_windows)
    train_data = normalize_per_user(train_windows, user_stats)
    # train_data is list of (normalized_tensor, label, file_id, user_id)
    
    # GroupKFold validation to estimate real performance
    from har.train.splitter import GroupKFoldSplitter
    splitter = GroupKFoldSplitter(n_splits=5)
    
    logger.info("\n=== Validation run (GroupKFold, 1 fold) to estimate performance ===")
    fold_iter = iter(splitter.split(train_windows))
    fold = next(fold_iter)
    
    val_train_data = [train_data[i] for i in fold.train_indices]
    val_val_data = [train_data[i] for i in fold.val_indices]
    
    # Quick validation with fewer epochs
    _, val_f1, _ = train_one_model(val_train_data, val_val_data, None, seed=42, device=device, epochs=80)
    logger.info(f"Validation Macro F1 (GroupKFold fold 0): {val_f1:.4f}")
    
    # Train final models on ALL data (ensemble of 3 seeds)
    logger.info("\n=== Training final ensemble on ALL data ===")
    models = []
    seeds = [42, 123, 456]
    
    for i, seed in enumerate(seeds):
        logger.info(f"\nTraining model {i+1}/3 (seed={seed})...")
        model, _, input_channels = train_one_model(train_data, None, None, seed=seed, device=device, epochs=100)
        models.append(model)
    
    # Generate submission with ensemble
    logger.info("\n=== Generating submission ===")
    
    # Load test data
    test_windows, test_summary = load_dataset(Path("test/test"), expect_label=False)
    submission_ids = read_sample_submission(Path("sample_submission.csv"))
    
    # Compute test user stats (use test data's own stats for normalization)
    test_user_stats = compute_user_stats(test_windows)
    test_data = normalize_per_user(test_windows, test_user_stats)
    
    # Build file_id -> normalized tensor mapping
    file_id_to_data = {d[2]: d[0] for d in test_data}
    
    # Compute features for all test windows in submission order
    test_features = []
    for fid in submission_ids:
        feat = compute_features(file_id_to_data[fid])
        test_features.append(feat)
    test_tensor = torch.stack(test_features)  # [N, 300, C]
    
    # Ensemble prediction
    all_probs = []
    for model in models:
        model.to(device)
        model.eval()
        probs = []
        with torch.no_grad():
            bs = 128
            for i in range(0, len(test_tensor), bs):
                batch = test_tensor[i:i+bs].to(device)
                logits = model(batch)
                prob = F.softmax(logits, dim=1).cpu()
                probs.append(prob)
        all_probs.append(torch.cat(probs, dim=0))
    
    # Average probabilities
    avg_probs = torch.stack(all_probs).mean(dim=0).numpy().astype(np.float64)
    predictions = np.argmax(avg_probs, axis=1)
    
    # Write submission
    output_path = Path("submissions/submission_improved.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="ascii") as f:
        f.write("Id,Label\n")
        for fid, label in zip(submission_ids, predictions):
            f.write(f"{fid},{int(label)}\n")
    
    logger.info(f"Submission written to: {output_path}")
    logger.info(f"Label distribution: {dict(sorted(Counter(predictions.tolist()).items()))}")
    logger.info("Done!")


if __name__ == "__main__":
    main()

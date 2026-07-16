#!/usr/bin/env python3
"""
Repeat experiment: train DAFN-T with 5 seeds, report mean ± std accuracy.

Usage:
    python experiments/repeat_experiment.py [--csv data/dataset_ready.csv]
                                           [--features data/X_visual_train.npy data/X_visual_val.npy]
                                           [--epochs 50] [--seed_list 42,123,456,789,999]
                                           [--device cuda]

Output:
    - result_seed{seed}.txt  — per-seed validation accuracy
    - repeat_summary.txt     — mean ± std for all seeds
    - checkpoints/           — best model per seed
"""
import os, sys, csv, json, argparse, time, warnings
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dafn import DAFN_T

# ─────────────────────────── Global config ───────────────────────────
AGRONOMIC_FEATURES = ['Plant Height','Stem Diameter','Leaf Width','Leaf Length',
                      'NDVI','RVI','LNC','LNA','LAI','LDW']
SENSOR_FEATURES   = ['Air_Temperature','Relative_Humidity','Light_Intensity',
                     'CO2','Soil_Moisture','Soil_Temperature']
ALL_FEATURES = AGRONOMIC_FEATURES + SENSOR_FEATURES
LABEL_MAP   = {'Healthy': 0, 'Stress': 1, 'Other': 1}  # binary: Other→Stress
HIDDEN_DIM  = 128   # FAM projection dimension (d)
WINDOW      = 5     # temporal window
BATCH_SIZE  = 16    # will auto-halve if OOM
EPOCHS      = 50
PATIENCE    = 10    # early stopping
SEED_LIST   = [42, 123, 456, 789, 999]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────── Data Loading ───────────────────────────
def load_data(csv_path, vis_feat_path=None, use_sensor=False):
    """Load CSV, visual features, return train/val rows + feature dict."""
    with open(csv_path) as f:
        all_rows = list(csv.DictReader(f))
    all_rows = [r for r in all_rows if r.get('Source') != '2023']
    valid = [r for r in all_rows if r.get('label_3class','Unknown') in {'Healthy','Stress','Other'}]
    feat = {}
    if vis_feat_path and os.path.exists(vis_feat_path):
        d = np.load(vis_feat_path, allow_pickle=True).item()
        feat = {int(k): np.array(v).flatten() for k, v in d.items()}
        print(f'  Loaded visual features: {len(feat)} samples')

    # Map from CSV Number column → feature index
    # finetuned_logits.npy uses enumerate(valid) index, not CSV row number
    valid_with_idx = [(i, r) for i, r in enumerate(valid)]
    train = [(i, r) for i, r in valid_with_idx if r.get('split') == 'train']
    val   = [(i, r) for i, r in valid_with_idx if r.get('split') == 'val']
    train_rows = [r for _, r in train]
    val_rows   = [r for _, r in val]

    # Standardize agronomic + sensor features
    scaler = StandardScaler()
    for rows in [train_rows, val_rows]:
        X = np.array([[float(r.get(f,'nan')) for f in ALL_FEATURES] for r in rows])
        X = np.nan_to_num(X, nan=0.0)
        X_s = scaler.fit_transform(X)
        for i, r in enumerate(rows):
            for j, f in enumerate(ALL_FEATURES):
                r[f] = X_s[i, j]

    # Class weights (binary)
    yt = np.array([LABEL_MAP.get(r['label_3class'],-1) for r in train_rows])
    cw = compute_class_weight('balanced', classes=np.unique(yt), y=yt)
    cw_tensor = torch.tensor(cw, dtype=torch.float32)

    # Build feature dict mapping (original idx) → feature vector
    # For finetuned_logits.npy, the keys are from enumerate(valid)
    # For pre-split visual features (X_visual_train.npy + X_visual_val.npy), we handle separately
    if not feat:
        # No feature dict loaded, we'll use pre-split visual npy files instead
        pass

    print(f'  Loaded {len(valid)} rows: train={len(train_rows)}, val={len(val_rows)}')
    print(f'  Train labels: {Counter(r["label_3class"] for r in train_rows)}')
    print(f'  Val labels:   {Counter(r["label_3class"] for r in val_rows)}')
    return train_rows, val_rows, feat, cw_tensor, valid

def load_visual_from_split(train_path, val_path, valid, train_rows, val_rows):
    """Load pre-split visual npy files (X_visual_train.npy / X_visual_val.npy)."""
    feat = {}
    if os.path.exists(train_path) and os.path.exists(val_path):
        Xtr = np.load(train_path, dtype=np.float32)
        Xva = np.load(val_path, dtype=np.float32)
        # Map row id → feature index in the split arrays
        # Since these are pre-split by row order in train/val lists
        for i, r in enumerate(train_rows):
            feat[id(r)] = Xtr[i].flatten()
        for i, r in enumerate(val_rows):
            feat[id(r)] = Xva[i].flatten()
        print(f'  Loaded visual features from split files: Xtr={Xtr.shape}, Xva={Xva.shape}')
    elif train_path and os.path.exists(train_path):
        # Single file with dict format
        d = np.load(train_path, allow_pickle=True).item()
        feat = {int(k): np.array(v).flatten() for k, v in d.items()}
        print(f'  Loaded visual features from dict: {len(feat)} samples')
    return feat

# ────────────────────── Temporal Dataset ───────────────────────────
class SequenceDataset(Dataset):
    """Sliding-window temporal sequences from plant-wise grouped rows."""
    def __init__(self, rows, feature_dict, window_size=WINDOW, use_sensor=False, stride=1):
        self.window_size = window_size
        self.features = feature_dict or {}
        self.use_sensor = use_sensor

        # Group by plant ID using the first number in Photo Path
        pg = defaultdict(list)
        for i, r in enumerate(rows):
            pp = r.get('Photo Path', '')
            if pp:
                import re
                m = re.match(r'images/(\d+)', pp.split(';')[0])
                pid = int(m.group(1)) if m else -1
            else:
                pid = -1
            pg[pid].append((r, i))

        self.samples = []
        for pid, items in pg.items():
            items.sort(key=lambda x: x[0].get('Date', ''))
            for i in range(0, len(items) - window_size + 1, stride):
                wins = items[i:i + window_size]
                label = LABEL_MAP.get(wins[-1][0].get('label_3class', 'Unknown'), -1)
                if label >= 0:
                    self.samples.append((wins, label))
        print(f'  SeqDataset: {len(self.samples)} windows (T={window_size}), {len(pg)} plants')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wins, label = self.samples[idx]
        T = self.window_size
        img_seq = torch.zeros(T, 2048)
        agr_seq = torch.zeros(T, len(AGRONOMIC_FEATURES))
        for t, (r, oi) in enumerate(wins):
            # Visual: try row id first, then feature dict key
            key = id(r)
            if key in self.features:
                img_seq[t] = torch.tensor(self.features[key], dtype=torch.float32)
            # Agronomic (already standardized in load_data)
            agr_seq[t] = torch.tensor([float(r.get(f, 0)) for f in AGRONOMIC_FEATURES], dtype=torch.float32)

        if self.use_sensor:
            sen_seq = torch.zeros(T, len(SENSOR_FEATURES))
            for t, (r, _) in enumerate(wins):
                sen_seq[t] = torch.tensor([float(r.get(f, 0)) for f in SENSOR_FEATURES], dtype=torch.float32)
            return img_seq, agr_seq, sen_seq, torch.tensor(label, dtype=torch.long)
        return img_seq, agr_seq, torch.tensor(label, dtype=torch.long)


# ────────────────────── Single Training Run ─────────────────────────
def train_dafn_t(seed, train_rows, val_rows, feat_dict, cw_tensor,
                 device='cuda', batch_size=BATCH_SIZE, use_sensor=False, num_epochs=EPOCHS):
    """Train DAFN-T with given seed, return best val accuracy."""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create datasets
    train_ds = SequenceDataset(train_rows, feat_dict, use_sensor=use_sensor)
    val_ds   = SequenceDataset(val_rows,   feat_dict, use_sensor=use_sensor)

    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f'  [Seed {seed}] SKIPPED: empty dataset')
        return 0.0

    # DataLoaders with adaptive batch size
    bs = batch_size
    while True:
        try:
            tl = DataLoader(train_ds, bs, shuffle=True)
            vl = DataLoader(val_ds, bs)
            # Test one batch
            for b in tl: break
            break
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and bs > 1:
                bs = bs // 2
                print(f'  [Seed {seed}] OOM → batch size halved to {bs}')
                torch.cuda.empty_cache()
                continue
            raise e

    # Model
    model = DAFN_T(window_size=WINDOW, hidden_dim=HIDDEN_DIM, use_sensor=use_sensor,
                   image_dim=2048, agronomic_dim=10, sensor_dim=6, num_classes=2)
    model.to(device)

    # Loss and optimizer
    crit = nn.CrossEntropyLoss(weight=cw_tensor.to(device))
    opt  = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)

    # Training loop
    best_acc = 0.0
    best_state = None
    patience_counter = 0
    t0 = time.time()

    print(f'\n{"="*60}')
    print(f'  Seed {seed} | DAFN-T (d={HIDDEN_DIM}, T={WINDOW})')
    print(f'  Train: {len(train_ds)} windows | Val: {len(val_ds)} windows')
    print(f'  Params: {sum(p.numel() for p in model.parameters()):,}')
    print(f'{"="*60}')

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(tl, desc=f'  Epoch {epoch:2d}/{num_epochs}', leave=False)
        for batch in pbar:
            lbl = batch[-1].to(device)
            n = len(batch)
            if n == 4:
                logits, _ = model(batch[0].to(device), batch[1].to(device), batch[2].to(device))
            elif n == 3:
                r = model(batch[0].to(device), batch[1].to(device))
                logits = r[0] if isinstance(r, tuple) else r
            else:
                logits = model(batch[0].to(device))

            loss = crit(logits, lbl)
            opt.zero_grad()
            loss.backward()
            opt.step()

            _, preds = logits.max(1)
            correct += (preds == lbl).sum().item()
            total += lbl.size(0)
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.3f}'})

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in vl:
                lbl = batch[-1].to(device)
                n = len(batch)
                if n == 4:
                    logits, _ = model(batch[0].to(device), batch[1].to(device), batch[2].to(device))
                elif n == 3:
                    r = model(batch[0].to(device), batch[1].to(device))
                    logits = r[0] if isinstance(r, tuple) else r
                else:
                    logits = model(batch[0].to(device))
                _, preds = logits.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(lbl.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1  = f1_score(all_labels, all_preds, average='macro')

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step(val_acc)

        # Log every 5 epochs + first and last
        if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs or patience_counter == PATIENCE:
            elapsed = time.time() - t0
            print(f'  [{seed}] E{epoch:2d}/{num_epochs} | train_acc={correct/total:.4f} | '
                  f'val_acc={val_acc:.4f} val_f1={val_f1:.4f} | '
                  f'best={best_acc:.4f} | {elapsed:6.1f}s')

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f'  [{seed}] Early stopping triggered (patience={PATIENCE})')
            break

    # Final evaluation
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in vl:
            lbl = batch[-1].to(device)
            n = len(batch)
            if n == 4:
                logits, _ = model(batch[0].to(device), batch[1].to(device), batch[2].to(device))
            elif n == 3:
                r = model(batch[0].to(device), batch[1].to(device))
                logits = r[0] if isinstance(r, tuple) else r
            else:
                logits = model(batch[0].to(device))
            _, preds = logits.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())

    final_acc = accuracy_score(all_labels, all_preds)
    final_f1  = f1_score(all_labels, all_preds, average='macro')

    # Save checkpoint
    ckpt_dir = os.path.join(RESULTS_DIR, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(best_state, os.path.join(ckpt_dir, f'dafn_t_seed{seed}.pth'))

    # Save per-seed result
    result_path = os.path.join(RESULTS_DIR, f'result_seed{seed}.txt')
    with open(result_path, 'w') as f:
        f.write(f'seed={seed}\n')
        f.write(f'best_val_acc={best_acc:.6f}\n')
        f.write(f'final_val_acc={final_acc:.6f}\n')
        f.write(f'final_macro_f1={final_f1:.6f}\n')
        f.write(f'epochs_completed={min(epoch, num_epochs)}\n')
    print(f'  [{seed}] Saved result → {result_path}')

    return best_acc


# ─────────────────────────── Main ──────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='data/dataset_ready.csv')
    parser.add_argument('--features', nargs='*', default=[],
                        help='Paths: [finetuned_logits.npy] OR [X_visual_train.npy X_visual_val.npy]')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--seed_list', default=','.join(str(s) for s in SEED_LIST))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--use_sensor', action='store_true', help='Include sensor modality')
    args = parser.parse_args()

    EPOCHS_G = args.epochs
    seeds = [int(s) for s in args.seed_list.split(',')]
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f'Device: {device}')
    print(f'Seeds: {seeds}')
    print(f'Epochs: {EPOCHS_G}')
    print(f'Batch size: {args.batch_size}')
    print(f'Use sensor: {args.use_sensor}')
    print(f'FAM dim (HIDDEN_DIM): {HIDDEN_DIM}')

    # ── Load data ──
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.csv)
    print(f'Loading CSV: {csv_path}')
    train_rows, val_rows, feat_dict, cw_tensor, valid = load_data(csv_path, use_sensor=args.use_sensor)

    # ── Load visual features ──
    if len(args.features) == 1:
        # Single dict file (finetuned_logits.npy style)
        fp = args.features[0]
        if os.path.exists(fp):
            d = np.load(fp, allow_pickle=True).item()
            # Map by enumerate(valid) index
            # The features dict uses enumerate(valid) index as key
            # We need to map each row in train/val to its index in valid
            valid_with_idx = [(i, r) for i, r in enumerate(valid)]
            for i, r in valid_with_idx:
                if i in d:
                    feat_dict[id(r)] = np.array(d[i]).flatten()
            print(f'  Mapped {len(feat_dict)} visual features from {fp}')
    elif len(args.features) == 2:
        # Pre-split visual features
        train_path = args.features[0]
        val_path   = args.features[1]
        if os.path.exists(train_path):
            Xtr = np.load(train_path).astype(np.float32)
            for i, r in enumerate(train_rows):
                feat_dict[id(r)] = Xtr[i].flatten()
        if os.path.exists(val_path):
            Xva = np.load(val_path).astype(np.float32)
            for i, r in enumerate(val_rows):
                feat_dict[id(r)] = Xva[i].flatten()
        print(f'  Loaded pre-split visual features: train={os.path.basename(train_path)}, val={os.path.basename(val_path)}')
    else:
        print('  WARNING: No visual features provided. Using zero vectors.')
        # Features remain empty; SequenceDataset will use zeros

    # ── Run experiments ──
    accuracies = []
    print(f'\n{"#"*60}')
    print(f'  Starting {len(seeds)} repeat experiments')
    print(f'{"#"*60}')

    epochs_to_run = EPOCHS_G

    for seed in seeds:
        acc = train_dafn_t(seed, train_rows, val_rows, feat_dict, cw_tensor,
                          device=device, batch_size=args.batch_size, use_sensor=args.use_sensor, num_epochs=epochs_to_run)
        accuracies.append(acc)
        print(f'  ✓ Seed {seed}: best_val_acc = {acc*100:.2f}%\n')

    # Restore

    # ── Summary ──
    acc_pct = np.array(accuracies) * 100
    mean_acc = np.mean(acc_pct)
    std_acc  = np.std(acc_pct, ddof=1)  # sample std

    print(f'\n{"="*60}')
    print(f'  REPEAT EXPERIMENT SUMMARY ({len(seeds)} seeds)')
    print(f'{"="*60}')
    print(f'  Model: DAFN-T (d={HIDDEN_DIM}, T={WINDOW})')
    print(f'  Seeds: {seeds}')
    for s, a in zip(seeds, acc_pct):
        print(f'    Seed {s:3d}: {a:.2f}%')
    print(f'  ─────────────────────────────')
    print(f'  Mean: {mean_acc:.2f}%')
    print(f'  Std:  {std_acc:.2f}%')
    print(f'  Formatted (LaTeX): ${mean_acc:.2f} \\pm {std_acc:.2f}\\%$')

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, 'repeat_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f'Repeat Experiment Summary\n')
        f.write(f'Model: DAFN-T (d={HIDDEN_DIM}, T={WINDOW})\n')
        f.write(f'Seeds: {seeds}\n')
        f.write(f'Epochs: {EPOCHS_G}\n')
        for s, a in zip(seeds, acc_pct):
            f.write(f'Seed {s:3d}: {a:.2f}%\n')
        f.write(f'Mean: {mean_acc:.2f}%\n')
        f.write(f'Std:  {std_acc:.2f}%\n')
        f.write(f'Mean ± Std: {mean_acc:.2f} ± {std_acc:.2f}%\n')
    print(f'\n  Summary saved → {summary_path}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()

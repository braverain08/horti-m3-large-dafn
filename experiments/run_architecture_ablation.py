#!/usr/bin/env python3
"""
Reproduce Table 8 (FAM/MRDS/dimension ablation) and Table 9 (temporal ablation).

Variants:
  dafn_t_d64      DAFN-T with FAM dim=64
  dafn_t_no_mrds  DAFN-T without MRDS (simple averaging), FAM dim=128
  dafn_t_no_fam   DAFN-T without FAM (raw concat + MRDS), FAM dim=128
  dafn_t_d32      DAFN-T with FAM dim=32
  dafn_t_d128     DAFN-T with FAM dim=128
  dafn_t1         DAFN-T with T=1 (temporal ablation)
  dafn_t5         DAFN-T with T=5 (temporal ablation reference)

Usage:
  python experiments/run_architecture_ablation.py --dry_run
  python experiments/run_architecture_ablation.py --epochs 50
"""
import os, sys, csv, re, json, argparse, time
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dafn import DAFN_T, DAFN_T_NoMRDS, DAFN_T_NoFAM


AGRO = ['Plant Height','Stem Diameter','Leaf Width','Leaf Length',
        'NDVI','RVI','LNC','LNA','LAI','LDW']
LABEL_MAP = {'Healthy': 0, 'Stress': 1, 'Other': 1}
BATCH = 16
EPOCHS = 50
PATIENCE = 10
SEED = 42

VARIANT_WINDOW = {
    'dafn_t_d64': 5,
    'dafn_t_no_mrds': 5,
    'dafn_t_no_fam': 5,
    'dafn_t_d32': 5,
    'dafn_t_d128': 5,
    'dafn_t1': 1,
    'dafn_t5': 5,
}


def make_model(name, window, use_sensor=False):
    common = dict(window_size=window, image_dim=2048, agronomic_dim=10,
                  sensor_dim=6, num_classes=2, use_sensor=use_sensor)
    if name == 'dafn_t_d64':
        return DAFN_T(hidden_dim=64, **common)
    if name == 'dafn_t_no_mrds':
        return DAFN_T_NoMRDS(hidden_dim=128, **common)
    if name == 'dafn_t_no_fam':
        return DAFN_T_NoFAM(hidden_dim=128, **common)
    if name == 'dafn_t_d32':
        return DAFN_T(hidden_dim=32, **common)
    if name == 'dafn_t1':
        return DAFN_T(hidden_dim=128, **common)
    return DAFN_T(hidden_dim=128, **common)  # dafn_t_d128 / dafn_t5


class SeqDataset(Dataset):
    def __init__(self, rows, feat, window=5):
        self.window = window
        self.feat = feat
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            pp = r.get('Photo Path', '')
            pid = -1
            if pp:
                m = re.match(r'images/(\d+)', pp.split(';')[0])
                if m:
                    pid = int(m.group(1))
            groups[pid].append((r, i))
        self.samples = []
        for items in groups.values():
            items.sort(key=lambda x: x[0].get('Date', ''))
            for i in range(len(items) - window + 1):
                wins = items[i:i+window]
                label = LABEL_MAP.get(wins[-1][0].get('label_3class', ''), -1)
                if label >= 0:
                    self.samples.append((wins, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wins, label = self.samples[idx]
        T = self.window
        img_seq = torch.zeros(T, 2048)
        agr_seq = torch.zeros(T, len(AGRO))
        for t, (r, oi) in enumerate(wins):
            img_seq[t] = torch.tensor(self.feat.get(oi, np.zeros(2048)), dtype=torch.float32)
            agr_seq[t] = torch.tensor([float(r.get(f, 0)) for f in AGRO], dtype=torch.float32)
        return img_seq, agr_seq, torch.tensor(label, dtype=torch.long)


def load(csv_path, ft_path, noisy_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get('Source') != '2023']
    valid = [r for r in rows if r.get('label_3class', '') in {'Healthy', 'Stress', 'Other'}]
    tr = [(i, r) for i, r in enumerate(valid) if r.get('split') == 'train']
    va = [(i, r) for i, r in enumerate(valid) if r.get('split') == 'val']

    scaler = StandardScaler()
    for rr in ([r for _, r in tr], [r for _, r in va]):
        X = np.array([[float(r.get(f, 'nan')) for f in AGRO] for r in rr])
        X = np.nan_to_num(X, nan=0.0)
        Xs = scaler.fit_transform(X)
        for i, r in enumerate(rr):
            for j, f in enumerate(AGRO):
                r[f] = Xs[i, j]

    y_tr = np.array([LABEL_MAP[r['label_3class']] for _, r in tr], dtype=np.int64)
    cw = torch.tensor(compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr),
                      dtype=torch.float32)

    d = np.load(ft_path, allow_pickle=True).item()
    clean_feat = {i: np.array(d.get(i, np.zeros(2048))).flatten() for i in range(len(valid))}
    v_noise = np.load(noisy_path).astype(np.float32) if noisy_path and os.path.exists(noisy_path) else None
    print(f'  Train:{len(tr)} Val:{len(va)} Features:{len(clean_feat)} Noisy:{v_noise.shape if v_noise is not None else "N/A"}')
    return valid, clean_feat, v_noise, cw


def train_variant(maker, seed, train_rows, train_feat, window, cw, device, epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    ds = SeqDataset(train_rows, train_feat, window)
    m = maker().to(device)
    dl = DataLoader(ds, min(BATCH, max(1, len(ds) // 4)), shuffle=True)
    crit = nn.CrossEntropyLoss(weight=cw.to(device))
    opt = optim.Adam(m.parameters(), lr=1e-4)
    best_acc = 0.0
    best_state = None
    patience = 0
    for ep in range(1, epochs + 1):
        m.train()
        c = 0
        t = 0
        for imgs, agrs, lbls in dl:
            imgs = imgs.to(device)
            agrs = agrs.to(device)
            lbls = lbls.to(device)
            logits, _ = m(imgs, agrs)
            loss = crit(logits, lbls)
            opt.zero_grad()
            loss.backward()
            opt.step()
            _, pr = logits.max(1)
            c += (pr == lbls).sum().item()
            t += lbls.size(0)
        acc = c / t
        if ep == 1 or ep % 10 == 0:
            print(f'    Ep{ep} train_acc={acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f'    early stop at Ep{ep}')
                break
    m.load_state_dict(best_state)
    return m


def evaluate_variant(m, rows, feat, window, device):
    ds = SeqDataset(rows, feat, window)
    dl = DataLoader(ds, BATCH)
    preds = []
    truths = []
    m.eval()
    with torch.no_grad():
        for imgs, agrs, lbls in dl:
            imgs = imgs.to(device)
            agrs = agrs.to(device)
            logits, _ = m(imgs, agrs)
            _, pr = logits.max(1)
            preds.extend(pr.cpu().numpy())
            truths.extend(lbls.numpy())
    return accuracy_score(truths, preds), f1_score(truths, preds, average='macro')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='data/dataset_ready.csv')
    p.add_argument('--features', default='data/finetuned_logits.npy')
    p.add_argument('--noisy', default='data/X_visual_val_noisy.npy')
    p.add_argument('--variants', default='dafn_t_d64,dafn_t_no_mrds,dafn_t_no_fam,dafn_t_d32,dafn_t_d128')
    p.add_argument('--seeds', default=str(SEED))
    p.add_argument('--epochs', type=int, default=EPOCHS)
    p.add_argument('--use_sensor', action='store_true')
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base, args.csv) if not os.path.exists(args.csv) else args.csv
    ft_path = os.path.join(base, args.features) if not os.path.exists(args.features) else args.features
    ny_path = os.path.join(base, args.noisy) if not os.path.exists(args.noisy) else args.noisy
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]

    print(f'Device: {device} | Variants: {variants} | Seeds: {seeds}')
    print('Loading data...')
    valid, clean_feat, v_noise, cw = load(csv_path, ft_path, ny_path)

    tr = [(i, r) for i, r in enumerate(valid) if r.get('split') == 'train']
    va = [(i, r) for i, r in enumerate(valid) if r.get('split') == 'val']
    train_rows = [r for _, r in tr]
    val_rows = [r for _, r in va]
    train_feat = {j: clean_feat[i] for j, (i, _) in enumerate(tr)}
    val_feat = {j: clean_feat[i] for j, (i, _) in enumerate(va)}
    noisy_feat = None
    if v_noise is not None:
        noisy_feat = {j: v_noise[j].flatten() for j in range(len(va))}

    if args.dry_run:
        for v in variants:
            w = VARIANT_WINDOW[v]
            m = make_model(v, w, args.use_sensor)
            n_params = sum(p.numel() for p in m.parameters())
            print(f'  {v}: window={w} params={n_params:,}')
        print('Dry run OK')
        return

    results = []
    print(f'\n{"="*80}\n  Architecture Ablation\n{"="*80}')
    for v in variants:
        w = VARIANT_WINDOW[v]
        print(f'\n--- {v} (T={w}) ---')
        clean_accs = []
        noisy_accs = []
        for seed in seeds:
            t0 = time.time()
            maker = lambda: make_model(v, w, args.use_sensor)
            m = train_variant(maker, seed, train_rows, train_feat, w, cw, device, args.epochs)
            c_acc, c_f1 = evaluate_variant(m, val_rows, val_feat, w, device)
            clean_accs.append(c_acc)
            n_acc = n_f1 = float('nan')
            if noisy_feat is not None:
                n_acc, n_f1 = evaluate_variant(m, val_rows, noisy_feat, w, device)
                noisy_accs.append(n_acc)
            print(f'  seed {seed}: clean={c_acc*100:.2f}% noisy={n_acc*100:.2f}% ({time.time()-t0:.0f}s)')
        c_mean = float(np.mean(clean_accs) * 100)
        c_std = float(np.std(clean_accs, ddof=1) * 100) if len(clean_accs) > 1 else 0.0
        n_mean = float(np.mean(noisy_accs) * 100) if noisy_accs else None
        n_std = float(np.std(noisy_accs, ddof=1) * 100) if len(noisy_accs) > 1 else 0.0
        results.append({'variant': v, 'window': w,
                        'clean_acc': round(c_mean, 2), 'clean_std': round(c_std, 2),
                        'noisy_acc': round(n_mean, 2) if n_mean is not None else None,
                        'noisy_std': round(n_std, 2) if noisy_accs else None})
        print(f'  >> {v}: Clean={c_mean:.2f}±{c_std:.2f}% Noisy={n_mean if n_mean is None else round(n_mean,2)}%')

    print(f'\n{"="*60}\n  SUMMARY\n{"="*60}')
    print(f'{"Variant":<18s} {"Clean":>10s} {"Noisy":>10s}')
    for r in results:
        n = f'{r["noisy_acc"]:.2f}±{r["noisy_std"]:.2f}' if r['noisy_acc'] is not None else 'N/A'
        print(f'{r["variant"]:<18s} {r["clean_acc"]:>6.2f}±{r["clean_std"]:.2f} {n:>10s}')

    out_dir = os.path.join(base, 'experiments', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'architecture_ablation.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()

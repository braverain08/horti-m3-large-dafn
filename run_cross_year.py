#!/usr/bin/env python3
"""
Reproduce Table 10 (cross-year generalization).

Trains on one or two growing seasons and evaluates on a held-out season:
  2024->2025, 2025->2024,
  2023+2024->2025, 2023+2025->2024

The 2023 splits need visual features for 2023 images. Pass them with
--features-2023 (dict keyed by the 2023 row index in labeled_agronomic.csv);
otherwise those splits are skipped with a warning.

Usage:
  python experiments/run_cross_year.py --dry_run
  python experiments/run_cross_year.py --epochs 50
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
from models.dafn import DAFN_T, ImageOnlyClassifier, AgronomicOnlyClassifier


AGRO = ['Plant Height','Stem Diameter','Leaf Width','Leaf Length',
        'NDVI','RVI','LNC','LNA','LAI','LDW']
SENS = ['Air_Temperature','Relative_Humidity','Light_Intensity',
        'CO2','Soil_Moisture','Soil_Temperature']
LABEL_MAP = {'Healthy': 0, 'Stress': 1, 'Other': 1}
BATCH = 16
EPOCHS = 50
PATIENCE = 10
SEED = 42
WINDOW = 5

SPLITS = [
    ('2024_to_2025', ['2024'], '2025'),
    ('2025_to_2024', ['2025'], '2024'),
    ('2023_2024_to_2025', ['2023', '2024'], '2025'),
    ('2023_2025_to_2024', ['2023', '2025'], '2024'),
]


class FeatDataset(Dataset):
    def __init__(self, vis, agro, y, need_agro=True):
        self.vis = vis
        self.agro = agro
        self.y = y
        self.need_agro = need_agro

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = torch.tensor(self.vis[i], dtype=torch.float32)
        lbl = torch.tensor(self.y[i], dtype=torch.long)
        if self.need_agro:
            return img, torch.tensor(self.agro[i], dtype=torch.float32), lbl
        return img, lbl


class SeqDataset(Dataset):
    def __init__(self, rows, feat_fn, window=5):
        self.window = window
        self.feat_fn = feat_fn
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
                if label >= 0 and all(self.feat_fn(r) is not None for r, _ in wins):
                    self.samples.append((wins, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wins, label = self.samples[idx]
        T = self.window
        img_seq = torch.zeros(T, 2048)
        agr_seq = torch.zeros(T, len(AGRO))
        for t, (r, _) in enumerate(wins):
            img_seq[t] = torch.tensor(self.feat_fn(r), dtype=torch.float32)
            agr_seq[t] = torch.tensor([float(r.get(f, 0)) for f in AGRO], dtype=torch.float32)
        return img_seq, agr_seq, torch.tensor(label, dtype=torch.long)


def load_sensors(path):
    data = {}
    if not os.path.exists(path):
        return data
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {'Date': r.get('Date', '')}
            for fld in SENS:
                try:
                    row[fld] = float(r.get(fld, 'nan'))
                except (TypeError, ValueError):
                    row[fld] = float('nan')
            data[row['Date']] = row
    return data


def load_rows(labeled_path, sensor_path):
    with open(labeled_path) as f:
        rows = list(csv.DictReader(f))
    sensors = load_sensors(sensor_path)
    for r in rows:
        date = r.get('Date', '')
        if date in sensors:
            for fld in SENS:
                r[fld] = sensors[date].get(fld, float('nan'))
        else:
            for fld in SENS:
                r[fld] = float('nan')
    for fld in AGRO + SENS:
        vals = []
        for r in rows:
            try:
                v = float(r.get(fld, 'nan'))
                if not np.isnan(v):
                    vals.append(v)
            except (TypeError, ValueError):
                pass
        if vals:
            med = float(np.median(vals))
            for r in rows:
                try:
                    if np.isnan(float(r.get(fld, 'nan'))):
                        r[fld] = med
                except (TypeError, ValueError):
                    r[fld] = med
    return [r for r in rows if r.get('label_3class', '') in {'Healthy', 'Stress', 'Other'}]


def build_feat_map(feat_dict, rows):
    m = {}
    for i, r in enumerate(rows):
        key = (r.get('Source', ''), r.get('Date', ''), r.get('Photo Path', ''))
        m[key] = np.array(feat_dict.get(i, np.zeros(2048))).flatten()
    return m


def standardize_agro(rows):
    scaler = StandardScaler()
    for year_rows in rows:
        X = np.array([[float(r.get(f, 'nan')) for f in AGRO] for r in year_rows])
        X = np.nan_to_num(X, nan=0.0)
        Xs = scaler.fit_transform(X)
        for i, r in enumerate(year_rows):
            for j, f in enumerate(AGRO):
                r[f] = Xs[i, j]


def scale_split(rows_tr, rows_te):
    scaler = StandardScaler()
    X_tr = np.array([[float(r.get(f, 'nan')) for f in AGRO] for r in rows_tr])
    X_tr = np.nan_to_num(X_tr, nan=0.0)
    scaler.fit(X_tr)
    for rr in (rows_tr, rows_te):
        X = np.array([[float(r.get(f, 'nan')) for f in AGRO] for r in rr])
        X = np.nan_to_num(X, nan=0.0)
        Xs = scaler.transform(X)
        for i, r in enumerate(rr):
            for j, f in enumerate(AGRO):
                r[f] = Xs[i, j]


def train_model(maker, mtype, vis_tr, agro_tr, y_tr, cw, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    m = maker().to(device)
    need_agro = mtype != 'VisionOnly'
    ds = FeatDataset(vis_tr, agro_tr, y_tr, need_agro)
    dl = DataLoader(ds, min(BATCH, max(1, len(ds) // 4)), shuffle=True)
    crit = nn.CrossEntropyLoss(weight=cw.to(device))
    opt = optim.Adam(m.parameters(), lr=1e-4)
    best = 0.0
    best_state = None
    patience = 0
    for ep in range(1, EPOCHS + 1):
        m.train()
        c = 0
        t = 0
        for batch in dl:
            if mtype == 'VisionOnly':
                imgs, lbls = batch[0].to(device), batch[-1].to(device)
                logits = m(imgs)
            elif mtype == 'AgronomicOnly':
                agrs, lbls = batch[1].to(device), batch[-1].to(device)
                logits = m(agrs)
            else:
                imgs, agrs, lbls = [b.to(device) for b in batch]
                logits = m(imgs, agrs)
            loss = crit(logits, lbls)
            opt.zero_grad()
            loss.backward()
            opt.step()
            _, pr = logits.max(1)
            c += (pr == lbls).sum().item()
            t += lbls.size(0)
        acc = c / t
        if acc > best:
            best = acc
            best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    m.load_state_dict(best_state)
    return m


def evaluate_model(m, mtype, vis, agro, y, device):
    need_agro = mtype != 'VisionOnly'
    ds = FeatDataset(vis, agro, y, need_agro)
    dl = DataLoader(ds, BATCH)
    preds = []
    truths = []
    m.eval()
    with torch.no_grad():
        for batch in dl:
            if mtype == 'VisionOnly':
                imgs, lbls = batch[0].to(device), batch[-1].to(device)
                logits = m(imgs)
            elif mtype == 'AgronomicOnly':
                agrs, lbls = batch[1].to(device), batch[-1].to(device)
                logits = m(agrs)
            else:
                imgs, agrs, lbls = [b.to(device) for b in batch]
                logits = m(imgs, agrs)
            _, pr = logits.max(1)
            preds.extend(pr.cpu().numpy())
            truths.extend(lbls.numpy())
    return accuracy_score(truths, preds), f1_score(truths, preds, average='macro')


def train_dafn(rows_tr, feat_fn, cw, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ds = SeqDataset(rows_tr, feat_fn, WINDOW)
    if len(ds) == 0:
        return None
    m = DAFN_T(window_size=WINDOW, hidden_dim=128, image_dim=2048,
               agronomic_dim=10, sensor_dim=6, num_classes=2, use_sensor=False).to(device)
    dl = DataLoader(ds, min(BATCH, max(1, len(ds) // 4)), shuffle=True)
    crit = nn.CrossEntropyLoss(weight=cw.to(device))
    opt = optim.Adam(m.parameters(), lr=1e-4)
    best = 0.0
    best_state = None
    patience = 0
    for ep in range(1, EPOCHS + 1):
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
        if acc > best:
            best = acc
            best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    m.load_state_dict(best_state)
    return m


def evaluate_dafn(m, rows, feat_fn, device):
    ds = SeqDataset(rows, feat_fn, WINDOW)
    if len(ds) == 0:
        return float('nan'), float('nan')
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


def build_arrays(rows, feat_fn):
    vis = []
    agro = []
    y = []
    for r in rows:
        f = feat_fn(r)
        if f is None:
            continue
        vis.append(f)
        agro.append([float(r.get(x, 0)) for x in AGRO])
        y.append(LABEL_MAP.get(r.get('label_3class', ''), -1))
    if not y:
        return None
    return (np.array(vis, dtype=np.float32),
            np.array(agro, dtype=np.float32),
            np.array(y, dtype=np.int64))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--labeled', default='data/labeled_agronomic.csv')
    p.add_argument('--sensor', default='data/sensor_daily.csv')
    p.add_argument('--ready', default='data/dataset_ready.csv')
    p.add_argument('--features', default='data/finetuned_logits.npy')
    p.add_argument('--features-2023', default='')
    p.add_argument('--splits', default='2024_to_2025,2025_to_2024,2023_2024_to_2025,2023_2025_to_2024')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()

    global EPOCHS
    EPOCHS = args.epochs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def resolve(x):
        return x if os.path.exists(x) else os.path.join(base, x)

    print(f'Device: {device}')
    print('Loading rows...')
    rows = load_rows(resolve(args.labeled), resolve(args.sensor))
    rows_ready = list(csv.DictReader(open(resolve(args.ready))))
    ready_valid = [r for r in rows_ready if r.get('label_3class', '') in {'Healthy', 'Stress', 'Other'}]
    feat_dict = np.load(resolve(args.features), allow_pickle=True).item()
    feat_by_key = build_feat_map(feat_dict, ready_valid)

    feat_2023 = {}
    if args.features_2023:
        d23 = np.load(resolve(args.features_2023), allow_pickle=True).item()
        rows_2023 = [r for r in rows if r.get('Source') == '2023']
        feat_2023 = build_feat_map(d23, rows_2023)

    def feat_fn(r):
        key = (r.get('Source', ''), r.get('Date', ''), r.get('Photo Path', ''))
        if r.get('Source') == '2023':
            return feat_2023.get(key)
        return feat_by_key.get(key)

    by_year = defaultdict(list)
    for r in rows:
        by_year[r.get('Source')].append(r)
    standardize_agro([by_year[y] for y in by_year])

    requested = [s.strip() for s in args.splits.split(',') if s.strip()]
    if args.dry_run:
        for split, train_years, test_year in SPLITS:
            if split not in requested:
                continue
            if '2023' in train_years and not args.features_2023:
                print(f'  {split}: skipped (2023 features not provided)')
            else:
                n_tr = sum(len(by_year[y]) for y in train_years)
                print(f'  {split}: train_years={train_years} test={test_year} train_rows={n_tr} test_rows={len(by_year[test_year])}')
        print('Dry run OK')
        return

    results = []
    print(f'\n{"="*80}\n  Cross-Year Generalization\n{"="*80}')
    for split, train_years, test_year in SPLITS:
        if split not in requested:
            continue
        if '2023' in train_years and not args.features_2023:
            print(f'\n--- {split}: skipped (pass --features-2023 to include 2023) ---')
            continue
        print(f'\n--- {split} (train {train_years} -> test {test_year}) ---')
        rows_tr = [r for y in train_years for r in by_year.get(y, [])]
        rows_te = by_year.get(test_year, [])
        if not rows_tr or not rows_te:
            print('  not enough rows; skipped')
            continue
        scale_split(rows_tr, rows_te)

        y_tr = np.array([LABEL_MAP[r['label_3class']] for r in rows_tr], dtype=np.int64)
        cw = torch.tensor(compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr),
                          dtype=torch.float32)

        tr = build_arrays(rows_tr, feat_fn)
        te = build_arrays(rows_te, feat_fn)
        row = {'split': split, 'train': '+'.join(train_years), 'test': test_year}
        if tr is None or te is None:
            print('  no visual features; only agronomic baseline')
            a_tr = np.array([[float(r.get(x, 0)) for x in AGRO] for r in rows_tr], dtype=np.float32)
            a_te = np.array([[float(r.get(x, 0)) for x in AGRO] for r in rows_te], dtype=np.float32)
            m = train_model(lambda: AgronomicOnlyClassifier(input_dim=10, num_classes=2), 'AgronomicOnly',
                            np.zeros((len(a_tr), 2048)), a_tr, y_tr, cw, device)
            acc, f1 = evaluate_model(m, 'AgronomicOnly', np.zeros((len(a_te), 2048)), a_te,
                                     np.array([LABEL_MAP[r['label_3class']] for r in rows_te], dtype=np.int64), device)
            row['agronomic_acc'] = round(float(acc) * 100, 2)
            row['vision_acc'] = None
            row['dafn_t_acc'] = None
            print(f'  Agronomic-only: {acc*100:.2f}%')
        else:
            vis_tr, agro_tr, y_tr2 = tr
            vis_te, agro_te, y_te = te
            m_vis = train_model(lambda: ImageOnlyClassifier(2048, 2), 'VisionOnly',
                                vis_tr, agro_tr, y_tr2, cw, device)
            v_acc, _ = evaluate_model(m_vis, 'VisionOnly', vis_te, agro_te, y_te, device)
            m_agro = train_model(lambda: AgronomicOnlyClassifier(input_dim=10, num_classes=2), 'AgronomicOnly',
                                 vis_tr, agro_tr, y_tr2, cw, device)
            a_acc, _ = evaluate_model(m_agro, 'AgronomicOnly', vis_te, agro_te, y_te, device)
            m_dafn = train_dafn(rows_tr, feat_fn, cw, device)
            d_acc = d_f1 = float('nan')
            if m_dafn is not None:
                d_acc, d_f1 = evaluate_dafn(m_dafn, rows_te, feat_fn, device)
            row['vision_acc'] = round(float(v_acc) * 100, 2)
            row['agronomic_acc'] = round(float(a_acc) * 100, 2)
            row['dafn_t_acc'] = round(float(d_acc) * 100, 2) if not np.isnan(d_acc) else None
            print(f'  Vision-only: {v_acc*100:.2f}% | Agronomic-only: {a_acc*100:.2f}% | DAFN-T: {d_acc*100:.2f}%')
        results.append(row)

    print(f'\n{"="*60}\n  SUMMARY\n{"="*60}')
    print(f'{"Split":<22s} {"Vision":>8s} {"Agro":>8s} {"DAFN-T":>8s}')
    for r in results:
        print(f'{r["split"]:<22s} {str(r.get("vision_acc")):>8s} {str(r.get("agronomic_acc")):>8s} {str(r.get("dafn_t_acc")):>8s}')

    out_dir = os.path.join(base, 'experiments', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'cross_year.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()

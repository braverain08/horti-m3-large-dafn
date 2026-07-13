#!/usr/bin/env python3
"""
Ablation study: robustness to agronomic data missingness.

Simulates scenarios where agronomic measurements are partially or fully
unavailable — a common real-world situation when sensors fail or samples
are missed during data collection. Compares DAFN/DAFN-T (multimodal)
against agronomic-only baselines under each scenario.

Usage:
    # Quick test (no image features required — runs agronomic-only models)
    python experiments/run_ablation.py --dry_run

    # Full ablation (requires image_features.npy)
    python experiments/run_ablation.py

    # Focus on specific missingness levels
    python experiments/run_ablation.py --missing_rates 0,0.3,0.5,0.7,0.9,1.0
"""
import os, sys, csv, json, argparse
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.dafn import DAFN, DAFN_T, ImageOnlyClassifier, AgronomicOnlyClassifier

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'experiments', 'results')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AGRONOMIC_FEATURES = ['Plant Height', 'Stem Diameter', 'Leaf Width', 'Leaf Length',
                       'NDVI', 'RVI', 'LNC', 'LNA', 'LAI', 'LDW']
SENSOR_FEATURES = ['Air_Temperature', 'Relative_Humidity', 'Light_Intensity',
                    'CO2', 'Soil_Moisture', 'Soil_Temperature']
ALL_FEATURES = AGRONOMIC_FEATURES + SENSOR_FEATURES
LABEL_MAP = {'Healthy': 0, 'Stress': 1, 'Other': 2}
NUM_CLASSES = 3; HIDDEN_DIM = 64; LR = 1e-4; EPOCHS = 40; BATCH_SIZE = 16
class_weights = None


def get_class_weights(y):
    cw = compute_class_weight('balanced', classes=np.unique(y), y=y)
    return torch.tensor(cw, dtype=torch.float32).to(DEVICE)


def load_data(csv_path, feature_npy_path=None, use_sensor=False):
    with open(csv_path) as f: all_rows = list(csv.DictReader(f))
    all_rows = [r for r in all_rows if r.get('Source') != '2023']
    valid = [r for r in all_rows if r.get('label_3class','Unknown') in LABEL_MAP]
    feat = {}
    if feature_npy_path and os.path.exists(feature_npy_path):
        d = np.load(feature_npy_path, allow_pickle=True).item()
        feat = {int(k): v for k, v in d.items()}
        print(f'  Loaded image features: {len(feat)} samples')
    else:
        print(f'  No image features found at {feature_npy_path}')
        print(f'  DAFN/DAFN-T image-based models will be skipped.')
    train = [r for i,r in enumerate(valid) if r.get('split')=='train']
    val = [r for i,r in enumerate(valid) if r.get('split')=='val']
    for label, rows in [('train', train), ('val', val)]:
        X = np.array([[float(r.get(f,'nan')) for f in ALL_FEATURES] for r in rows])
        X = np.nan_to_num(X, nan=0.0)
        scaler = StandardScaler(); scaler.fit(X); X_s = scaler.transform(X)
        for i, r in enumerate(rows):
            for j, f in enumerate(ALL_FEATURES): r[f] = X_s[i,j]
    yt = np.array([LABEL_MAP.get(r['label_3class'], -1) for r in train])
    global class_weights; class_weights = get_class_weights(yt)
    print(f'  Loaded {len(valid)}: train={len(train)} val={len(val)}')
    return train, val, feat


def apply_missingness(rows, features, missing_rate, rng):
    """Randomly zero out `missing_rate` fraction of each row's features."""
    if missing_rate <= 0: return
    n_feat = len(features); n_mask = max(1, int(n_feat * missing_rate))
    for r in rows:
        idxs = rng.choice(n_feat, size=n_mask, replace=False)
        for i in idxs: r[features[i]] = 0.0


class AblationDataset(Dataset):
    def __init__(self, rows, feature_dict=None, use_sensor=False):
        self.rows = rows; self.features = feature_dict or {}; self.use_sensor = use_sensor
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        agr = torch.tensor([float(r.get(f,0)) for f in AGRONOMIC_FEATURES], dtype=torch.float32)
        img = torch.tensor(self.features.get(idx, np.zeros(2048,dtype=np.float32)), dtype=torch.float32)
        lbl = torch.tensor(LABEL_MAP.get(r.get('label_3class','Unknown'), -1), dtype=torch.long)
        if self.use_sensor:
            sen = torch.tensor([float(r.get(f,0)) for f in SENSOR_FEATURES], dtype=torch.float32)
            return img, agr, sen, lbl
        return img, agr, lbl


def run_torch(model_class, train_ds, val_ds, name, **kw):
    global class_weights
    tl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    vl = DataLoader(val_ds, BATCH_SIZE)
    m = model_class(**kw).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=class_weights)
    opt = optim.Adam(m.parameters(), LR)
    params = sum(p.numel() for p in m.parameters())
    best_acc = 0.0; best_state = None
    for ep in range(EPOCHS):
        m.train(); tl_loss=0.0; c=0; t=0
        for batch in tl:
            lbl = batch[-1].to(DEVICE); n=len(batch)
            if n==4: logits,_ = m(batch[0].to(DEVICE),batch[1].to(DEVICE),batch[2].to(DEVICE))
            elif n==3:
                r = m(batch[0].to(DEVICE),batch[1].to(DEVICE))
                logits = r[0] if isinstance(r,tuple) else r
            else: logits = m(batch[0].to(DEVICE))
            loss = crit(logits,lbl); opt.zero_grad(); loss.backward(); opt.step()
            _,pr=logits.max(1); c+=(pr==lbl).sum().item(); t+=lbl.size(0); tl_loss+=loss.item()
        m.eval(); ap=[]; al=[]
        with torch.no_grad():
            for batch in vl:
                lbl = batch[-1].to(DEVICE); n=len(batch)
                if n==4: logits,_ = m(batch[0].to(DEVICE),batch[1].to(DEVICE),batch[2].to(DEVICE))
                elif n==3:
                    r = m(batch[0].to(DEVICE),batch[1].to(DEVICE))
                    logits = r[0] if isinstance(r,tuple) else r
                else: logits = m(batch[0].to(DEVICE))
                _,pr=logits.max(1); ap.extend(pr.cpu().numpy()); al.extend(lbl.cpu().numpy())
        vacc = accuracy_score(al,ap); vf1 = f1_score(al,ap,average='macro')
        if vacc > best_acc:
            best_acc = vacc; best_state = {k:v.cpu().clone() for k,v in m.state_dict().items()}
        if (ep+1)%20==0: print(f'      [{name}] E{ep+1}/{EPOCHS} val_acc={vacc:.4f}')
    m.load_state_dict(best_state)
    m.eval(); ap=[]; al=[]
    with torch.no_grad():
        for batch in vl:
            lbl = batch[-1].to(DEVICE); n=len(batch)
            if n==4: logits,_ = m(batch[0].to(DEVICE),batch[1].to(DEVICE),batch[2].to(DEVICE))
            elif n==3:
                r = m(batch[0].to(DEVICE),batch[1].to(DEVICE))
                logits = r[0] if isinstance(r,tuple) else r
            else: logits = m(batch[0].to(DEVICE))
            _,pr=logits.max(1); ap.extend(pr.cpu().numpy()); al.extend(lbl.cpu().numpy())
    res = {'model':name,'accuracy':accuracy_score(al,ap),'macro_f1':f1_score(al,ap,average='macro'),'params':params}
    pc=precision_recall_fscore_support(al,ap,labels=[0,1,2],zero_division=0)
    res['recall']=pc[1].tolist(); res['f1_per_class']=pc[2].tolist()
    return res


def run_scenario(train, val, feat, missing_rate, use_sensor, has_features):
    rng = np.random.RandomState(SEED + int(missing_rate*100))
    label = f'{missing_rate*100:.0f}% missing'
    print(f'\n  --> Scenario: {label}')
    apply_missingness(train, AGRONOMIC_FEATURES, missing_rate, rng)
    apply_missingness(val, AGRONOMIC_FEATURES, missing_rate, rng)
    results = []
    td = AblationDataset(train, feat, use_sensor)
    vd = AblationDataset(val, feat, use_sensor)

    # Agronomic-only baseline
    print('    [Agronomic-only] ...', end=' ', flush=True)
    agr_ds1 = torch.utils.data.TensorDataset(
        torch.stack([td[i][1] for i in range(len(td))]),
        torch.tensor([td[i][-1] for i in range(len(td))]))
    agr_ds2 = torch.utils.data.TensorDataset(
        torch.stack([vd[i][1] for i in range(len(vd))]),
        torch.tensor([vd[i][-1] for i in range(len(vd))]))
    res = run_torch(AgronomicOnlyClassifier, agr_ds1, agr_ds2,
                    f'Agri-only (miss={missing_rate:.2f})',
                    input_dim=10, num_classes=NUM_CLASSES)
    results.append(res); print(f'Acc={res["accuracy"]:.4f}')

    # SVM & RF
    Xtr = np.array([[float(r.get(f,0)) for f in AGRONOMIC_FEATURES] for r in train])
    ytr = np.array([LABEL_MAP.get(r['label_3class'],-1) for r in train])
    Xv = np.array([[float(r.get(f,0)) for f in AGRONOMIC_FEATURES] for r in val])
    yv = np.array([LABEL_MAP.get(r['label_3class'],-1) for r in val])
    for clf_name, clf_fn in [('SVM', lambda: SVC(kernel='rbf',gamma='scale',C=1.0)),
                              ('RF', lambda: RandomForestClassifier(n_estimators=100,random_state=SEED))]:
        m = clf_fn(); m.fit(Xtr,ytr); p = m.predict(Xv)
        acc = accuracy_score(yv,p); f1 = f1_score(yv,p,average='macro')
        pc = precision_recall_fscore_support(yv,p,labels=[0,1,2],zero_division=0)
        results.append({'model':f'{clf_name} (miss={missing_rate:.2f})','accuracy':acc,'macro_f1':f1,
                        'recall':pc[1].tolist(),'f1_per_class':pc[2].tolist(),'params':0})
        print(f'    [{clf_name}] ... Acc={acc:.4f}')

    # DAFN (requires image features)
    if has_features:
        print('    [DAFN] ...', end=' ', flush=True)
        res = run_torch(DAFN, td, vd, f'DAFN (miss={missing_rate:.2f})',
                        image_dim=2048, agronomic_dim=10, sensor_dim=6,
                        hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES, use_sensor=use_sensor)
        results.append(res); print(f'Acc={res["accuracy"]:.4f}')

        # DAFN-T
        print('    [DAFN-T] ...', end=' ', flush=True)
        from collections import defaultdict
        pg = defaultdict(list)
        for i,r in enumerate(train): pg[r['Number']].append((r,i))
        for p in pg: pg[p].sort(key=lambda x: x[0]['Date'])
        sq_t_samples = []
        for plant, items in pg.items():
            for i in range(0, len(items)-5+1, 1):
                wins = items[i:i+5]; lbl = LABEL_MAP.get(wins[-1][0].get('label_3class','Unknown'), -1)
                if lbl >= 0: sq_t_samples.append((wins,lbl))
        pg = defaultdict(list)
        for i,r in enumerate(val): pg[r['Number']].append((r,i))
        for p in pg: pg[p].sort(key=lambda x: x[0]['Date'])
        sq_v_samples = []
        for plant, items in pg.items():
            for i in range(0, len(items)-5+1, 1):
                wins = items[i:i+5]; lbl = LABEL_MAP.get(wins[-1][0].get('label_3class','Unknown'), -1)
                if lbl >= 0: sq_v_samples.append((wins,lbl))
        if sq_t_samples and sq_v_samples:
            class SeqDs(Dataset):
                def __init__(self, samples, feat_dict, use_sensor):
                    self.samples=samples; self.features=feat_dict or {}; self.use_sensor=use_sensor
                def __len__(self): return len(self.samples)
                def __getitem__(self, idx):
                    wins,lbl=self.samples[idx]; T=5
                    img_seq=torch.zeros(T,2048); agr_seq=torch.zeros(T,10)
                    for t,(r,oi) in enumerate(wins):
                        img_seq[t]=torch.tensor(self.features.get(oi,np.zeros(2048)),dtype=torch.float32)
                        agr_seq[t]=torch.tensor([float(r.get(f,0)) for f in AGRONOMIC_FEATURES],dtype=torch.float32)
                    if self.use_sensor:
                        sen_seq=torch.zeros(T,6)
                        for t,(r,_) in enumerate(wins):
                            sen_seq[t]=torch.tensor([float(r.get(f,0)) for f in SENSOR_FEATURES],dtype=torch.float32)
                        return img_seq,agr_seq,sen_seq,torch.tensor(lbl,dtype=torch.long)
                    return img_seq,agr_seq,torch.tensor(lbl,dtype=torch.long)
            sq_t = SeqDs(sq_t_samples, feat, use_sensor)
            sq_v = SeqDs(sq_v_samples, feat, use_sensor)
            res = run_torch(DAFN_T, sq_t, sq_v, f'DAFN-T (miss={missing_rate:.2f})',
                            window_size=5, image_dim=2048, agronomic_dim=10,
                            sensor_dim=6, hidden_dim=HIDDEN_DIM,
                            num_classes=NUM_CLASSES, use_sensor=use_sensor)
            results.append(res); print(f'Acc={res["accuracy"]:.4f}')
        else:
            print('    [DAFN-T] skipped (no sequences)')
    else:
        print('    [DAFN / DAFN-T] SKIPPED (needs image_features.npy)')

    return results


SCENARIO_LABELS = {0.0: '0% (baseline)', 0.3: '30% missing', 0.5: '50% missing',
                    0.7: '70% missing', 0.9: '90% missing', 1.0: '100% missing'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'experiments', 'results'))
    p.add_argument('--missing_rates', default='0,0.3,0.5,0.7,0.9,1.0')
    p.add_argument('--csv', default=os.path.join(DATA_DIR, 'dataset_ready.csv'))
    p.add_argument('--features', default=os.path.join(DATA_DIR, 'image_features.npy'))
    p.add_argument('--use_sensor', action='store_true')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()
    global EPOCHS; EPOCHS = args.epochs
    rates = [float(x) for x in args.missing_rates.split(',')]
    global RESULTS_DIR; RESULTS_DIR = args.results_dir
    if not args.dry_run:
        os.makedirs(RESULTS_DIR, exist_ok=True)
    has_features = os.path.exists(args.features)

    print('='*90)
    print('  Ablation Study: Robustness to Agronomic Data Missingness')
    print(f'  Device: {DEVICE} | Epochs: {EPOCHS} | Sensor: {args.use_sensor}')
    print(f'  Missing rates: {rates}')
    print(f'  Image features: {"AVAILABLE" if has_features else "NOT FOUND"}')
    print('='*90)

    if args.dry_run:
        print('\n  [DRY RUN] Would execute:')
        for r in rates:
            models = '+'.join(['Agri-only','SVM','RF'] + (['DAFN','DAFN-T'] if has_features else []))
            print(f'    missing={r:.0%}: {models}')
        print(); return

    train, val, feat = load_data(args.csv, args.features, args.use_sensor)
    all_results = []
    for rate in rates:
        all_results.extend(run_scenario(train, val, feat, rate, args.use_sensor, has_features))

    # Results table
    print('\n' + '='*90)
    print('  ABLATION RESULTS: Accuracy Under Agronomic Data Missingness')
    print('='*90)
    print(f'  {"Model":<28s} {"Missing":<10s} {"Acc":>7s} {"F1":>7s} {"Params":>8s}')
    print('  ' + '-'*62)
    for r in all_results:
        mn = r['model']; miss = ''
        if '(miss=' in mn:
            base = mn.split('(miss=')[0].strip()
            miss = f'{float(mn.split("(miss=")[1].rstrip(")")):.0%}'
        else: base = mn
        print(f'  {base:<28s} {miss:<10s} {r["accuracy"]:>7.3f} {r.get("macro_f1",0):>7.3f} {r.get("params",0):>8,d}')
    print('='*90)

    # Accuracy drop analysis
    print('\n  Accuracy Drop (relative to 0% missing):')
    print('  ' + '-'*60)
    for base in ['Agri-only','SVM','RF','DAFN','DAFN-T']:
        group = [r for r in all_results if r['model'].startswith(base)]
        if not group: continue
        full = max(r['accuracy'] for r in group if '(miss=0' in r['model'])
        print(f'  {base:<25s}', end='')
        for r in group:
            if '(miss=' in r['model']:
                m = float(r['model'].split('(miss=')[1].rstrip(')'))
                print(f'  {m:.0%}: d={full-r["accuracy"]:+.4f}', end='')
        print()
    print('='*90)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(RESULTS_DIR, f'ablation_{ts}.json')
    with open(out, 'w') as f: json.dump(all_results, f, indent=2)
    print(f'\nSaved to {out}')
    print('Done.')

if __name__ == '__main__': main()

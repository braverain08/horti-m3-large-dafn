#!/usr/bin/env python3
"""Error analysis for DAFN-T on validation set."""
import os, sys, csv, re, json
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.checkpoint_loader import load_dafn_t

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BS = 64
AGRO = ['Plant Height','Stem Diameter','Leaf Width','Leaf Length',
        'NDVI','RVI','LNC','LNA','LAI','LDW']
LABEL_MAP_BIN = {'Healthy':0,'Stress':1,'Other':1}

class SeqDataset(Dataset):
    def __init__(self, rows, feat_dict, window=5, stride=1):
        self.window = window; self.feat = feat_dict or {}
        pg = defaultdict(list)
        for oi, r in rows:
            pp = r.get('Photo Path','').split(';')[0].strip()
            pid_match = re.match(r'images/(\d+)', pp) if pp else None
            pid = int(pid_match.group(1)) if pid_match else -1
            pg[pid].append((r, oi))
        self.samples = []
        for pid, items in pg.items():
            items.sort(key=lambda x: x[0].get('Date',''))
            for i in range(0, len(items) - window + 1, stride):
                wins = items[i:i+window]
                label = LABEL_MAP_BIN.get(wins[-1][0].get('label_3class','Unknown'), -1)
                if label >= 0:
                    orig_label = wins[-1][0].get('label_3class','')
                    self.samples.append((wins, label, orig_label))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        wins, label, orig = self.samples[idx]; T = self.window
        img_seq = torch.zeros(T, 2048); agr_seq = torch.zeros(T, len(AGRO))
        for t, (r, oi) in enumerate(wins):
            img_seq[t] = torch.tensor(self.feat.get(oi, np.zeros(2048)), dtype=torch.float32)
            agr_seq[t] = torch.tensor([float(r.get(f,0)) for f in AGRO], dtype=torch.float32)
        return img_seq, agr_seq, torch.tensor(label, dtype=torch.long), orig

def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load data
    print("Loading validation set...")
    with open(os.path.join(base, 'data', 'dataset_ready.csv')) as f:
        rows = [r for r in csv.DictReader(f) if r.get('Source')!='2023'
                and r.get('label_3class','') in {'Healthy','Stress','Other'}]
    val = [(i, r) for i, r in enumerate(rows) if r.get('split')=='val']
    
    print("Loading features...")
    d = np.load(os.path.join(base, 'data', 'finetuned_logits.npy'), allow_pickle=True).item()
    valid_all = [r for r in csv.DictReader(open(os.path.join(base, 'data', 'dataset_ready.csv')))
                 if r.get('Source')!='2023' and r.get('label_3class','') in {'Healthy','Stress','Other'}]
    feat_dict = {i: np.array(d.get(i, np.zeros(2048))).flatten() for i in range(len(valid_all))}
    
    ds = SeqDataset(val, feat_dict, window=5)
    dl = DataLoader(ds, min(BS, len(ds)//2))
    print(f"Val windows: {len(ds)}")
    
    print("Loading model...")
    m = load_dafn_t(os.path.join(base, 'data', 'dafn_t_dim128_best.pth'))
    m.to(DEVICE)
    m.eval()
    
    # Inference
    preds, truths, confs, orig_labels = [], [], [], []
    for batch in tqdm(dl, desc='Inference'):
        imgs, agrs, lbls, origs = batch
        imgs, agrs, lbls = imgs.to(DEVICE), agrs.to(DEVICE), lbls.to(DEVICE)
        with torch.no_grad():
            logits, _ = m(imgs, agrs)
        probs = torch.softmax(logits, dim=1)
        conf, pr = probs.max(1)
        preds.extend(pr.cpu().numpy())
        truths.extend(lbls.cpu().numpy())
        confs.extend(conf.cpu().numpy())
        orig_labels.extend(origs)
    
    acc = np.mean(np.array(preds) == np.array(truths)) * 100
    cm = confusion_matrix(truths, preds)
    
    print(f"\nOverall accuracy: {acc:.2f}%")
    print(f"Error rate: {100-acc:.2f}%")
    print(f"\nConfusion matrix:")
    print(f"              Pred H    Pred S")
    print(f"  Actual H    {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"  Actual S    {cm[1,0]:5d}    {cm[1,1]:5d}")
    print(f"\nClassification report:")
    print(classification_report(truths, preds, target_names=['Healthy','Stress']))
    
    # Error analysis
    errors = [(t, p, c, o) for t, p, c, o in zip(truths, preds, confs, orig_labels) if t != p]
    n_healthy_as_stress = sum(1 for t, p, _, _ in errors if t == 0 and p == 1)  # FN for stress
    n_stress_as_healthy = sum(1 for t, p, _, _ in errors if t == 1 and p == 0)  # FP for stress
    
    # Estimate error types based on confidence
    # Low confidence errors (probs close to 0.5) → early stress
    # High confidence errors → occlusion/labeling
    low_conf = sum(1 for _, _, c, _ in errors if c < 0.7)
    high_conf = len(errors) - low_conf
    
    # Early stress: model predicted Healthy but ground truth is Stress with low confidence
    early_stress = sum(1 for t, p, c, o in errors if t == 1 and p == 0 and c < 0.7)
    occlusion = sum(1 for t, p, c, _ in errors if c >= 0.7)
    other = low_conf - early_stress + (len(errors) - low_conf - occlusion)
    
    print(f"\n=== Error Analysis ===")
    print(f"Total errors: {len(errors)}")
    print(f"  H→S (FP): {n_healthy_as_stress} ({n_healthy_as_stress/max(len(errors),1)*100:.0f}%)")
    print(f"  S→H (FN): {n_stress_as_healthy} ({n_stress_as_healthy/max(len(errors),1)*100:.0f}%)")
    print(f"\nEstimated error categories:")
    print(f"  Early stress (subtle symptoms): {early_stress} ({early_stress/max(len(errors),1)*100:.0f}%)")
    print(f"  Occlusion/overlap:              {occlusion} ({occlusion/max(len(errors),1)*100:.0f}%)")
    print(f"  Other:                           {other} ({other/max(len(errors),1)*100:.0f}%)")
    
    results = {
        'overall_accuracy': f"{acc:.2f}%",
        'error_rate': f"{100-acc:.2f}%",
        'total_errors': len(errors),
        'confusion_matrix': cm.tolist(),
        'error_categories': {
            'early_stress_pct': round(early_stress/max(len(errors),1)*100),
            'occlusion_pct': round(occlusion/max(len(errors),1)*100),
            'other_pct': round(other/max(len(errors),1)*100),
        }
    }
    out_path = os.path.join(base, 'experiments', 'results', 'error_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""MRDS weight analysis on real degraded vs clean validation images."""
import os, sys, csv, re, json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.checkpoint_loader import load_dafn_t

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BS = 64
AGRO = ['Plant Height','Stem Diameter','Leaf Width','Leaf Length',
        'NDVI','RVI','LNC','LNA','LAI','LDW']
LABEL_MAP_BIN = {'Healthy':0,'Stress':1,'Other':1}

class SeqDataset(Dataset):
    """Temporal sequence dataset. Marks windows as degraded if last image in window matches degraded set."""
    def __init__(self, rows, feat_dict, window=5, degraded_files=None, stride=1):
        self.window = window
        self.feat = feat_dict or {}
        self.degraded = set()
        if degraded_files:
            for fn in degraded_files:
                with open(fn) as f:
                    for row in csv.DictReader(f):
                        self.degraded.add(row.get('photo_path','').split(';')[0].strip())
        
        pg = defaultdict(list)
        for oi, r in rows:
            pp = r.get('Photo Path','').split(';')[0].strip()
            pid_match = re.match(r'images/(\d+)', pp) if pp else None
            pid = int(pid_match.group(1)) if pid_match else -1
            pg[pid].append((r, oi, pp))
        
        self.samples = []
        for pid, items in pg.items():
            items.sort(key=lambda x: x[0].get('Date',''))
            for i in range(0, len(items) - window + 1, stride):
                wins = items[i:i+window]
                label = LABEL_MAP_BIN.get(wins[-1][0].get('label_3class','Unknown'), -1)
                if label >= 0:
                    # Check if last image in window is degraded
                    last_pp = wins[-1][2]
                    is_degraded = last_pp in self.degraded
                    self.samples.append((wins, label, is_degraded))
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        wins, label, is_degraded = self.samples[idx]
        T = self.window
        img_seq = torch.zeros(T, 2048); agr_seq = torch.zeros(T, len(AGRO))
        for t, (r, oi, _) in enumerate(wins):
            img_seq[t] = torch.tensor(self.feat.get(oi, np.zeros(2048)), dtype=torch.float32)
            agr_seq[t] = torch.tensor([float(r.get(f,0)) for f in AGRO], dtype=torch.float32)
        return img_seq, agr_seq, torch.tensor(label, dtype=torch.long), torch.tensor(is_degraded, dtype=torch.bool)

def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load CSV
    print("Loading validation set...")
    with open(os.path.join(base, 'data', 'dataset_ready.csv')) as f:
        rows = [r for r in csv.DictReader(f) if r.get('Source')!='2023'
                and r.get('label_3class','') in {'Healthy','Stress','Other'}]
    val = [(i, r) for i, r in enumerate(rows) if r.get('split')=='val']
    print(f"  Val samples: {len(val)}")
    
    # Load features
    print("Loading finetuned features...")
    d = np.load(os.path.join(base, 'data', 'finetuned_logits.npy'), allow_pickle=True).item()
    valid_all = [r for r in csv.DictReader(open(os.path.join(base, 'data', 'dataset_ready.csv')))
                 if r.get('Source')!='2023' and r.get('label_3class','') in {'Healthy','Stress','Other'}]
    feat_dict = {i: np.array(d.get(i, np.zeros(2048))).flatten() for i in range(len(valid_all))}
    print(f"  Features: {len(feat_dict)}")
    
    # Degraded subsets
    subdir = os.path.join(base, 'experiments', 'results', 'degraded_subsets')
    degraded_files = [os.path.join(subdir, f) for f in os.listdir(subdir) 
                      if f.startswith('degraded_') and f.endswith('.csv')]
    
    # Dataset
    ds = SeqDataset(val, feat_dict, window=5, degraded_files=degraded_files)
    dl = DataLoader(ds, min(BS, len(ds)//2))
    
    # Load model
    print("Loading DAFN-T model...")
    m = load_dafn_t(os.path.join(base, 'data', 'dafn_t_dim128_best.pth'))
    m.to(DEVICE)
    m.eval()
    
    # Inference
    clean_weights, degraded_weights = [], []
    correct, total = 0, 0
    for batch in tqdm(dl, desc='Inference'):
        imgs, agrs, lbls, is_deg = [b.to(DEVICE) for b in batch]
        with torch.no_grad():
            logits, w_list = m(imgs, agrs)
        # w_list: list of T tensors [B, num_modalities]; avg over timesteps
        w_avg = torch.stack(w_list).mean(dim=0)  # [B, num_modalities]
        
        _, preds = logits.max(1)
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)
        
        for i in range(len(lbls)):
            w = w_avg[i].cpu().numpy()
            if is_deg[i]:
                degraded_weights.append(w)
            else:
                clean_weights.append(w)
    
    # Results
    acc = correct / total * 100
    print(f"\nOverall accuracy: {acc:.2f}%")
    
    if clean_weights:
        cw = np.mean(clean_weights, axis=0)
        print(f"  Clean windows: {len(clean_weights)}")
        print(f"    vis={cw[0]:.3f}, agr={cw[1]:.3f}" + (f", sen={cw[2]:.3f}" if len(cw)>2 else ""))
    
    if degraded_weights:
        dw = np.mean(degraded_weights, axis=0)
        print(f"  Degraded windows: {len(degraded_weights)}")
        print(f"    vis={dw[0]:.3f}, agr={dw[1]:.3f}" + (f", sen={dw[2]:.3f}" if len(dw)>2 else ""))
        print(f"  Δvis={dw[0]-cw[0]:+.3f}, Δagr={dw[1]-cw[1]:+.3f}")
    
    # Save JSON
    results = {
        'n_clean': len(clean_weights), 'n_degraded': len(degraded_weights),
        'accuracy_pct': acc,
        'clean_vis': float(np.mean([w[0] for w in clean_weights])) if clean_weights else 0,
        'clean_agr': float(np.mean([w[1] for w in clean_weights])) if clean_weights else 0,
        'degraded_vis': float(np.mean([w[0] for w in degraded_weights])) if degraded_weights else 0,
        'degraded_agr': float(np.mean([w[1] for w in degraded_weights])) if degraded_weights else 0,
    }
    out_path = os.path.join(base, 'experiments', 'results', 'mrds_real_weights.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()

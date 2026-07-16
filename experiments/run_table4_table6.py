#!/usr/bin/env python3
"""
Focused repeat experiment for Table 4 (Vision-only) and Table 6 (DAFN-T noisy).
3 seeds each: [42, 123, 456]
"""
import os, sys, csv, argparse, time, re
from collections import defaultdict, Counter
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dafn import ImageOnlyClassifier, DAFN_T

AGRO = ['Plant Height','Stem Diameter','Leaf Width','Leaf Length','NDVI','RVI','LNC','LNA','LAI','LDW']
SENS = ['Air_Temperature','Relative_Humidity','Light_Intensity','CO2','Soil_Moisture','Soil_Temperature']
LABEL_MAP = {'Healthy':0,'Stress':1,'Other':1}
SEEDS = [42,123,456]; BS=16; EPOCHS=50; PATIENCE=10
RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','table46')
os.makedirs(RES_DIR, exist_ok=True)

# ── Datasets ──
class FeatDataset(Dataset):
    def __init__(self, vis, agro, y, need_agro=True):
        self.vis=vis; self.agro=agro; self.y=y; self.need_agro=need_agro
    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        img=torch.tensor(self.vis[i],dtype=torch.float32)
        lbl=torch.tensor(self.y[i],dtype=torch.long)
        if self.need_agro: return img,torch.tensor(self.agro[i],dtype=torch.float32),lbl
        return img,lbl

class SeqDataset(Dataset):
    """Sliding-window temporal sequences. If noisy_feat is given, val features are overridden."""
    def __init__(self, rows, clean_feat, window=5, noisy_feat=None, val_rows=None):
        self.window=window
        # Build clean feature dict keyed by valid index
        self.feat = {i: np.array(v).flatten() for i, v in clean_feat.items()}
        # Override val rows with noisy features if provided
        if noisy_feat is not None and val_rows is not None:
            # Map val rows (in valid list order) → noisy features (in val set order)
            valid_with_idx = [(i, r) for i, r in enumerate(rows)]
            val_idx_in_valid = [i for i, r in valid_with_idx if r.get('split') == 'val']
            for j, vidx in enumerate(val_idx_in_valid):
                if j < len(noisy_feat):
                    self.feat[vidx] = noisy_feat[j].flatten()
        # Group by plant ID
        pg = defaultdict(list)
        for i, r in enumerate(rows):
            pp = r.get('Photo Path','')
            pid=-1
            if pp:
                m=re.match(r'images/(\d+)',pp.split(';')[0])
                if m: pid=int(m.group(1))
            pg[pid].append((r,i))
        self.samples=[]
        for pid,items in pg.items():
            items.sort(key=lambda x:x[0].get('Date',''))
            for i in range(len(items)-window+1):
                wins=items[i:i+window]
                label=LABEL_MAP.get(wins[-1][0].get('label_3class','Unknown'),-1)
                if label>=0: self.samples.append((wins,label))
    def __len__(self): return len(self.samples)
    def __getitem__(self,idx):
        wins,label=self.samples[idx]; T=self.window
        img_seq=torch.zeros(T,2048); agr_seq=torch.zeros(T,len(AGRO))
        for t,(r,oi) in enumerate(wins):
            img_seq[t]=torch.tensor(self.feat.get(oi,np.zeros(2048)),dtype=torch.float32)
            agr_seq[t]=torch.tensor([float(r.get(f,0)) for f in AGRO],dtype=torch.float32)
        return img_seq,agr_seq,torch.tensor(label,dtype=torch.long)

# ── Load data ──
def load(csv_path, ft_path, noisy_path):
    print("Loading...")
    with open(csv_path) as f: rows=list(csv.DictReader(f))
    rows=[r for r in rows if r.get('Source')!='2023']
    valid=[r for r in rows if r.get('label_3class','') in {'Healthy','Stress','Other'}]
    valid_with_idx=[(i,r) for i,r in enumerate(valid)]
    tr=[(i,r) for i,r in valid_with_idx if r.get('split')=='train']
    va=[(i,r) for i,r in valid_with_idx if r.get('split')=='val']
    train_rows=[r for _,r in tr]; val_rows=[r for _,r in va]

    scaler=StandardScaler()
    for rr in [train_rows,val_rows]:
        X=np.array([[float(r.get(f,'nan')) for f in AGRO] for r in rr]); X=np.nan_to_num(X,nan=0.0)
        Xs=scaler.fit_transform(X)
        for i,r in enumerate(rr):
            for j,f in enumerate(AGRO): r[f]=Xs[i,j]

    a_tr=np.array([[float(r.get(f,0)) for f in AGRO] for r in train_rows],dtype=np.float32)
    a_va=np.array([[float(r.get(f,0)) for f in AGRO] for r in val_rows],dtype=np.float32)
    y_tr=np.array([LABEL_MAP[r['label_3class']] for r in train_rows],dtype=np.int64)
    y_va=np.array([LABEL_MAP[r['label_3class']] for r in val_rows],dtype=np.int64)
    cw=torch.tensor(compute_class_weight('balanced',classes=np.unique(y_tr),y=y_tr),dtype=torch.float32)
    # Clean features
    d=np.load(ft_path,allow_pickle=True).item()
    v_tr=np.array([np.array(d.get(i,np.zeros(2048)),dtype=np.float32).flatten() for i,_ in tr])
    v_va=np.array([np.array(d.get(i,np.zeros(2048)),dtype=np.float32).flatten() for i,_ in va])
    v_noise=np.load(noisy_path).astype(np.float32) if noisy_path and os.path.exists(noisy_path) else None
    print(f"  Train:{len(tr)} Val:{len(va)} Clean:{v_tr.shape},{v_va.shape} Noisy:{v_noise.shape if v_noise is not None else 'N/A'}")
    return v_tr,v_va,v_noise,a_tr,a_va,y_tr,y_va,cw,valid,val_rows

# ── Training ──
def train_model(mfn, mtype, v_tr, a_tr, y_tr, cw, seed, device, val_data=None):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    m=mfn().to(device)
    need_agro=(mtype not in ['VisionOnly','DAFN-T'])
    dl=DataLoader(FeatDataset(v_tr,a_tr,y_tr,need_agro),min(BS,len(v_tr)//4),shuffle=True)
    crit=nn.CrossEntropyLoss(weight=cw.to(device)); opt=optim.Adam(m.parameters(),lr=1e-4)
    best_acc=0.0; best_state=None; patience=0
    for ep in range(1,EPOCHS+1):
        m.train(); c=0; t=0
        for batch in dl:
            if mtype=='VisionOnly':
                imgs,lbls=batch[0].to(device),batch[-1].to(device); logits=m(imgs)
            else:
                imgs,agrs,lbls=[b.to(device) for b in batch]; logits=m(imgs,agrs)
            loss=crit(logits,lbls); opt.zero_grad(); loss.backward(); opt.step()
            _,pr=logits.max(1); c+=(pr==lbls).sum().item(); t+=lbls.size(0)
        if c/t>best_acc: best_acc=c/t; patience=0; best_state={k:v.cpu().clone() for k,v in m.state_dict().items()}
        else: patience+=1
        if patience>=PATIENCE:
            if ep==1 or ep%10==0: print(f"    [{mtype}] Seed {seed} Ep{ep} done")
            break
    m.load_state_dict(best_state); return m

def evaluate(m, mtype, vis, agro, y, device):
    if mtype=='VisionOnly':
        dl=DataLoader(FeatDataset(vis,agro,y,False),BS)
    else:
        dl=DataLoader(FeatDataset(vis,agro,y,True),BS)
    pr_all,lb_all=[],[]
    m.eval()
    with torch.no_grad():
        for batch in dl:
            if mtype=='VisionOnly':
                imgs,lbls=batch[0].to(device),batch[-1].to(device); logits=m(imgs)
            else:
                imgs,agrs,lbls=[b.to(device) for b in batch]; logits=m(imgs,agrs)
            _,pr=logits.max(1); pr_all.extend(pr.cpu().numpy()); lb_all.extend(lbls.cpu().numpy())
    return accuracy_score(lb_all,pr_all),f1_score(lb_all,pr_all,average='macro')

# ── DAFN-T Training (with SequenceDataset) ──
def train_dafn_t(seed, valid_rows, clean_feat, cw, device):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    # Training: only use training rows
    train_rows_only = [r for r in valid_rows if r.get('split')=='train']
    # Build clean feat dict for training-only using valid indices
    train_feat = {i: clean_feat[i] for i in range(len(valid_rows)) if valid_rows[i].get('split')=='train'}
    train_feat_rekeyed = {j: train_feat[orig_i] for j, (orig_i, r) in enumerate([(i,r) for i,r in enumerate(valid_rows) if r.get('split')=='train'])}
    train_ds = SeqDataset(train_rows_only, train_feat_rekeyed, window=5)
    m = DAFN_T(window_size=5, hidden_dim=128, image_dim=2048, agronomic_dim=10,
               sensor_dim=6, num_classes=2, use_sensor=False).to(device)
    dl = DataLoader(train_ds, min(BS,len(train_ds)//2), shuffle=True)
    crit = nn.CrossEntropyLoss(weight=cw.to(device)); opt = optim.Adam(m.parameters(), lr=1e-4)
    best_acc=0.0; best_state=None; patience=0
    for ep in range(1, EPOCHS+1):
        m.train(); c=0; t=0
        for batch in dl:
            imgs,agrs,lbls=[batch[i].to(device) for i in [0,1,2]]
            logits,_ = m(imgs, agrs)
            loss=crit(logits,lbls); opt.zero_grad(); loss.backward(); opt.step()
            _,pr=logits.max(1); c+=(pr==lbls).sum().item(); t+=lbls.size(0)
        if ep==1 or ep%10==0: print(f"    [DAFN-T] Seed {seed} Ep{ep} train_acc={c/t:.4f}")
        if c/t>best_acc: best_acc=c/t; patience=0; best_state={k:v.cpu().clone() for k,v in m.state_dict().items()}
        else: patience+=1
        if patience>=PATIENCE: break
    m.load_state_dict(best_state); return m

def evaluate_dafn_t(m, valid_rows, clean_feat, noisy_feat, val_rows, device):
    """Evaluate on val-only clean and noisy data."""
    # Map val rows to their indices in the valid list
    val_indices = [i for i, r in enumerate(valid_rows) if r.get('split') == 'val']
    # Clean: use val_rows with clean features keyed by val-list index
    clean_val_feat = {j: clean_feat.get(val_indices[j], np.zeros(2048)).flatten() for j in range(len(val_rows))}
    clean_ds = SeqDataset(val_rows, clean_val_feat, window=5)
    c_ac,c_f1 = _eval_dafn(m, clean_ds, device)
    # Noisy: use val_rows with noisy features
    noisy_val_feat = {j: noisy_feat[j].flatten() if j < len(noisy_feat) else np.zeros(2048) for j in range(len(val_rows))}
    noisy_ds = SeqDataset(val_rows, noisy_val_feat, window=5)
    n_ac,n_f1 = _eval_dafn(m, noisy_ds, device)
    return c_ac,c_f1,n_ac,n_f1

def _eval_dafn(m, ds, device):
    dl=DataLoader(ds, BS)
    pr_all,lb_all=[],[]
    m.eval()
    with torch.no_grad():
        for batch in dl:
            imgs,agrs,lbls=[batch[i].to(device) for i in [0,1,2]]
            logits,_=m(imgs,agrs)
            _,pr=logits.max(1); pr_all.extend(pr.cpu().numpy()); lb_all.extend(lbls.cpu().numpy())
    return accuracy_score(lb_all,pr_all), f1_score(lb_all,pr_all,average='macro')

# ── Main ──
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='data/dataset_ready.csv')
    p.add_argument('--features',default='/root/autodl-tmp/finetuned_logits.npy')
    p.add_argument('--noisy',default='data/X_visual_val_noisy.npy')
    args=p.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    base=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path=os.path.join(base,args.csv)
    ft_path=args.features if os.path.exists(args.features) else os.path.join(base,args.features)
    ny_path=args.noisy if os.path.exists(args.noisy) else os.path.join(base,args.noisy)
    print(f"Device: {device}")

    vt,vv,vn,at,av,yt,yv,cw,valid_rows,val_rows=load(csv_path,ft_path,ny_path)
    # Build clean feature dict for DAFN-T (enumerate valid)
    d=np.load(ft_path,allow_pickle=True).item()
    clean_feat={i:np.array(d.get(i,np.zeros(2048))).flatten() for i in range(len(valid_rows))}

    all_r=[]

    # ── 1. Vision-only (Table 4) ──
    print(f"\n{'='*50}\n  Vision-Only (Table 4)\n{'='*50}")
    ca_list=[]
    for seed in SEEDS:
        t0=time.time()
        m=train_model(lambda: ImageOnlyClassifier(2048,2),'VisionOnly',
                      vt,at,yt,cw,seed,device)
        cac,_=evaluate(m,'VisionOnly',vv,av,yv,device); ca_list.append(cac)
        nac,_=evaluate(m,'VisionOnly',vn,av,yv,device)
        rp=os.path.join(RES_DIR,f'result_VisionOnly_seed{seed}.txt')
        with open(rp,'w') as f:
            f.write(f'model=VisionOnly\nseed={seed}\nclean_acc={cac:.6f}\nnoisy_acc={nac:.6f}\nelapsed={time.time()-t0:.0f}s\n')
        print(f"  Seed {seed}: clean={cac*100:.2f}% noisy={nac*100:.2f}%")
    ca2=np.array(ca_list)*100
    print(f"  >> VisionOnly: Clean={np.mean(ca2):.2f}±{np.std(ca2,ddof=1):.2f}%")
    all_r.append(('VisionOnly',ca2))

    # ── 2. DAFN-T + Noisy (Table 6) ──
    print(f"\n{'='*50}\n  DAFN-T Noisy (Table 6)\n{'='*50}")
    clean_ac_list=[]
    noisy_ac_list=[]
    for seed in SEEDS:
        t0=time.time()
        m=train_dafn_t(seed,valid_rows,clean_feat,cw,device)
        cac,cf1,nac,nf1=evaluate_dafn_t(m,valid_rows,clean_feat,vn,val_rows,device)
        clean_ac_list.append(cac); noisy_ac_list.append(nac)
        rp=os.path.join(RES_DIR,f'result_DAFN-T-Noisy_seed{seed}.txt')
        with open(rp,'w') as f:
            f.write(f'model=DAFN-T-Noisy\nseed={seed}\nclean_acc={cac:.6f}\nnoisy_acc={nac:.6f}\nclean_f1={cf1:.6f}\nnoisy_f1={nf1:.6f}\nelapsed={time.time()-t0:.0f}s\n')
        print(f"  Seed {seed}: clean={cac*100:.2f}% noisy={nac*100:.2f}%")
    ca2=np.array(clean_ac_list)*100; na2=np.array(noisy_ac_list)*100
    print(f"  >> DAFN-T Noisy: Clean={np.mean(ca2):.2f}±{np.std(ca2,ddof=1):.2f}% Noisy={np.mean(na2):.2f}±{np.std(na2,ddof=1):.2f}%")
    all_r.append(('DAFN-T-Noisy',na2))

    # ── Summary ──
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    for name,vals in all_r:
        vals_pct=vals if np.max(vals)>1 else vals*100
        print(f"  {name:<20s}: {np.mean(vals_pct):.2f}±{np.std(vals_pct,ddof=1):.2f}%")
    sp=os.path.join(RES_DIR,'summary.txt')
    with open(sp,'w') as f:
        f.write("Table 4 & Table 6 Repeat Experiment Summary (3 seeds)\n\n")
        for name,vals in all_r:
            vals_pct=vals if np.max(vals)>1 else vals*100
            f.write(f"{name:<20s}: {np.mean(vals_pct):.2f}±{np.std(vals_pct,ddof=1):.2f}%\n")
    print(f"\nSaved → {sp}")

if __name__=='__main__': main()

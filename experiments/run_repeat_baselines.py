#!/usr/bin/env python3
"""Repeat experiment for baseline fusion models (Table 10).
5 seeds per model, clean + noisy evaluation."""
import os, sys, csv, argparse, time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dafn import ImageOnlyClassifier, SimpleFusion, ConcatCBAM, GatedFusion, CrossAttentionFusion

AGRO = ['Plant Height','Stem Diameter','Leaf Width','Leaf Length','NDVI','RVI','LNC','LNA','LAI','LDW']
LABEL_MAP = {'Healthy':0,'Stress':1,'Other':1}
SEEDS = [42,123,456,789,999]; BATCH=16; EPOCHS=50; PATIENCE=10
MODELS = {
    'Simple':     (lambda: SimpleFusion(2048,10,64,2),['img','agr']),
    'CBAM':       (lambda: ConcatCBAM(2048,10,128,2),['img','agr']),
    'Gated':      (lambda: GatedFusion(2048,10,64,2),['img','agr']),
    'CrossAttn':  (lambda: CrossAttentionFusion(2048,10,64,2),['img','agr']),
    'VisionOnly': (lambda: ImageOnlyClassifier(2048,2),['img']),
}
RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','baselines')
os.makedirs(RES_DIR,exist_ok=True)

class FeatDataset(Dataset):
    def __init__(self, vis, agro, y, need_agro=True):
        self.vis=vis; self.agro=agro; self.y=y; self.need_agro=need_agro
    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        img=torch.tensor(self.vis[i],dtype=torch.float32)
        lbl=torch.tensor(self.y[i],dtype=torch.long)
        if self.need_agro:
            return img,torch.tensor(self.agro[i],dtype=torch.float32),lbl
        return img,lbl

def load(csv_path, ft_path, noisy_path):
    print("Loading...")
    with open(csv_path) as f: rows=list(csv.DictReader(f))
    rows=[r for r in rows if r.get('Source')!='2023']
    valid=[r for r in rows if r.get('label_3class','') in {'Healthy','Stress','Other'}]
    tr=[(i,r) for i,r in enumerate(valid) if r.get('split')=='train']
    va=[(i,r) for i,r in enumerate(valid) if r.get('split')=='val']
    scaler=StandardScaler()
    for rr in [[r for _,r in tr],[r for _,r in va]]:
        X=np.array([[float(r.get(f,'nan')) for f in AGRO] for r in rr]); X=np.nan_to_num(X,nan=0.0)
        Xs=scaler.fit_transform(X)
        for i,r in enumerate(rr):
            for j,f in enumerate(AGRO): r[f]=Xs[i,j]
    a_tr=np.array([[float(r.get(f,0)) for f in AGRO] for _,r in tr],dtype=np.float32)
    a_va=np.array([[float(r.get(f,0)) for f in AGRO] for _,r in va],dtype=np.float32)
    y_tr=np.array([LABEL_MAP[r['label_3class']] for _,r in tr],dtype=np.int64)
    y_va=np.array([LABEL_MAP[r['label_3class']] for _,r in va],dtype=np.int64)
    cw=torch.tensor(compute_class_weight('balanced',classes=np.unique(y_tr),y=y_tr),dtype=torch.float32)
    d=np.load(ft_path,allow_pickle=True).item()
    v_tr=np.array([d.get(i,np.zeros(2048)).flatten() for i,_ in tr],dtype=np.float32)
    v_va=np.array([d.get(i,np.zeros(2048)).flatten() for i,_ in va],dtype=np.float32)
    v_noise=np.load(noisy_path).astype(np.float32) if noisy_path and os.path.exists(noisy_path) else None
    print(f"  Train:{len(tr)} Val:{len(va)} Clean:{v_tr.shape},{v_va.shape} Noisy:{v_noise.shape if v_noise is not None else 'N/A'}")
    return v_tr,v_va,v_noise,a_tr,a_va,y_tr,y_va,cw

def train_one(mfn, mtype, v_tr, a_tr, y_tr, cw, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    m=mfn().to(device)
    dl=DataLoader(FeatDataset(v_tr,a_tr,y_tr,need_agro=(mtype!='VisionOnly')),min(BATCH,len(v_tr)//4),shuffle=True)
    crit=nn.CrossEntropyLoss(weight=cw.to(device)); opt=optim.Adam(m.parameters(),lr=1e-4)
    best_acc=0.0; best_state=None; patience=0
    for ep in range(1,EPOCHS+1):
        m.train(); c=0; t=0
        for batch in dl:
            if mtype=='VisionOnly':
                imgs,lbls=batch[0].to(device),batch[-1].to(device)
                logits=m(imgs)
            else:
                imgs,agrs,lbls=[b.to(device) for b in batch]
                logits=m(imgs,agrs)
            loss=crit(logits,lbls); opt.zero_grad(); loss.backward(); opt.step()
            _,pr=logits.max(1); c+=(pr==lbls).sum().item(); t+=lbls.size(0)
        if ep==1 or ep%10==0:
            print(f"    [{mtype}] Seed {seed} Ep{ep}/{EPOCHS} train_acc={c/t:.4f}")
        # Early stopping based on training accuracy improvement
        if c/t > best_acc:
            best_acc=c/t; patience=0; best_state={k:v.cpu().clone() for k,v in m.state_dict().items()}
        else:
            patience+=1
            if patience>=PATIENCE:
                print(f"    [{mtype}] Seed {seed} early stop Ep{ep}")
                break
    m.load_state_dict(best_state); return m

def evaluate(m, mtype, vis, agro, y, device):
    dl=DataLoader(FeatDataset(vis,agro,y,need_agro=(mtype!='VisionOnly')),BATCH)
    pr_all,lb_all=[],[]
    m.eval()
    with torch.no_grad():
        for batch in dl:
            if mtype=='VisionOnly':
                imgs,lbls=batch[0].to(device),batch[-1].to(device)
                logits=m(imgs)
            else:
                imgs,agrs,lbls=[b.to(device) for b in batch]
                logits=m(imgs,agrs)
            _,pr=logits.max(1)
            pr_all.extend(pr.cpu().numpy()); lb_all.extend(lbls.cpu().numpy())
    return accuracy_score(lb_all,pr_all), f1_score(lb_all,pr_all,average='macro')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='data/dataset_ready.csv')
    p.add_argument('--features',default='/root/autodl-tmp/finetuned_logits.npy')
    p.add_argument('--noisy',default='data/X_visual_val_noisy.npy')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args()
    device=args.device if torch.cuda.is_available() else 'cpu'
    base=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path=os.path.join(base,args.csv)
    ft_path=args.features if os.path.exists(args.features) else os.path.join(base,args.features)
    ny_path=args.noisy if os.path.exists(args.noisy) else os.path.join(base,args.noisy)
    print(f"Device: {device}")
    vt,vv,vn,at,av,yt,yv,cw=load(csv_path,ft_path,ny_path)
    all_r=[]
    for mname,(mfn,_) in MODELS.items():
        print(f"\n{'='*50}\n  {mname}\n{'='*50}")
        ca,na=[],[]
        for seed in SEEDS:
            t0=time.time()
            m=train_one(mfn,mname,vt,at,yt,cw,seed,device)
            ca1,_=evaluate(m,mname,vv,av,yv,device); ca.append(ca1)
            na1,_=evaluate(m,mname,vn,av,yv,device) if vn is not None else (0,0); na.append(na1)
            rp=os.path.join(RES_DIR,f'result_{mname}_seed{seed}.txt')
            with open(rp,'w') as f:
                f.write(f'model={mname}\nseed={seed}\nclean_acc={ca1:.6f}\nnoisy_acc={na1:.6f}\nelapsed={time.time()-t0:.0f}s\n')
            print(f"  [{mname}] Seed {seed}: clean={ca1*100:.2f}% noisy={na1*100:.2f}%")
        ca2=np.array(ca)*100; na2=np.array(na)*100
        print(f"  >> {mname}: Clean={np.mean(ca2):.2f}±{np.std(ca2,ddof=1):.2f}% Noisy={np.mean(na2):.2f}±{np.std(na2,ddof=1):.2f}%")
        all_r.append((mname,ca2,na2))
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    for m,c,n in all_r:
        print(f"  {m:<12s} Clean={np.mean(c):.2f}±{np.std(c,ddof=1):.2f}% Noisy={np.mean(n):.2f}±{np.std(n,ddof=1):.2f}%")
    sp=os.path.join(RES_DIR,'summary.txt')
    with open(sp,'w') as f:
        f.write(f'{"Model":<12s} {"Clean Acc":>12s} {"Noisy Acc":>12s}\n'+'-'*40+'\n')
        for m,c,n in all_r:
            f.write(f'{m:<12s} {np.mean(c):>6.2f}±{np.std(c,ddof=1):>.2f}% {np.mean(n):>6.2f}±{np.std(n,ddof=1):>.2f}%\n')
    print(f"\nSaved → {sp}")

if __name__=='__main__': main()

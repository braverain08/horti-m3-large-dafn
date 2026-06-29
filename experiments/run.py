#!/usr/bin/env python3
"""Experiment runner for Q1 DAFN paper."""
import os, sys, csv, json, argparse
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.dafn import DAFN, DAFN_T, ImageOnlyClassifier, AgronomicOnlyClassifier, SimpleFusion, ConcatCBAM, GatedFusion, CrossAttentionFusion
from models.dafn_ft import DAFN_ImageNet

SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AGRONOMIC_FEATURES = ['Plant Height','Stem Diameter','Leaf Width','Leaf Length','NDVI','RVI','LNC','LNA','LAI','LDW']
SENSOR_FEATURES   = ['Air_Temperature','Relative_Humidity','Light_Intensity','CO2','Soil_Moisture','Soil_Temperature']
ALL_FEATURES = AGRONOMIC_FEATURES + SENSOR_FEATURES
LABEL_MAP   = {'Healthy': 0, 'Stress': 1, 'Other': 2}
NUM_CLASSES = 3; HIDDEN_DIM = 64; LR = 1e-4; EPOCHS = 80; BATCH_SIZE = 16
class_weights = None

class StressDataset(Dataset):
    def __init__(self, rows, feature_dict=None, use_sensor=False):
        self.rows = rows; self.features = feature_dict or {}; self.use_sensor = use_sensor
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        agr = torch.tensor([float(r.get(f,0)) for f in AGRONOMIC_FEATURES], dtype=torch.float32)
        img = torch.tensor(self.features.get(idx, np.zeros(2048,dtype=np.float32)), dtype=torch.float32)
        lbl = torch.tensor(LABEL_MAP.get(r.get('label_3class','Unknown'), -1), dtype=torch.long)
        if self.use_sensor:
            return img, agr, torch.tensor([float(r.get(f,0)) for f in SENSOR_FEATURES], dtype=torch.float32), lbl
        return img, agr, lbl

class SequenceDataset(Dataset):
    def __init__(self, rows, feature_dict=None, window_size=5, use_sensor=False, stride=1):
        self.window_size = window_size; self.features = feature_dict or {}; self.use_sensor = use_sensor
        pg = defaultdict(list)
        for i, r in enumerate(rows): pg[r['Number']].append((r, i))
        for p in pg: pg[p].sort(key=lambda x: x[0]['Date'])
        self.samples = []
        for plant, items in pg.items():
            for i in range(0, len(items)-window_size+1, stride):
                wins = items[i:i+window_size]; label = LABEL_MAP.get(wins[-1][0].get('label_3class','Unknown'), -1)
                if label >= 0: self.samples.append((wins, label))
        print(f'  Sequences: {len(self.samples)} (T={window_size})')
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        wins, label = self.samples[idx]; T = self.window_size
        img_seq = torch.zeros(T, 2048); agr_seq = torch.zeros(T, len(AGRONOMIC_FEATURES))
        for t, (r, oi) in enumerate(wins):
            img_seq[t] = torch.tensor(self.features.get(oi, np.zeros(2048)), dtype=torch.float32)
            agr_seq[t] = torch.tensor([float(r.get(f,0)) for f in AGRONOMIC_FEATURES], dtype=torch.float32)
        if self.use_sensor:
            sen_seq = torch.zeros(T, len(SENSOR_FEATURES))
            for t, (r, _) in enumerate(wins):
                sen_seq[t] = torch.tensor([float(r.get(f,0)) for f in SENSOR_FEATURES], dtype=torch.float32)
            return img_seq, agr_seq, sen_seq, torch.tensor(label, dtype=torch.long)
        return img_seq, agr_seq, torch.tensor(label, dtype=torch.long)

def load_data(csv_path, feature_npy_path=None, use_sensor=False):
    with open(csv_path) as f: all_rows = list(csv.DictReader(f))
    all_rows = [r for r in all_rows if r.get('Source') != '2023']
    valid = [r for r in all_rows if r.get('label_3class','Unknown') in LABEL_MAP]
    feat = {}
    if feature_npy_path and os.path.exists(feature_npy_path):
        d = np.load(feature_npy_path, allow_pickle=True).item()
        feat = {int(k): v for k, v in d.items()}
    train = [r for i,r in enumerate(valid) if r.get('split')=='train']
    val = [r for i,r in enumerate(valid) if r.get('split')=='val']
    # Standardize each row
    for label, rows in [('train', train), ('val', val)]:
        X = np.array([[float(r.get(f,'nan')) for f in ALL_FEATURES] for r in rows])
        X = np.nan_to_num(X, nan=0.0)
        scaler = StandardScaler()
        scaler.fit(X)
        X_s = scaler.transform(X)
        for i, r in enumerate(rows):
            for j, f in enumerate(ALL_FEATURES): r[f] = X_s[i,j]
    yt = np.array([LABEL_MAP.get(r['label_3class'],-1) for r in train])
    cw = compute_class_weight('balanced', classes=np.unique(yt), y=yt)
    global class_weights; class_weights = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
    print(f'Loaded {len(valid)}: train={len(train)} val={len(val)}')
    print(f'  Train labels: {Counter(r["label_3class"] for r in train)}')
    print(f'  Val labels:   {Counter(r["label_3class"] for r in val)}')
    print(f'  Class weights: {dict(zip(range(3),cw))}')
    return train, val, feat

def train_epoch(model, loader, crit, opt):
    model.train(); tl=0.0; c=0; t=0
    for batch in loader:
        lbl = batch[-1].to(DEVICE); n=len(batch)
        if n==4: logits,_ = model(batch[0].to(DEVICE),batch[1].to(DEVICE),batch[2].to(DEVICE))
        elif n==3: r=model(batch[0].to(DEVICE),batch[1].to(DEVICE)); logits=r[0] if isinstance(r,tuple) else r
        else: logits = model(batch[0].to(DEVICE))
        loss=crit(logits,lbl); opt.zero_grad(); loss.backward(); opt.step()
        _,pr=logits.max(1); c+=(pr==lbl).sum().item(); t+=lbl.size(0); tl+=loss.item()
    return tl/len(loader), c/t

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); ap=[]; al=[]
    for batch in loader:
        lbl=batch[-1].to(DEVICE); n=len(batch)
        if n==4: logits,_=model(batch[0].to(DEVICE),batch[1].to(DEVICE),batch[2].to(DEVICE))
        elif n==3: r=model(batch[0].to(DEVICE),batch[1].to(DEVICE)); logits=r[0] if isinstance(r,tuple) else r
        else: logits=model(batch[0].to(DEVICE))
        _,pr=logits.max(1); ap.extend(pr.cpu().numpy()); al.extend(lbl.cpu().numpy())
    acc=accuracy_score(al,ap); f1=f1_score(al,ap,average='macro')
    pc=precision_recall_fscore_support(al,ap,labels=[0,1,2],zero_division=0)
    return {'accuracy':acc,'macro_f1':f1,'precision':pc[0].tolist(),'recall':pc[1].tolist(),'f1_per_class':pc[2].tolist()}

def run(model_class, train_ds, val_ds, name, **kw):
    tl=DataLoader(train_ds,BATCH_SIZE,shuffle=True); vl=DataLoader(val_ds,BATCH_SIZE)
    m=model_class(**kw).to(DEVICE); crit=nn.CrossEntropyLoss(weight=class_weights); opt=optim.Adam(m.parameters(),LR)
    params=sum(p.numel() for p in m.parameters()); ba=0; bs=None
    for ep in range(EPOCHS):
        loss,acc=train_epoch(m,tl,crit,opt); res=evaluate(m,vl)
        if res['accuracy']>ba: ba=res['accuracy']; bs={k:v.cpu().clone() for k,v in m.state_dict().items()}
        if (ep+1)%30==0: print(f'  [{name}] E{ep+1}/{EPOCHS} loss={loss:.4f} val={res["accuracy"]:.4f}')
    m.load_state_dict(bs); res=evaluate(m,vl); res['model']=name; res['params']=params
    return res

def main():
    global EPOCHS
    p=argparse.ArgumentParser()
    p.add_argument('--methods',default='all'); p.add_argument('--csv',default=os.path.join(DATA_DIR,'dataset_ready.csv'))
    p.add_argument('--features',default=os.path.join(DATA_DIR,'image_features.npy')); p.add_argument('--use_sensor',action='store_true')
    p.add_argument('--window',type=int,default=5); p.add_argument('--dry_run',action='store_true'); p.add_argument('--epochs',type=int,default=EPOCHS)
    args=p.parse_args()
    EPOCHS=args.epochs
    if args.dry_run: print(f'Epochs={EPOCHS} Sensor={args.use_sensor} Window={args.window} Device={DEVICE}'); return

    train,val,feat=load_data(args.csv,args.features,args.use_sensor)
    img_dim = feat.get(0, np.zeros(2048)).shape[0] if feat else 2048
    ms=args.methods.split(',') if args.methods!='all' else ['agronomic_only','simple_fusion','dafn','dafn_t','svm','rf']
    results=[]

    if 'image_only' in ms:
        print('\n--- Image-only ---')
        ds1=DataLoader(torch.utils.data.TensorDataset(torch.stack([StressDataset(train,feat)[i][0] for i in range(len(train))]),torch.tensor([StressDataset(train,feat)[i][-1] for i in range(len(train))])),BATCH_SIZE)
        ds2=DataLoader(torch.utils.data.TensorDataset(torch.stack([StressDataset(val,feat)[i][0] for i in range(len(val))]),torch.tensor([StressDataset(val,feat)[i][-1] for i in range(len(val))])),BATCH_SIZE)
        m=ImageOnlyClassifier(input_dim=img_dim,num_classes=NUM_CLASSES).to(DEVICE)
        params=sum(p.numel() for p in m.parameters())
        import copy; m2=copy.deepcopy; crit=nn.CrossEntropyLoss(weight=class_weights); opt=optim.Adam(m.parameters(),LR); ba=0; bs=None
        for ep in range(EPOCHS):
            m.train(); tl=0.0; c=0; t=0
            for batch in ds1: x=l=batch[-1].to(DEVICE); logits=m(batch[0].to(DEVICE)); loss=crit(logits,l); opt.zero_grad(); loss.backward(); opt.step(); _,pr=logits.max(1); c+=(pr==l).sum().item(); t+=l.size(0); tl+=loss.item()
            m.eval(); ap=[]; al=[]
            with torch.no_grad():
                for batch in ds2: l=batch[-1].to(DEVICE); logits=m(batch[0].to(DEVICE)); _,pr=logits.max(1); ap.extend(pr.cpu().numpy()); al.extend(l.cpu().numpy())
            vacc=accuracy_score(al,ap); vf1=f1_score(al,ap,average='macro')
            if vacc>ba: ba=vacc; bs={k:v.cpu().clone() for k,v in m.state_dict().items()}
            if (ep+1)%30==0: print(f'  [Image-only] E{ep+1}/{EPOCHS} val={vacc:.4f}')

    td=StressDataset(train,feat,args.use_sensor); vd=StressDataset(val,feat,args.use_sensor)
    if 'agronomic_only' in ms:
        ds1=torch.utils.data.TensorDataset(torch.stack([td[i][1] for i in range(len(td))]),torch.tensor([td[i][-1] for i in range(len(td))]))
        ds2=torch.utils.data.TensorDataset(torch.stack([vd[i][1] for i in range(len(vd))]),torch.tensor([vd[i][-1] for i in range(len(vd))]))
        results.append(run(AgronomicOnlyClassifier,ds1,ds2,'Agronomic-only',input_dim=10,num_classes=NUM_CLASSES))
    if 'simple_fusion' in ms:
        results.append(run(SimpleFusion,td,vd,'SimpleFusion',image_dim=img_dim,agronomic_dim=10,hidden_dim=HIDDEN_DIM,num_classes=NUM_CLASSES))
    if 'concat_cbam' in ms:
        results.append(run(ConcatCBAM,td,vd,'Concat+CBAM',image_dim=2048,agronomic_dim=10,hidden_dim=128,num_classes=NUM_CLASSES))
    if 'gated' in ms:
        results.append(run(GatedFusion,td,vd,'Gated Fusion',image_dim=2048,agronomic_dim=10,hidden_dim=HIDDEN_DIM,num_classes=NUM_CLASSES))
    if 'cross_attn' in ms:
        results.append(run(CrossAttentionFusion,td,vd,'Cross-Attention',image_dim=2048,agronomic_dim=10,hidden_dim=HIDDEN_DIM,num_classes=NUM_CLASSES))
    if 'dafn' in ms:
        results.append(run(DAFN,td,vd,'DAFN',image_dim=img_dim,agronomic_dim=10,sensor_dim=6,hidden_dim=HIDDEN_DIM,num_classes=NUM_CLASSES,use_sensor=args.use_sensor))
    if 'dafn_ft' in ms:
        results.append(run(DAFN_ImageNet,td,vd,'DAFN_ImageNet',logit_dim=3,agronomic_dim=10,sensor_dim=6,hidden=HIDDEN_DIM,num_classes=NUM_CLASSES,use_sensor=args.use_sensor))
    if 'dafn_t' in ms:
        sq_t=SequenceDataset(train,feat,args.window,args.use_sensor); sq_v=SequenceDataset(val,feat,args.window,args.use_sensor)
        if len(sq_t)>0 and len(sq_v)>0: results.append(run(DAFN_T,sq_t,sq_v,f'DAFN-T(T={args.window})',window_size=args.window,image_dim=2048,agronomic_dim=10,sensor_dim=6,hidden_dim=HIDDEN_DIM,num_classes=NUM_CLASSES,use_sensor=args.use_sensor))

    if 'svm' in ms:
        Xtr=np.nan_to_num(np.array([[float(r.get(f,0)) for f in AGRONOMIC_FEATURES] for r in train]),nan=0.0); ytr=np.array([LABEL_MAP.get(r['label_3class'],-1) for r in train])
        Xv=np.nan_to_num(np.array([[float(r.get(f,0)) for f in AGRONOMIC_FEATURES] for r in val]),nan=0.0); yv=np.array([LABEL_MAP.get(r['label_3class'],-1) for r in val])
        m=SVC(kernel='rbf',gamma='scale',C=1.0); m.fit(Xtr,ytr); p=m.predict(Xv); acc=accuracy_score(yv,p)
        pc=precision_recall_fscore_support(yv,p,labels=[0,1,2],zero_division=0)
        results.append({'model':'SVM','accuracy':acc,'macro_f1':f1_score(yv,p,average='macro'),'recall':pc[1].tolist(),'f1_per_class':pc[2].tolist(),'params':0})
    if 'rf' in ms:
        Xtr=np.nan_to_num(np.array([[float(r.get(f,0)) for f in AGRONOMIC_FEATURES] for r in train]),nan=0.0); ytr=np.array([LABEL_MAP.get(r['label_3class'],-1) for r in train])
        Xv=np.nan_to_num(np.array([[float(r.get(f,0)) for f in AGRONOMIC_FEATURES] for r in val]),nan=0.0); yv=np.array([LABEL_MAP.get(r['label_3class'],-1) for r in val])
        m=RandomForestClassifier(n_estimators=100,random_state=SEED); m.fit(Xtr,ytr); p=m.predict(Xv); acc=accuracy_score(yv,p)
        pc=precision_recall_fscore_support(yv,p,labels=[0,1,2],zero_division=0)
        results.append({'model':'RF','accuracy':acc,'macro_f1':f1_score(yv,p,average='macro'),'recall':pc[1].tolist(),'f1_per_class':pc[2].tolist(),'params':0})

    print('\n'+'='*90); print('  RESULTS'); print('='*90)
    print('  Model                   Acc    F1   S_R  Params')
    print('  ' + '-'*22 + ' ' + '-'*6 + ' ' + '-'*6 + ' ' + '-'*6 + ' ' + '-'*8)
    for r in results: sr=r['recall'][1] if len(r['recall'])>1 else 0; print(f'  {r["model"]:<22s} {r["accuracy"]:>6.3f} {r.get("macro_f1",0):>6.3f} {sr:>6.3f} {r.get("params",0):>8,d}')
    print('='*90)
    ts=datetime.now().strftime('%Y%m%d_%H%M%S'); out=os.path.join(RESULTS_DIR,f'results_{ts}.json')
    with open(out,'w') as f: json.dump(results,f,indent=2)
    print(f'Saved to {out}')


#!/usr/bin/env python3
"""Evaluate vision model on degraded vs clean validation subsets."""
import os, sys, csv, argparse, json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch, torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dafn import DAFN_T

TF = T.Compose([
    T.Resize(256), T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
LABEL_MAP = {'Healthy': 0, 'Stress': 1, 'Other': 1}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BS = 32

def load_vision_model(path):
    sd = torch.load(path, map_location='cpu')
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(2048, 3)
    m.load_state_dict(sd, strict=False)
    m.to(DEVICE); m.eval()
    return m

def load_dafn_model(path):
    sd = torch.load(path, map_location='cpu')
    m = DAFN_T(window_size=5, hidden_dim=128, image_dim=2048, agronomic_dim=10,
               sensor_dim=6, num_classes=2, use_sensor=False)
    m.load_state_dict(sd, strict=False)
    m.to(DEVICE); m.eval()
    return m

def extract_features_single(model, img_path):
    img = Image.open(img_path).convert('RGB')
    x = TF(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        x = model.conv1(x); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
        x = model.layer1(x); x = model.layer2(x); x = model.layer3(x); x = model.layer4(x)
        x = model.avgpool(x)
        feat = x.view(x.size(0), -1)
    return feat.cpu().numpy().flatten()

def predict_vision(model, img_path):
    img = Image.open(img_path).convert('RGB')
    x = TF(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
    return logits.argmax(1).item()

def evaluate_subset(model, subset_csv, data_dir, idx, model_type='vision'):
    with open(subset_csv) as f:
        rows = list(csv.DictReader(f))
    print(f"  Evaluating {len(rows)} images from {os.path.basename(subset_csv)}")

    preds, truths = [], []
    for r in tqdm(rows, desc=f'{model_type}'):
        pp = r.get('photo_path','')
        fn = os.path.basename(pp.split(';')[0])
        ip = idx.get(fn)
        if not ip or not os.path.exists(ip):
            continue
        lbl = LABEL_MAP.get(r.get('label',''), -1)
        if lbl < 0: continue

        if model_type == 'vision':
            pred = predict_vision(model, ip)
        else:
            pred = 0  # DAFN-T not supported for single images

        # Map 3-class to binary
        if model_type == 'vision' and isinstance(pred, int):
            pred = 1 if pred in [1,2] else 0

        preds.append(pred); truths.append(lbl)

    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average='macro')
    cm = confusion_matrix(truths, preds) if len(set(truths+preds))>1 else None
    return {'accuracy': acc, 'macro_f1': f1, 'confusion_matrix': cm.tolist() if cm is not None else [],
            'n': len(truths), 'predictions': preds, 'truths': truths}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vision-model', default='data/resnet50_finetuned.pth')
    p.add_argument('--dafn-model', default='data/dafn_t_dim128_best.pth')
    p.add_argument('--data-dir', default='/Users/rainxu/Downloads/2023-2025 Tomato dataset')
    p.add_argument('--subset-dir', default='experiments/results/degraded_subsets')
    p.add_argument('--output', default='experiments/results/degraded_eval.json')
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Build image index
    print("Building image index...")
    idx = {}
    for year in ['2024','2025']:
        yp = os.path.join(args.data_dir, year)
        if not os.path.isdir(yp): continue
        for r, d, fs in os.walk(yp):
            for f in fs:
                if f.lower().endswith(('.jpg','.jpeg','.png')):
                    idx[f] = os.path.join(r, f)
    print(f"  Indexed {len(idx)} images")

    # Load model
    print(f"Loading vision model from {args.vision_model}...")
    vmodel = load_vision_model(args.vision_model)
    print("  Vision model loaded")

    # Find subset CSV files
    subset_files = [f for f in os.listdir(args.subset_dir) if f.endswith('.csv') and f != 'subset_summary.csv']
    if not subset_files:
        print(f"No subset CSVs found in {args.subset_dir}")
        return

    results = {}
    for sf in sorted(subset_files):
        name = sf.replace('.csv','')
        sp = os.path.join(args.subset_dir, sf)
        res = evaluate_subset(vmodel, sp, args.data_dir, idx, 'vision')
        results[name] = {'accuracy': res['accuracy'], 'macro_f1': res['macro_f1'],
                         'n': res['n'], 'confusion_matrix': res['confusion_matrix']}
        print(f"  {name}: acc={res['accuracy']*100:.2f}%, f1={res['macro_f1']:.4f}, n={res['n']}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.output}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Extract 3-dim logits from fine-tuned ResNet50 as image features for DAFN_ImageNet.
These logits alone achieve 97.4% validation accuracy."""
import os, csv, argparse
import numpy as np
from PIL import Image
import torch, torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABEL_MAP = {'Healthy': 0, 'Stress': 1, 'Other': 2}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-dir', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()

    with open(args.csv) as f: rows = list(csv.DictReader(f))
    valid = [r for r in rows if r.get('label_3class','Unknown') in LABEL_MAP]

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(2048, 3)
    state = torch.load(args.checkpoint, map_location=DEVICE)
    # Handle DataParallel wrapping
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE).eval()
    print(f'Loaded fine-tuned model. FC weights: {model.fc.weight.shape}')

    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # Build image index
    idx = {}
    for year in ['2024','2025']:
        for root, dirs, files in os.walk(os.path.join(args.data_dir, year)):
            for f in files:
                if f.lower().endswith(('.jpg','.jpeg','.png')):
                    idx[f] = os.path.join(root, f)
    print(f'Index: {len(idx)} images')

    feats = {}
    for i, r in enumerate(tqdm(valid)):
        paths = r.get('Photo Path','').split(';')
        img_path = None
        for p in paths:
            bn = os.path.basename(p.strip())
            if bn in idx: img_path = idx[bn]; break
        if img_path:
            try:
                img = Image.open(img_path).convert('RGB')
                t = tfm(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = model(t).flatten().cpu().numpy()
                feats[i] = logits.astype(np.float32)
            except:
                feats[i] = np.zeros(3, dtype=np.float32)
        else:
            feats[i] = np.zeros(3, dtype=np.float32)

    np.save(args.output, feats)
    nz = sum(1 for v in feats.values() if np.all(v==0))
    print(f'Saved {len(feats)} 3-dim logits to {args.output}, zeros={nz}')

if __name__ == '__main__':
    main()

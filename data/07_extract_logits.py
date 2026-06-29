#!/usr/bin/env python3
"""Extract 3-class logits from fine-tuned ResNet50 as image features for DAFN."""
import os, sys, csv, argparse, glob
import numpy as np
from PIL import Image
import torch, torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABEL_MAP = {'Healthy': 0, 'Stress': 1, 'Other': 2}

def build_image_index(data_base):
    idx = {}
    for year in ['2024', '2025']:
        for root, dirs, files in os.walk(os.path.join(data_base, year)):
            for f in files:
                if f.lower().endswith(('.jpg','.jpeg','.png')):
                    idx[f] = os.path.join(root, f)
    return idx

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True); p.add_argument('--data-dir', required=True)
    p.add_argument('--output', required=True); p.add_argument('--checkpoint')
    args = p.parse_args()

    with open(args.csv) as f: rows = list(csv.DictReader(f))
    valid = [r for r in rows if r.get('label_3class','Unknown') in LABEL_MAP]

    # Load fine-tuned model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(2048, 3)
    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
        print(f'Loaded checkpoint: {args.checkpoint}')
    else:
        ckpt = os.path.join(os.path.dirname(args.output), 'resnet50_finetuned.pth')
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            print(f'Loaded checkpoint: {ckpt}')
    model = model.to(DEVICE).eval()
    print(f'Model loaded (fc weights: {model.fc.weight.shape})')

    image_index = build_image_index(args.data_dir)
    print(f'Image index: {len(image_index)} files')

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # Remove FC to get 2048-dim features BEFORE the classifier layer
    feature_extractor = nn.Sequential(*list(model.children())[:-1])  # 2048-dim avgpool features
    feature_extractor.eval()

    feat_dict = {}
    failures = 0
    for i, r in enumerate(tqdm(valid, desc='Extracting logits')):
        paths = r.get('Photo Path', '').split(';')
        found = None
        for p in paths:
            bn = os.path.basename(p.strip())
            if bn in image_index:
                found = image_index[bn]; break
        if found:
            try:
                img = Image.open(found).convert('RGB')
                t = transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    feat = feature_extractor(t).flatten().cpu().numpy()
                feat_dict[i] = feat.astype(np.float32)
            except:
                feat_dict[i] = np.zeros(2048, dtype=np.float32); failures += 1
        else:
            feat_dict[i] = np.zeros(2048, dtype=np.float32); failures += 1

    np.save(args.output, feat_dict)
    nz = sum(1 for v in feat_dict.values() if np.all(v==0))
    print(f'Saved {len(feat_dict)} features to {args.output}')
    print(f'Zero: {nz}/{len(feat_dict)}, Failures: {failures}')
    print(f'Feature dim: {list(feat_dict.values())[0].shape}')

if __name__ == '__main__':
    main()

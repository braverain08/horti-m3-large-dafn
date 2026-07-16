#!/usr/bin/env python3
"""Extract fine-tuned ResNet50 2048-d features from validation images
with three single noise types (gaussian, low_light, occlusion).
Uses os.walk() for image indexing (robust to any directory structure)."""
import os, sys, csv, argparse
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

def build_image_index(data_base):
    """Index all images by basename via os.walk()."""
    idx = {}
    for year in os.listdir(data_base):
        yp = os.path.join(data_base, year)
        if not os.path.isdir(yp) or not year.isdigit(): continue
        for r, d, fs in os.walk(yp):
            for f in fs:
                if f.lower().endswith(('.jpg','.jpeg','.png')):
                    idx[f] = os.path.join(r, f)
    return idx

def noise_gaussian(img, rng):
    k = rng.choice([5, 7, 9, 11])
    return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.0, 3.0)))
def noise_low_light(img, rng):
    return ImageEnhance.Brightness(img).enhance(rng.uniform(0.3, 0.7))
def noise_occlusion(img, rng):
    w, h = img.size; out = img.copy()
    for _ in range(rng.randint(1, 4)):
        bw = rng.randint(w//8, w//3); bh = rng.randint(h//8, h//3)
        out.paste(Image.new('RGB', (bw, bh), (rng.randint(0,40),)*3),
                  (rng.randint(0,w-bw), rng.randint(0,h-bh)))
    return out

NOISE_MAP = {'gaussian': noise_gaussian, 'low_light': noise_low_light, 'occlusion': noise_occlusion}
TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

def extract_features(model, x):
    x = model.conv1(x); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
    x = model.layer1(x); x = model.layer2(x); x = model.layer3(x); x = model.layer4(x)
    x = model.avgpool(x); return x.view(x.size(0), -1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='/Users/rainxu/Downloads/2023-2025 Tomato dataset')
    p.add_argument('--csv', default='data/dataset_ready.csv')
    p.add_argument('--model', default='data/resnet50_finetuned.pth')
    p.add_argument('--out-dir', default='data/')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(); rng = np.random.RandomState(args.seed)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    idx = build_image_index(args.data_dir); print(f"Index: {len(idx)} images")
    with open(args.csv) as f:
        rows = [r for r in csv.DictReader(f) if r.get('Source')!='2023'
                and r.get('label_3class','') in {'Healthy','Stress','Other'}]
    val = [r for r in rows if r.get('split')=='val']; print(f"Val: {len(val)}")

    sd = torch.load(args.model, map_location='cpu')
    model = models.resnet50(weights=None); model.fc = nn.Linear(2048, 3)
    model.load_state_dict(sd, strict=False); model.to(DEVICE); model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters())} params)")

    for noise_name, noise_fn in NOISE_MAP.items():
        out_path = os.path.join(args.out_dir, f'X_visual_val_{noise_name}.npy')
        print(f"\n=== {noise_name} ===")
        feats, batch = [], []
        for r in tqdm(val, desc=noise_name):
            pp = r.get('Photo Path','')
            ip = idx.get(os.path.basename(pp.split(';')[0])) if pp else None
            if ip is None or not os.path.exists(ip):
                feats.append(np.zeros(2048, dtype=np.float32)); continue
            batch.append(noise_fn(Image.open(ip).convert('RGB'), rng))
            if len(batch) >= args.batch_size:
                with torch.no_grad():
                    f = extract_features(model, torch.stack([TF(im) for im in batch]).to(DEVICE))
                feats.extend(f.cpu().numpy()); batch = []
        if batch:
            with torch.no_grad():
                f = extract_features(model, torch.stack([TF(im) for im in batch]).to(DEVICE))
            feats.extend(f.cpu().numpy())
        X = np.array(feats, dtype=np.float32)
        np.save(out_path, X); print(f"  {X.shape} -> {out_path} ({X.nbytes>>20}MB)")

if __name__ == '__main__': main()

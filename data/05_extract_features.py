#!/usr/bin/env python3
"""
Step 5 [GPU]: Extract ResNet50 image features for all samples.
Run this on AutoDL (GPU).

Input: dataset_ready.csv (from Step 4) — contains 'Photo Path' column
Output: image_features.npy — dict mapping sample index -> 2048-dim feature vector

Usage:
  python data/05_extract_features.py --input data/dataset_ready.csv --output data/image_features.npy
"""
import os, sys, csv, argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

BASE = r'/Users/rainxu/Downloads/2023-2025 Tomato dataset'

import argparse
def get_base_path(args):
    if args.data_dir:
        return args.data_dir
    return BASE

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image(image_path):
    """Load and preprocess a single image."""
    try:
        img = Image.open(image_path).convert('RGB')
        return TRANSFORM(img)
    except Exception as e:
        print(f'  Warning: cannot load {image_path}: {e}')
        return None


CACHED_IMAGE_DIRS = None

def build_image_index(data_base=None):
    """Build a {filename: full_path} index for all images to avoid repeated os.walk."""
    index = {}
    root_dir = data_base or BASE
    for year in ['2023', '2024', '2025']:
        search_path = os.path.join(root_dir, year)
        for root, dirs, files in os.walk(search_path):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    index[f] = os.path.join(root, f)
    print(f'Indexed {len(index)} images')
    return index


def resolve_image_paths(photo_path, year_str, month_str, data_base=None):
    """Photo Path: 'images/111,211/1725149753_1725149464_4.jpg;...'"""
    global CACHED_IMAGE_DIRS
    if CACHED_IMAGE_DIRS is None:
        CACHED_IMAGE_DIRS = build_image_index(data_base)

    paths = []
    for p in photo_path.split(';'):
        p = p.strip()
        if not p:
            continue
        img_name = os.path.basename(p)
        if img_name in CACHED_IMAGE_DIRS:
            paths.append(CACHED_IMAGE_DIRS[img_name])
    return paths


def extract_features(model, image_paths, device):
    """Extract 2048-dim feature by averaging multiple images for this sample."""
    tensors = []
    for img_path in image_paths:
        tensor = load_image(img_path)
        if tensor is not None:
            tensors.append(tensor)

    if not tensors:
        return np.zeros(2048, dtype=np.float32)

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        features = model(batch).cpu().numpy()
    return features.mean(axis=0)  # Average multiple images per plant-day


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='dataset_ready.csv path')
    parser.add_argument('--output', required=True, help='output .npy path')
    parser.add_argument('--data-dir', default=None, help='Dataset root dir (default: local path)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--use_pca', type=int, default=0,
                        help='If >0, reduce to this dim with PCA after extraction')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load ResNet50 (remove final FC layer for 2048-dim features)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(model.children())[:-1])  # Remove FC + avgpool
    model = model.to(device)
    model.eval()
    print('ResNet50 loaded (output: 2048-dim features)')

    # Load dataset
    with open(args.input) as f:
        rows = list(csv.DictReader(f))
    print(f'Processing {len(rows)} samples')

    # Read CSV file info to resolve image paths
    # We need to know the year and month for each row to resolve photo paths
    features = {}
    failures = 0

    for i, row in enumerate(tqdm(rows, desc='Extracting features')):
        photo_path = row.get('Photo Path', '')
        if not photo_path:
            features[i] = np.zeros(2048, dtype=np.float32)
            failures += 1
            continue

        year_str = row.get('Year', row.get('Source', ''))
        month_str = row['Date'][:6] if 'Date' in row else ''
        img_paths = resolve_image_paths(photo_path, year_str, month_str, get_base_path(args))
        if not img_paths:
            features[i] = np.zeros(2048, dtype=np.float32)
            failures += 1
            continue

        feat = extract_features(model, img_paths, device)
        features[i] = feat

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, features)

    print(f'Saved {len(features)} features to {args.output}')
    print(f'Failed/resolved: {failures}/{len(rows)}')
    print(f'Feature shape: {list(features.values())[0].shape}')

    if args.use_pca > 0:
        from sklearn.decomposition import PCA
        X = np.array([features[i] for i in range(len(rows))])
        pca = PCA(n_components=args.use_pca)
        X_pca = pca.fit_transform(X)
        np.save(args.output.replace('.npy', f'_pca{args.use_pca}.npy'), X_pca)
        print(f'PCA reduced to {args.use_pca} dim, explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Step 6 [GPU]: Fine-tune ResNet50 on tomato stress dataset.
Then extract fine-tuned 2048-dim features for DAFN training.

Run on AutoDL after 05_extract_features.py has verified image paths.

Usage:
  python data/06_finetune_resnet.py \
    --csv data/dataset_ready.csv \
    --data-dir "/root/autodl-tmp/2023-2025 Tomato dataset" \
    --output data/image_features_finetuned.npy \
    --epochs 10
"""
import os, sys, csv, argparse, glob, random
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LABEL_MAP = {'Healthy': 0, 'Stress': 1, 'Other': 2}
NUM_CLASSES = 3

# Build a filename→path index once
def build_image_index(data_base):
    idx = {}
    for year in ['2024', '2025']:
        search = os.path.join(data_base, year)
        for root, dirs, files in os.walk(search):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    idx[f] = os.path.join(root, f)
    return idx


class TomatoImageDataset(Dataset):
    """Loads images on-the-fly from Photo Path strings."""
    def __init__(self, rows, image_index, use_all_images=True):
        self.samples = []
        self.image_index = image_index
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for r in rows:
            label = LABEL_MAP.get(r.get('label_3class', 'Unknown'), -1)
            if label < 0:
                continue
            paths = r.get('Photo Path', '').split(';')
            found = []
            for p in paths:
                p = p.strip()
                if not p:
                    continue
                img_name = os.path.basename(p)
                if img_name in self.image_index:
                    found.append(self.image_index[img_name])
                # Also try searching subdirectories
                for img_key, img_path in self.image_index.items():
                    if p.replace('\\', '/').endswith(img_key):
                        found.append(img_path)
            if found:
                # Use the first available image per sample
                self.samples.append((found[0], label))

        print(f'  Dataset: {len(self.samples)} samples with images')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
        except Exception as e:
            print(f'  Error loading {img_path}: {e}')
            img = torch.zeros(3, 224, 224)
        return img, torch.tensor(label, dtype=torch.long)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--data-dir', required=True, help='Path to the dataset (with 2024/ and 2025/)')
    p.add_argument('--output', required=True, help='Output .npy for fine-tuned features')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=32)
    args = p.parse_args()

    # Load CSV
    with open(args.csv) as f:
        all_rows = list(csv.DictReader(f))
    all_rows = [r for r in all_rows if r.get('Source') != '2023']
    valid = [r for r in all_rows if r.get('label_3class', 'Unknown') in LABEL_MAP]
    train = [r for i, r in enumerate(valid) if r.get('split') == 'train']
    val = [r for i, r in enumerate(valid) if r.get('split') == 'val']
    print(f'Rows: train={len(train)} val={len(val)}')

    # Build image index
    print('Building image index...')
    image_index = build_image_index(args.data_dir)
    print(f'  Found {len(image_index)} images')

    # Create datasets
    train_ds = TomatoImageDataset(train, image_index)
    val_ds = TomatoImageDataset(val, image_index)

    if len(train_ds) == 0:
        print('No images found! Check data-dir path.')
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

    # Model: ResNet50 with fine-tuning
    print(f'Building model on {DEVICE}...')
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze early layers, unfreeze last 2 blocks
    for name, param in model.named_parameters():
        param.requires_grad = False
    # Unfreeze layer4 and layer3 (last 2 blocks)
    for name, param in model.named_parameters():
        if 'layer4' in name or 'layer3' in name:
            param.requires_grad = True

    # Replace classifier
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    # Class weights
    y_train = np.array([LABEL_MAP.get(r['label_3class'], -1) for r in train
                        if LABEL_MAP.get(r['label_3class'], -1) >= 0])
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
    print(f'  Class weights: {dict(zip(range(3), cw))}')

    # Training
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)
    # Different LR for pretrained vs new layers
    params = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'fc' not in n], 'lr': args.lr / 10},
        {'params': model.fc.parameters(), 'lr': args.lr},
    ]
    optimizer = optim.Adam(params, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0
    for ep in range(args.epochs):
        model.train()
        tl, tc, tt = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = outputs.max(1)
            tc += (preds == labels).sum().item()
            tt += labels.size(0)
            tl += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        va, vc, vt = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = outputs.max(1)
                vc += (preds == labels).sum().item()
                vt += labels.size(0)
        vacc = vc / vt
        if vacc > best_acc:
            best_acc = vacc
            torch.save(model.state_dict(), os.path.join(os.path.dirname(args.output), 'resnet50_finetuned.pth'))
        if (ep + 1) % 2 == 0:
            print(f'  Epoch {ep+1}/{args.epochs} | Train Loss: {tl/len(train_loader):.4f} | Train Acc: {tc/tt:.4f} | Val Acc: {vacc:.4f}')

    print(f'Best val acc: {best_acc:.4f}')

    # Load best model and extract features
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(args.output), 'resnet50_finetuned.pth')))
    feature_extractor = nn.Sequential(*list(model.children())[:-1])  # Remove FC
    feature_extractor.eval()

    feat_dict = {}
    with torch.no_grad():
        for i in tqdm(range(len(valid)), desc='Extracting features'):
            row = valid[i]
            paths = row.get('Photo Path', '').split(';')
            found_img = None
            for p in paths:
                p = p.strip()
                if not p:
                    continue
                img_name = os.path.basename(p)
                if img_name in image_index:
                    found_img = image_index[img_name]
                    break
            if found_img:
                try:
                    img = Image.open(found_img).convert('RGB')
                    img_tensor = transforms.Compose([
                        transforms.Resize(256), transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])(img).unsqueeze(0).to(DEVICE)
                    feat = feature_extractor(img_tensor).cpu().numpy().flatten()
                except:
                    feat = np.zeros(2048, dtype=np.float32)
            else:
                feat = np.zeros(2048, dtype=np.float32)
            feat_dict[i] = feat

    np.save(args.output, feat_dict)
    nz = sum(1 for v in feat_dict.values() if np.all(v == 0))
    print(f'Saved {len(feat_dict)} fine-tuned features to {args.output}')
    print(f'  Zero features (no image): {nz}/{len(feat_dict)}')
    print(f'  Best validation accuracy: {best_acc:.4f}')


if __name__ == '__main__':
    main()

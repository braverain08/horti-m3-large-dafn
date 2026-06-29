#!/usr/bin/env python3
"""
Generate Grad-CAM visualizations comparing frozen vs. fine-tuned ResNet50 attention.
Run on AutoDL after fine-tuning (Step 6).

Usage:
  python data/99_gradcam.py \
    --csv data/dataset_ready.csv \
    --data-dir "/root/autodl-tmp/2023-2025 Tomato dataset" \
    --checkpoint data/resnet50_finetuned.pth \
    --output paper_q1/figures/
"""
import os, sys, csv, argparse, random
import numpy as np
from PIL import Image
import torch, torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torchvision.io import read_image

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABEL_MAP = {'Healthy': 0, 'Stress': 1, 'Other': 2}

def build_image_index(data_base):
    idx = {}
    for year in ['2024','2025']:
        for r,d,f in os.walk(os.path.join(data_base, year)):
            for fn in f:
                if fn.lower().endswith('.jpg'): idx[fn] = os.path.join(r,fn)
    return idx

class GradCAM:
    """Simplified Grad-CAM for ResNet50."""
    def __init__(self, model):
        self.model = model.eval()
        self.gradients = None
        self.activations = None
        # Register hooks on layer4
        def fwd_hook(m,i,o): self.activations = o
        def bwd_hook(m,i,o): self.gradients = o[0]
        model.layer4.register_forward_hook(fwd_hook)
        model.layer4.register_full_backward_hook(bwd_hook)

    def generate(self, img_tensor, target_class=None):
        """Generate Grad-CAM heatmap for target_class."""
        img_tensor = img_tensor.unsqueeze(0).requires_grad_(True)
        out = self.model(img_tensor)
        if target_class is None:
            target_class = out.argmax(1).item()
        self.model.zero_grad()
        out[0, target_class].backward()
        pool = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (self.activations * pool).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224,224), mode='bilinear', align_corners=False)
        cam = cam.detach().squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True); p.add_argument('--data-dir', required=True)
    p.add_argument('--checkpoint', required=True); p.add_argument('--output', default='figures')
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load images
    idx = build_image_index(args.data_dir)
    print(f'Index: {len(idx)} images')

    with open(args.csv) as f:
        rows = [r for r in csv.DictReader(f) if r.get('Source')!='2023' and r.get('label_3class') in LABEL_MAP]

    # Build frozen model
    frozen = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for p in frozen.parameters(): p.requires_grad = False
    frozen.fc = nn.Linear(2048, 3)
    frozen = frozen.to(DEVICE).eval()

    # Build fine-tuned model
    ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    ft.fc = nn.Linear(2048, 3)
    ft.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    ft = ft.to(DEVICE).eval()

    cam_frozen = GradCAM(frozen)
    cam_ft = GradCAM(ft)

    tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    # --- Select sample images (3 stress, 3 healthy) ---
    # Pre-filter: only keep samples whose images are actually available on disk
    def _find_image(r):
        for p_str in r.get('Photo Path','').split(';'):
            bn = os.path.basename(p_str.strip())
            if bn in idx:
                return idx[bn]
        return None

    stress_candidates = [(r, _find_image(r)) for r in rows if r['label_3class']=='Stress']
    stress_candidates = [(r, p) for r, p in stress_candidates if p is not None]
    healthy_candidates = [(r, _find_image(r)) for r in rows if r['label_3class']=='Healthy']
    healthy_candidates = [(r, p) for r, p in healthy_candidates if p is not None]

    random.seed(42)
    n_stress = min(3, len(stress_candidates))
    n_healthy = min(3, len(healthy_candidates))
    print(f'Candidates: {len(stress_candidates)} stress, {len(healthy_candidates)} healthy; sampling {n_stress}+{n_healthy}')
    samples = random.sample(stress_candidates, n_stress) + random.sample(healthy_candidates, n_healthy)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print(f'Generating {len(samples)} Grad-CAM visualizations...')
    for si, (r, img_path) in enumerate(samples):
        # img_path is guaranteed non-None from pre-filtering


        img = Image.open(img_path).convert('RGB')
        tensor = tfm(img).to(DEVICE)

        # Get CAMs
        cam_f, cls_f = cam_frozen.generate(tensor)
        cam_t, cls_t = cam_ft.generate(tensor)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # Original
        axes[0].imshow(img.resize((224,224))); axes[0].set_title(f'Original ({r["label_3class"]})')
        axes[0].axis('off')
        # Frozen CAM
        axes[1].imshow(img.resize((224,224))); axes[1].imshow(cam_f, cmap='jet', alpha=0.5)
        axes[1].set_title(f'Frozen (pred={["H","S","O"][cls_f]})')
        axes[1].axis('off')
        # Fine-tuned CAM
        axes[2].imshow(img.resize((224,224))); axes[2].imshow(cam_t, cmap='jet', alpha=0.5)
        axes[2].set_title(f'Fine-tuned (pred={["H","S","O"][cls_t]})')
        axes[2].axis('off')

        plt.tight_layout()
        out_fn = os.path.join(args.output, f'gradcam_{si}_{r["label_3class"]}.png')
        plt.savefig(out_fn, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {out_fn}')

    print(f'Done. Generated {len(samples)} Grad-CAM visualizations in {args.output}/')

if __name__ == '__main__':
    main()

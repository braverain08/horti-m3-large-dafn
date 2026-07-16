#!/usr/bin/env python3
"""Analyze validation image quality (brightness, blur, contrast)."""
import os, sys, csv, argparse, json, re
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def build_image_index(data_base):
    idx = {}
    for year in ['2024','2025']:
        yp = os.path.join(data_base, year)
        if not os.path.isdir(yp): continue
        for r, d, fs in os.walk(yp):
            for f in fs:
                if f.lower().endswith(('.jpg','.jpeg','.png')):
                    idx[f] = os.path.join(r, f)
    return idx

def compute_laplacian_var(gray):
    """Variance of Laplacian via numpy convolution."""
    from scipy import ndimage
    lap = ndimage.laplace(gray.astype(np.float32))
    return lap.var()

def compute_brightness(gray):
    return gray.mean()

def compute_contrast(gray):
    return gray.std()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='/Users/rainxu/Downloads/2023-2025 Tomato dataset')
    p.add_argument('--csv', default='data/dataset_ready.csv')
    p.add_argument('--output', default='experiments/results/quality_analysis.json')
    p.add_argument('--sample', type=int, default=0, help='Only process N images (0=all)')
    args = p.parse_args()
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("Building image index...")
    idx = build_image_index(args.data_dir)
    print(f"  Indexed {len(idx)} images")

    print("Loading CSV (val set)...")
    csv_path = os.path.join(base, args.csv)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get('Source') != '2023']
    valid = [r for r in rows if r.get('label_3class','') in {'Healthy','Stress','Other'}]
    val = [r for r in valid if r.get('split') == 'val']
    print(f"  {len(val)} validation samples")

    results = []
    processed = 0
    for r in tqdm(val, desc='Analyzing'):
        pp = r.get('Photo Path','')
        if not pp: continue
        img_name = os.path.basename(pp.split(';')[0])
        img_path = idx.get(img_name)
        if img_path is None or not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        img_small = img.resize((w//4, h//4), Image.LANCZOS)  # downsample 16x for speed
        gray = np.array(img_small.convert('L'), dtype=np.float32)

        brightness = compute_brightness(gray)
        contrast = compute_contrast(gray)
        try:
            blur = compute_laplacian_var(gray)
        except:
            blur = 0.0

        results.append({
            'filename': os.path.basename(img_path),
            'photo_path': pp.split(';')[0],
            'label': r.get('label_3class',''),
            'plant_id': re.match(r'images/(\d+)', pp.split(';')[0]).group(1) if re.match(r'images/(\d+)', pp) else '',
            'brightness': float(brightness),
            'blur_laplacian': float(blur),
            'contrast': float(contrast),
        })
        processed += 1
        if args.sample and processed >= args.sample:
            break

    print(f"\nProcessed {len(results)} images")

    # Statistics
    b = np.array([r['brightness'] for r in results])
    bl = np.array([r['blur_laplacian'] for r in results])
    c = np.array([r['contrast'] for r in results])

    stats = {
        'count': len(results),
        'brightness': {'mean': float(b.mean()), 'std': float(b.std()), 
                       'p15': float(np.percentile(b, 15)), 'p50': float(np.percentile(b, 50)),
                       'min': float(b.min()), 'max': float(b.max())},
        'blur_laplacian': {'mean': float(bl.mean()), 'std': float(bl.std()),
                           'p15': float(np.percentile(bl, 15)), 'p50': float(np.percentile(bl, 50)),
                           'min': float(bl.min()), 'max': float(bl.max())},
        'contrast': {'mean': float(c.mean()), 'std': float(c.std()),
                     'p15': float(np.percentile(c, 15)), 'p50': float(np.percentile(c, 50)),
                     'min': float(c.min()), 'max': float(c.max())},
        'thresholds': {
            'low_brightness': float(np.percentile(b, 15)),
            'high_blur': float(np.percentile(bl, 15)),
            'low_contrast': float(np.percentile(c, 15)),
        }
    }

    output = {'stats': stats, 'per_image': results}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {args.output}")

    print(f"\n=== Quality Thresholds (15th percentile) ===")
    print(f"  Low brightness < {stats['thresholds']['low_brightness']:.1f}")
    print(f"  High blur (low Laplacian var) < {stats['thresholds']['high_blur']:.1f}")
    print(f"  Low contrast < {stats['thresholds']['low_contrast']:.1f}")
    print(f"\n=== Distribution Summary ===")
    print(f"  Brightness: mean={stats['brightness']['mean']:.1f}±{stats['brightness']['std']:.1f} [{stats['brightness']['min']:.0f}-{stats['brightness']['max']:.0f}]")
    print(f"  Laplacian var: mean={stats['blur_laplacian']['mean']:.1f}±{stats['blur_laplacian']['std']:.1f}")
    print(f"  Contrast: mean={stats['contrast']['mean']:.1f}±{stats['contrast']['std']:.1f}")

if __name__ == '__main__':
    main()

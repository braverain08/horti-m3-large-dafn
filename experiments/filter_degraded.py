#!/usr/bin/env python3
"""Filter validation images into degraded/clean subsets based on quality analysis."""
import os, sys, json, argparse, random
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='experiments/results/quality_analysis.json')
    p.add_argument('--output', default='experiments/results/degraded_subsets')
    p.add_argument('--n-per-class', type=int, default=200, help='Samples per degradation type')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # Load quality analysis
    with open(args.input) as f:
        data = json.load(f)
    images = data['per_image']
    thresh = data['stats']['thresholds']
    print(f"Loaded {len(images)} images with quality metrics")

    # Classify each image
    degraded = {'low_brightness': [], 'blurry': [], 'low_contrast': []}
    clean = []

    for img in images:
        reasons = []
        if img['brightness'] < thresh['low_brightness']:
            reasons.append('low_brightness')
        if img['blur_laplacian'] < thresh['high_blur']:
            reasons.append('blurry')
        if img['contrast'] < thresh['low_contrast']:
            reasons.append('low_contrast')
        
        lbl = img.get('label', '')
        if reasons:
            for r in reasons:
                degraded[r].append((img, r, lbl))
        else:
            clean.append((img, '', lbl))

    print(f"\n=== Degradation Statistics ===")
    for k, v in degraded.items():
        print(f"  {k}: {len(v)} images")
    print(f"  Clean: {len(clean)} images")

    # Sample subsets
    subsets = {}
    for dtype, imgs in degraded.items():
        n = min(args.n_per_class, len(imgs))
        sampled = random.sample(imgs, n)
        subsets[f'degraded_{dtype}'] = sampled
        print(f"  Sampled {n} for {dtype}")

    n_clean = min(args.n_per_class, len(clean))
    subsets['clean'] = random.sample(clean, n_clean)
    print(f"  Sampled {n_clean} for clean")

    # Save subsets
    import csv
    for name, imgs in subsets.items():
        out_path = os.path.join(args.output, f'{name}.csv')
        with open(out_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['filename','photo_path','label','degradation_type','brightness','blur_laplacian','contrast'])
            for img, dtype, lbl in imgs:
                w.writerow([
                    img.get('filename',''),
                    img.get('photo_path',''),
                    lbl,
                    dtype if dtype else 'none',
                    f"{img['brightness']:.1f}",
                    f"{img['blur_laplacian']:.1f}",
                    f"{img['contrast']:.1f}"
                ])
        print(f"  Saved {out_path} ({len(imgs)} rows)")

    # Summary CSV
    sum_path = os.path.join(args.output, 'subset_summary.csv')
    with open(sum_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['subset','count'])
        for name, imgs in sorted(subsets.items()):
            w.writerow([name, len(imgs)])
    print(f"\nSaved subset summary → {sum_path}")

if __name__ == '__main__':
    main()

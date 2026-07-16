#!/usr/bin/env python3
"""Generate visualizations for degradation analysis."""
import os, sys, json, csv, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--quality', default='experiments/results/quality_analysis.json')
    p.add_argument('--eval', default='experiments/results/degraded_eval.json')
    p.add_argument('--output', default='experiments/results/figures/degraded')
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load quality data
    with open(args.quality) as f:
        qdata = json.load(f)
    images = qdata['per_image']
    stats = qdata['stats']
    thresh = stats['thresholds']

    # 1. Distribution histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(axes,
        ['brightness','blur_laplacian','contrast'],
        ['Brightness (Grayscale Mean)', 'Blur (Laplacian Variance)', 'Contrast (Grayscale Std)']):
        vals = np.array([i[key] for i in images])
        ax.hist(vals, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
        ax.axvline(thresh.get(f'low_{key.split()[0].lower()}', thresh.get(f'high_{key.split()[0].lower()}', 0)),
                   color='red', linestyle='--', label=f'Threshold ({thresh.get(f"low_{key.split()[0].lower()}", thresh.get(f"high_{key.split()[0].lower()}", 0)):.0f})')
        ax.set_xlabel(title); ax.set_ylabel('Count')
        ax.legend(); ax.set_title(f'{title} Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'quality_distributions.png'), dpi=150)
    plt.close()
    print(f"Saved quality_distributions.png")

    # 2. Performance by degradation type (bar chart)
    if os.path.exists(args.eval):
        with open(args.eval) as f:
            edata = json.load(f)
        names = list(edata.keys())
        accs = [edata[n]['accuracy']*100 for n in names]
        colors = ['#e74c3c' if 'degraded' in n else '#2ecc71' for n in names]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(names, accs, color=colors, alpha=0.8)
        ax.set_ylabel('Accuracy (%)'); ax.set_title('Vision Model Performance by Degradation Type')
        ax.set_ylim(0, 100)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{acc:.1f}%',
                    ha='center', va='bottom', fontsize=9)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, 'performance_by_degradation.png'), dpi=150)
        plt.close()
        print(f"Saved performance_by_degradation.png")

    # 3. Degradation correlation scatter
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    b = np.array([i['brightness'] for i in images])
    bl = np.array([i['blur_laplacian'] for i in images])
    c = np.array([i['contrast'] for i in images])
    lbls = np.array([0 if i.get('label','')=='Healthy' else 1 for i in images])
    
    for ax, x, y, xl, yl in zip(axes,
        [b, b, bl], [bl, c, c],
        ['Brightness', 'Brightness', 'Blur (Laplacian)'],
        ['Blur (Laplacian)', 'Contrast', 'Contrast']):
        colors = ['#2ecc71' if l==0 else '#e74c3c' for l in lbls]
        ax.scatter(x, y, c=colors, alpha=0.3, s=5)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'quality_correlation.png'), dpi=150)
    plt.close()
    print(f"Saved quality_correlation.png")

    print(f"\nAll figures saved to {args.output}")

if __name__ == '__main__':
    main()

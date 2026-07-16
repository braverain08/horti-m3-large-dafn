#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计显著性检验：对每个基线模型与 DAFN-T 做配对 T 检验。
读取结果文件，输出可直接用于 Table 11 的 p 值和显著性标记。"""
import os, re, sys
import numpy as np
from scipy.stats import ttest_rel
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, 'experiments', 'results')
OUT = os.path.join(RES, 'significance_results.txt')

def read_result(path):
    """读取 result_*.txt 返回 {seed: clean_acc, noisy_acc}"""
    data = {}
    with open(path) as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                data[k.strip()] = float(v) if k.strip() in ['clean_acc','noisy_acc','best_val_acc'] else v
    return data

def load_all():
    """扫描所有结果文件，返回 {model: {seed: {{clean,noisy}_acc}}}"""
    all_models = defaultdict(dict)
    
    # DAFN-T clean (results/result_seed*.txt)
    for f in sorted(os.listdir(RES)):
        m = re.match(r'result_seed(\d+)\.txt', f)
        if m:
            fp = os.path.join(RES, f)
            d = read_result(fp)
            seed = int(m.group(1))
            all_models['DAFN-T'][seed] = {'clean_acc': d.get('best_val_acc', 0)}
    
    # Baselines (results/baselines/result_{Model}_seed*.txt)
    bdir = os.path.join(RES, 'baselines')
    if os.path.isdir(bdir):
        for f in sorted(os.listdir(bdir)):
            m = re.match(r'result_(\w+)_seed(\d+)\.txt', f)
            if m:
                name, seed = m.group(1), int(m.group(2))
                fp = os.path.join(bdir, f)
                d = read_result(fp)
                all_models[name][seed] = {'clean_acc': d.get('clean_acc', 0),
                                          'noisy_acc': d.get('noisy_acc', 0)}
    
    # DAFN-T noisy (table46/result_DAFN-T-Noisy_seed*.txt)
    tdir = os.path.join(RES, 'table46')
    if os.path.isdir(tdir):
        for f in sorted(os.listdir(tdir)):
            m = re.match(r'result_DAFN-T-Noisy_seed(\d+)\.txt', f)
            if m:
                seed = int(m.group(1))
                fp = os.path.join(tdir, f)
                d = read_result(fp)
                # Add noisy key to existing DAFN-T entry
                if seed in all_models['DAFN-T']:
                    all_models['DAFN-T'][seed]['noisy_acc'] = d.get('noisy_acc', 0)
    
    return all_models

def fmt_mean_std(vals):
    """格式化均值±标准差"""
    return f"{np.mean(vals)*100:.2f}±{np.std(vals,ddof=1)*100:.2f}"

def sig_mark(p):
    """显著性标记"""
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'

def paired_ttest(baseline_vals, dafn_vals):
    """配对 T 检验（单侧：DAFN-T > 基线）"""
    if len(baseline_vals) < 2 or len(dafn_vals) < 2:
        return 1.0, '—'
    diffs = np.array(dafn_vals) - np.array(baseline_vals)
    stat, p_two = ttest_rel(dafn_vals, baseline_vals, alternative='greater')
    return stat, p_two

def main():
    models = load_all()
    
    # 模型名称映射
    name_map = {
        'Simple': 'Simple Concat + MLP',
        'CBAM': 'ConcatCBAM',
        'Gated': 'Gated Fusion',
        'CrossAttn': 'CrossAttention Fusion',
        'VisionOnly': 'Vision-Only (ResNet50 FT)',
    }
    baseline_names = [n for n in name_map.keys() if n in models]
    
    ref_model = 'DAFN-T'
    dafn_clean = {s: v.get('clean_acc', 0) for s, v in models[ref_model].items()}
    dafn_noisy = {s: v.get('noisy_acc', 0) for s, v in models[ref_model].items() if 'noisy_acc' in v}
    
    lines = []
    lines.append("=" * 80)
    lines.append(" 统计显著性检验：配对 T 检验（单侧，DAFN-T > 基线）")
    lines.append("=" * 80)
    lines.append("")
    
    # Clean 条件
    lines.append("-" * 80)
    lines.append(" [Clean 条件]")
    lines.append("-" * 80)
    lines.append(f"  DAFN-T: {fmt_mean_std(list(dafn_clean.values()))} (n={len(dafn_clean)})")
    lines.append("")
    
    table_clean = []
    for name in baseline_names:
        display = name_map.get(name, name)
        vals = {s: v.get('clean_acc', 0) for s, v in models[name].items()}
        common_seeds = sorted(set(vals.keys()) & set(dafn_clean.keys()))
        bv = [vals[s] for s in common_seeds]
        dv = [dafn_clean[s] for s in common_seeds]
        
        stat, p = paired_ttest(bv, dv)
        mark = sig_mark(p)
        lines.append(f"  {display:<30s}: {fmt_mean_std(bv):>15s}  vs  "
                     f"{fmt_mean_std(dv):>15s}  p={p:.4f}  {mark}")
        table_clean.append((display, fmt_mean_std(bv), p, mark))
    
    # Noisy 条件
    lines.append("")
    lines.append("-" * 80)
    lines.append(" [Noisy 条件]")
    lines.append("-" * 80)
    if dafn_noisy:
        lines.append(f"  DAFN-T: {fmt_mean_std(list(dafn_noisy.values()))} (n={len(dafn_noisy)})")
    lines.append("")
    
    table_noisy = []
    for name in baseline_names:
        display = name_map.get(name, name)
        vals = {s: v.get('noisy_acc', 0) for s, v in models[name].items()}
        common_seeds = sorted(set(vals.keys()) & set(dafn_noisy.keys()))
        if len(common_seeds) < 2:
            lines.append(f"  {display:<30s}: 数据不足（共同种子少于2个），跳过")
            table_noisy.append((display, '—', 1.0, '—'))
            continue
        bv = [vals[s] for s in common_seeds]
        dv = [dafn_noisy[s] for s in common_seeds]
        
        stat, p = paired_ttest(bv, dv)
        mark = sig_mark(p)
        lines.append(f"  {display:<30s}: {fmt_mean_std(bv):>15s}  vs  "
                     f"{fmt_mean_std(dv):>15s}  p={p:.4f}  {mark}")
        table_noisy.append((display, fmt_mean_std(bv), p, mark))
    
    # Table 11 格式输出
    lines.append("")
    lines.append("=" * 80)
    lines.append(" Table 11 格式化输出")
    lines.append("=" * 80)
    lines.append("")
    lines.append("| Method | Clean Acc. | Noisy Acc. | Parameters |")
    lines.append("|---|---|---|---|")
    
    param_map = {'Simple': '68 K', 'CBAM': '72 K', 'Gated': '78 K',
                 'CrossAttn': '185 K', 'VisionOnly': '25.6 M', 'DAFN-T': '80 K'}
    
    # Main table
    for name in baseline_names:
        display = name_map.get(name, name)
        cv = fmt_mean_std([v.get('clean_acc',0) for v in models[name].values()])
        nv = fmt_mean_std([v.get('noisy_acc',0) for v in models[name].values()])
        params = param_map.get(name, '—')
        # Find p-values
        cp = next((p for n,_,p,_ in table_clean if n==display), 1.0)
        np2 = next((p for n,_,p,_ in table_noisy if n==display), 1.0)
        lines.append(f"| {display} | {cv}{sig_mark(cp)} | {nv}{sig_mark(np2)} | {params} |")
    
    # DAFN-T row
    dafn_cv = fmt_mean_std(list(dafn_clean.values()))
    dafn_nv = fmt_mean_std(list(dafn_noisy.values())) if dafn_noisy else '—'
    lines.append(f"| **DAFN-T (ours)** | **{dafn_cv}** | **{dafn_nv}** | **80 K** |")
    
    lines.append("")
    lines.append("Significance: * p<0.05, ** p<0.01, *** p<0.001, n.s. = not significant")
    lines.append("P-values from one-sided paired t-test (DAFN-T > baseline)")
    
    result = "\n".join(lines)
    print(result)
    
    with open(OUT, 'w') as f:
        f.write(result)
    print(f"\n  Results saved to {OUT}")

if __name__ == '__main__':
    main()

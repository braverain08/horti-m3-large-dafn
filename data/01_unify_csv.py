#!/usr/bin/env python3
"""
Step 1: Unify all growth index CSVs (2023-2025) into a single DataFrame.

Key column difference:
  - 2024/2025: Number, NDVI, RVI, LNC, LNA, LAI, LDW, Plant Height, Stem Diameter, Leaf Width, Leaf Length, Photo Path, Sensor Path
  - 2023: Number, Leaf Area, Leaf Angle, Plant Height, NDVI, RVI, LNC, LNA, LAI, LDW, Photo Path
"""
import csv, os, glob, argparse
from collections import defaultdict

BASE = r'/Users/rainxu/Downloads/2023-2025 Tomato dataset'
TARGET_FEATURES = ['Plant Height', 'Stem Diameter', 'Leaf Width', 'Leaf Length',
                   'NDVI', 'RVI', 'LNC', 'LNA', 'LAI', 'LDW']

def safe_float(v, default=float('nan')):
    try: return float(v) if v else default
    except: return default

def parse_2024_2025(filepath):
    rows = []
    with open(filepath, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            row = {'Number': r.get('Number', '').strip()}
            for feat in TARGET_FEATURES:
                row[feat] = safe_float(r.get(feat, ''))
            row['Photo Path'] = r.get('Photo Path', '').strip()
            row['Sensor Path'] = r.get('Sensor Path', '').strip()
            fname = os.path.basename(filepath).replace('.csv', '')
            row['Year'] = fname[:4]
            row['Date'] = fname
            rows.append(row)
    return rows

def parse_2023(filepath):
    rows = []
    with open(filepath, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            row = {'Number': r.get('Number', '').strip()}
            for feat in TARGET_FEATURES:
                row[feat] = safe_float(r.get(feat, '')) if feat not in ('Stem Diameter','Leaf Width','Leaf Length') else float('nan')
            row['Photo Path'] = r.get('Photo Path', '').strip()
            row['Sensor Path'] = ''
            fname = os.path.basename(filepath).replace('.csv', '')
            row['Year'] = fname[:4]
            row['Date'] = fname
            rows.append(row)
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default=os.path.join(os.path.dirname(__file__), 'unified_agronomic.csv'))
    args = p.parse_args()

    all_rows, stats = [], defaultdict(lambda: {'files': 0, 'rows': 0, 'plants': set()})

    for year in ['2023', '2024', '2025']:
        if year == '2023':
            pattern = os.path.join(BASE, year, '*', 'processed_data', 'growth_index(final)', '*.csv')
        else:
            sub = '2024' if year == '2024' else '2025'
            pattern = os.path.join(BASE, year, sub, '*', 'processed_data', 'growth_index(final)', '*.csv')
        files = sorted(glob.glob(pattern))
        stats[year]['files'] = len(files)
        parse_fn = parse_2023 if year == '2023' else parse_2024_2025
        for fpath in files:
            rows = parse_fn(fpath)
            for r in rows:
                r['Source'] = year
            all_rows.extend(rows)
            stats[year]['rows'] += len(rows)
            for r in rows:
                stats[year]['plants'].add(r['Number'])

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fieldnames = ['Number', 'Year', 'Date', 'Source'] + TARGET_FEATURES + ['Photo Path', 'Sensor Path']
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(all_rows)
    print(f'Written {len(all_rows)} rows to {args.output}')
    for year in ['2023','2024','2025']:
        s = stats[year]
        print(f'  {year}: {s["files"]} files, {s["rows"]} rows, {len(s["plants"])} plants')

if __name__ == '__main__':
    main()

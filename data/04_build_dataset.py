#!/usr/bin/env python3
"""
Step 4: Build final training/validation dataset.
- Merge agronomic data + sensor data + image features
- Plant-wise split
- Handle NaN/missing values
- Output: cleaned CSV for training + optional numpy features

Run order: 01 → 02 → 03 → 04
"""
import os, csv, random, argparse
from collections import defaultdict, Counter
import numpy as np

AGRONOMIC_FEATURES = ['Plant Height', 'Stem Diameter', 'Leaf Width', 'Leaf Length',
                      'NDVI', 'RVI', 'LNC', 'LNA', 'LAI', 'LDW']
SENSOR_FEATURES = ['Air_Temperature', 'Relative_Humidity', 'Light_Intensity',
                   'CO2', 'Soil_Moisture', 'Soil_Temperature']
ALL_FEATURES = AGRONOMIC_FEATURES + SENSOR_FEATURES
LABEL_3CLASS = {'Healthy': 0, 'Stress': 1, 'Other': 2}


def load_agronomic(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_sensors(path):
    """Return {date_str: {field: value}}."""
    data = {}
    if not os.path.exists(path):
        return data
    with open(path) as f:
        for r in csv.DictReader(f):
            d = {'Date': r['Date']}
            for fld in SENSOR_FEATURES:
                try:
                    d[fld] = float(r.get(fld, 'nan'))
                except:
                    d[fld] = float('nan')
            data[r['Date']] = d
    return data


def load_image_features(path):
    """Return {sample_id: np.array} or empty if file doesn't exist."""
    if not os.path.exists(path):
        return {}
    data = np.load(path, allow_pickle=True).item()
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=os.path.join(os.path.dirname(__file__), 'dataset_ready.csv'))
    parser.add_argument('--min_samples_per_plant', type=int, default=1)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--fill_na', action='store_true', default=True,
                        help='Fill NaN with column median')
    parser.add_argument('--use_2023', action='store_true',
                        help='Include 2023 data (missing 3 agronomic features)')
    args = parser.parse_args()

    DATA_DIR = os.path.dirname(__file__)

    # Load data
    agronomic = load_agronomic(os.path.join(DATA_DIR, 'labeled_agronomic.csv'))
    sensors = load_sensors(os.path.join(DATA_DIR, 'sensor_daily.csv'))

    # Merge sensor data
    for row in agronomic:
        date = row['Date']
        if date in sensors:
            for fld in SENSOR_FEATURES:
                row[fld] = sensors[date].get(fld, float('nan'))
        else:
            for fld in SENSOR_FEATURES:
                row[fld] = float('nan')

    # Filter options
    if not args.use_2023:
        agronomic = [r for r in agronomic if r['Source'] != '2023']
        print(f'After excluding 2023: {len(agronomic)} rows')

    if args.fill_na:
        # Fill NaN with median of each numeric column
        for fld in ALL_FEATURES:
            vals = []
            for r in agronomic:
                try:
                    v = float(r.get(fld, 'nan'))
                    if not np.isnan(v):
                        vals.append(v)
                except:
                    pass
            if vals:
                median_val = np.median(vals)
                for r in agronomic:
                    try:
                        if np.isnan(float(r.get(fld, 'nan'))):
                            r[fld] = median_val
                    except:
                        r[fld] = median_val

    # Plant-wise split: collect unique plants
    plants = sorted(set(r['Number'] for r in agronomic))
    random.seed(42)
    random.shuffle(plants)
    n_val = max(1, int(len(plants) * args.val_split))
    val_plants = set(plants[:n_val])
    train_plants = set(plants[n_val:])

    print(f'Plant-wise split: {len(train_plants)} train, {len(val_plants)} val')
    print(f'  Train plants: {sorted(train_plants)[:10]}...')
    print(f'  Val plants: {sorted(val_plants)[:10]}...')

    # Assign split
    for row in agronomic:
        row['split'] = 'val' if row['Number'] in val_plants else 'train'

    # Filter unknown labels
    valid = [r for r in agronomic if r['label_3class'] != 'Unknown']
    print(f'Rows with valid labels: {len(valid)} / {len(agronomic)}')

    # Stats
    for split_name in ['train', 'val']:
        split_rows = [r for r in valid if r['split'] == split_name]
        label_dist = Counter(r['label_3class'] for r in split_rows)
        print(f'  {split_name}: {len(split_rows)} samples, labels={dict(label_dist)}')

    # Write output
    fieldnames = list(agronomic[0].keys())
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(valid)

    print(f'\nWritten {len(valid)} rows to {args.output}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Post-processing script: after finetuned_logits.npy is ready,
extract all .npy files in the format required by experiments.

Usage:
    python data/prepare_experiment_data.py [--finetuned PATH]
    
Outputs in data/:
    X_visual_train.npy, X_visual_val.npy  (N, 2048) — fine-tuned ResNet50 features
    X_agro_train.npy, X_agro_val.npy      (N, 10)
    X_sensor_train.npy, X_sensor_val.npy  (N, 6)
    y_train.npy, y_val.npy                (N,) — binary: 0=Healthy, 1=Stress+Other
    plant_id_train.npy, plant_id_val.npy  (N,) — from Photo Path
"""
import os, sys, csv, re, argparse
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, 'dataset_ready.csv')

AGRO_FEATS = ['Plant Height', 'Stem Diameter', 'Leaf Width', 'Leaf Length',
              'NDVI', 'RVI', 'LNC', 'LNA', 'LAI', 'LDW']
SENSOR_FEATS = ['Air_Temperature', 'Relative_Humidity', 'Light_Intensity',
                'CO2', 'Soil_Moisture', 'Soil_Temperature']
LABEL_MAP = {'Healthy': 0, 'Stress': 1, 'Other': 1}  # binary

def extract_plant_id(photo_path):
    """Extract plant number from 'images/PLANT,*/...' path."""
    if not photo_path:
        return -1
    m = re.match(r'images/(\d+)', photo_path.split(';')[0])
    return int(m.group(1)) if m else -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetuned', default=os.path.join(DATA_DIR, 'finetuned_logits.npy'),
                        help='Path to finetuned_logits.npy')
    args = parser.parse_args()

    # 1. Load CSV
    print("Loading CSV...")
    with open(CSV_PATH) as f:
        all_rows = list(csv.DictReader(f))
    all_rows = [r for r in all_rows if r.get('Source') != '2023']
    valid = [r for r in all_rows if r.get('label_3class','Unknown') in {'Healthy','Stress','Other'}]
    train = [r for i, r in enumerate(valid) if r.get('split') == 'train']
    val  = [r for i, r in enumerate(valid) if r.get('split') == 'val']
    print(f"  Train: {len(train)}, Val: {len(val)}")

    # 2. Agronomic features
    print("Extracting agronomic features...")
    X_agro_train = np.array([[float(r.get(f,0)) for f in AGRO_FEATS] for r in train], dtype=np.float32)
    X_agro_val   = np.array([[float(r.get(f,0)) for f in AGRO_FEATS] for r in val], dtype=np.float32)
    np.save(os.path.join(DATA_DIR, 'X_agro_train.npy'), X_agro_train)
    np.save(os.path.join(DATA_DIR, 'X_agro_val.npy'), X_agro_val)
    print(f"  X_agro_train: {X_agro_train.shape}")
    print(f"  X_agro_val:   {X_agro_val.shape}")

    # 3. Sensor features
    print("Extracting sensor features...")
    X_sen_train = np.array([[float(r.get(f,0)) for f in SENSOR_FEATS] for r in train], dtype=np.float32)
    X_sen_val   = np.array([[float(r.get(f,0)) for f in SENSOR_FEATS] for r in val], dtype=np.float32)
    np.save(os.path.join(DATA_DIR, 'X_sensor_train.npy'), X_sen_train)
    np.save(os.path.join(DATA_DIR, 'X_sensor_val.npy'), X_sen_val)
    print(f"  X_sensor_train: {X_sen_train.shape}")
    print(f"  X_sensor_val:   {X_sen_val.shape}")

    # 4. Binary labels
    print("Extracting labels (binary)...")
    y_train = np.array([LABEL_MAP[r['label_3class']] for r in train], dtype=np.int64)
    y_val   = np.array([LABEL_MAP[r['label_3class']] for r in val], dtype=np.int64)
    np.save(os.path.join(DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(DATA_DIR, 'y_val.npy'), y_val)
    c0_t = np.sum(y_train==0); c1_t = np.sum(y_train==1)
    c0_v = np.sum(y_val==0);   c1_v = np.sum(y_val==1)
    print(f"  y_train: {y_train.shape} (H:{c0_t} S:{c1_t})")
    print(f"  y_val:   {y_val.shape}   (H:{c0_v} S:{c1_v})")

    # 5. Plant IDs
    print("Extracting plant IDs from Photo Path...")
    def get_plant_ids(rows):
        return np.array([extract_plant_id(r.get('Photo Path','')) for r in rows], dtype=np.int64)
    pid_train = get_plant_ids(train)
    pid_val   = get_plant_ids(val)
    np.save(os.path.join(DATA_DIR, 'plant_id_train.npy'), pid_train)
    np.save(os.path.join(DATA_DIR, 'plant_id_val.npy'), pid_val)
    print(f"  plant_id_train: {len(pid_train)} ({len(np.unique(pid_train))} unique)")
    print(f"  plant_id_val:   {len(pid_val)} ({len(np.unique(pid_val))} unique)")

    # 6. Visual features (from fine-tuned model)
    if os.path.exists(args.finetuned):
        print(f"Loading finetuned features from {args.finetuned}...")
        d = np.load(args.finetuned, allow_pickle=True).item()
        # Dict {index: 2048-d feature}
        print(f"  Loaded {len(d)} features")

        def get_vis_feats(rows):
            arr = []
            for i, r in enumerate(rows):
                idx = int(r.get('Number', i))
                feat = np.array(d.get(idx, np.zeros(2048, dtype=np.float32)), dtype=np.float32)
                arr.append(feat.flatten())
            return np.array(arr, dtype=np.float32)

        X_vis_train = get_vis_feats(train)
        X_vis_val   = get_vis_feats(val)
        np.save(os.path.join(DATA_DIR, 'X_visual_train.npy'), X_vis_train)
        np.save(os.path.join(DATA_DIR, 'X_visual_val.npy'), X_vis_val)
        print(f"  X_visual_train: {X_vis_train.shape}")
        print(f"  X_visual_val:   {X_vis_val.shape}")
    else:
        print(f"  finetuned_logits.npy not found at {args.finetuned}")
        print("  Visual features will be prepared after fine-tuning completes.")

    print("\nDone! Prepared .npy files for experiments.")

if __name__ == '__main__':
    main()

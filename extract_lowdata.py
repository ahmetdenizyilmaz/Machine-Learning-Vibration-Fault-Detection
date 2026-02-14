"""
Memory-efficient feature extraction from HuggingFace dataset.
Run in Colab: %run extract_lowdata.py
"""
import numpy as np
import json, os, sys
from collections import defaultdict

sys.path.insert(0, '/content/mlvfd')
os.chdir('/content/mlvfd')

from feature_extraction import extract_features
from datasets import load_dataset
import pandas as pd

SIGNAL_FIELD = 'low_data'
HF_DATASET = 'adyady/bearing-fault-dataset'

ds = load_dataset(HF_DATASET, split='train')

print(f'Total rows: {len(ds)}')
print('Building file_name index...')

# Build index: file_name -> list of row indices
file_index = defaultdict(list)
for i in range(len(ds)):
    file_index[ds[i]['file_name']].append(i)

print(f'Unique segments: {len(file_index)}')
print('Extracting features...')

rows = []
for j, (seg_name, indices) in enumerate(file_index.items()):
    try:
        seg_rows = [ds[i] for i in indices]
        axes_data = {}
        for row in seg_rows:
            signal = row[SIGNAL_FIELD]
            if signal is not None:
                axes_data[row['axis']] = np.asarray(signal, dtype=np.float64)

        x = axes_data.get('x')
        if x is None:
            continue

        y = axes_data.get('y')
        z = axes_data.get('z')
        first = seg_rows[0]
        fs = float(first.get('target_sample_rate') or first.get('original_sample_rate'))

        rpm = None
        meta_str = first.get('metadata_json', '{}')
        if isinstance(meta_str, str) and meta_str:
            try:
                meta = json.loads(meta_str)
                rpm = meta.get('rpm') or meta.get('operating_conditions', {}).get('rpm')
            except:
                pass
        if rpm is not None:
            rpm = float(rpm)

        feats = extract_features(x, y, z, fs=fs, rpm=rpm)
        feats['filename'] = seg_name
        feats['fault_category'] = first.get('fault_category', '')
        feats['fault_type'] = first.get('fault_type', '')
        feats['dataset'] = first.get('source_dataset', '')
        feats['sample_rate_hz'] = fs
        rows.append(feats)
    except Exception as e:
        if j < 5:
            print(f'  SKIP {seg_name}: {e}')

    if (j + 1) % 1000 == 0:
        print(f'  Processed {j+1}/{len(file_index)} segments')

df = pd.DataFrame(rows)
df.to_csv('features_low_data.csv', index=False)
print(f'\nDone! Extracted {len(df)} segments -> features_low_data.csv')

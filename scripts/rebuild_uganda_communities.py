"""
Rebuild uganda_communities.json and update nexis_activations.json act dicts
to use RCT site keys (geo_long_lat_key 1-331) instead of UgandaGeocodes keys.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Load CSV
csv_path = ROOT / "data/uganda/UgandaDataProcessed.csv"
df = pd.read_csv(csv_path)

# Geographic language labels (matching UG_COL in index.html)
LANG_LABELS = {
    1: 'Alur',
    2: 'Lugbara',
    3: 'Madi',
    4: 'Karamojong',
    5: 'Teso',
    6: 'Langi',
    7: 'Other',
}

# Use ALL Prithvi sites (all 331 RCT keys), not just geolocated_failed=FALSE.
# geolocated_failed=TRUE sites still have approximate coordinates (geo_long/geo_lat)
# which are accurate enough for a map dot. Filtering to 182 would leave SAE top/bottom
# activation sites off the map because they often fall in the 149 non-geocoded sites.
npz = np.load(ROOT / "results/uganda/prithvi_l5_1024/site_features.npz")
prithvi_keys = set(npz['site_keys'].astype(int))

sites = (
    df.groupby('geo_long_lat_key')
    .agg(lat=('geo_lat', 'first'), lon=('geo_long', 'first'), lang_group=('lang_group', 'first'))
    .reset_index()
    .rename(columns={'geo_long_lat_key': 'geokey'})
)
# Keep only sites that have Prithvi features, drop the 1 site with NaN coords
sites = sites[sites['geokey'].isin(prithvi_keys)].dropna(subset=['lat', 'lon'])

print(f"Total sites with Prithvi features: {len(sites)}")
print(f"Geokey range: {sites['geokey'].min()} – {sites['geokey'].max()}")
print("\nSites per lang_group:")
for lg, cnt in sites.groupby('lang_group').size().items():
    print(f"  lang_group={lg} ({LANG_LABELS.get(lg,'?')}): {cnt} sites")

# Build communities list
communities = []
for _, row in sites.iterrows():
    communities.append({
        "geokey": int(row['geokey']),
        "lat": round(float(row['lat']), 6),
        "lon": round(float(row['lon']), 6),
        "lang": LANG_LABELS.get(int(row['lang_group']), 'Other'),
    })

out_path = ROOT / "docs/assets/uganda_communities.json"
with open(out_path, 'w') as f:
    json.dump(communities, f, separators=(',', ':'))
print(f"\nWrote {len(communities)} communities to {out_path}")

# --------------------------------------------------------------------------
# Rebuild act dicts in nexis_activations.json
# --------------------------------------------------------------------------
act_json_path = ROOT / "docs/assets/nexis_activations.json"
with open(act_json_path) as f:
    nexis = json.load(f)

# Build site->lang_group lookup
site_lang = dict(zip(sites['geokey'].astype(int), sites['lang_group'].astype(int)))

# Load SAE features
npz = np.load(ROOT / "results/uganda/prithvi_l5_1024/site_features.npz")
feat = npz['site_features']   # (331, 1024)
keys = npz['site_keys'].astype(int)   # (331,)
site_to_row = {k: i for i, k in enumerate(keys)}

print("\nRebuilding act dicts for uganda_skilled...")

for mod_key, mod in nexis.get('uganda_skilled', {}).items():
    new_act = {}

    if mod_key.startswith('lang_'):
        lg = int(mod_key.split('_')[1])
        for sk, lg_val in site_lang.items():
            new_act[str(sk)] = 1.0 if lg_val == lg else 0.0
        n_active = sum(1 for v in new_act.values() if v > 0)
        print(f"  {mod_key}: {n_active} active sites (lang_group={lg} = {LANG_LABELS.get(lg,'?')})")

    elif mod_key.isdigit():
        feat_idx = int(mod_key)
        col = feat[:, feat_idx]
        for i, sk in enumerate(keys):
            v = float(col[i])
            if v != 0.0:
                new_act[str(int(sk))] = v
        n_active = len(new_act)
        print(f"  Z_{mod_key}: {n_active} active sites (feature col {feat_idx})")

    else:
        print(f"  {mod_key}: skipped (unknown format)")
        continue

    mod['act'] = new_act

# Also handle uganda_biz if present
for mod_key, mod in nexis.get('uganda_biz', {}).items():
    new_act = {}

    if mod_key.startswith('lang_'):
        lg = int(mod_key.split('_')[1])
        for sk, lg_val in site_lang.items():
            new_act[str(sk)] = 1.0 if lg_val == lg else 0.0
        n_active = sum(1 for v in new_act.values() if v > 0)
        print(f"  biz/{mod_key}: {n_active} active sites")

    elif mod_key.isdigit():
        feat_idx = int(mod_key)
        col = feat[:, feat_idx]
        for i, sk in enumerate(keys):
            v = float(col[i])
            if v != 0.0:
                new_act[str(int(sk))] = v
        print(f"  biz/Z_{mod_key}: {len(new_act)} active sites")

    else:
        print(f"  biz/{mod_key}: skipped")
        continue

    mod['act'] = new_act

with open(act_json_path, 'w') as f:
    json.dump(nexis, f, separators=(',', ':'))
print(f"\nWrote updated nexis_activations.json")

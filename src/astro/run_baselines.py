"""
Run unimodal baselines + supervised ceiling on cached features.
No CA/CP training — just evaluation on frozen representations.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# ============================================================================
# Paths
# ============================================================================
MULTIDESA_ROOT = os.environ.get('MULTIDESA_ROOT', str(Path(__file__).parent))
KEPLER_CACHE = os.path.join(MULTIDESA_ROOT, 'logs/cross_modal/cached_features')
TESS_CACHE = os.path.join(MULTIDESA_ROOT, 'logs/cross_modal/cached_features_tess')

AGE_CATALOG = os.environ.get('AGE_CATALOG', '/path/to/ages_dataset_gyro.csv')
NSS_CATALOG = os.environ.get('NSS_CATALOG', '/path/to/nss_dataset.csv')
TESS_CROSSMATCH = os.environ.get('TESS_CROSSMATCH', os.path.join(MULTIDESA_ROOT, 'dataset/multimodal_catalog.parquet'))

EVAL_LABELS = ['teff', 'logg', 'feh', 'prot']


# ============================================================================
# Data loading
# ============================================================================
def load_kepler_data():
    splits = {}
    for name in ['train', 'test']:
        z_l = np.load(os.path.join(KEPLER_CACHE, f'z_lamost_{name}.npy'))
        z_k = np.load(os.path.join(KEPLER_CACHE, f'z_kepler_{name}.npy'))
        meta = pd.read_parquet(os.path.join(KEPLER_CACHE, f'meta_{name}.parquet'))
        label_cols = [c for c in EVAL_LABELS if c in meta.columns]
        splits[name] = {
            'z_a': z_l, 'z_b': z_k,
            'labels': meta[label_cols].values.astype(np.float32),
            'label_names': label_cols,
            'meta': meta,
        }
        print(f"  Kepler {name}: {len(z_l)} samples, LAMOST={z_l.shape[1]}d, Kepler={z_k.shape[1]}d")
    return splits['train'], splits['test']


def load_tess_data():
    splits = {}
    for name in ['train', 'test']:
        z_l = np.load(os.path.join(TESS_CACHE, f'z_lamost_{name}.npy'))
        z_t = np.load(os.path.join(TESS_CACHE, f'z_tess_{name}.npy'))
        meta = pd.read_parquet(os.path.join(TESS_CACHE, f'meta_{name}.parquet'))
        label_cols = [c for c in EVAL_LABELS if c in meta.columns]
        splits[name] = {
            'z_a': z_l, 'z_b': z_t,
            'labels': meta[label_cols].values.astype(np.float32),
            'label_names': label_cols,
            'meta': meta,
        }
        print(f"  TESS {name}: {len(z_l)} samples, LAMOST={z_l.shape[1]}d, TESS={z_t.shape[1]}d")
    return splits['train'], splits['test']


# ============================================================================
# Label loading (shared-signal tasks)
# ============================================================================
def load_age_labels(kids):
    age_df = pd.read_csv(AGE_CATALOG, low_memory=False)
    age_df = age_df.dropna(subset=['final_age', 'age_error'])
    age_map = dict(zip(age_df['KID'].values, age_df['final_age'].values))
    return np.array([age_map.get(k, np.nan) for k in kids], dtype=np.float32)


def load_binarity_labels(kids):
    nss_df = pd.read_csv(NSS_CATALOG, low_memory=False)
    nss_df = nss_df[nss_df['binarity_class'] != 3]
    nss_df.loc[nss_df['binarity_class'] == 4, 'binarity_class'] = 3
    nss_df['binary_hard'] = (nss_df['binarity_class'] > 0).astype(int)
    nss_map = dict(zip(nss_df['KID'].values, nss_df['binary_hard'].values))
    return np.array([nss_map.get(k, np.nan) for k in kids], dtype=np.float32)


def get_kids_kepler(meta):
    return meta['KID'].values


def get_kids_tess(meta):
    if 'KID' in meta.columns and meta['KID'].notna().any():
        return meta['KID'].fillna(-1).astype(int).values
    # Fallback: gaia -> KID mapping
    df = pd.read_parquet(TESS_CROSSMATCH, columns=['gaia_source_id', 'KID'])
    df = df.dropna(subset=['KID'])
    df['KID'] = df['KID'].astype(int)
    df = df.drop_duplicates('gaia_source_id')
    gaia2kid = dict(zip(df['gaia_source_id'].values, df['KID'].values))
    return np.array([gaia2kid.get(g, -1) for g in meta['gaia_source_id'].values])


# ============================================================================
# Probes
# ============================================================================
def ridge_probe(X_tr, y_tr, X_te, y_te):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xte = scaler.transform(X_te)
    model = Ridge(alpha=1.0).fit(Xtr, y_tr)
    pred = model.predict(Xte)
    return {'r2': float(r2_score(y_te, pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_te, pred)))}


def logreg_probe(X_tr, y_tr, X_te, y_te):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xte = scaler.transform(X_te)
    model = LogisticRegression(max_iter=5000, C=1.0, class_weight='balanced')
    model.fit(Xtr, y_tr.astype(int))
    pred = model.predict(Xte)
    return {'accuracy': float(accuracy_score(y_te, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_te, pred)),
            'f1': float(f1_score(y_te, pred, average='binary'))}


# ============================================================================
# Supervised ceiling (MLP)
# ============================================================================
def supervised_mlp_regression(X_train, y_train, X_test, y_test,
                              hidden_dim=256, epochs=200, lr=1e-3,
                              patience=20, batch_size=256):
    from torch.utils.data import TensorDataset, DataLoader

    scaler = StandardScaler()
    X_tr = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    X_te = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_te_np = y_test

    device = torch.device('cuda')
    model = nn.Sequential(
        nn.Linear(X_tr.shape[1], hidden_dim), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden_dim, 1),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    best_loss, wait, best_state = float('inf'), 0, None
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.mse_loss(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(X_te.to(device)).cpu().numpy().ravel()
        loss = float(np.mean((pred - y_te_np) ** 2))
        if loss < best_loss:
            best_loss, wait = loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(X_te.to(device)).cpu().numpy().ravel()
    return {'r2': float(r2_score(y_te_np, pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_te_np, pred)))}


def supervised_mlp_classification(X_train, y_train, X_test, y_test,
                                  hidden_dim=256, epochs=200, lr=1e-3,
                                  patience=20, batch_size=256):
    from torch.utils.data import TensorDataset, DataLoader

    scaler = StandardScaler()
    X_tr = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    X_te = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_te_np = y_test.astype(int)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    device = torch.device('cuda')
    model = nn.Sequential(
        nn.Linear(X_tr.shape[1], hidden_dim), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden_dim, 1),
    ).to(device)
    pos_weight = pos_weight.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    best_loss, wait, best_state = float('inf'), 0, None
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pos_weight).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(X_te.to(device)).cpu().numpy().ravel()
        preds = (logits > 0).astype(int)
        bal_acc = balanced_accuracy_score(y_te_np, preds)
        loss = -bal_acc
        if loss < best_loss:
            best_loss, wait = loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(X_te.to(device)).cpu().numpy().ravel()
    preds = (logits > 0).astype(int)
    return {'accuracy': float(accuracy_score(y_te_np, preds)),
            'balanced_accuracy': float(balanced_accuracy_score(y_te_np, preds)),
            'f1': float(f1_score(y_te_np, preds, average='binary'))}


# ============================================================================
# Main evaluation
# ============================================================================
def evaluate_experiment(name, train, test, get_kids_fn, weak_name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    z_a_tr, z_b_tr = train['z_a'], train['z_b']
    z_a_te, z_b_te = test['z_a'], test['z_b']
    labels_tr, labels_te = train['labels'], test['labels']
    label_names = train['label_names']

    concat_tr = np.concatenate([z_a_tr, z_b_tr], axis=1)
    concat_te = np.concatenate([z_a_te, z_b_te], axis=1)

    reps = {
        'frozen_concat': (concat_tr, concat_te),
        'lamost_only': (z_a_tr, z_a_te),
        f'{weak_name}_only': (z_b_tr, z_b_te),
    }

    results = {}

    # --- Modality-specific labels (Ridge probe) ---
    print(f"\n--- Modality-specific labels (Ridge probe) ---")
    for rep_name, (Xtr, Xte) in reps.items():
        for j, lbl in enumerate(label_names):
            mask_tr = ~np.isnan(labels_tr[:, j])
            mask_te = ~np.isnan(labels_te[:, j])
            if mask_tr.sum() < 50 or mask_te.sum() < 20:
                continue
            res = ridge_probe(Xtr[mask_tr], labels_tr[mask_tr, j],
                              Xte[mask_te], labels_te[mask_te, j])
            res['n_test'] = int(mask_te.sum())
            key = f'{rep_name}/{lbl}'
            results[key] = res
            print(f"  {key}: R²={res['r2']:.4f}")

    # Supervised ceiling for modality-specific
    print(f"\n--- Supervised ceiling (modality-specific) ---")
    for j, lbl in enumerate(label_names):
        mask_tr = ~np.isnan(labels_tr[:, j])
        mask_te = ~np.isnan(labels_te[:, j])
        if mask_tr.sum() < 50 or mask_te.sum() < 20:
            continue
        print(f"  Training MLP for {lbl}...")
        res = supervised_mlp_regression(
            concat_tr[mask_tr], labels_tr[mask_tr, j],
            concat_te[mask_te], labels_te[mask_te, j])
        res['n_test'] = int(mask_te.sum())
        key = f'supervised_ceiling/{lbl}'
        results[key] = res
        print(f"  {key}: R²={res['r2']:.4f}")

    # --- Shared-signal tasks (age, binarity) ---
    kids_tr = get_kids_fn(train['meta'])
    kids_te = get_kids_fn(test['meta'])
    age_tr = load_age_labels(kids_tr)
    age_te = load_age_labels(kids_te)
    bin_tr = load_binarity_labels(kids_tr)
    bin_te = load_binarity_labels(kids_te)

    n_age_tr = (~np.isnan(age_tr)).sum()
    n_age_te = (~np.isnan(age_te)).sum()
    n_bin_tr = (~np.isnan(bin_tr)).sum()
    n_bin_te = (~np.isnan(bin_te)).sum()
    print(f"\n--- Shared-signal labels ---")
    print(f"  age: train={n_age_tr}, test={n_age_te}")
    print(f"  binarity: train={n_bin_tr}, test={n_bin_te}")

    # Age regression
    mask_tr_age = ~np.isnan(age_tr)
    mask_te_age = ~np.isnan(age_te)
    if mask_tr_age.sum() >= 50 and mask_te_age.sum() >= 20:
        for rep_name, (Xtr, Xte) in reps.items():
            res = ridge_probe(Xtr[mask_tr_age], age_tr[mask_tr_age],
                              Xte[mask_te_age], age_te[mask_te_age])
            res['n_train'] = int(mask_tr_age.sum())
            res['n_test'] = int(mask_te_age.sum())
            key = f'{rep_name}/age'
            results[key] = res
            print(f"  {key}: R²={res['r2']:.4f}")

        # Supervised ceiling
        print(f"  Training MLP for age...")
        res = supervised_mlp_regression(
            concat_tr[mask_tr_age], age_tr[mask_tr_age],
            concat_te[mask_te_age], age_te[mask_te_age])
        res['n_train'] = int(mask_tr_age.sum())
        res['n_test'] = int(mask_te_age.sum())
        results['supervised_ceiling/age'] = res
        print(f"  supervised_ceiling/age: R²={res['r2']:.4f}")

    # Binarity classification
    mask_tr_bin = ~np.isnan(bin_tr)
    mask_te_bin = ~np.isnan(bin_te)
    if mask_tr_bin.sum() >= 50 and mask_te_bin.sum() >= 20:
        for rep_name, (Xtr, Xte) in reps.items():
            res = logreg_probe(Xtr[mask_tr_bin], bin_tr[mask_tr_bin],
                               Xte[mask_te_bin], bin_te[mask_te_bin])
            res['n_train'] = int(mask_tr_bin.sum())
            res['n_test'] = int(mask_te_bin.sum())
            key = f'{rep_name}/binarity'
            results[key] = res
            print(f"  {key}: bal_acc={res['balanced_accuracy']:.4f}, f1={res['f1']:.4f}")

        # Supervised ceiling
        print(f"  Training MLP for binarity...")
        res = supervised_mlp_classification(
            concat_tr[mask_tr_bin], bin_tr[mask_tr_bin],
            concat_te[mask_te_bin], bin_te[mask_te_bin])
        res['n_train'] = int(mask_tr_bin.sum())
        res['n_test'] = int(mask_te_bin.sum())
        results['supervised_ceiling/binarity'] = res
        print(f"  supervised_ceiling/binarity: bal_acc={res['balanced_accuracy']:.4f}, f1={res['f1']:.4f}")

    return results


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    all_results = {}

    # LAMOST x Kepler
    print("\nLoading LAMOST x Kepler cached features...")
    kepler_train, kepler_test = load_kepler_data()
    all_results['kepler'] = evaluate_experiment(
        'LAMOST x Kepler', kepler_train, kepler_test,
        get_kids_kepler, 'kepler')

    # LAMOST x TESS
    print("\nLoading LAMOST x TESS cached features...")
    tess_train, tess_test = load_tess_data()
    all_results['tess'] = evaluate_experiment(
        'LAMOST x TESS', tess_train, tess_test,
        get_kids_tess, 'tess')

    # Save
    out_path = os.path.join(MULTIDESA_ROOT, 'logs/cross_modal/baseline_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for exp_name, results in all_results.items():
        print(f"\n  {exp_name.upper()}")
        print(f"  {'-'*50}")
        for key in sorted(results.keys()):
            val = results[key]
            if 'balanced_accuracy' in val:
                print(f"    {key:40s} bal_acc={val['balanced_accuracy']:.4f}  f1={val['f1']:.4f}")
            elif 'r2' in val:
                print(f"    {key:40s} R²={val['r2']:.4f}")


if __name__ == '__main__':
    main()

"""Diagnostic: compare canonical stored spectra vs recovery_analysis run for
LAMOST x Kepler. Read-only. No estimation/classifier logic modified."""
import sys, os, hashlib, glob
import numpy as np
import pandas as pd
sys.path.insert(0, '/rg/perets_prj/ilay.kamai/mm_align_vs_pred')
sys.path.insert(0, '/rg/perets_prj/ilay.kamai/MultiDESA/analysis')

from src.analyze_phase_diagram import (
    _make_cv_folds, _r2_ridge_1d, _r2_ridge_2d, _classify_signal_elbow, recovery_count)
import src.recovery_analysis as RA

NPZ = '/rg/perets_prj/ilay.kamai/MultiDESA/analysis/multimodal_comparison/phase_lamost_x_kepler_classification_data.npz'
LAM = '/rg/perets_prj/ilay.kamai/MultiDESA/logs/lamost/2026-02-12/44644'
KEP = '/rg/perets_prj/ilay.kamai/MultiDESA/logs/kepler/2026-02-12-07-02'


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def breakpoint_of(r2):
    """elbow n_signal (best_k) = # signal from the elbow classifier (unchanged)."""
    return int(_classify_signal_elbow(np.asarray(r2), log_transform=False)['n_signal'])


# ── canonical (stored) ──────────────────────────────────────────────────────
d = np.load(NPZ, allow_pickle=True)
c_cca_sv = d['cca_sv'].astype(float)
c_cca_r2 = d['cca_r2'].astype(float)              # (50,1)
c_cca_sig = d['cca_is_signal'].astype(bool)
c_cp_sv = d['cp_direct_sv'].astype(float)
c_cp_sum = d['cp_direct_sum_r2'].astype(float)    # (50,)
c_cp_sig = d['cp_direct_is_signal'].astype(bool)

# ── recovery_analysis run (fresh) ───────────────────────────────────────────
ds = RA.load_astro('kepler')
n_match = ds['X'].shape[0]
Z1c, Z2c = RA._prereduce_and_center(ds['X'], ds['Y'])
lab = ds['labels']
folds = _make_cv_folds(Z1c.shape[0], cv=RA.CV)
NCOMP = ds['n_components']
reg = 1e-6

# CCA (mirror cca_recovery, keep r2)
n = Z1c.shape[0]
Z1 = Z1c - Z1c.mean(0); Z2 = Z2c - Z2c.mean(0)
Sxx = Z1.T@Z1/n + reg*np.eye(Z1.shape[1]); Syy = Z2.T@Z2/n + reg*np.eye(Z2.shape[1])
Sxy = Z1.T@Z2/n
Sxi = RA._eigh_inv_sqrt(Sxx); Syi = RA._eigh_inv_sqrt(Syy)
M = Sxi@Sxy@Syi
P, phi, Qt = np.linalg.svd(M, full_matrices=False)
nc = min(NCOMP, len(phi)); phi = np.clip(phi[:nc], 0, 1)
Wx = Sxi@P[:, :nc]; Wy = Syi@Qt.T[:, :nc]
m_cca_r2 = np.zeros((nc, lab.shape[1]))
for j in range(nc):
    m_cca_r2[j] = _r2_ridge_2d(np.column_stack([Z1@Wx[:, j], Z2@Wy[:, j]]), lab, folds)
m_cca_sig = _classify_signal_elbow(m_cca_r2, log_transform=False)['is_signal']

# A-SVD (mirror asvd_recovery, keep r2)
Syx = Z2.T@Z1/n
A = Syx@Sxi
Ua_, sigma_a, Vt_a = np.linalg.svd(A, full_matrices=False)
nc2 = min(NCOMP, len(sigma_a)); sigma_a = sigma_a[:nc2]
Wsrc = Sxi@Vt_a.T[:, :nc2]
m_cp_r2 = np.zeros((nc2, lab.shape[1]))
for j in range(nc2):
    m_cp_r2[j] = _r2_ridge_1d(Z1@Wsrc[:, j], lab, folds)
m_cp_sig = _classify_signal_elbow(m_cp_r2, log_transform=False)['is_signal']

m_cca_sum = m_cca_r2.sum(1); m_cp_sum = m_cp_r2.sum(1)
c_cca_sum = c_cca_r2.sum(1)


def dump(tag, c_sv, c_sum, c_sig, m_sv, m_sum, m_sig):
    print(f"\n{'='*104}\n{tag}: per-component  (canon vs recovery)\n{'='*104}")
    print(f"{'idx':>3} | {'canon_sv':>10} {'mine_sv':>10} {'|Δsv|':>9} | "
          f"{'canon_R2':>9} {'mine_R2':>9} | {'c_sig':>5} {'m_sig':>5}")
    print('-'*104)
    K = max(len(c_sv), len(m_sv))
    for i in range(K):
        cs = c_sv[i] if i < len(c_sv) else np.nan
        ms = m_sv[i] if i < len(m_sv) else np.nan
        cr = c_sum[i] if i < len(c_sum) else np.nan
        mr = m_sum[i] if i < len(m_sum) else np.nan
        csig = int(c_sig[i]) if i < len(c_sig) else -1
        msig = int(m_sig[i]) if i < len(m_sig) else -1
        mark = '  <<' if (i < len(c_sig) and i < len(m_sig) and c_sig[i] != m_sig[i]) else ''
        print(f"{i:>3} | {cs:>10.5f} {ms:>10.5f} {abs(cs-ms):>9.5f} | "
              f"{cr:>9.5f} {mr:>9.5f} | {csig:>5} {msig:>5}{mark}")


dump('CCA', c_cca_sv, c_cca_sum, c_cca_sig, phi, m_cca_sum, m_cca_sig)
dump('A-SVD', c_cp_sv, c_cp_sum, c_cp_sig, sigma_a, m_cp_sum, m_cp_sig)

# ── STEP 3 answers ──────────────────────────────────────────────────────────
print(f"\n{'#'*104}\nSTEP 3 ANSWERS\n{'#'*104}")
K = min(len(c_cca_sv), len(phi))
absd = np.abs(c_cca_sv[:K]-phi[:K]); reld = absd/np.maximum(np.abs(c_cca_sv[:K]), 1e-12)
print(f"a) CCA singular values: max|Δ|={absd.max():.6f} @ idx {int(absd.argmax())}; "
      f"max rel Δ={reld.max():.6f} @ idx {int(reld.argmax())}")
Ka = min(len(c_cp_sv), len(sigma_a))
absa = np.abs(c_cp_sv[:Ka]-sigma_a[:Ka]); rela = absa/np.maximum(np.abs(c_cp_sv[:Ka]), 1e-12)
print(f"   A-SVD singular values: max|Δ|={absa.max():.6f} @ idx {int(absa.argmax())}; "
      f"max rel Δ={rela.max():.6f} @ idx {int(rela.argmax())}")
print(f"b) CCA R^2 at index 11:  canon={c_cca_sum[11]:.6f}   mine={m_cca_sum[11]:.6f}")
print(f"   CCA elbow breakpoint (n_signal): canon(stored)={int(d['cca_n_signal'])}  "
      f"canon(reclassify stored R2)={breakpoint_of(c_cca_r2)}  mine={breakpoint_of(m_cca_r2)}")
print(f"   -> index 11 is {'BELOW' if 11>=breakpoint_of(c_cca_r2) else 'WITHIN'} canon breakpoint, "
      f"{'BELOW' if 11>=breakpoint_of(m_cca_r2) else 'WITHIN'} mine breakpoint")
print(f"c) canon CCA signal set contains index 11? {bool(c_cca_sig[11])}  "
      f"(mine? {bool(m_cca_sig[11])})")
print(f"   canon cca_is_signal idx = {list(np.where(c_cca_sig)[0])}")
print(f"   mine  cca_is_signal idx = {list(np.where(m_cca_sig)[0])}")

# Δ̂ / r̂ / floor each
for tag, sv, sig in [('canon CCA', c_cca_sv, c_cca_sig), ('mine  CCA', phi, m_cca_sig),
                     ('canon ASVD', c_cp_sv, c_cp_sig), ('mine  ASVD', sigma_a, m_cp_sig)]:
    rec = recovery_count(sv, sig)
    dd = (sv[sig].min()/sv[~sig].max()) if sig.any() and (~sig).any() else float('nan')
    print(f"   {tag}: k̂={rec['k_hat']} r̂={rec['r_hat']} floor={rec['floor']:.5f}@idx{rec['floor_index']} Δ̂={dd:.5f}")

# ── STEP 4 config ───────────────────────────────────────────────────────────
print(f"\n{'#'*104}\nSTEP 4 CONFIG\n{'#'*104}")
def files_of(logdir):
    out = []
    for pat in ('svd/U_test.npy', 'svd/S_test.npy', 'svd/features_output_space.npy',
                'svd/features_input_space.npy', '*_features.npy', '*test_preds.csv', '*_preds.csv'):
        for f in sorted(glob.glob(os.path.join(logdir, pat))):
            out.append(f)
    seen = set(); uniq = [x for x in out if not (x in seen or seen.add(x))]
    return uniq
for name, ld in [('LAMOST', LAM), ('Kepler', KEP)]:
    print(f"[{name}] {ld}")
    for f in files_of(ld):
        try:
            if f.endswith('.npy'):
                shp = np.load(f, mmap_mode='r').shape
            else:
                shp = ('csv', sum(1 for _ in open(f)))
            print(f"   {os.path.relpath(f, ld):40s} shape={shp} md5={md5(f)} mtime={pd.Timestamp(os.path.getmtime(f), unit='s')}")
        except Exception as e:
            print(f"   {f}: ERR {e}")
print(f"n_matched: canon=891  mine={n_match}")
print(f"n_components: canon=50 mine={NCOMP} | ε: canon=1e-6 mine={reg} | pca_prereduce: canon=100 mine=100")
print(f"shared label: canon={d['shared_label_names'].tolist()} mine=log g | NaN handling: fillna(col mean)")
import numpy, scipy, sklearn
print(f"versions THIS run: numpy {numpy.__version__} scipy {scipy.__version__} sklearn {sklearn.__version__}")

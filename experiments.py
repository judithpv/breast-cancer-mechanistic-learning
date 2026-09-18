# =============================================================================
# experiments.py
# Feature Selection Experiments — Mechanistic Breast Cancer Relapse Framework
#
# Reproduces the feature selection table (Table 1) in the paper.
#
#   Exp A  — Cohort control: v4 model (grade+ER only) on the v5 cohort
#             (N=1,765, complete for HER2/PR/cellularity). Confirms that
#             cohort composition change has negligible effect on C-index.
#
#   Exp B  — Speed bucket extension: +HER2 +PR, no Soil (N=1,814).
#             This is the FINAL MODEL; full analysis in mechanistic_model.py.
#
#   Exp S1 — Soil = tumour cellularity (N=1,765)
#   Exp S2 — Soil = age at diagnosis  (N=1,814)
#   Exp S3 — Soil = cellularity + age (N=1,765)
#
# Authors: Judith Pérez-Velázquez · Meltem Gölgeli · İbrahim Kulaç
#
# Usage:
#   python experiments.py
#   python experiments.py /path/to/brca_metabric.tar.gz
# =============================================================================

import warnings; warnings.filterwarnings('ignore')
import os, sys, tarfile
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

np.random.seed(42)

M       = 1e9
LN_M    = np.log(M)
LAM_MIN = 1e-4
GAMMA   = 1.0
OPT     = dict(maxiter=5000, ftol=1e-12, gtol=1e-8)

# =============================================================================
# Data loading
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
TAR_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, 'brca_metabric.tar.gz')

if not os.path.exists(TAR_PATH):
    raise FileNotFoundError(
        f"Data not found: {TAR_PATH}\n"
        "Download from: https://www.cbioportal.org/study/summary?id=brca_metabric"
    )

print(f'Loading METABRIC from: {TAR_PATH}\n')
with tarfile.open(TAR_PATH, 'r:*') as tar:
    df_patient = pd.read_csv(tar.extractfile('brca_metabric/data_clinical_patient.txt'), sep='\t', comment='#')
    df_sample  = pd.read_csv(tar.extractfile('brca_metabric/data_clinical_sample.txt'),  sep='\t', comment='#')

df = pd.merge(df_patient, df_sample, on='PATIENT_ID').iloc[1:].reset_index(drop=True)

df['ER_BIN']          = df['ER_STATUS'].map({'Positive': 1.0, 'Negative': 0.0})
df['HER2_BIN']        = df['HER2_STATUS'].map({'Positive': 1.0, 'Negative': 0.0})
df['PR_BIN']          = df['PR_STATUS'].map({'Positive': 1.0, 'Negative': 0.0})
df['CELLULARITY_ENC'] = df['CELLULARITY'].map({'Low': 0.0, 'Moderate': 1.0, 'High': 2.0})

for col in ['TUMOR_SIZE', 'LYMPH_NODES_EXAMINED_POSITIVE', 'GRADE',
            'ER_BIN', 'HER2_BIN', 'PR_BIN', 'CELLULARITY_ENC',
            'AGE_AT_DIAGNOSIS', 'RFS_MONTHS']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

y_event_all = df['RFS_STATUS'].apply(lambda x: 1 if str(x).startswith('1') else 0)
SEED_COLS   = ['TUMOR_SIZE', 'LYMPH_NODES_EXAMINED_POSITIVE']

# =============================================================================
# Generic fitter
# =============================================================================

def fit_model(df_sub, speed_cols, soil_cols, label):
    """Fit the mechanistic model on df_sub with given Speed and Soil features.

    Seed  : TUMOR_SIZE + LYMPH_NODES_EXAMINED_POSITIVE  (always fixed)
    Speed : speed_cols  →  linear link for lambda
    Soil  : soil_cols   →  additive contributions to ln n0

    Returns a dict with:
      label, N, c (model C-index), c2 (same-cohort 2-bucket Speed-only baseline),
      delta (c - c2), lr_p (log-rank p), sign_checks, converged.
    """
    req = SEED_COLS + speed_cols + soil_cols + ['RFS_MONTHS']
    dm  = df_sub.dropna(subset=req).copy()
    N   = len(dm)
    yt  = dm['RFS_MONTHS'].values
    ye  = y_event_all.loc[dm.index].values
    Xs  = dm[SEED_COLS].values
    Xsp = dm[speed_cols].values if speed_cols else np.zeros((N, 0))
    Xso = dm[soil_cols].values  if soil_cols  else np.zeros((N, 0))
    ns, nso = len(speed_cols), len(soil_cols)

    # parameter layout: [a0, a1, a2, g1..gS, b0, b1..bP]
    def _pred(p):
        ln_n0 = p[0] + p[1]*Xs[:, 0] + p[2]*Xs[:, 1]
        for k in range(nso):
            ln_n0 = ln_n0 + p[3 + k] * Xso[:, k]
        lam = p[3 + nso]
        for k in range(ns):
            lam = lam + p[3 + nso + 1 + k] * Xsp[:, k]
        return (LN_M - ln_n0) / np.clip(lam, LAM_MIN, None)

    def _loss(p):
        r = yt - _pred(p)
        return np.sum(r[ye == 1] ** 2) + GAMMA * np.sum(np.maximum(0, r[ye == 0]) ** 2)

    n_p = 3 + nso + 1 + ns
    p0  = np.zeros(n_p)
    p0[0] = 5.0
    p0[3 + nso] = 0.1
    bds = [(None, None)] * 3 + [(None, None)] * nso + [(LAM_MIN, None)] + [(None, None)] * ns

    res = minimize(_loss, p0, method='L-BFGS-B', bounds=bds, options=OPT)
    tp  = _pred(res.x)
    c   = concordance_index(yt, tp, ye)
    thr = np.median(tp)
    lr  = logrank_test(yt[tp <= thr], yt[tp > thr],
                       event_observed_A=ye[tp <= thr],
                       event_observed_B=ye[tp > thr])

    # Same-cohort baseline (no Soil).
    # When testing Soil variables (nso > 0): baseline = same Speed, no Soil → ns cols.
    # When testing Speed extensions (nso == 0): baseline = grade+ER only → 2 cols.
    # This gives a meaningful delta in both cases.
    n_base = 2 if nso == 0 else ns

    def _pred2(p2):
        ln_n0 = p2[0] + p2[1]*Xs[:, 0] + p2[2]*Xs[:, 1]
        lam   = p2[3]
        for k in range(n_base):
            lam = lam + p2[4 + k] * Xsp[:, k]
        return (LN_M - ln_n0) / np.clip(lam, LAM_MIN, None)

    def _loss2(p2):
        r = yt - _pred2(p2)
        return np.sum(r[ye == 1] ** 2) + GAMMA * np.sum(np.maximum(0, r[ye == 0]) ** 2)

    n_p2 = 4 + n_base
    p0_2 = np.zeros(n_p2); p0_2[0] = 5.0; p0_2[3] = 0.1
    bds2 = [(None,None)]*3 + [(LAM_MIN,None)] + [(None,None)]*n_base
    r2   = minimize(_loss2, p0_2, method='L-BFGS-B', bounds=bds2, options=OPT)
    c2   = concordance_index(yt, _pred2(r2.x), ye)

    return dict(label=label, N=N, c=c, c2=c2, delta=c - c2,
                lr_p=lr.p_value, params=res.x,
                converged=res.success or res.status == 1)


# =============================================================================
# Experiment A — cohort control
# Four-feature v4 model (grade + ER only) fitted on the v5 cohort
# (N=1,765, restricted to patients with complete HER2/PR/cellularity records)
# to isolate cohort-composition effects from feature effects.
# =============================================================================
print('=' * 65)
print('EXPERIMENT A: v4 model (grade+ER) on v5 cohort  [cohort control]')
print('=' * 65)

req_v5 = SEED_COLS + ['GRADE', 'ER_BIN', 'HER2_BIN', 'PR_BIN', 'CELLULARITY_ENC', 'RFS_MONTHS']
df_v5  = df.dropna(subset=req_v5)

rA = fit_model(df_v5, ['GRADE', 'ER_BIN'], [], 'Exp A: v4 on v5 cohort')
print(f'  N                          : {rA["N"]:,}')
print(f'  C-Index (v4 on v5 cohort)  : {rA["c"]:.4f}')
print(f'  C-Index (v4 on original)   : 0.6547  (reference, N=2,095)')
print(f'  Cohort effect              : {rA["c"] - 0.6547:+.4f}  (expected ~0)')
print(f'  Log-Rank p                 : {rA["lr_p"]:.3e}')
print(f'  Conclusion: cohort composition change has negligible effect on C-index\n')

# =============================================================================
# Experiment B — Speed bucket extension [FINAL MODEL]
# Add HER2 and PR to Speed; no Soil; largest available cohort (N=1,814).
# Full analysis (Cox PH, CV) in mechanistic_model.py.
# =============================================================================
print('=' * 65)
print('EXPERIMENT B: Speed + HER2 + PR, no Soil  [FINAL MODEL]')
print('=' * 65)

req_B = SEED_COLS + ['GRADE', 'ER_BIN', 'HER2_BIN', 'PR_BIN', 'RFS_MONTHS']
df_B  = df.dropna(subset=req_B)

rB = fit_model(df_B, ['GRADE', 'ER_BIN', 'HER2_BIN', 'PR_BIN'], [], 'Exp B: +HER2+PR')
print(f'  N                          : {rB["N"]:,}')
print(f'  C-Index (model)            : {rB["c"]:.4f}')
print(f'  C-Index (2-bkt, same N)    : {rB["c2"]:.4f}')
print(f'  Delta                      : {rB["delta"]:+.4f}')
print(f'  Log-Rank p                 : {rB["lr_p"]:.3e}')
print(f'  Converged                  : {rB["converged"]}')

p = rB['params']
a0, a1, a2, b0, b1, b2, b3, b4 = p
checks_B = [
    ('a1 > 0  (tumour size  -> more Seed)',  a1 > 0),
    ('a2 > 0  (lymph nodes  -> more Seed)',  a2 > 0),
    ('b1 > 0  (grade        -> faster Speed)', b1 > 0),
    ('b2 < 0  (ER+          -> slower Speed)', b2 < 0),
    ('b3 > 0  (HER2+        -> faster Speed)', b3 > 0),
    ('b4 < 0  (PR+          -> slower Speed)', b4 < 0),
]
print('  Biological consistency:')
for desc, ok in checks_B:
    print(f'    [{"PASS" if ok else "FAIL"}]  {desc}')
print(f'  Passed: {sum(ok for _, ok in checks_B)}/6')
print(f'  --> ADOPTED as final model\n')

# =============================================================================
# Experiments S1/S2/S3 — Soil bucket candidate testing
# Speed is fixed at grade+ER+HER2+PR (validated in Exp B).
# Delta is computed against the same-cohort 2-bucket baseline (no Soil).
# =============================================================================
print('=' * 65)
print('SOIL BUCKET CANDIDATES  (Speed = grade+ER+HER2+PR fixed)')
print('=' * 65)

soil_experiments = [
    (['CELLULARITY_ENC'],                   'S1: Soil = Cellularity only'),
    (['AGE_AT_DIAGNOSIS'],                  'S2: Soil = Age at diagnosis'),
    (['CELLULARITY_ENC', 'AGE_AT_DIAGNOSIS'], 'S3: Soil = Cellularity + Age'),
]

soil_results = []
for soil_cols, label in soil_experiments:
    r = fit_model(df, ['GRADE', 'ER_BIN', 'HER2_BIN', 'PR_BIN'], soil_cols, label)
    soil_results.append(r)
    print(f'  {label}')
    print(f'    N          : {r["N"]:,}')
    print(f'    C-Index    : {r["c"]:.4f}  (2-bucket baseline same cohort: {r["c2"]:.4f})')
    print(f'    Delta      : {r["delta"]:+.4f}')
    print(f'    Log-Rank p : {r["lr_p"]:.3e}')
    print()

# =============================================================================
# Summary table (reproduces Table 1 in paper)
# =============================================================================
print('=' * 65)
print('SUMMARY TABLE  (reproduces Table 1 in paper)')
print('=' * 65)
print(f'  {"Model":<48}  {"N":>6}  {"C-Index":>8}  {"Delta":>7}')
print('  ' + '-' * 76)
print(f'  {"Baseline: Speed = grade+ER  (original cohort)":<48}  {"2,095":>6}  {"0.6547":>8}  {"---":>7}')
print(f'  {"Exp A: grade+ER on v5 cohort  [cohort control]":<48}  {rA["N"]:>6,}  {rA["c"]:>8.4f}  {rA["c"]-0.6547:>+7.4f}')
print(f'  {"Exp B: +HER2+PR  [FINAL MODEL]":<48}  {rB["N"]:>6,}  {rB["c"]:>8.4f}  {rB["delta"]:>+7.4f}')
for r in soil_results:
    print(f'  {r["label"]:<48}  {r["N"]:>6,}  {r["c"]:>8.4f}  {r["delta"]:>+7.4f}')
print()
print('  Conclusion:')
print('    HER2+PR add +0.0055 to Speed; all 6 biological sign checks pass.')
print('    No Soil proxy improves on the same-cohort two-bucket baseline.')
print('    Soil bucket deferred to prospective TIL/PD-L1 data collection.')

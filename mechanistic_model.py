# =============================================================================
# mechanistic_model.py
# A Scalable Mechanistic Learning Framework for Breast Cancer Relapse
#
# Final model: Seed (tumour size, lymph nodes) + Speed (grade, ER, HER2, PR)
# 8-parameter L-BFGS-B inference; Cox PH multivariate baseline; 5-fold CV.
#
# Authors: Judith Pérez-Velázquez · Meltem Gölgeli · İbrahim Kulaç
#   TU München · TOBB University · Koç University School of Medicine
#
# Usage:
#   python mechanistic_model.py
#   python mechanistic_model.py /path/to/brca_metabric.tar.gz
#
# Data: METABRIC via cBioPortal
#   https://www.cbioportal.org/study/summary?id=brca_metabric
# =============================================================================

import warnings; warnings.filterwarnings('ignore')
import os, sys, tarfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import platform
print(f'Python     : {sys.version.split()[0]}')
print(f'Platform   : {platform.system()} {platform.release()}')
print(f'NumPy      : {np.__version__}')
print(f'Pandas     : {pd.__version__}')
print(f'SciPy      : {scipy.__version__}')
import lifelines as _lf; print(f'Lifelines  : {_lf.__version__}')
import matplotlib as _mpl; print(f'Matplotlib : {_mpl.__version__}')
print()

np.random.seed(42)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 10, 'figure.dpi': 150,
})

# =============================================================================
# Constants
# =============================================================================
M       = 1e9          # detection threshold; ~1 cm lesion (~10^9 cells)
LN_M    = np.log(M)   # = 20.7233; ln(10^9); absorbed into alpha_0 during inference
LAM_MIN = 1e-4         # positivity floor for lambda (biological feasibility)
GAMMA   = 1.0          # censoring penalty weight (fixed; see §2.4)
K_FOLDS = 5            # cross-validation folds
SEED    = 42

print(f'Detection threshold M  : {M:.2e} cells')
print(f'ln(M)                  : {LN_M:.4f}')
print(f'Censoring penalty gamma: {GAMMA}   (fixed)')
print(f'CV folds               : {K_FOLDS}')
print()

# =============================================================================
# 1. Data loading
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
TAR_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, 'brca_metabric.tar.gz')

if not os.path.exists(TAR_PATH):
    raise FileNotFoundError(
        f"Data not found: {TAR_PATH}\n"
        "Download brca_metabric.tar.gz from: "
        "https://www.cbioportal.org/study/summary?id=brca_metabric"
    )

print(f'Loading METABRIC from: {TAR_PATH}')
with tarfile.open(TAR_PATH, 'r:*') as tar:
    df_patient = pd.read_csv(tar.extractfile('brca_metabric/data_clinical_patient.txt'), sep='\t', comment='#')
    df_sample  = pd.read_csv(tar.extractfile('brca_metabric/data_clinical_sample.txt'),  sep='\t', comment='#')

df = pd.merge(df_patient, df_sample, on='PATIENT_ID').iloc[1:].reset_index(drop=True)
print(f'Merged dataset: {len(df):,} rows, {df.shape[1]} columns\n')

# =============================================================================
# 2. Feature encoding and analytic cohort
# =============================================================================
df['ER_BIN']   = df['ER_STATUS'].map({'Positive': 1.0, 'Negative': 0.0})
df['HER2_BIN'] = df['HER2_STATUS'].map({'Positive': 1.0, 'Negative': 0.0})
df['PR_BIN']   = df['PR_STATUS'].map({'Positive': 1.0, 'Negative': 0.0})

SEED_COLS  = ['TUMOR_SIZE', 'LYMPH_NODES_EXAMINED_POSITIVE']
SPEED_COLS = ['GRADE', 'ER_BIN', 'HER2_BIN', 'PR_BIN']
TIME_COL   = 'RFS_MONTHS'

for col in SEED_COLS + SPEED_COLS + [TIME_COL]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

y_event_all = df['RFS_STATUS'].apply(lambda x: 1 if str(x).startswith('1') else 0)

df_model = df.dropna(subset=SEED_COLS + SPEED_COLS + [TIME_COL]).copy()
df_model['EVENT'] = y_event_all.loc[df_model.index].values

y_time  = df_model[TIME_COL].values
y_event = df_model['EVENT'].values
X_seed  = df_model[SEED_COLS].values
X_speed = df_model[SPEED_COLS].values
N = len(df_model)

print(f'Analytic cohort : {N:,} patients')
print(f'  Events        : {y_event.sum():,}  ({100*y_event.mean():.1f}%)')
print(f'  Censored      : {(1-y_event).sum():,}  ({100*(1-y_event.mean()):.1f}%)')
print(f'  Median follow-up: {np.median(y_time):.1f} months\n')

# =============================================================================
# 3. Mechanistic model functions
# =============================================================================

def predict_t(params, Xs, Xsp):
    """Deterministic projection T_pred = (ln M - ln n0) / lambda.  Eq. (5)."""
    a0, a1, a2, b0, b1, b2, b3, b4 = params
    ln_n0 = a0 + a1*Xs[:, 0] + a2*Xs[:, 1]
    lam   = np.clip(b0 + b1*Xsp[:, 0] + b2*Xsp[:, 1] + b3*Xsp[:, 2] + b4*Xsp[:, 3],
                    LAM_MIN, None)
    return (LN_M - ln_n0) / lam


def hinge_loss(params, Xs, Xsp, t_obs, event):
    """Censoring-aware hinge-MSE loss.  Eq. (10)."""
    t_pred = predict_t(params, Xs, Xsp)
    resid  = t_obs - t_pred
    return (np.sum(resid[event == 1] ** 2)
            + GAMMA * np.sum(np.maximum(0.0, resid[event == 0]) ** 2))


THETA_0  = np.array([5.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
BOUNDS   = [(None, None), (None, None), (None, None),
            (LAM_MIN, None), (None, None), (None, None), (None, None), (None, None)]
OPT_OPTS = dict(maxiter=5000, ftol=1e-12, gtol=1e-8)

# =============================================================================
# 4. Full-cohort parameter inference
# =============================================================================
print('=' * 60)
print('FULL-COHORT INFERENCE (L-BFGS-B, 8 parameters)')
print('=' * 60)

res = minimize(hinge_loss, THETA_0, args=(X_seed, X_speed, y_time, y_event),
               method='L-BFGS-B', bounds=BOUNDS, options=OPT_OPTS)

if not (res.success or res.status == 1):
    print(f'WARNING: optimiser — {res.message}')
else:
    print(f'Converged ({res.nit} iterations, loss = {res.fun:.2f})\n')

THETA_HAT = res.x
a0, a1, a2, b0, b1, b2, b3, b4 = THETA_HAT

print('Inferred parameter vector:')
for name, val in [
    ('a0  Seed intercept',     a0),
    ('a1  tumour size',        a1),
    ('a2  lymph nodes',        a2),
    ('b0  Speed intercept',    b0),
    ('b1  grade',              b1),
    ('b2  ER status',          b2),
    ('b3  HER2 status',        b3),
    ('b4  PR status',          b4),
]:
    print(f'  {name:<25} = {val:+.6f}')

print('\nBiological consistency (6 non-intercept checks):')
checks = [
    ('a1 > 0  (size  -> more Seed)',     a1 > 0),
    ('a2 > 0  (nodes -> more Seed)',     a2 > 0),
    ('b1 > 0  (grade -> faster Speed)',  b1 > 0),
    ('b2 < 0  (ER+   -> slower Speed)', b2 < 0),
    ('b3 > 0  (HER2+ -> faster Speed)', b3 > 0),
    ('b4 < 0  (PR+   -> slower Speed)', b4 < 0),
]
for desc, ok in checks:
    print(f'  [{"PASS" if ok else "FAIL"}]  {desc}')
print(f'  Passed: {sum(ok for _, ok in checks)}/6\n')

lam_vals = np.clip(
    b0 + b1*X_speed[:, 0] + b2*X_speed[:, 1] + b3*X_speed[:, 2] + b4*X_speed[:, 3],
    LAM_MIN, None)
print(f'Sensitivity dT_pred/dX_nodes ~ {-a2/np.mean(lam_vals):.2f} months per additional positive node\n')

t_pred_full = predict_t(THETA_HAT, X_seed, X_speed)
c_mech_insample = concordance_index(y_time, t_pred_full, y_event)
print(f'In-sample C-Index (mechanistic): {c_mech_insample:.4f}')

# =============================================================================
# 5. Cox PH multivariate baseline (same 6 features, in-sample)
#    This is the key comparison against the standard Cox baseline.
# =============================================================================
print()
print('=' * 60)
print('COX PH BASELINE (same 6 features, in-sample)')
print('=' * 60)

cox_df = df_model[SEED_COLS + SPEED_COLS].copy()
cox_df['T'] = y_time
cox_df['E'] = y_event

cph = CoxPHFitter(penalizer=0.0)
cph.fit(cox_df, duration_col='T', event_col='E')
c_cox_insample = cph.concordance_index_

print(f'In-sample C-Index (Cox PH, 6 features): {c_cox_insample:.4f}')
print('\nCox PH log-hazard ratios:')
print(cph.summary[['coef', 'exp(coef)', 'p']].rename(
    columns={'coef': 'log-HR', 'exp(coef)': 'HR', 'p': 'p-value'}).round(4).to_string())

# =============================================================================
# 6. 5-fold cross-validated C-index (mechanistic and Cox PH)
# =============================================================================
print()
print('=' * 60)
print(f'{K_FOLDS}-FOLD CROSS-VALIDATED C-INDEX')
print('=' * 60)

rng  = np.random.default_rng(SEED)
perm = rng.permutation(N)
folds = [perm[i * N // K_FOLDS : (i + 1) * N // K_FOLDS] for i in range(K_FOLDS)]

mech_cv_c = []
cox_cv_c  = []

for fold_i, test_idx in enumerate(folds):
    train_idx = np.concatenate([folds[j] for j in range(K_FOLDS) if j != fold_i])

    # Mechanistic model — fit on training fold, predict on test fold
    res_cv = minimize(
        hinge_loss, THETA_0,
        args=(X_seed[train_idx], X_speed[train_idx], y_time[train_idx], y_event[train_idx]),
        method='L-BFGS-B', bounds=BOUNDS, options=OPT_OPTS)
    t_test = predict_t(res_cv.x, X_seed[test_idx], X_speed[test_idx])
    c_m = concordance_index(y_time[test_idx], t_test, y_event[test_idx])
    mech_cv_c.append(c_m)

    # Cox PH — fit on training fold, predict on test fold
    tr_df = pd.DataFrame(np.hstack([X_seed[train_idx], X_speed[train_idx]]),
                         columns=SEED_COLS + SPEED_COLS)
    tr_df['T'] = y_time[train_idx]
    tr_df['E'] = y_event[train_idx]
    te_df = pd.DataFrame(np.hstack([X_seed[test_idx], X_speed[test_idx]]),
                         columns=SEED_COLS + SPEED_COLS)
    cph_cv = CoxPHFitter(penalizer=0.0)
    cph_cv.fit(tr_df, duration_col='T', event_col='E')
    hz = cph_cv.predict_partial_hazard(te_df).values
    c_c = concordance_index(y_time[test_idx], -hz, y_event[test_idx])
    cox_cv_c.append(c_c)

    print(f'  Fold {fold_i+1}:  mechanistic = {c_m:.4f}   Cox PH = {c_c:.4f}')

c_mech_cv = np.mean(mech_cv_c)
c_cox_cv  = np.mean(cox_cv_c)
print(f'\n  Mean {K_FOLDS}-fold CV:  mechanistic = {c_mech_cv:.4f}   Cox PH = {c_cox_cv:.4f}')
delta_cv = c_mech_cv - c_cox_cv
print(f'  Delta (mechanistic - Cox): {delta_cv:+.4f}  '
      f'({"mechanistic is better" if delta_cv > 0 else "Cox PH is better by interpretability price"})')

# =============================================================================
# 7. C-index summary table
# =============================================================================
print()
print('=' * 60)
print('C-INDEX SUMMARY')
print('=' * 60)

c_size  = concordance_index(y_time, -X_seed[:, 0],  y_event)
c_nodes = concordance_index(y_time, -X_seed[:, 1],  y_event)
c_grade = concordance_index(y_time, -X_speed[:, 0], y_event)
c_er    = concordance_index(y_time,  X_speed[:, 1], y_event)
c_her2  = concordance_index(y_time, -X_speed[:, 2], y_event)
c_pr    = concordance_index(y_time,  X_speed[:, 3], y_event)

rows = [
    ('Mechanistic model  (in-sample, 8 params)',  c_mech_insample),
    (f'Mechanistic model  ({K_FOLDS}-fold CV)',   c_mech_cv),
    ('Cox PH             (in-sample, 6 features)', c_cox_insample),
    (f'Cox PH             ({K_FOLDS}-fold CV)',   c_cox_cv),
    ('--- univariate baselines ---',               None),
    ('Tumour size',                                c_size),
    ('Lymph node count',                           c_nodes),
    ('Histological grade',                         c_grade),
    ('ER status',                                  c_er),
    ('HER2 status',                                c_her2),
    ('PR status',                                  c_pr),
]
print(f'  {"Model":<52}  C-Index')
print('  ' + '-' * 62)
for name, c in rows:
    if c is None:
        print(f'  {name}')
    else:
        print(f'  {name:<52}  {c:.4f}')

# =============================================================================
# 8. Risk stratification and Kaplan-Meier
# =============================================================================
print()
print('=' * 60)
print('RISK STRATIFICATION AND KAPLAN-MEIER')
print('=' * 60)

threshold = np.median(t_pred_full)
mask_hi   = t_pred_full <= threshold
mask_lo   = ~mask_hi

lr = logrank_test(y_time[mask_hi], y_time[mask_lo],
                  event_observed_A=y_event[mask_hi],
                  event_observed_B=y_event[mask_lo])

print(f'Median T_pred threshold : {threshold:.1f} months')
print(f'High-Risk group         : {mask_hi.sum():,} patients')
print(f'Low-Risk group          : {mask_lo.sum():,} patients')
print(f'Log-Rank statistic      : {lr.test_statistic:.4f}')
print(f'Log-Rank p-value        : {lr.p_value:.3e}')

fig, ax = plt.subplots(figsize=(8, 5))
kmf = KaplanMeierFitter()
for label, mask, color in [('High Risk', mask_hi, '#d62728'),
                             ('Low Risk',  mask_lo,  '#1f77b4')]:
    kmf.fit(y_time[mask], event_observed=y_event[mask], label=label)
    kmf.plot_survival_function(ax=ax, color=color, ci_show=True, ci_alpha=0.12)
ax.set_xlabel('Time from surgery (months)')
ax.set_ylabel('Relapse-Free Survival')
ax.set_ylim(0, 1.05)
ax.set_xlim(left=0)
ax.grid(True, alpha=0.2, linestyle='--')
ax.spines[['top', 'right']].set_visible(False)
ax.text(0.97, 0.97, f'Log-Rank p = {lr.p_value:.2e}',
        transform=ax.transAxes, ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='grey', alpha=0.8))
ax.set_title(f'Mechanistic Risk Stratification — METABRIC (N={N:,})')
ax.legend(framealpha=0.9)
plt.tight_layout()
km_out = os.path.join(script_dir, 'fig_km.pdf')
plt.savefig(km_out, bbox_inches='tight')
print(f'Saved: {km_out}')

# =============================================================================
# 9. Summary (for paper numbers)
# =============================================================================
print()
print('=' * 60)
print('SUMMARY')
print('=' * 60)
print(f'  N = {N:,}  |  Events = {y_event.sum():,}  |  M = {M:.0e}  |  gamma = {GAMMA}')
print()
print(f'  Mechanistic in-sample C        : {c_mech_insample:.4f}')
print(f'  Mechanistic {K_FOLDS}-fold CV C         : {c_mech_cv:.4f}')
print(f'  Cox PH in-sample C             : {c_cox_insample:.4f}')
print(f'  Cox PH {K_FOLDS}-fold CV C              : {c_cox_cv:.4f}')
print(f'  Interpretability price (in-sample): {c_mech_insample - c_cox_insample:+.4f}')
print(f'  Interpretability price (CV)    : {c_mech_cv - c_cox_cv:+.4f}')
print()
print(f'  Log-Rank statistic             : {lr.test_statistic:.4f}')
print(f'  Log-Rank p-value               : {lr.p_value:.3e}')
print('=' * 60)

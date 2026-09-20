# =============================================================================
# revision_analysis.py
# Extended analyses for "A Mechanistic Learning Framework for Breast Cancer
# Relapse Prediction" (Perez-Velazquez, Golgeli, Kulac)
#
# Companion to mechanistic_model.py (the base implementation: unanchored,
# linear nodal link). Implements:
#
#   1  Scale anchoring of (n0, lambda) on a reference phenotype -> identifiability,
#      with a numerical demonstration of the flat ridge (loss invariance, Hessian eigenvalues)
#   2  Plausibility of predicted times and saturating Seed links (log1p nodes, pN stage, bounded)
#      2b  refits of the log1p and bounded links for other doubling-time anchors and detection thresholds M
#   3  Illustrative patients and sensitivity of absolute n0 to the anchor
#   4  Out-of-fold performance, time-dependent AUC, Kaplan-Meier risk groups
#   5  ER/PR association and treatment structure
#   6  Treatment, continuous-receptor and immune-Soil variants
#   7  Paired bootstrap: mechanistic vs Cox delta-C, confidence intervals for every coefficient
#
# Usage (run from this folder):
#   python revision_analysis.py --tar brca_metabric.tar.gz [--n-boot 1000] [--quick]
#
# Outputs -> outputs_revision/  (results.json, summary.md, tables_*.csv, fig_*.pdf)
# =============================================================================

import warnings; warnings.filterwarnings('ignore')
import argparse, io, json, os, sys, tarfile, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from joblib import Parallel, delayed
from scipy.special import expit as scipy_expit
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ----------------------------------------------------------------------------- constants
M        = 1e9
LN_M     = np.log(M)
LAM_MIN  = 1e-4
GAMMA    = 1.0
K_FOLDS  = 5
SEED     = 42
OPT_OPTS = dict(maxiter=5000, ftol=1e-12, gtol=1e-8)

# --- Scale anchor --------------------------------------------------------------
# lambda is fixed for a REFERENCE PHENOTYPE to a literature volume-doubling time.
# VERIFIED 2026-09-13: Nakashima et al. 2019, Breast Cancer 26:206-214,
#   doi:10.1007/s12282-018-0914-0 (N=265, serial ultrasonography) report a
#   volume doubling time of 185 days for ER+/HER2- breast tumours (vs 124 days
#   for triple-negative, p=0.027) -- the closest direct literature match to the
#   reference phenotype (Grade 2, ER+, HER2-, PR+). This is corroborated by
#   Dahan et al. 2021, Cancer Med 10(15):5203-5217, doi:10.1002/cam4.3939, a
#   systematic review of 80 years of doubling-time literature reporting a
#   pooled average of 180 days across all subtypes.
VDT_REF_DAYS   = 185.0
DAYS_PER_MONTH = 365.25 / 12
LAM_REF        = np.log(2.0) / (VDT_REF_DAYS / DAYS_PER_MONTH)     # month^-1
X_REF          = {'GRADE': 2.0, 'ER_BIN': 1.0, 'HER2_BIN': 0.0, 'PR_BIN': 1.0}  # luminal-A-like, grade 2
# any other Speed covariate (z-scored expression, treatment flags) has reference value 0

SEED_COLS  = ['TUMOR_SIZE', 'LYMPH_NODES_EXAMINED_POSITIVE']
BASE_SPEED = ['GRADE', 'ER_BIN', 'HER2_BIN', 'PR_BIN']
GENES      = ['ESR1', 'PGR', 'ERBB2', 'MKI67',
              'PTPRC', 'CD3E', 'CD3D', 'CD2', 'CD8A', 'CD8B', 'CD4', 'GZMA', 'GZMB', 'GZMK', 'PRF1', 'NKG7',
              'CD274', 'PDCD1', 'CTLA4', 'FOXP3', 'CXCL9', 'CXCL10', 'CXCL13', 'IFNG', 'STAT1', 'IDO1',
              'LAG3', 'HLA-DRA', 'MS4A1', 'CD19', 'CD79A', 'IGKC', 'CCL5']
IMMUNE_GENES = GENES[4:]

# ----------------------------------------------------------------------------- anchor / threshold override
from contextlib import contextmanager

@contextmanager
def anchor_and_threshold(vdt_days=None, m_cells=None):
    """Temporarily override the doubling-time anchor and/or the detection threshold M (module globals), then restore."""
    global M, LN_M, LAM_REF
    saved = (M, LN_M, LAM_REF)
    try:
        if m_cells is not None:
            M = float(m_cells); LN_M = np.log(M)
        if vdt_days is not None:
            LAM_REF = np.log(2.0) / (vdt_days / DAYS_PER_MONTH)
        yield
    finally:
        M, LN_M, LAM_REF = saved

# ----------------------------------------------------------------------------- data
def load_cohort(tar_path, cache_csv):
    if os.path.exists(cache_csv):
        return pd.read_csv(cache_csv)
    with tarfile.open(tar_path, 'r:*') as tar:
        dp = pd.read_csv(tar.extractfile('brca_metabric/data_clinical_patient.txt'), sep='\t', comment='#')
        ds = pd.read_csv(tar.extractfile('brca_metabric/data_clinical_sample.txt'), sep='\t', comment='#')
        df = pd.merge(dp, ds, on='PATIENT_ID').iloc[1:].reset_index(drop=True)   # identical to mechanistic_model.py
        f = io.TextIOWrapper(tar.extractfile('brca_metabric/data_mrna_illumina_microarray.txt'), encoding='utf-8')
        header = next(f).rstrip('\n').split('\t'); samples = header[2:]
        expr = {}
        want = set(GENES)
        for line in f:
            g = line.split('\t', 1)[0]
            if g in want:
                parts = line.rstrip('\n').split('\t')
                expr[g] = pd.Series(pd.to_numeric(parts[2:], errors='coerce'), index=samples)
        expr = pd.DataFrame(expr)
    for c in ['ER', 'HER2', 'PR']:
        df[c + '_BIN'] = df[c + '_STATUS'].map({'Positive': 1.0, 'Negative': 0.0})
    for c in SEED_COLS + ['GRADE', 'RFS_MONTHS', 'AGE_AT_DIAGNOSIS']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['EVENT']    = df['RFS_STATUS'].apply(lambda x: 1 if str(x).startswith('1') else 0)
    df['HT_BIN']   = df['HORMONE_THERAPY'].map({'YES': 1.0, 'NO': 0.0})
    df['CHEMO_BIN']= df['CHEMOTHERAPY'].map({'YES': 1.0, 'NO': 0.0})
    df['HER2_CN']  = df['HER2_SNP6'].map({'LOSS': -1.0, 'NEUTRAL': 0.0, 'GAIN': 1.0})
    df = df.set_index('PATIENT_ID')
    z = lambda s: (s - s.mean()) / s.std()
    for g in ['ESR1', 'PGR', 'ERBB2', 'MKI67']:
        df[g + '_Z'] = z(expr[g])
    df['IMM_SIG_Z'] = z(expr[IMMUNE_GENES].mean(axis=1))
    df['CYT_Z']     = z(expr[['GZMA', 'PRF1']].mean(axis=1))
    df['TCELL_Z']   = z(expr[['CD3E', 'CD3D', 'CD2', 'CD8A', 'CD8B']].mean(axis=1))
    df['PDL1_Z']    = z(expr['CD274'])
    df = df.reset_index()
    df = df.dropna(subset=SEED_COLS + BASE_SPEED + ['RFS_MONTHS']).copy()    # analytic cohort N=1,814
    df.to_csv(cache_csv, index=False)
    return df

# ----------------------------------------------------------------------------- model
def nodes_transform(nodes, link):
    if link in ('linear', 'bounded'):
        return nodes
    if link == 'log1p':
        return np.log1p(nodes)
    if link == 'pN':               # AJCC pN: 0 / 1-3 / 4-9 / >=10
        return np.select([nodes == 0, nodes <= 3, nodes <= 9], [0.0, 1.0, 2.0], 3.0)
    raise ValueError(link)

def design(df, spec):
    Xs  = np.column_stack([df['TUMOR_SIZE'].values,
                           nodes_transform(df['LYMPH_NODES_EXAMINED_POSITIVE'].values, spec['link'])])
    Xsp = df[spec['speed']].values.astype(float)
    Xso = df[spec['soil']].values.astype(float) if spec['soil'] else np.zeros((len(df), 0))
    return Xs, Xsp, Xso

def xref_vec(spec):
    return np.array([X_REF.get(c, 0.0) for c in spec['speed']])

def unpack(theta, spec):
    nso, ns = len(spec['soil']), len(spec['speed'])
    a = theta[:3 + nso]
    if spec['anchored']:
        b  = theta[3 + nso:]
        b0 = LAM_REF - b @ xref_vec(spec)
    else:
        b0 = theta[3 + nso]; b = theta[4 + nso:]
    return a, b0, b

def ln_n0_of(theta, spec, Xs, Xso):
    a, _, _ = unpack(theta, spec)
    z = a[0] + a[1] * Xs[:, 0] + a[2] * Xs[:, 1] + (Xso @ a[3:] if Xso.shape[1] else 0.0)
    if spec['link'] == 'bounded':
        return LN_M * scipy_expit(z)          # ln n0 in (0, ln M)  =>  n0 < M always; expit avoids exp overflow
    return z

def lam_of(theta, spec, Xsp):
    _, b0, b = unpack(theta, spec)
    return np.clip(b0 + Xsp @ b, LAM_MIN, None)

def predict_t(theta, spec, Xs, Xsp, Xso):
    return (LN_M - ln_n0_of(theta, spec, Xs, Xso)) / lam_of(theta, spec, Xsp)

def loss(theta, spec, Xs, Xsp, Xso, t, e):
    r = t - predict_t(theta, spec, Xs, Xsp, Xso)
    return np.sum(r[e == 1] ** 2) + GAMMA * np.sum(np.maximum(0.0, r[e == 0]) ** 2)

def theta0_bounds(spec):
    nso, ns = len(spec['soil']), len(spec['speed'])
    if spec['anchored']:
        th = np.zeros(3 + nso + ns); bds = [(None, None)] * (3 + nso + ns)
    else:
        th = np.zeros(4 + nso + ns); th[3 + nso] = 0.1
        bds = [(None, None)] * (3 + nso) + [(LAM_MIN, None)] + [(None, None)] * ns
    th[0] = -1.5 if spec['link'] == 'bounded' else 5.0
    return th, bds

def fit(spec, Xs, Xsp, Xso, t, e):
    th0, bds = theta0_bounds(spec)
    res = minimize(loss, th0, args=(spec, Xs, Xsp, Xso, t, e), method='L-BFGS-B', bounds=bds, options=OPT_OPTS)
    return res.x, res

def param_names(spec):
    names = ['alpha0', 'alpha1_size', 'alpha2_nodes'] + ['soil_' + c for c in spec['soil']]
    if not spec['anchored']:
        names.append('beta0')
    return names + ['beta_' + c for c in spec['speed']]

def cox_fit(df_tr, feats):
    d = df_tr[feats].copy(); d['T'] = df_tr['RFS_MONTHS'].values; d['E'] = df_tr['EVENT'].values
    return CoxPHFitter(penalizer=0.0).fit(d, 'T', 'E')

# ----------------------------------------------------------------------------- evaluation helpers
def cv_oof(df, spec, feats_cox=None):
    """5-fold CV with a fixed fold assignment (seed 42). Returns out-of-fold T_pred (mech) and
    partial hazard (Cox), per-fold C-indices, and OOF risk-group labels (threshold from training fold)."""
    N = len(df); Xs, Xsp, Xso = design(df, spec); t, e = df['RFS_MONTHS'].values, df['EVENT'].values
    rng = np.random.default_rng(SEED); perm = rng.permutation(N)
    folds = [perm[i * N // K_FOLDS:(i + 1) * N // K_FOLDS] for i in range(K_FOLDS)]
    oof_t = np.full(N, np.nan); oof_hz = np.full(N, np.nan); oof_hi = np.zeros(N, dtype=bool)
    c_m, c_c = [], []
    feats = feats_cox or (SEED_COLS + spec['speed'] + spec['soil'])
    for i, te in enumerate(folds):
        tr = np.concatenate([folds[j] for j in range(K_FOLDS) if j != i])
        th, _ = fit(spec, Xs[tr], Xsp[tr], Xso[tr], t[tr], e[tr])
        tp_tr = predict_t(th, spec, Xs[tr], Xsp[tr], Xso[tr])
        tp_te = predict_t(th, spec, Xs[te], Xsp[te], Xso[te])
        oof_t[te] = tp_te; oof_hi[te] = tp_te <= np.median(tp_tr)
        c_m.append(concordance_index(t[te], tp_te, e[te]))
        cph = cox_fit(df.iloc[tr], feats)
        hz = cph.predict_partial_hazard(df.iloc[te][feats]).values
        oof_hz[te] = hz
        c_c.append(concordance_index(t[te], -hz, e[te]))
    return dict(oof_t=oof_t, oof_hz=oof_hz, oof_hi=oof_hi, c_mech_folds=c_m, c_cox_folds=c_c)

def ipcw_auc(t, e, risk, tau):
    """Cumulative/dynamic AUC at time tau with inverse-probability-of-censoring weights (Uno et al. 2007).
    Cases: t<=tau & event; controls: t>tau. Higher risk = earlier event."""
    kmc = KaplanMeierFitter().fit(t, event_observed=1 - e)          # censoring survival G
    G = lambda x: np.clip(kmc.survival_function_at_times(x).values, 1e-6, None)
    cases = (t <= tau) & (e == 1); ctrls = t > tau
    if cases.sum() == 0 or ctrls.sum() == 0:
        return np.nan
    w = 1.0 / G(t[cases])
    rc, rk = risk[cases], risk[ctrls]
    conc = (rc[:, None] > rk[None, :]).astype(float) + 0.5 * (rc[:, None] == rk[None, :])
    return float((w[:, None] * conc).sum() / (w.sum() * ctrls.sum()))

def numerical_hessian(f, x, rel=1e-4):
    n = len(x); H = np.zeros((n, n)); h = rel * np.maximum(np.abs(x), 1e-2)
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ej = np.zeros(n); ei[i] = h[i]; ej[j] = h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H

def ridge_point(theta, spec, c):
    """Move the (unanchored, linear-link) parameter vector along the scale-invariance direction."""
    a0, a1, a2, b0, b1, b2, b3, b4 = theta
    return np.array([LN_M - c * (LN_M - a0), c * a1, c * a2, c * b0, c * b1, c * b2, c * b3, c * b4])

# ----------------------------------------------------------------------------- bootstrap
def _boot_worker(b, df, specs, main_key, cox_feats):
    rng = np.random.default_rng(SEED + 1000 + b)
    N = len(df); idx = rng.integers(0, N, N); oob = np.setdiff1d(np.arange(N), idx)
    dtr, dte = df.iloc[idx], df.iloc[oob]
    t_te, e_te = dte['RFS_MONTHS'].values, dte['EVENT'].values
    out = {'b': b}
    for key, spec in specs.items():
        Xs, Xsp, Xso = design(dtr, spec); Xs2, Xsp2, Xso2 = design(dte, spec)
        try:
            th, _ = fit(spec, Xs, Xsp, Xso, dtr['RFS_MONTHS'].values, dtr['EVENT'].values)
            out['C_' + key] = concordance_index(t_te, predict_t(th, spec, Xs2, Xsp2, Xso2), e_te)
            if key == main_key or key.startswith('treat'):
                out['theta_' + key] = th.tolist()
        except Exception:
            out['C_' + key] = np.nan
    for key, feats in cox_feats.items():
        try:
            cph = cox_fit(dtr, feats)
            out['C_' + key] = concordance_index(t_te, -cph.predict_partial_hazard(dte[feats]).values, e_te)
        except Exception:
            out['C_' + key] = np.nan
    return out

def md_table(dfx, floatfmt='%.4f'):
    """Minimal markdown table writer (no tabulate dependency)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return 'nan' if np.isnan(v) else (floatfmt % v)
        return str(v)
    cols = list(dfx.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in dfx.iterrows():
        lines.append('| ' + ' | '.join(fmt(r[c]) for c in cols) + ' |')
    return '\n'.join(lines)

def pct_ci(x, lo=2.5, hi=97.5):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float(np.nanmean(x)), float(np.percentile(x, lo)), float(np.percentile(x, hi)), int(len(x))

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tar', default='brca_metabric.tar.gz')
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--quick', action='store_true', help='n_boot=50, for a smoke test')
    ap.add_argument('--main-link', default='log1p', choices=['linear', 'log1p', 'pN', 'bounded'])
    ap.add_argument('--out', default='outputs_revision')
    ap.add_argument('--jobs', type=int, default=-1)
    args = ap.parse_args()
    n_boot = 50 if args.quick else args.n_boot
    here = os.path.dirname(os.path.abspath(__file__)); out = os.path.join(here, args.out); os.makedirs(out, exist_ok=True)
    R = {'settings': dict(M=M, LAM_MIN=LAM_MIN, GAMMA=GAMMA, K_FOLDS=K_FOLDS, SEED=SEED, VDT_REF_DAYS=VDT_REF_DAYS,
                          LAM_REF=LAM_REF, X_REF=X_REF, main_link=args.main_link, n_boot=n_boot)}
    md = []
    def sec(title): md.append(f'\n## {title}\n'); print('\n' + '=' * 70 + f'\n{title}\n' + '=' * 70)
    def line(s=''): md.append(s); print(s)
    def table(dfx, name, floatfmt='%.4f'):
        dfx.to_csv(os.path.join(out, f'table_{name}.csv'), index=False)
        md.append(md_table(dfx, floatfmt))
        print(dfx.to_string(index=False))

    t0 = time.time()
    df = load_cohort(args.tar, os.path.join(out, 'cohort_cache.csv'))
    N = len(df); t, e = df['RFS_MONTHS'].values, df['EVENT'].values
    line(f'Analytic cohort N={N:,}, events={int(e.sum()):,} ({100*e.mean():.1f}%), median follow-up {np.median(t):.1f} months')
    R['cohort'] = dict(N=N, events=int(e.sum()))

    # ================================================================== 1. unanchored model: ridge demonstration
    sec('1. Identifiability: the unanchored, linear-link model lies on a flat ridge')
    spec_sub = dict(name='unanchored', speed=BASE_SPEED, soil=[], link='linear', anchored=False)
    Xs, Xsp, Xso = design(df, spec_sub)
    th_sub, res_sub = fit(spec_sub, Xs, Xsp, Xso, t, e)
    L0 = loss(th_sub, spec_sub, Xs, Xsp, Xso, t, e)
    line(f'Unanchored linear-link fit: theta = {np.round(th_sub, 5).tolist()}, loss = {L0:.4f}, '
         f'in-sample C = {concordance_index(t, predict_t(th_sub, spec_sub, Xs, Xsp, Xso), e):.4f}')
    line(f'alpha0 hat = {th_sub[0]:.4f} vs initial value 5.0  (initialisation-dependent: ridge has no gradient)')
    ridge_rows = []
    for c in [0.5, 0.8, 1.0, 1.25, 2.0]:
        thc = ridge_point(th_sub, spec_sub, c)
        Lc = loss(thc, spec_sub, Xs, Xsp, Xso, t, e)
        ridge_rows.append(dict(c=c, alpha0=thc[0], beta0=thc[3], loss=Lc, rel_loss_change=(Lc - L0) / L0,
                               C_index=concordance_index(t, predict_t(thc, spec_sub, Xs, Xsp, Xso), e),
                               n0_mean_patient=float(np.exp(thc[0] + thc[1] * df['TUMOR_SIZE'].mean() + thc[2] * df['LYMPH_NODES_EXAMINED_POSITIVE'].mean()))))
    table(pd.DataFrame(ridge_rows), 'ridge_invariance', '%.6g')
    line('Loss and C-index are exactly invariant along the ridge; the implied absolute n0 is not.')
    H = numerical_hessian(lambda x: loss(x, spec_sub, Xs, Xsp, Xso, t, e), th_sub)
    S = np.diag(1 / np.maximum(np.abs(th_sub), 1e-2)); Hs = S @ H @ S     # scale-normalised Hessian
    ev = np.sort(np.linalg.eigvalsh(Hs))
    line(f'Scale-normalised Hessian eigenvalues (unanchored linear-link model): {np.array2string(ev, precision=3)}')
    line(f'  smallest/largest = {ev[0]/ev[-1]:.2e}  -> one (near-)null direction')
    R['ridge'] = dict(rows=ridge_rows, hessian_eigs_unanchored=ev.tolist())

    # ================================================================== 2. anchored fits, seed-link comparison
    sec('2. Anchored model and saturating Seed links')
    line(f'Anchor: lambda(reference phenotype grade 2, ER+, HER2-, PR+) fixed to ln2 / VDT, VDT = {VDT_REF_DAYS:.0f} days '
         f'-> lambda_ref = {LAM_REF:.4f} month^-1  (Nakashima et al. 2019; Dahan et al. 2021)')
    link_rows = []; fits = {}
    for link in ['linear', 'log1p', 'pN', 'bounded']:
        spec = dict(name=f'anchored_{link}', speed=BASE_SPEED, soil=[], link=link, anchored=True)
        Xs, Xsp, Xso = design(df, spec)
        th, res = fit(spec, Xs, Xsp, Xso, t, e)
        tp = predict_t(th, spec, Xs, Xsp, Xso)
        Ha = numerical_hessian(lambda x: loss(x, spec, Xs, Xsp, Xso, t, e), th)
        Sa = np.diag(1 / np.maximum(np.abs(th), 1e-2)); eva = np.sort(np.linalg.eigvalsh(Sa @ Ha @ Sa))
        cv = cv_oof(df, spec)
        lnn0 = ln_n0_of(th, spec, Xs, Xso)
        fits[link] = dict(spec=spec, theta=th, tp=tp, cv=cv)
        a, b0, b = unpack(th, spec)
        link_rows.append(dict(seed_link=link, n_params=len(th), loss=res.fun, C_in=concordance_index(t, tp, e),
                              C_cv=np.mean(cv['c_mech_folds']), C_cv_sd=np.std(cv['c_mech_folds'], ddof=1),
                              n_Tpred_le_0=int((tp <= 0).sum()), min_Tpred=float(tp.min()), max_ln_n0=float(lnn0.max()),
                              hess_min_over_max=eva[0] / eva[-1], alpha1_size=a[1], alpha2_nodes=a[2], beta0_implied=b0,
                              **{f'beta_{c}': v for c, v in zip(BASE_SPEED, b)}))
    table(pd.DataFrame(link_rows), 'seed_links', '%.5g')
    R['seed_links'] = link_rows
    main_link = args.main_link
    main = fits[main_link]; spec_main = main['spec']; th_main = main['theta']
    line(f'\nMAIN MODEL = anchored + {main_link} node link  (change with --main-link)')

    # seed-link figure: n0 vs nodes at cohort-median size
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    nodes_grid = np.arange(0, 46); size_med = np.median(df['TUMOR_SIZE'])
    for link, ls in [('linear', '--'), ('log1p', '-'), ('pN', ':'), ('bounded', '-.')]:
        sp = fits[link]['spec']; th = fits[link]['theta']
        Xg = np.column_stack([np.full_like(nodes_grid, size_med, dtype=float), nodes_transform(nodes_grid.astype(float), link)])
        ax.plot(nodes_grid, ln_n0_of(th, sp, Xg, np.zeros((len(nodes_grid), 0))), ls, label=f'{link} link')
    ax.axhline(LN_M, color='k', lw=0.8); ax.text(1, LN_M + 0.3, r'$\ln M$ (detection threshold)', fontsize=9)
    ax.set_xlabel('Positive lymph nodes'); ax.set_ylabel(r'$\ln n_0$ at median tumour size'); ax.legend(); ax.grid(alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False); plt.tight_layout(); plt.savefig(os.path.join(out, 'fig_seed_link.pdf')); plt.close()

    # ================================================================== 2b. dependence of the fit on the anchor and on M, log1p vs bounded
    sec('2b. Does the fit depend on the doubling-time anchor and on the detection threshold M? (log1p vs bounded link)')
    line('Each configuration is refitted from scratch (not rescaled). For the log-linear log1p link the scale invariance of Section 1 implies that\n'
         'T_pred and all rank metrics are unchanged and absolute n0 scales with M; the bounded link (ln n0 = ln M * logistic(z)) has no such invariance.')
    am_rows = []
    for link in ['log1p', 'bounded']:
        for vdt, m_cells in [(150, 1e9), (185, 1e9), (250, 1e9), (185, 1e8)]:
            with anchor_and_threshold(vdt, m_cells):
                spec = dict(name=f'anchored_{link}', speed=BASE_SPEED, soil=[], link=link, anchored=True)
                Xs, Xsp, Xso = design(df, spec)
                th, res = fit(spec, Xs, Xsp, Xso, t, e)
                tp = predict_t(th, spec, Xs, Xsp, Xso); lnn0 = ln_n0_of(th, spec, Xs, Xso)
                cvx = cv_oof(df, spec)
                a, b0, b = unpack(th, spec)
                am_rows.append(dict(seed_link=link, VDT_days=vdt, M_cells=m_cells, loss=res.fun, C_in=concordance_index(t, tp, e),
                                    C_cv=np.mean(cvx['c_mech_folds']), n_Tpred_le_0=int((tp <= 0).sum()),
                                    mean_Tpred_months=float(tp.mean()), median_n0_cells=float(np.exp(np.median(lnn0))),
                                    alpha0=a[0], alpha1_size=a[1], alpha2_nodes=a[2], **{f'beta_{c}': v for c, v in zip(BASE_SPEED, b)}))
    table(pd.DataFrame(am_rows), 'anchor_M_sensitivity', '%.5g')
    R['anchor_M_sensitivity'] = am_rows

    # ================================================================== 3. illustrative patients, VDT sensitivity
    sec('3. Illustrative patients and sensitivity of absolute n0 to the anchor')
    pts = pd.DataFrame([dict(Patient='A (aggressive)', TUMOR_SIZE=30, LYMPH_NODES_EXAMINED_POSITIVE=3, GRADE=3, ER_BIN=0, HER2_BIN=1, PR_BIN=0),
                        dict(Patient='B (favourable)', TUMOR_SIZE=15, LYMPH_NODES_EXAMINED_POSITIVE=0, GRADE=1, ER_BIN=1, HER2_BIN=0, PR_BIN=1),
                        dict(Patient='C (extreme nodal burden: 26 mm, 42 nodes, A-speed)', TUMOR_SIZE=26, LYMPH_NODES_EXAMINED_POSITIVE=42, GRADE=3, ER_BIN=0, HER2_BIN=1, PR_BIN=0),
                        dict(Patient='Cohort mean', TUMOR_SIZE=df['TUMOR_SIZE'].mean(), LYMPH_NODES_EXAMINED_POSITIVE=df['LYMPH_NODES_EXAMINED_POSITIVE'].mean(),
                             GRADE=df['GRADE'].mean(), ER_BIN=df['ER_BIN'].mean(), HER2_BIN=df['HER2_BIN'].mean(), PR_BIN=df['PR_BIN'].mean())])
    pt_rows = []
    for link in ['linear', main_link] if main_link != 'linear' else ['linear']:
        sp = fits[link]['spec']; th = fits[link]['theta']
        Xs_p, Xsp_p, Xso_p = design(pts, sp)
        for i, r in pts.iterrows():
            ln0 = ln_n0_of(th, sp, Xs_p, Xso_p)[i]; lam = lam_of(th, sp, Xsp_p)[i]
            pt_rows.append(dict(model=f'anchored_{link}', patient=r['Patient'], ln_n0=ln0, n0_cells=np.exp(ln0), lambda_per_month=lam,
                                doubling_time_days=np.log(2) / lam * DAYS_PER_MONTH, T_pred_months=(LN_M - ln0) / lam,
                                risk_group='High' if (LN_M - ln0) / lam <= np.median(fits[link]['tp']) else 'Low'))
    table(pd.DataFrame(pt_rows), 'illustrative_patients', '%.4g')
    line(f'Median T_pred (main model, in-sample) = {np.median(main["tp"]):.1f} months')
    R['patients'] = pt_rows
    # VDT sensitivity (ratios are invariant; absolute n0 scales with the anchor)
    vdt_rows = []
    for vdt in [100, 126, 150, 164, 185, 200, 250, 268, 332, 400, 500]:  # spans published subtype/receptor-specific doubling times
        lam_ref = np.log(2) / (vdt / DAYS_PER_MONTH); c = lam_ref / LAM_REF
        a, b0, b = unpack(th_main, spec_main)
        # along the ridge: D -> c*D  (D = ln M - ln n0), lambda -> c*lambda ; T_pred unchanged
        Xs_m, _, Xso_m = design(pts.iloc[[3]], spec_main)
        ln0_mean = ln_n0_of(th_main, spec_main, Xs_m, Xso_m)[0]
        vdt_rows.append(dict(VDT_days=vdt, lambda_ref=lam_ref, scale_c=c, ln_n0_cohort_mean=LN_M - c * (LN_M - ln0_mean),
                             n0_cohort_mean_cells=np.exp(LN_M - c * (LN_M - ln0_mean)), T_pred_cohort_mean=(LN_M - ln0_mean) / lam_of(th_main, spec_main, design(pts.iloc[[3]], spec_main)[1])[0]))
    table(pd.DataFrame(vdt_rows), 'vdt_sensitivity', '%.4g')
    R['vdt_sensitivity'] = vdt_rows

    # ================================================================== 4. CV performance, time-dependent AUC, KM
    sec('4. Out-of-fold performance of the main model vs Cox')
    cvm = main['cv']
    cph_full = cox_fit(df, SEED_COLS + BASE_SPEED)
    perf_rows = []
    for name, risk in [('mechanistic (main)', -cvm['oof_t']), ('Cox PH, 6 features', cvm['oof_hz'])]:
        perf_rows.append(dict(model=name, C_cv=concordance_index(t, -risk, e),
                              AUC_5y=ipcw_auc(t, e, risk, 60.0), AUC_10y=ipcw_auc(t, e, risk, 120.0), AUC_15y=ipcw_auc(t, e, risk, 180.0)))
    perf_rows[0]['C_in'] = concordance_index(t, main['tp'], e); perf_rows[1]['C_in'] = cph_full.concordance_index_
    table(pd.DataFrame(perf_rows), 'performance_oof', '%.4f')
    d_folds = np.array(cvm['c_mech_folds']) - np.array(cvm['c_cox_folds'])
    line(f'Per-fold C (mech):  {np.round(cvm["c_mech_folds"], 4).tolist()}')
    line(f'Per-fold C (Cox) :  {np.round(cvm["c_cox_folds"], 4).tolist()}')
    line(f'Paired per-fold delta C (mech - Cox): mean {d_folds.mean():+.4f}, sd {d_folds.std(ddof=1):.4f}')
    line('\nCox PH coefficients (full cohort):'); line(md_table(cph_full.summary[['coef', 'exp(coef)', 'coef lower 95%', 'coef upper 95%', 'p']].reset_index().rename(columns={'covariate': 'covariate'}), '%.4f'))
    R['performance'] = dict(rows=perf_rows, fold_delta=d_folds.tolist())
    # KM: in-sample median split and out-of-fold split (threshold from training fold)
    km_rows = []
    for name, hi in [('in-sample median split', main['tp'] <= np.median(main['tp'])), ('out-of-fold split (training-fold threshold)', cvm['oof_hi'])]:
        lr = logrank_test(t[hi], t[~hi], event_observed_A=e[hi], event_observed_B=e[~hi])
        km_rows.append(dict(split=name, n_high=int(hi.sum()), n_low=int((~hi).sum()), events_high=int(e[hi].sum()), events_low=int(e[~hi].sum()),
                            logrank_stat=lr.test_statistic, p_value=lr.p_value))
    table(pd.DataFrame(km_rows), 'km_splits', '%.4g')
    R['km'] = km_rows
    hi = main['tp'] <= np.median(main['tp'])
    fig, ax = plt.subplots(figsize=(8, 5)); kmf = KaplanMeierFitter()
    for label, mask, color in [('High Risk', hi, '#d62728'), ('Low Risk', ~hi, '#1f77b4')]:
        kmf.fit(t[mask], event_observed=e[mask], label=label); kmf.plot_survival_function(ax=ax, color=color, ci_show=True, ci_alpha=0.12)
    ax.set_xlabel('Time from surgery (months)'); ax.set_ylabel('Relapse-Free Survival'); ax.set_ylim(0, 1.05); ax.set_xlim(left=0)
    ax.grid(True, alpha=0.2, linestyle='--'); ax.spines[['top', 'right']].set_visible(False)
    ax.text(0.97, 0.97, f'Log-Rank p = {km_rows[0]["p_value"]:.2e}', transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='grey', alpha=0.8))
    ax.set_title(f'Mechanistic Risk Stratification — METABRIC (N={N:,})'); ax.legend(framealpha=0.9); plt.tight_layout()
    plt.savefig(os.path.join(out, 'fig_km_revised.pdf'), bbox_inches='tight'); plt.close()

    # ================================================================== 5. ER/PR collinearity, treatment structure
    sec('5. Collinearity and treatment confounding structure')
    ct = pd.crosstab(df['ER_BIN'], df['PR_BIN']); phi = np.corrcoef(df['ER_BIN'], df['PR_BIN'])[0, 1]
    line(f'ER x PR cross-tab:\n{ct.to_string()}\nphi(ER, PR) = {phi:.3f}')
    line(f'\nER x hormone therapy:\n{pd.crosstab(df["ER_BIN"], df["HT_BIN"]).to_string()}')
    line(f'\nHER2 x chemotherapy:\n{pd.crosstab(df["HER2_BIN"], df["CHEMO_BIN"]).to_string()}')
    Xd = df[BASE_SPEED + ['HT_BIN', 'CHEMO_BIN']].dropna()
    vif = {}
    for c in Xd.columns:
        y = Xd[c].values; X1 = np.column_stack([np.ones(len(Xd)), Xd.drop(columns=c).values])
        r2 = 1 - np.sum((y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]) ** 2) / np.sum((y - y.mean()) ** 2)
        vif[c] = 1 / (1 - r2)
    line(f'Variance inflation factors: { {k: round(v, 2) for k, v in vif.items()} }')
    R['collinearity'] = dict(phi_ER_PR=phi, vif=vif)

    # ================================================================== 6. sensitivity specs (treatment, receptors, immune)
    sec('6. Treatment, continuous-receptor and immune-Soil variants: point estimates + CV')
    specs = {
        'main':        spec_main,
        'treat_HT':    dict(name='treat_HT', speed=BASE_SPEED + ['HT_BIN'], soil=[], link=main_link, anchored=True),
        'treat_HTCT':  dict(name='treat_HTCT', speed=BASE_SPEED + ['HT_BIN', 'CHEMO_BIN'], soil=[], link=main_link, anchored=True),
        'cont_recept': dict(name='cont_recept', speed=['GRADE', 'ESR1_Z', 'ERBB2_Z', 'PGR_Z'], soil=[], link=main_link, anchored=True),
        'her2_cn':     dict(name='her2_cn', speed=['GRADE', 'ER_BIN', 'HER2_CN', 'PR_BIN'], soil=[], link=main_link, anchored=True),
        'mki67':       dict(name='mki67', speed=BASE_SPEED + ['MKI67_Z'], soil=[], link=main_link, anchored=True),
        'imm_sig':     dict(name='imm_sig', speed=BASE_SPEED, soil=['IMM_SIG_Z'], link=main_link, anchored=True),
        'imm_cyt':     dict(name='imm_cyt', speed=BASE_SPEED, soil=['CYT_Z'], link=main_link, anchored=True),
        'imm_pdl1':    dict(name='imm_pdl1', speed=BASE_SPEED, soil=['PDL1_Z'], link=main_link, anchored=True),
        'mki67_imm':   dict(name='mki67_imm', speed=BASE_SPEED + ['MKI67_Z'], soil=['IMM_SIG_Z'], link=main_link, anchored=True),
    }
    var_rows = []; var_params = {}
    for key, sp in specs.items():
        d = df.dropna(subset=SEED_COLS + sp['speed'] + sp['soil'])
        Xs, Xsp, Xso = design(d, sp); tt, ee = d['RFS_MONTHS'].values, d['EVENT'].values
        th, _ = fit(sp, Xs, Xsp, Xso, tt, ee); cv = cv_oof(d, sp)
        var_params[key] = dict(zip(param_names(sp), np.round(th, 6).tolist()))
        var_rows.append(dict(variant=key, N=len(d), C_in=concordance_index(tt, predict_t(th, sp, Xs, Xsp, Xso), ee),
                             C_cv=np.mean(cv['c_mech_folds']), C_cv_cox_same_feats=np.mean(cv['c_cox_folds']),
                             n_Tpred_le_0=int((predict_t(th, sp, Xs, Xsp, Xso) <= 0).sum()),
                             coefficients=' '.join(f'{k.replace("beta_", "").replace("soil_", "soil:")}={v:+.4g}' for k, v in var_params[key].items() if not k.startswith('alpha'))))
    table(pd.DataFrame(var_rows), 'variants_pointestimates', '%.4f')
    R['variants'] = dict(rows=var_rows, params=var_params)

    # ================================================================== 7. paired bootstrap
    sec(f'7. Paired bootstrap, {n_boot} resamples, out-of-bag scoring')
    cox_feats = {'cox6': SEED_COLS + BASE_SPEED}
    boot_specs = {k: v for k, v in specs.items() if k != 'her2_cn'}       # her2_cn drops 4 patients; handled separately
    tb = time.time()
    res = Parallel(n_jobs=args.jobs, verbose=0)(delayed(_boot_worker)(b, df, boot_specs, 'main', cox_feats) for b in range(n_boot))
    line(f'bootstrap wall time {time.time()-tb:.0f} s')
    B = pd.DataFrame(res)
    # 7a: parameter CIs, main model
    TH = np.array([x for x in B['theta_main'] if isinstance(x, list)])
    names = param_names(spec_main)
    prow = []
    a_hat, b0_hat, b_hat = unpack(th_main, spec_main)
    expected = {'alpha1_size': '+', 'alpha2_nodes': '+', 'beta_GRADE': '+', 'beta_ER_BIN': '-', 'beta_HER2_BIN': '+', 'beta_PR_BIN': '-'}
    for j, nm in enumerate(names):
        mean, lo, hi_, n = pct_ci(TH[:, j])
        prow.append(dict(parameter=nm, estimate=th_main[j], boot_mean=mean, ci_lo=lo, ci_hi=hi_, expected_sign=expected.get(nm, ''),
                         ci_excludes_zero=(lo > 0) or (hi_ < 0), sign_ok=(expected.get(nm) == ('+' if th_main[j] > 0 else '-')) if nm in expected else ''))
    b0_boot = np.array([LAM_REF - np.array(th[3:]) @ xref_vec(spec_main) for th in TH])
    m0, l0, h0, _ = pct_ci(b0_boot)
    prow.append(dict(parameter='beta0 (implied by anchor)', estimate=b0_hat, boot_mean=m0, ci_lo=l0, ci_hi=h0, expected_sign='', ci_excludes_zero=l0 > 0, sign_ok=''))
    table(pd.DataFrame(prow), 'params_ci_main', '%.5g')
    R['params_ci_main'] = prow
    # 7b: treatment variants parameter CIs
    for key in ['treat_HT', 'treat_HTCT']:
        THt = np.array([x for x in B['theta_' + key] if isinstance(x, list)]); nm_t = param_names(specs[key])
        rows_t = [dict(parameter=nm, estimate=var_params[key][nm], ci_lo=pct_ci(THt[:, j])[1], ci_hi=pct_ci(THt[:, j])[2]) for j, nm in enumerate(nm_t)]
        line(f'\nParameter CIs, variant {key}:'); table(pd.DataFrame(rows_t), f'params_ci_{key}', '%.5g'); R[f'params_ci_{key}'] = rows_t
    # 7c: delta-C CIs
    drow = []
    def add_delta(label, a, bcol):
        d = B[a] - B[bcol]; mean, lo, hi_, n = pct_ci(d)
        drow.append(dict(comparison=label, delta_C_mean=mean, ci_lo=lo, ci_hi=hi_, n_valid=n, frac_positive=float((d.dropna() > 0).mean()),
                         C_A_oob=float(B[a].mean()), C_B_oob=float(B[bcol].mean())))
    add_delta('mechanistic (main) - Cox PH (same 6 features)', 'C_main', 'C_cox6')
    for key, label in [('cont_recept', 'continuous ESR1/PGR/ERBB2 mRNA - main'), ('mki67', '+MKI67 mRNA - main'),
                       ('treat_HT', '+hormone therapy - main'), ('treat_HTCT', '+hormone & chemo - main'),
                       ('imm_sig', 'Soil=immune signature - main'), ('imm_cyt', 'Soil=cytolytic score - main'),
                       ('imm_pdl1', 'Soil=PD-L1 mRNA - main'), ('mki67_imm', '+MKI67 & immune Soil - main')]:
        add_delta(label, 'C_' + key, 'C_main')
    table(pd.DataFrame(drow), 'deltaC_ci', '%.4f')
    R['deltaC_ci'] = drow
    B.drop(columns=[c for c in B.columns if c.startswith('theta_')]).to_csv(os.path.join(out, 'bootstrap_C_values.csv'), index=False)

    # coefficient figure with CIs
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    P = pd.DataFrame(prow)
    for k, (bucket, sel) in enumerate([('Seed', P['parameter'].str.startswith('alpha')), ('Speed', P['parameter'].str.startswith('beta_'))]):
        sub = P[sel]; y = np.arange(len(sub))
        ax[k].barh(y, sub['estimate'], color=['#1f77b4' if v > 0 else '#d62728' for v in sub['estimate']], alpha=0.85)
        ax[k].errorbar(sub['estimate'], y, xerr=[sub['estimate'] - sub['ci_lo'], sub['ci_hi'] - sub['estimate']], fmt='none', ecolor='k', capsize=3)
        ax[k].set_yticks(y); ax[k].set_yticklabels([p.replace('alpha', 'α').replace('beta_', 'β ').replace('_BIN', '').replace('_', ' ') for p in sub['parameter']])
        ax[k].axvline(0, color='k', lw=0.8); ax[k].set_title(f'{bucket} bucket (95% bootstrap CI)'); ax[k].grid(alpha=0.25, axis='x')
        ax[k].spines[['top', 'right']].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(out, 'fig_params_ci.pdf')); plt.close()

    # ================================================================== save
    R['runtime_s'] = time.time() - t0
    json.dump(R, open(os.path.join(out, 'results.json'), 'w'), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    with open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write(f'# Analysis summary\n\nGenerated by revision_analysis.py; n_boot={n_boot}; main link={main_link}; '
                f'VDT_ref={VDT_REF_DAYS:.0f} d; runtime {R["runtime_s"]/60:.1f} min\n')
        f.write('\n'.join(md))
    line(f'\nDone in {R["runtime_s"]/60:.1f} min. Outputs in {os.path.relpath(out)}')

if __name__ == '__main__':
    main()

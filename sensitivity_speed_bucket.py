#!/usr/bin/env python3
"""Sensitivity analysis for the Speed bucket: are ER and PR needed for discrimination?

Companion to `revision_analysis.py` (imported unchanged: same cohort, 5-fold split, optimiser and
paired out-of-bag bootstrap resamples). Refits the anchored log1p model with reduced Speed buckets:

  main       Grade, ER, HER2, PR            (primary model of the paper)
  four       Grade, HER2                    (size, nodes, grade and HER2 = the four features whose
                                             bootstrap intervals exclude zero)
  noER       Grade, HER2, PR                (ER dropped; its coefficient is ~0)
  hr_merged  Grade, HER2, HR-negative       (ER and PR merged: HR-negative = ER- and PR-;
                                             removes the ER-PR collinearity)

and reports cross-validated concordance plus paired-bootstrap differences against the primary model.
Rank-based metrics do not depend on the doubling-time anchor for the log1p link, so these concordance
values are unaffected by the anchor; the anchor's reference phenotype (Grade 2, ER+, HER2-, PR+) is
defined with ER and PR, which is why the paper retains them (see Section 3.1).

Usage:
    python sensitivity_speed_bucket.py --tar brca_metabric.tar.gz --n-boot 1000
Outputs are written to `outputs_sensitivity_speed/` (CSV tables and a `summary.md`).
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from lifelines.utils import concordance_index

import revision_analysis as ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tar', default='brca_metabric.tar.gz')
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--out', default='outputs_sensitivity_speed')
    ap.add_argument('--jobs', type=int, default=-1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = ra.load_cohort(args.tar, os.path.join(args.out, 'cohort_cache.csv'))
    df['HR_NEG'] = ((df['ER_BIN'] == 0) & (df['PR_BIN'] == 0)).astype(float)   # reference phenotype (ER+ and/or PR+) = 0
    N = len(df)
    print(f'Analytic cohort N={N:,}, events={int(df["EVENT"].sum()):,}; '
          f'ER-PR phi = {np.corrcoef(df["ER_BIN"], df["PR_BIN"])[0, 1]:.3f}; '
          f'ER+/PR- n={int(((df.ER_BIN == 1) & (df.PR_BIN == 0)).sum())}, ER-/PR+ n={int(((df.ER_BIN == 0) & (df.PR_BIN == 1)).sum())}')

    def spec(name, speed):
        return dict(name=name, speed=speed, soil=[], link='log1p', anchored=True)
    specs = {
        'main':      spec('main', ra.BASE_SPEED),
        'four':      spec('four', ['GRADE', 'HER2_BIN']),
        'noER':      spec('noER', ['GRADE', 'HER2_BIN', 'PR_BIN']),
        'hr_merged': spec('hr_merged', ['GRADE', 'HR_NEG', 'HER2_BIN']),
    }
    cox_feats = {'cox6': ra.SEED_COLS + ra.BASE_SPEED}

    # ---- point estimates and 5-fold cross-validation
    rows = []
    for key, sp in specs.items():
        d = df.dropna(subset=ra.SEED_COLS + sp['speed'])
        Xs, Xsp, Xso = ra.design(d, sp); t, e = d['RFS_MONTHS'].values, d['EVENT'].values
        th, _ = ra.fit(sp, Xs, Xsp, Xso, t, e)
        cv = ra.cv_oof(d, sp)
        tp = ra.predict_t(th, sp, Xs, Xsp, Xso)
        rows.append(dict(model=key, n_speed_features=len(sp['speed']), N=len(d),
                         C_in=concordance_index(t, tp, e), C_cv=np.mean(cv['c_mech_folds']),
                         C_cv_cox_same_feats=np.mean(cv['c_cox_folds']), n_Tpred_le_0=int((tp <= 0).sum()),
                         coefficients=' '.join(f'{k.replace("beta_", "")}={v:+.4g}'
                                               for k, v in dict(zip(ra.param_names(sp), th)).items() if not k.startswith('alpha'))))
    P = pd.DataFrame(rows)
    print(P.to_string(index=False))
    P.to_csv(os.path.join(args.out, 'table_point_estimates.csv'), index=False)

    # ---- paired bootstrap (same seeds as revision_analysis.py, so 'main - Cox' reproduces the paper)
    tb = time.time()
    res = Parallel(n_jobs=args.jobs, verbose=0)(delayed(ra._boot_worker)(b, df, specs, 'main', cox_feats) for b in range(args.n_boot))
    print(f'bootstrap {args.n_boot} resamples, wall time {time.time() - tb:.0f} s')
    B = pd.DataFrame(res)
    drow = []
    for a, b, label in [('C_main', 'C_cox6', 'mechanistic (main) - Cox PH (same 6 features)  [reproduces Section 3.2]'),
                        ('C_four', 'C_main', 'Grade + HER2 only - main'),
                        ('C_noER', 'C_main', 'drop ER only - main'),
                        ('C_hr_merged', 'C_main', 'ER and PR merged (HR-negative) - main')]:
        dlt = B[a] - B[b]; m, lo, hi, n = ra.pct_ci(dlt)
        drow.append(dict(comparison=label, delta_C_mean=m, ci_lo=lo, ci_hi=hi, n_valid=n,
                         frac_positive=float((dlt.dropna() > 0).mean()),
                         C_A_oob=float(B[a].mean()), C_B_oob=float(B[b].mean())))
    D = pd.DataFrame(drow)
    print(D.to_string(index=False))
    D.to_csv(os.path.join(args.out, 'table_deltaC_ci.csv'), index=False)

    with open(os.path.join(args.out, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('# Speed-bucket sensitivity (anchored log1p model)\n\n'
                f'Cohort N={N:,}; {args.n_boot} paired out-of-bag bootstrap resamples (seeds as in `revision_analysis.py`).\n\n'
                '## Point estimates and 5-fold cross-validation\n\n' + ra.md_table(P.drop(columns=['coefficients']), '%.4f') + '\n\n'
                '## Paired bootstrap differences in concordance\n\n' + ra.md_table(D, '%.4f') + '\n')


if __name__ == '__main__':
    main()

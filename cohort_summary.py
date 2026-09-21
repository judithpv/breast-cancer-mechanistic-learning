#!/usr/bin/env python3
"""Cohort summary and descriptive checks quoted in the paper.

Companion to `revision_analysis.py` (imported unchanged: same cohort loader, model and optimiser).
Writes, for the analytic METABRIC cohort (N = 1,814):

  table1_baseline_characteristics.csv   Table 1 of the paper (baseline clinicopathological characteristics)
  treatment_by_receptor.csv             hormone therapy by ER status and chemotherapy by HER2 status (Section 4.1)
  negative_predicted_times.csv          every patient with a predicted relapse time <= 0 under each Seed link
                                        (linear, log1p, pN, bounded), with covariates (Sections 3.1 and 4.2)
  doubling_time_range.csv               model-implied doubling times across the cohort, primary model (Section 4.2)
  summary.md                            human-readable summary of the above

Usage:
    python cohort_summary.py --tar brca_metabric.tar.gz
Outputs are written to `outputs_cohort_summary/` (or --out).
"""
import argparse, os
import numpy as np, pandas as pd

import revision_analysis as ra


def fmt_pct(n, total):
    return f'{int(n):,} ({100 * n / total:.1f}%)'


def baseline_table(df):
    N = len(df); e = df['EVENT'].values.astype(int)
    t = df['RFS_MONTHS']; nodes = df['LYMPH_NODES_EXAMINED_POSITIVE']
    rows = [('Cohort size (N)', 'Patients', f'{N:,}'),
            ('Distant relapse', 'Event observed', fmt_pct(e.sum(), N)),
            ('', 'Right-censored', fmt_pct(N - e.sum(), N)),
            ('Follow-up time', 'Months, median [IQR]', f'{t.median():.1f} [{t.quantile(.25):.1f} - {t.quantile(.75):.1f}]'),
            ('Tumour size', 'mm, mean +- SD', f'{df["TUMOR_SIZE"].mean():.1f} +- {df["TUMOR_SIZE"].std():.1f}'),
            ('Positive lymph nodes', 'Count, median [range]', f'{nodes.median():.0f} [{nodes.min():.0f} - {nodes.max():.0f}]')]
    for g in (1, 2, 3):
        rows.append(('Histological grade' if g == 1 else '', f'Grade {g}', fmt_pct((df['GRADE'] == g).sum(), N)))
    for col, name in [('ER_BIN', 'ER status'), ('HER2_BIN', 'HER2 status'), ('PR_BIN', 'PR status')]:
        rows.append((name, 'Negative (0)', fmt_pct((df[col] == 0).sum(), N)))
        rows.append(('', 'Positive (1)', fmt_pct((df[col] == 1).sum(), N)))
    return pd.DataFrame(rows, columns=['variable', 'category_or_unit', 'summary'])


def treatment_table(df):
    rows = []
    for er, lab in [(1, 'ER-positive'), (0, 'ER-negative')]:
        d = df[df['ER_BIN'] == er]
        rows.append(dict(comparison=f'hormone therapy, {lab}', n_treated=int((d['HT_BIN'] == 1).sum()), n=len(d),
                         n_missing=int(d['HT_BIN'].isna().sum())))
    for h, lab in [(1, 'HER2-positive'), (0, 'HER2-negative')]:
        d = df[df['HER2_BIN'] == h]
        rows.append(dict(comparison=f'chemotherapy, {lab}', n_treated=int((d['CHEMO_BIN'] == 1).sum()), n=len(d),
                         n_missing=int(d['CHEMO_BIN'].isna().sum())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tar', default='brca_metabric.tar.gz')
    ap.add_argument('--out', default='outputs_cohort_summary')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = ra.load_cohort(args.tar, os.path.join(args.out, 'cohort_cache.csv'))
    N = len(df); t, e = df['RFS_MONTHS'].values, df['EVENT'].values
    print(f'Analytic cohort N={N:,}, events={int(e.sum()):,}')

    # ---- Table 1 and treatment structure
    T1 = baseline_table(df); print(T1.to_string(index=False))
    T1.to_csv(os.path.join(args.out, 'table1_baseline_characteristics.csv'), index=False)
    TR = treatment_table(df); print(TR.to_string(index=False))
    TR.to_csv(os.path.join(args.out, 'treatment_by_receptor.csv'), index=False)

    # ---- predicted relapse times <= 0 under each Seed link (anchored model, Speed = grade, ER, HER2, PR)
    neg_rows, counts, primary = [], {}, None
    for link in ['linear', 'log1p', 'pN', 'bounded']:
        spec = dict(name=f'anchored_{link}', speed=ra.BASE_SPEED, soil=[], link=link, anchored=True)
        Xs, Xsp, Xso = ra.design(df, spec)
        th, _ = ra.fit(spec, Xs, Xsp, Xso, t, e)
        tp = ra.predict_t(th, spec, Xs, Xsp, Xso)
        counts[link] = dict(n_negative=int((tp <= 0).sum()), min_Tpred=float(tp.min()))
        for i in np.where(tp <= 0)[0]:
            r = df.iloc[i]
            neg_rows.append(dict(seed_link=link, patient_id=r['PATIENT_ID'], tumour_size_mm=r['TUMOR_SIZE'],
                                 positive_nodes=r['LYMPH_NODES_EXAMINED_POSITIVE'], grade=r['GRADE'],
                                 ER=r['ER_BIN'], HER2=r['HER2_BIN'], PR=r['PR_BIN'], T_pred_months=float(tp[i])))
        if link == 'log1p':
            primary = (spec, th, Xsp)
    NEG = pd.DataFrame(neg_rows).sort_values(['seed_link', 'T_pred_months'])
    print(NEG.to_string(index=False))
    NEG.to_csv(os.path.join(args.out, 'negative_predicted_times.csv'), index=False)

    # ---- model-implied doubling times across the cohort (primary model)
    spec, th, Xsp = primary
    lam = ra.lam_of(th, spec, Xsp); dt = np.log(2) / lam * ra.DAYS_PER_MONTH
    DT = pd.DataFrame([dict(min_days=dt.min(), median_days=float(np.median(dt)), max_days=dt.max(),
                            lambda_min=lam.min(), lambda_max=lam.max())])
    print(DT.to_string(index=False))
    DT.to_csv(os.path.join(args.out, 'doubling_time_range.csv'), index=False)

    nodes = df['LYMPH_NODES_EXAMINED_POSITIVE']
    with open(os.path.join(args.out, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write(f'# Cohort summary (analytic cohort N={N:,}, {int(e.sum()):,} distant relapses)\n\n'
                '## Table 1: baseline clinicopathological characteristics\n\n' + ra.md_table(T1, '%s') + '\n\n'
                '## Treatment by receptor status\n\n' + ra.md_table(TR, '%s') + '\n\n'
                '## Predicted relapse times <= 0 by Seed link (anchored model)\n\n'
                + ra.md_table(pd.DataFrame(counts).T.reset_index().rename(columns={'index': 'seed_link'}), '%.4g') + '\n\n'
                + ra.md_table(NEG, '%.4g') + '\n\n'
                f'Maximum recorded positive-node count in the analytic cohort: {nodes.max():.0f}; '
                f'patients with >= 33 positive nodes: {int((nodes >= 33).sum())}.\n\n'
                '## Model-implied doubling times across the cohort (primary log1p model)\n\n' + ra.md_table(DT, '%.4g') + '\n')


if __name__ == '__main__':
    main()

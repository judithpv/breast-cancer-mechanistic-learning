# Cohort summary (analytic cohort N=1,814, 737 distant relapses)

## Table 1: baseline clinicopathological characteristics

| variable | category_or_unit | summary |
|---|---|---|
| Cohort size (N) | Patients | 1,814 |
| Distant relapse | Event observed | 737 (40.6%) |
|  | Right-censored | 1,077 (59.4%) |
| Follow-up time | Months, median [IQR] | 101.7 [41.8 - 169.2] |
| Tumour size | mm, mean +- SD | 26.2 +- 15.3 |
| Positive lymph nodes | Count, median [range] | 0 [0 - 45] |
| Histological grade | Grade 1 | 163 (9.0%) |
|  | Grade 2 | 732 (40.4%) |
|  | Grade 3 | 919 (50.7%) |
| ER status | Negative (0) | 431 (23.8%) |
|  | Positive (1) | 1,383 (76.2%) |
| HER2 status | Negative (0) | 1,587 (87.5%) |
|  | Positive (1) | 227 (12.5%) |
| PR status | Negative (0) | 863 (47.6%) |
|  | Positive (1) | 951 (52.4%) |

## Treatment by receptor status

| comparison | n_treated | n | n_missing |
|---|---|---|---|
| hormone therapy, ER-positive | 999 | 1383 | 0 |
| hormone therapy, ER-negative | 124 | 431 | 0 |
| chemotherapy, HER2-positive | 99 | 227 | 0 |
| chemotherapy, HER2-negative | 288 | 1587 | 0 |

## Predicted relapse times <= 0 by Seed link (anchored model)

| seed_link | n_negative | min_Tpred |
|---|---|---|
| linear | 7 | -63.07 |
| log1p | 3 | -32.44 |
| pN | 3 | -33.1 |
| bounded | 0 | 0.5022 |

| seed_link | patient_id | tumour_size_mm | positive_nodes | grade | ER | HER2 | PR | T_pred_months |
|---|---|---|---|---|---|---|---|---|
| linear | MB-7297 | 25 | 45 | 3 | 1 | 0 | 1 | -63.07 |
| linear | MB-0660 | 160 | 22 | 3 | 0 | 0 | 0 | -58.96 |
| linear | MB-0406 | 180 | 17 | 3 | 1 | 0 | 0 | -52.49 |
| linear | MB-0361 | 24 | 41 | 3 | 0 | 1 | 0 | -34.36 |
| linear | MB-0112 | 150 | 14 | 3 | 1 | 0 | 0 | -18.92 |
| linear | MB-7283 | 30 | 33 | 2 | 1 | 0 | 0 | -14.18 |
| linear | MB-6063 | 120 | 16 | 3 | 0 | 1 | 0 | -4.961 |
| log1p | MB-0406 | 180 | 17 | 3 | 1 | 0 | 0 | -32.44 |
| log1p | MB-0660 | 160 | 22 | 3 | 0 | 0 | 0 | -25.09 |
| log1p | MB-0112 | 150 | 14 | 3 | 1 | 0 | 0 | -8.721 |
| pN | MB-0406 | 180 | 17 | 3 | 1 | 0 | 0 | -33.1 |
| pN | MB-0660 | 160 | 22 | 3 | 0 | 0 | 0 | -19.57 |
| pN | MB-0112 | 150 | 14 | 3 | 1 | 0 | 0 | -12.96 |

Maximum recorded positive-node count in the analytic cohort: 45; patients with >= 33 positive nodes: 3.

## Model-implied doubling times across the cohort (primary log1p model)

| min_days | median_days | max_days | lambda_min | lambda_max |
|---|---|---|---|---|
| 134 | 172.4 | 199.6 | 0.1057 | 0.1575 |

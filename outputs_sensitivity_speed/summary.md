# Speed-bucket sensitivity (anchored log1p model)

Cohort N=1,814; 1000 paired out-of-bag bootstrap resamples (seeds as in `revision_analysis.py`).

## Point estimates and 5-fold cross-validation

| model | n_speed_features | N | C_in | C_cv | C_cv_cox_same_feats | n_Tpred_le_0 |
|---|---|---|---|---|---|---|
| main | 4 | 1814 | 0.6554 | 0.6540 | 0.6564 | 3 |
| four | 2 | 1814 | 0.6542 | 0.6538 | 0.6565 | 3 |
| noER | 3 | 1814 | 0.6554 | 0.6540 | 0.6567 | 3 |
| hr_merged | 3 | 1814 | 0.6544 | 0.6531 | 0.6561 | 3 |

## Paired bootstrap differences in concordance

| comparison | delta_C_mean | ci_lo | ci_hi | n_valid | frac_positive | C_A_oob | C_B_oob |
|---|---|---|---|---|---|---|---|
| mechanistic (main) - Cox PH (same 6 features)  [reproduces Section 3.2] | -0.0020 | -0.0159 | 0.0111 | 1000 | 0.3930 | 0.6507 | 0.6527 |
| Grade + HER2 only - main | 0.0017 | -0.0038 | 0.0104 | 1000 | 0.6220 | 0.6524 | 0.6507 |
| drop ER only - main | 0.0015 | -0.0006 | 0.0081 | 1000 | 0.7940 | 0.6523 | 0.6507 |
| ER and PR merged (HR-negative) - main | -0.0000 | -0.0046 | 0.0074 | 1000 | 0.4350 | 0.6507 | 0.6507 |

# Mechanistic Learning Framework for Breast Cancer Relapse Prediction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22830698.svg)](https://doi.org/10.5281/zenodo.22830698)

Code accompanying:

> Pérez-Velázquez, J., Gölgeli, M., Kulaç, İ. *A Mechanistic Learning
> Framework for Breast Cancer Relapse Prediction.* Mathematical
> Biosciences (manuscript MBS-D-26-00734).

A modular **Seed / Speed / Soil** framework that maps routine
histopathological variables onto a mechanistic exponential-growth model
of metastatic relapse, fitted and evaluated on the METABRIC cohort. See
the paper for the full derivation and biological rationale.

## What's here

| File | Purpose |
|---|---|
| `mechanistic_model.py` | Base implementation: data loading, ODE model, domain mapping (Seed/Speed), censoring-aware loss, L-BFGS-B fitting, Cox PH baseline, 5-fold CV, using an unanchored model with a linear nodal link. |
| `experiments.py` | Feature-selection ablation experiments (Table 2 in the paper): isolates the marginal contribution of HER2/PR and candidate Soil proxies (cellularity, age). |
| `revision_analysis.py` | Primary analysis and sensitivity analyses: scale-anchored model (log1p nodal link), identifiability analysis, alternative Seed links (linear / pN-stage / bounded), paired-bootstrap confidence intervals, treatment-confounding sensitivity, molecular/immune Soil extensions, and sensitivity to the doubling-time anchor and detection threshold. Companion to `mechanistic_model.py`. This is the entry point for reproducing the main results reported in the paper. |
| `sensitivity_speed_bucket.py` | Speed-bucket sensitivity analysis (Section 3.1 of the paper): refits the anchored log1p model with Grade + HER2 only, with ER dropped, and with ER and PR merged into one hormone-receptor variable, and reports cross-validated concordance and paired-bootstrap differences against the primary model. Imports `revision_analysis.py` unchanged (same cohort, folds and bootstrap seeds). |
| `make_fig1.py` | Regenerates the Figure 1 pipeline schematic. |
| `outputs_revision/`, `outputs_revision_bounded/` | Example outputs (tables, figures, bootstrap results) for the two analyses with the log1p link (`outputs_revision/`) and the bounded link (`outputs_revision_bounded/`) as the main model, included for reference. |
| `outputs_sensitivity_speed/` | Outputs of `sensitivity_speed_bucket.py` (CSV tables and `summary.md`). |

## Data

This repository does **not** bundle patient data. All scripts read the
public METABRIC cohort directly from a local copy of the cBioPortal
archive:

1. Download `brca_metabric.tar.gz` from
   <https://www.cbioportal.org/study/summary?id=brca_metabric>
2. Run any script with `--tar path/to/brca_metabric.tar.gz` (or place
   the file in the working directory, which is the default).

## Requirements

```
pip install -r requirements.txt
```

Developed and tested with the versions pinned in `requirements.txt`
(Python 3.11+).

## Usage

```bash
# Main model fit, C-index, Cox PH baseline, 5-fold CV
python mechanistic_model.py brca_metabric.tar.gz

# Feature-selection ablation table
python experiments.py brca_metabric.tar.gz

# Full analysis (anchored model, identifiability, bootstrap CIs, sensitivity analyses)
python revision_analysis.py --tar brca_metabric.tar.gz --n-boot 1000
# add --quick for a fast smoke test (n_boot=50)

# Speed-bucket sensitivity analysis (Section 3.1); needs revision_analysis.py in the same folder
python sensitivity_speed_bucket.py --tar brca_metabric.tar.gz --n-boot 1000
```

Outputs are written to `outputs_revision/` (tables as CSV, figures as
PDF, a `results.json`, and a human-readable `summary.md`); the sensitivity
analysis writes to `outputs_sensitivity_speed/`.

`table_vdt_sensitivity.csv` shows how the absolute scale of the initial burden
$n_0$ depends on the literature doubling-time anchor (100-500 days); rank-based
results do not depend on it for the log1p link.
`table_anchor_M_sensitivity.csv` refits the log1p and bounded links from scratch for different
anchors (150, 185, 250 days) and detection thresholds ($M=10^8$, $10^9$ cells): for log1p, $T_{\mathrm{pred}}$ and the C-index are
unchanged, whereas the bounded link's fit depends on both. The badge above resolves to the latest archived
version; the citation below is for the version matching these outputs.

## Status

This is a research proof of concept accompanying a manuscript currently
under peer review. Internal five-fold cross-validation is reported;
external multi-cohort validation has not yet been performed. See the
paper's Limitations and Future Directions sections.

## Citation

If you use this code, please cite the archived release and the paper
above (full paper citation to be updated on acceptance):

> Pérez-Velázquez, J., Gölgeli, M., Kulaç, İ. (2026). Mechanistic
> Learning Framework for Breast Cancer Relapse Prediction (v1.0.5)
> [Software]. Zenodo. https://doi.org/10.5281/zenodo.22871858

And the METABRIC dataset:

> Curtis, C. et al. The genomic and transcriptomic architecture of 2,000
> breast tumours reveals novel subgroups. *Nature* 486, 346–352 (2012).
> Pereira, B. et al. The somatic mutation profiles of 2,433 breast
> cancers refine their genomic and transcriptomic landscapes. *Nature
> Communications* 7, 11479 (2016).

## License

MIT — see `LICENSE`.

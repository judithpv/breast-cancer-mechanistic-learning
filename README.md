# Mechanistic Learning Framework for Breast Cancer Relapse Prediction

Code accompanying:

> Pérez-Velázquez, J., Gölgeli, M., Kulaç, İ. *A Mechanistic Learning
> Framework for Breast Cancer Relapse Prediction.* Mathematical
> Biosciences (under revision, manuscript MBS-D-26-00734).

A modular **Seed / Speed / Soil** framework that maps routine
histopathological variables onto a mechanistic exponential-growth model
of metastatic relapse, fitted and evaluated on the METABRIC cohort. See
the paper for the full derivation and biological rationale.

## What's here

| File | Purpose |
|---|---|
| `mechanistic_model.py` | Core framework: data loading, ODE model, domain mapping (Seed/Speed), censoring-aware loss, L-BFGS-B fitting, Cox PH baseline, 5-fold CV. This is the entry point for reproducing the main results. |
| `experiments.py` | Feature-selection ablation experiments (Table 1 in the paper): isolates the marginal contribution of HER2/PR and candidate Soil proxies (cellularity, age). |
| `revision_analysis.py` | Major-revision analyses: scale-anchored identifiability fix, saturating Seed links (log1p / pN-stage / bounded), paired-bootstrap confidence intervals, treatment-confounding sensitivity, molecular/immune Soil extensions. Companion to `mechanistic_model.py`, which is left as originally submitted. |
| `make_fig1.py` | Regenerates the Figure 1 pipeline schematic. |
| `outputs_revision/`, `outputs_revision_bounded/` | Example outputs (tables, figures, bootstrap results) from the two saturating-link variants discussed in the revision, included for reference. |

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

# Full revision analysis (identifiability, bootstrap CIs, sensitivity analyses)
python revision_analysis.py --tar brca_metabric.tar.gz --n-boot 1000
# add --quick for a fast smoke test (n_boot=50)
```

Outputs are written to `outputs_revision/` (tables as CSV, figures as
PDF, a `results.json`, and a human-readable `summary.md`).

## Status

This is a research proof of concept accompanying a manuscript currently
under peer review. Internal five-fold cross-validation is reported;
external multi-cohort validation has not yet been performed. See the
paper's Limitations and Future Directions sections.

## Citation

If you use this code, please cite the paper above (full citation to be
updated on acceptance) and the METABRIC dataset:

> Curtis, C. et al. The genomic and transcriptomic architecture of 2,000
> breast tumours reveals novel subgroups. *Nature* 486, 346–352 (2012).
> Pereira, B. et al. The somatic mutation profiles of 2,433 breast
> cancers refine their genomic and transcriptomic landscapes. *Nature
> Communications* 7, 11479 (2016).

## License

MIT — see `LICENSE`.

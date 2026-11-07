# Replication package

**Paper.** Bennedsen, Hillebrand, Koopman (2026), *Auditing the Global
Carbon Budget: exploring the 2024--2025 vintage shift.* Submitted to
Earth System Science Data.

This package reproduces every table and figure in the paper from the
raw inputs in `data/`. The fitted model artifacts behind the paper's
tables are shipped in `results/`, so the tables and figures can also be
re-rendered in under a minute without re-estimating anything.

## Contents

```
gcb_audit_replication/
├── README.md                this file
├── LICENSE                  MIT licence
├── requirements.txt         Python dependencies
├── run_all.py               master driver (stages 1-7)
├── code/                    all Python scripts
├── data/                    nine GCB vintages, climate indices, SAOD,
│                            GCB2024/consolidated deposit files
├── results/                 fitted artifacts per vintage (parameters,
│                            Hessians, covariances, smoothed states)
└── output/
    ├── tables/              paper tables (LaTeX fragments)
    ├── figures/             paper figures (PDF, some PNG)
    └── csv/                 intermediate and cited numbers
```

## Requirements

Python 3.10 or later with numpy, pandas, scipy, statsmodels,
matplotlib and openpyxl (`pip install -r requirements.txt`). The
shipped outputs were produced with numpy 2.2.6, pandas 2.3.3, scipy
1.15.3, statsmodels 0.15.0, matplotlib 3.10.9, openpyxl 3.1.5.

## How to run

```bash
pip install -r requirements.txt
python run_all.py                    # full reproduction (~30-60 min)
python run_all.py --from 5_tables    # use shipped fits, render outputs (<1 min)
python run_all.py --to 1_panel       # model-free audit only (~2 min)
```

| Stage | What it does | Approx. time |
| --- | --- | --- |
| `1_panel` | Model-free audit: Tables 1-4 and A1-A4, Figures 1-2, H&C2023 counterfactual | ~2 min |
| `2_estimate` | Baseline MDS-GCB maximum likelihood on each vintage 2017-2025 (multistart) | ~10-20 min |
| `3_hessian` | Step-checked Hessians and covariances for the baseline fits | ~1 min |
| `4_muC` | mu_C-augmented model on each vintage (multistart), summary CSV | ~10-20 min |
| `5_tables` | Tables 5, 6 and C1 from the artifacts in `results/` | seconds |
| `6_figures` | Figures 3, 4, 5 and the full-sample fit behind Figures D1-D2 | ~5 min |
| `7_supplementary` | Diagnostics not shown in the paper (panel gaps, AF/SR bands) | ~10 min |

The driver stops at the first failure. Stages 2 and 4 overwrite the
artifacts in `results/`; run from a fresh copy of the package to keep
the shipped ones.

## Code-to-paper map

| Paper item | File in `output/` | Producer script |
| --- | --- | --- |
| Table 1 | `tables/table_bim_three_vintages.tex` | `table_bim_three_vintages.py` |
| Table 2 | `tables/table_step1_subperiods.tex` | `audit_step1_subperiods.py` |
| Table 3 | `tables/table_step2_vintages.tex` | `audit_step2_vintages.py` |
| Table 4 | `tables/table_ob_2024_2025.tex` | `audit_step4_appendixB.py` + `table_ob_2024_2025.py` |
| Table 5 | `tables/table_muC_2025_comparison.tex` | `table_muC_2025_comparison.py` |
| Table 6 | `tables/table_muC_vintages.tex` | `jrssa_estimate_muC_all_vintages.py` + `table_muC_vintages.py` |
| Tables A1-A4 | `tables/table_step4_appendixB.tex` | `audit_step4_appendixB.py` |
| Table C1 | `tables/appendixA_full_parameters_vintages.tex` | `table_jrssa_full_vintages.py` |
| Figure 1 | `figures/figure_bim_vintage_means.pdf` | `bim_vintage_means.py` |
| Figure 2 | `figures/figure_panel_af_sr_vintages.pdf` | `figure_panel_af_sr_vintages.py` |
| Figure 3 | `figures/figure_jrssa_beta_vintages.pdf` | `figure_jrssa_beta_vintages.py` |
| Figure 4 | `figures/figure_smoothed_sinks_2025.pdf` | `figure_smoothed_sinks_2025.py` |
| Figure 5, D1, D2 | `figures/figure_X1_all_vintages.pdf`, `figure_muC_full_states_2x2.pdf`, `figure_muC_full_X1.pdf` | `appendix_muC_drift_figures.py` |
| Counterfactual (Sect. 2.4) | `csv/hc_counterfactual.csv` | `audit_step5_hc_counterfactual.py` |

## Estimation notes

Every maximum-likelihood fit runs through a seeded multistart
(`code/estimation_utils.py`): about a dozen polished starts per fit, a
degeneracy screen that rejects fits where the Kalman filter has broken
down, and a console report of every distinct local maximum found. The
likelihood is multimodal in the persistence parameter phi_1 for some
vintages; on the 2018 and 2024 vintages a stationary and a
boundary fit lie within 0.2 log-likelihood units of each other, as
discussed in the paper's Limitations section. The multistart keeps
the highest non-degenerate likelihood.

Standard errors come from a step-checked central-difference Hessian
with a relative eigenvalue floor (`robust_hessian` in the same
module). The script warns when a standard error is not identified by
the curvature.

Reproducibility: the modes and the reported estimates are stable
across machines; the last digits of an optimum and the second decimal
of some finite-difference standard errors can differ across BLAS
builds. The shipped artifacts in `results/` are the fits reported in
the paper.

## Data

- Nine vintages of the Global Carbon Budget master spreadsheet
  (2017-2025), distributed by the Global Carbon Project under CC BY
  4.0 (Friedlingstein et al., various years; the 2025 operational
  file is `Global_Carbon_Budget_2025_v0.7.xlsx`).
- `Global_Carbon_Budget_2024_v1.0.csv` and `GCB_adjustments.csv`:
  the data deposit of the consolidated reanalysis (Friedlingstein et
  al., 2026, Nature 649, 98-103; Zenodo 10.5281/zenodo.16367993).
- `climate_indices_annual_1959_2024.csv` (calendar-year ENSO 3.4,
  NAO, PDO) and `climate_indices_annual_litstd_1959_2024.csv` (the
  shifted windows used in the sink state equations), built from the
  NOAA CPC and NCEI monthly series (ERSSTv5).
- `saod_annual_1959_2024.csv`: stratospheric aerosol optical depth,
  GISS (Sato et al.) spliced with GloSSAC v2.22, area-weighted global
  mean.

Appendix B of the paper documents the series and windows in detail.
Please obtain current versions of these inputs from their original
providers for any use beyond replication.

## Licence

The code is released under the MIT licence (see `LICENSE`). The data
files remain under their original providers' terms.

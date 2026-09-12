# CHARLS–HRS persistent metabolic burden analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22334763.svg)](https://doi.org/10.5281/zenodo.22334763)

This repository contains the statistical analysis code used for the harmonized
CHARLS–HRS study of depressive symptoms, memory, sustained activities-of-daily-
living (ADL) independence, disability transition, and death among adults with
persistent metabolic burden.

## Scope and disclosure boundary

The repository contains code and reproducibility metadata only. It does **not**
contain participant-level data, derived data, numerical result tables, figures,
manuscripts, submission files, credentials, contact details, or local machine
paths. Generated outputs are ignored by Git.

CHARLS and HRS files are third-party cohort data and are not redistributed.
Researchers must obtain their own authorized copies and comply with the
respective data-use conditions:

- CHARLS: <https://charls.pku.edu.cn/en/>
- HRS: <https://hrs.isr.umich.edu/data-products>

HRS biomarker and sensitive-health products may require an additional data-use
agreement beyond ordinary HRS registration.

## Analysis design implemented

- Persistent metabolic burden is defined from repeated biomarker assessments.
- The cross-cohort bridge uses four harmonized components: waist circumference,
  blood pressure/hypertension, HbA1c/diabetes, and HDL cholesterol.
- The bridge threshold is at least three abnormal components at both exposure
  waves (CHARLS 2011/2015; HRS 2010/2014).
- Functional outcomes are evaluated in 2018 and 2020.
- Models use multiple imputation, cohort biomarker weights, stabilized response
  weights, complex-survey covariance estimation, and prespecified multiplicity
  adjustment.
- The fixed random seed for the harmonized analysis is `20260903`.

## Repository structure

```text
config/example.env                    Environment-variable template
src/formal_analysis.py                CHARLS variable construction utilities
src/advanced_inference.py             CHARLS cohort and survey-model routines
src/hrs_external_validation.py        HRS construction and inference routines
src/charls_hrs_harmonized_analysis.py Cross-cohort analysis entry point
src/enhanced_sensitivity_analysis.py  Standardized risks, E-values, physical-activity adjustment, and landmark models
src/make_tables.py                    Descriptive and comparison tables
src/make_figures.R                    Forest plots and sensitivity figure
tests/test_code_release.py            Offline syntax and disclosure checks
requirements.txt                      Python package versions
environment-r.txt                     R and R-package versions
```

## Expected local data layout

Set `CHARLS_DATA_DIR` to a directory containing the following licensed files:

```text
Harmonized CHARLS/H_CHARLS_D_Data/H_CHARLS_D_Data.dta
2011/Blood_20140429/Blood_20140429.dta
2011/weight/weight.dta
2011/PSU/PSU.dta
2015/Blood/Blood.dta
2015/Weights/Weights.dta
2018/CHARLS2018r/Sample_Infor.dta
2020/CHARLS2020r/Health_Status_and_Functioning.dta
2020/CHARLS2020r/Exit_Module.dta
2020/CHARLS2020r/Sample_Infor.dta
```

Set `HRS_DATA_DIR` to a directory containing:

```text
01_rand_longitudinal/extracted/randhrs1992_2022v1_STATA/randhrs1992_2022v1.dta
06_harmonized_hrs/extracted/H_HRS_d.dta
DBS_CRP/HRS_CRP_XWAVE.dta
```

Set `HRS_BIOMARKER_DIR` to the directory containing the registered biomarker
wave files used by the bridge:

```text
biomk10bl_r.dta
biomk14bl.dta
```

File and variable names follow the releases used in the analysis. If a provider
revises a filename while retaining equivalent variables, update the path mapping
at the top of the corresponding source module and record the change in a new
software release.

## Reproduction

Python 3.12 was used for the archived release. Create an isolated environment
and install the pinned packages:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Define the environment variables shown in `config/example.env`, then run:

```bash
python src/charls_hrs_harmonized_analysis.py
python src/enhanced_sensitivity_analysis.py
python src/make_tables.py
Rscript src/make_figures.R
```

The first command writes the main harmonized estimates and diagnostics. The
second writes standardized state probabilities, marginal risk contrasts,
E-values, physical-activity-adjusted estimates, and landmark estimates. The
remaining commands create tables and figures from generated files. None of
these outputs should be committed to this repository.

## Verification without cohort data

Run the offline checks from the repository root:

```bash
python -m unittest discover -s tests -v
```

These checks compile every Python script and scan the release tree for prohibited
data/manuscript formats, absolute local paths, contact details, and common secret
patterns. They do not access cohort data.

## Licence and citation

The analysis code is released under the MIT License. Cite the versioned archival
release rather than an unversioned branch: <https://doi.org/10.5281/zenodo.22334763>.
Machine-readable citation metadata are provided in `CITATION.cff`.

## Data and code availability wording

The statistical code is openly available from this repository and its versioned
archival DOI. The participant-level CHARLS and HRS data are not redistributed by
the authors because they are third-party cohort resources; qualified researchers
may obtain access directly from the respective study websites under their data-
use procedures.

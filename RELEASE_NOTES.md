# Release notes

## Version 2.1.0 — 2026-09-14

- Adds full adjustment for demographic, socioeconomic, behavioral, morbidity,
  self-rated-health, and metabolic covariates.
- Restores stabilized inverse-probability response weighting by interval.
- Adds disability-or-death and two-or-more-ADL-difficulty sensitivities.
- Re-estimates the ELSA Waves 2/4 temporal window with the same four-component
  phenotype, logistic model, covariate set, and OR estimand as the main window.
- Retains a code-only disclosure boundary; no manuscripts or participant-level
  or aggregate results are included.

## Version 2.0.0 — 2026-09-14

- Adds the ELSA Wave 4/Wave 6 persistent metabolic-burden cohort.
- Extends HRS functional follow-up through 2022.
- Recasts functional follow-up as repeated independence-to-disability,
  disability-to-independence, and data-supported mortality transitions.
- Adds multiple-imputation person-period models, random-effects summaries,
  interval-specific checks, and continuous-time Markov sensitivity models.
- Adds R code for transition-design, forest, and interval-stability figures.
- Retains a code-only disclosure boundary: no data, derived results, figures,
  manuscripts, contact details, local paths, or credentials are included.

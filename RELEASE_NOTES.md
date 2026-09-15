# Release notes

## Version 2.2.2 — 2026-09-15

- Adds the principal continuous two-wave depressive symptom-burden analysis to
  the ELSA Waves 2/4 temporal-window sensitivity analysis.
- Uses Wave 2 morbidity and self-rated health, biomarker and response weights,
  and the same survey-weighted logistic OR estimand as the principal ELSA window.
- Retains the threshold-based and memory estimates as secondary analyses.
- Maintains the code-only release boundary; no manuscripts, data, numerical
  outputs, figures, local paths, credentials, or personal contact details are included.

## Version 2.2.1 — 2026-09-15

- Adds Rubin-pooled multivariable Wald tests for the nonlinear restricted-cubic-spline terms.
- Adds fully adjusted, response-weighted estimates for each follow-up interval and after exclusion of the first post-anchor interval.
- Annotates the spline figure with cohort-specific nonlinearity P values.
- Maintains the code-only release boundary; no manuscripts, data, numerical outputs, figures, local paths, credentials, or personal contact details are included.

## Version 2.2.0 — 2026-09-15

- Makes the cohort-standardized continuous two-wave depressive symptom burden
  the principal harmonized exposure for the disability-transition analysis.
- Adds full models using first-biomarker-wave morbidity and self-rated health,
  with anchor-wave covariate timing as a sensitivity analysis.
- Adds restricted cubic spline exposure–response estimates and a staged
  covariate-attenuation matrix with and without longitudinal response weights.
- Adds JAD-oriented forest, spline, and attenuation figure code.
- Maintains the code-only release boundary; no manuscripts, data, numerical
  outputs, figures, local paths, credentials, or personal contact details are included.

## Version 2.1.1 — 2026-09-14

- Refreshes the transition and cohort-flow figure code and makes the enhanced
  analysis output directory configurable.

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

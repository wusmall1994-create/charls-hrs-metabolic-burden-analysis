from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", PROJECT / "results"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import charls_hrs_harmonized_analysis as bridge
import hrs_external_validation as hrs


SEED = 20260903

RAW_PREDICTORS = [
    "age", "female", "education", "rural", "partnered", "smoking", "drinking",
    "comorbidity", "self_health", "met_anchor", "memory_first", "memory_anchor",
    "cesd_first", "cesd_anchor", "grip", "physical_activity", "social_participation",
    "log_crp", "log_cystatin",
]
FULL_CORE = [
    "age_z", "female", "education_z", "rural", "partnered", "smoking", "drinking",
    "comorbidity_z", "self_health_z", "met_anchor_z",
]
TARGETS = ["persistent_low_depression", "memory_z"]
TRANSITION_PREDICTORS = ["age_z", "female", "met_anchor_z"] + TARGETS


def multinomial_standardization(
    fit: dict,
    data: pd.DataFrame,
    predictor: str,
    standard_weight: str,
) -> dict:
    names = fit["names"]
    predictors = names[1:]
    d = data[[standard_weight] + predictors].dropna().copy()
    base = np.column_stack([np.ones(len(d)), d[predictors].astype(float).to_numpy()])
    j = names.index(predictor)
    beta = fit["beta"].reshape(2, len(names))
    cov = fit["cov"]
    weights = d[standard_weight].to_numpy(float, copy=True)
    weights /= weights.sum()

    scenarios: dict[int, dict] = {}
    for value in (0, 1):
        x = base.copy()
        x[:, j] = value
        eta = np.clip(x @ beta.T, -25, 25)
        exp_eta = np.exp(eta)
        denom = 1 + exp_eta.sum(axis=1)
        probs = np.column_stack([1 / denom, exp_eta[:, 0] / denom, exp_eta[:, 1] / denom])
        means = np.sum(weights[:, None] * probs, axis=0)
        gradients = []
        for state in (0, 1, 2):
            blocks = []
            for modeled_state in (1, 2):
                derivative = probs[:, state] * (
                    (1.0 if state == modeled_state else 0.0) - probs[:, modeled_state]
                )
                blocks.append(np.sum(weights[:, None] * derivative[:, None] * x, axis=0))
            gradients.append(np.concatenate(blocks))
        scenarios[value] = {"means": means, "gradients": gradients}

    result = {"scenarios": {}}
    for value in (0, 1):
        result["scenarios"][value] = {}
        for state in (0, 1, 2):
            gradient = scenarios[value]["gradients"][state]
            result["scenarios"][value][state] = (
                float(scenarios[value]["means"][state]),
                float(gradient @ cov @ gradient),
            )

    p0 = scenarios[0]["means"][1]
    p1 = scenarios[1]["means"][1]
    g0 = scenarios[0]["gradients"][1]
    g1 = scenarios[1]["gradients"][1]
    difference_gradient = g1 - g0
    log_ratio_gradient = g1 / p1 - g0 / p0
    result["disability_difference"] = (
        float(p1 - p0),
        float(difference_gradient @ cov @ difference_gradient),
    )
    result["disability_log_ratio"] = (
        float(math.log(p1 / p0)),
        float(log_ratio_gradient @ cov @ log_ratio_gradient),
    )
    return result


def pooled_row(values: list[tuple[float, float, float]], transform: str = "identity") -> dict:
    if transform == "logit":
        converted = []
        for estimate, variance, df in values:
            bounded = min(max(estimate, 1e-8), 1 - 1e-8)
            converted.append((
                math.log(bounded / (1 - bounded)),
                variance / (bounded**2 * (1 - bounded) ** 2),
                df,
            ))
        values = converted
    q, se, df, p, lo, hi, within, between = hrs.pool_scalar(
        [value[0] for value in values],
        [value[1] for value in values],
        [value[2] for value in values],
    )
    if transform == "exp":
        estimate, lower, upper = math.exp(q), math.exp(lo), math.exp(hi)
    elif transform == "logit":
        estimate = 1 / (1 + math.exp(-q))
        lower = 1 / (1 + math.exp(-lo))
        upper = 1 / (1 + math.exp(-hi))
    else:
        estimate, lower, upper = q, lo, hi
    return {
        "estimate": estimate,
        "ci_low": lower,
        "ci_high": upper,
        "p_value": p,
        "df": df,
        "within_variance": within,
        "between_variance": between,
    }


def e_value(rr: float) -> float:
    value = rr if rr >= 1 else 1 / rr
    return value + math.sqrt(value * (value - 1))


def analyse_cohort(cohort: pd.DataFrame, cesd_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(SEED)
    auxiliary = ["durable_composite_independent", "f1_state", "f2_state"]
    original_cols = [
        "durable_composite_independent", "observed_durable_composite", "f1_state", "f2_state",
        "observed_f1_composite", "observed_f2_composite", "blood_weight", "design_stratum",
        "design_psu",
    ]
    original = cohort[original_cols].copy()
    mice = hrs.MICEData(
        cohort[RAW_PREDICTORS + auxiliary].astype(float),
        perturbation_method="gaussian",
        k_pmm=20,
    )
    mice.update_all(10)

    standardization = []
    physical_sustained = []
    physical_transition = []
    landmark_transition = []
    landmark_counts = []

    for _ in range(hrs.M):
        mice.update_all(5)
        d = hrs.prepare_completed(mice.data.copy(), cesd_threshold=cesd_threshold)
        for column in original_cols:
            d[column] = original[column].to_numpy()

        ipw, _ = hrs.response_ipw(
            d, "observed_f1_composite", TRANSITION_PREDICTORS, "blood_weight"
        )
        d["transition_weight"] = d["blood_weight"] * ipw
        base_transition = hrs.multinomial_fit(
            d, "f1_state", TRANSITION_PREDICTORS, "transition_weight"
        )
        standardized = multinomial_standardization(
            base_transition, d, "persistent_low_depression", "blood_weight"
        )
        standardization.append((
            standardized,
            base_transition["design_df"],
            base_transition["converged"],
        ))

        sustained_predictors = FULL_CORE + ["physical_activity"] + TARGETS
        ipw, _ = hrs.response_ipw(
            d, "observed_durable_composite", sustained_predictors, "blood_weight"
        )
        d["physical_sustained_weight"] = d["blood_weight"] * ipw
        physical_sustained.append(
            hrs.logistic_fit(
                d, "durable_composite_independent", sustained_predictors,
                "physical_sustained_weight",
            )
        )

        activity_transition_predictors = TRANSITION_PREDICTORS + ["physical_activity"]
        ipw, _ = hrs.response_ipw(
            d, "observed_f1_composite", activity_transition_predictors, "blood_weight"
        )
        d["physical_transition_weight"] = d["blood_weight"] * ipw
        physical_transition.append(
            hrs.multinomial_fit(
                d, "f1_state", activity_transition_predictors, "physical_transition_weight"
            )
        )

        landmark = d[d["f1_state"] == 0].copy()
        ipw, _ = hrs.response_ipw(
            landmark, "observed_f2_composite", TRANSITION_PREDICTORS, "blood_weight"
        )
        landmark["landmark_weight"] = landmark["blood_weight"] * ipw
        fit = hrs.multinomial_fit(
            landmark, "f2_state", TRANSITION_PREDICTORS, "landmark_weight"
        )
        landmark_transition.append(fit)
        landmark_counts.append(fit["counts"])

    probability_rows = []
    labels = {0: "Independent", 1: "ADL disability", 2: "Death"}
    for exposure_value, exposure_label in [(0, "Not persistently low"), (1, "Persistently low")]:
        for state in (0, 1, 2):
            pooled = pooled_row([
                (
                    item[0]["scenarios"][exposure_value][state][0],
                    item[0]["scenarios"][exposure_value][state][1],
                    item[1],
                )
                for item in standardization
            ], transform="logit")
            probability_rows.append({
                "exposure_group": exposure_label,
                "outcome_state": labels[state],
                **pooled,
            })
    difference = pooled_row([
        (item[0]["disability_difference"][0], item[0]["disability_difference"][1], item[1])
        for item in standardization
    ])
    log_ratio = pooled_row([
        (item[0]["disability_log_ratio"][0], item[0]["disability_log_ratio"][1], item[1])
        for item in standardization
    ], transform="exp")
    nearest_null = log_ratio["ci_high"] if log_ratio["estimate"] < 1 else log_ratio["ci_low"]
    contrast_rows = [{
        "analysis": "Standardized first-follow-up ADL disability",
        "contrast": "Persistently low vs not persistently low depressive symptoms",
        "effect_metric": "Risk difference",
        "all_converged": bool(all(item[2] for item in standardization)),
        **difference,
    }, {
        "analysis": "Standardized first-follow-up ADL disability",
        "contrast": "Persistently low vs not persistently low depressive symptoms",
        "effect_metric": "Risk ratio",
        "all_converged": bool(all(item[2] for item in standardization)),
        **log_ratio,
        "e_value_estimate": e_value(log_ratio["estimate"]),
        "e_value_ci_limit": e_value(nearest_null) if nearest_null < 1 or nearest_null > 1 else 1.0,
    }]

    def pooled_coefficient(fits: list[dict], variable: str, block_index: int | None = None) -> dict:
        names = fits[0]["names"]
        index = names.index(variable)
        if block_index is not None:
            index += block_index * len(names)
        return pooled_row([
            (fit["beta"][index], fit["cov"][index, index], fit["design_df"])
            for fit in fits
        ], transform="exp")

    contrast_rows.append({
        "analysis": "Physical-activity-adjusted sustained independence",
        "contrast": "Persistently low vs not persistently low depressive symptoms",
        "effect_metric": "Odds ratio",
        "all_converged": bool(all(fit["converged"] for fit in physical_sustained)),
        **pooled_coefficient(physical_sustained, "persistent_low_depression"),
    })
    contrast_rows.append({
        "analysis": "Physical-activity-adjusted first-follow-up ADL disability",
        "contrast": "Persistently low vs not persistently low depressive symptoms",
        "effect_metric": "Relative risk ratio",
        "all_converged": bool(all(fit["converged"] for fit in physical_transition)),
        **pooled_coefficient(physical_transition, "persistent_low_depression", 0),
    })
    contrast_rows.append({
        "analysis": "Landmark ADL disability at second follow-up among first-follow-up survivors who remained independent",
        "contrast": "Persistently low vs not persistently low depressive symptoms",
        "effect_metric": "Relative risk ratio",
        "all_converged": bool(all(fit["converged"] for fit in landmark_transition)),
        **pooled_coefficient(landmark_transition, "persistent_low_depression", 0),
        "analysis_n": int(np.median([fit["N"] for fit in landmark_transition])),
        "independent_n": int(np.median([counts["0"] for counts in landmark_counts])),
        "disability_n": int(np.median([counts["1"] for counts in landmark_counts])),
        "death_n": int(np.median([counts["2"] for counts in landmark_counts])),
    })
    return pd.DataFrame(probability_rows), pd.DataFrame(contrast_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohorts = {
        "CHARLS": bridge.charls_bridge_cohort(),
        "HRS": bridge.hrs_bridge_cohort(),
    }
    probability_parts = []
    contrast_parts = []
    for name, cohort in cohorts.items():
        probabilities, contrasts = analyse_cohort(cohort, cohort.attrs["cesd_threshold"])
        probabilities.insert(0, "cohort", name)
        contrasts.insert(0, "cohort", name)
        probability_parts.append(probabilities)
        contrast_parts.append(contrasts)
    probabilities = pd.concat(probability_parts, ignore_index=True)
    contrasts = pd.concat(contrast_parts, ignore_index=True)
    probabilities.to_csv(
        OUT / "charls_hrs_standardized_state_probabilities.csv", index=False, encoding="utf-8-sig"
    )
    contrasts.to_csv(
        OUT / "charls_hrs_enhanced_sensitivity_analyses.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "standardized_state_probabilities": json.loads(probabilities.to_json(orient="records")),
        "enhanced_sensitivity_analyses": json.loads(contrasts.to_json(orient="records")),
        "notes": {
            "physical_activity": "Any vigorous or moderate activity at the anchor wave; CHARLS required at least 10 minutes and HRS included any reported non-never frequency.",
            "landmark": "Conditional analysis among participants alive and ADL independent at the first follow-up.",
            "e_value": "Calculated from the standardized marginal risk ratio, not the multinomial relative risk ratio.",
        },
    }
    (OUT / "charls_hrs_enhanced_sensitivity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("STANDARDIZED STATE PROBABILITIES")
    print(probabilities.to_string(index=False))
    print("\nENHANCED SENSITIVITY ANALYSES")
    print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()

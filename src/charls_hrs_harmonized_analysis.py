from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", PROJECT / "results"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats

import advanced_inference as charls
import formal_analysis as charls_formal
import hrs_external_validation as hrs


SEED = 20260903


def charls_bridge_cohort() -> pd.DataFrame:
    """Map CHARLS to the HRS analysis contract using the common four components."""
    source = charls.build_cohort("persistent3_of_4_no_tg").copy()
    original_five = charls.build_cohort("persistent3")
    source_ids = set(source["ID"].dropna().astype(str))
    original_ids = set(original_five["ID"].dropna().astype(str))

    six_comorbidities = ["r3cancre", "r3lunge", "r3hearte", "r3stroke", "r3psyche", "r3arthre"]
    disease = pd.concat([charls_formal.binary(source[col]) for col in six_comorbidities], axis=1)
    disease.columns = six_comorbidities
    source["comorbidity_bridge"] = disease.sum(axis=1, min_count=len(six_comorbidities))

    d = pd.DataFrame(index=source.index)
    d["cohort"] = "CHARLS"
    d["age"] = pd.to_numeric(source["r1agey"], errors="coerce")
    d["female"] = pd.to_numeric(source["female"], errors="coerce")
    d["education"] = pd.to_numeric(source["education"], errors="coerce")
    d["rural"] = pd.to_numeric(source["rural"], errors="coerce")
    d["partnered"] = pd.to_numeric(source["partnered"], errors="coerce")
    d["smoking"] = pd.to_numeric(source["smoking"], errors="coerce")
    d["drinking"] = pd.to_numeric(source["drinking"], errors="coerce")
    d["comorbidity"] = source["comorbidity_bridge"]
    d["self_health"] = pd.to_numeric(source["self_health"], errors="coerce")
    d["met_anchor"] = pd.to_numeric(source["met3_4c"], errors="coerce")
    d["memory_first"] = pd.to_numeric(source["r1tr20"], errors="coerce")
    d["memory_anchor"] = pd.to_numeric(source["r3tr20"], errors="coerce")
    d["cesd_first"] = pd.to_numeric(source["r1cesd10"], errors="coerce")
    d["cesd_anchor"] = pd.to_numeric(source["r3cesd10"], errors="coerce")
    d["grip"] = pd.to_numeric(source["r3gripsum"], errors="coerce")
    d["physical_activity"] = charls_formal.any_yes(
        charls_formal.binary(source["r3vgact_c"]),
        charls_formal.binary(source["r3mdact_c"]),
    )
    d["social_participation"] = pd.to_numeric(source["social_reserve"], errors="coerce")
    d["log_crp"] = pd.to_numeric(source["log_crp"], errors="coerce")
    d["log_cystatin"] = pd.to_numeric(source["log_cysc"], errors="coerce")
    d["blood_weight"] = pd.to_numeric(source["weight2015"], errors="coerce")
    d["design_stratum"] = pd.to_numeric(source["stratum_prov_urban"], errors="coerce")
    d["design_psu"] = pd.to_numeric(source["psu_code"], errors="coerce")

    f1 = pd.to_numeric(source["state_2018"], errors="coerce")
    f2 = pd.to_numeric(source["state_2020"], errors="coerce")
    d["f1_state"] = f1
    d["f2_state"] = f2
    d["observed_f1_composite"] = f1.notna().astype(float)
    d["observed_f2_composite"] = f2.notna().astype(float)
    d["f1_composite_independent"] = np.where(f1.notna(), (f1 == 0).astype(float), np.nan)
    d["f2_composite_independent"] = np.where(f2.notna(), (f2 == 0).astype(float), np.nan)
    d["observed_f1_function"] = f1.isin([0, 1]).astype(float)
    d["observed_f2_function"] = f2.isin([0, 1]).astype(float)
    d["f1_independent"] = np.where(f1.isin([0, 1]), (f1 == 0).astype(float), np.nan)
    d["f2_independent"] = np.where(f2.isin([0, 1]), (f2 == 0).astype(float), np.nan)
    d["observed_durable"] = (f1.isin([0, 1]) & f2.isin([0, 1])).astype(float)
    d["durable_independent"] = np.where(
        d["observed_durable"] == 1,
        ((f1 == 0) & (f2 == 0)).astype(float),
        np.nan,
    )
    d["observed_durable_composite"] = (f1.notna() & f2.notna()).astype(float)
    d["durable_composite_independent"] = np.where(
        d["observed_durable_composite"] == 1,
        ((f1 == 0) & (f2 == 0)).astype(float),
        np.nan,
    )
    d["f1_disabled"] = (f1 == 1).astype(float)
    d["observed_recovery"] = np.where(f1 == 1, f2.isin([0, 1]).astype(float), 0.0)
    d["recovered_f2"] = np.where((f1 == 1) & f2.isin([0, 1]), (f2 == 0).astype(float), np.nan)

    eligible = (
        (d["blood_weight"] > 0)
        & d["design_stratum"].notna()
        & d["design_psu"].notna()
    )
    result = d.loc[eligible].reset_index(drop=True)
    result.attrs.update(
        exposure="2011 and 2015, >=3/4 abnormalities at both waves",
        followups="2018 and 2020",
        cesd_threshold=10,
        source_n=len(source),
        original_five_component_n=len(original_five),
        four_five_overlap_n=len(source_ids & original_ids),
    )
    return result


def hrs_bridge_cohort() -> pd.DataFrame:
    biomarkers = hrs.load_biomarkers(requested_waves=(10, 12))
    longitudinal = hrs.load_longitudinal()
    source = hrs.build_window(
        longitudinal,
        biomarkers,
        10,
        12,
        followup_offsets=(2, 3),
    )
    result = hrs.select_cohort(source, 10, 12, threshold=3)
    result["cohort"] = "HRS"
    result.attrs.update(
        exposure="2010 and 2014, >=3/4 abnormalities at both waves",
        followups="2018 and 2020",
        cesd_threshold=4,
        source_n=len(source),
    )
    return result


def flow_row(cohort: pd.DataFrame) -> dict:
    return {
        "cohort": cohort["cohort"].iloc[0],
        "exposure": cohort.attrs["exposure"],
        "followups": cohort.attrs["followups"],
        "eligible_n": len(cohort),
        "primary_observed_n": int(cohort["durable_composite_independent"].notna().sum()),
        "sustained_independent_n": int((cohort["durable_composite_independent"] == 1).sum()),
        "f1_independent_n": int((cohort["f1_state"] == 0).sum()),
        "f1_disability_n": int((cohort["f1_state"] == 1).sum()),
        "f1_death_n": int((cohort["f1_state"] == 2).sum()),
        "original_five_component_n": cohort.attrs.get("original_five_component_n", np.nan),
        "four_five_overlap_n": cohort.attrs.get("four_five_overlap_n", np.nan),
    }


def result_bundle(cohort: pd.DataFrame, cesd_threshold: float):
    return hrs.run_mi(
        cohort,
        primary_outcome="durable_composite_independent",
        primary_observed="observed_durable_composite",
        cesd_threshold=cesd_threshold,
    )


def continuous_depression_sensitivity(cohort: pd.DataFrame) -> pd.DataFrame:
    """Repeat the two-marker bridge using a harmonized continuous CES-D burden.

    CES-D scores are standardized within cohort and exposure wave, averaged
    across the two exposure waves, standardized again, and reverse-coded so
    that a one-unit increase represents one SD lower long-term symptom burden.
    """
    np.random.seed(SEED)
    raw_predictors = [
        "age", "female", "education", "rural", "partnered", "smoking", "drinking",
        "comorbidity", "self_health", "met_anchor", "memory_first", "memory_anchor",
        "cesd_first", "cesd_anchor", "grip", "social_participation", "log_crp", "log_cystatin",
    ]
    primary_outcome = "durable_composite_independent"
    primary_observed = "observed_durable_composite"
    auxiliary_outcomes = [primary_outcome, "f1_state", "f2_independent"]
    original_cols = [
        primary_outcome, primary_observed, "f1_state", "observed_f1_composite",
        "blood_weight", "design_stratum", "design_psu",
    ]
    original = cohort[original_cols].copy()
    mice = hrs.MICEData(
        cohort[raw_predictors + auxiliary_outcomes].astype(float),
        perturbation_method="gaussian",
        k_pmm=20,
    )
    mice.update_all(10)

    core = [
        "age_z", "female", "education_z", "rural", "partnered", "smoking", "drinking",
        "comorbidity_z", "self_health_z", "met_anchor_z",
    ]
    continuous = "favorable_depressive_burden_z"
    sustained_predictors = core + [continuous, "memory_z"]
    response_predictors = core + [
        continuous, "memory_z", "grip_z", "social_participation",
        "favorable_crp_z", "favorable_cystatin_z",
    ]
    transition_predictors = ["age_z", "female", "met_anchor_z", continuous, "memory_z"]
    sustained_fits = []
    multistate_fits = []
    contrasts = []

    for _ in range(hrs.M):
        mice.update_all(5)
        d = hrs.prepare_completed(mice.data.copy())
        first_z = hrs.zscore(d["cesd_first"])
        anchor_z = hrs.zscore(d["cesd_anchor"])
        d[continuous] = -hrs.zscore((first_z + anchor_z) / 2)
        for col in original:
            d[col] = original[col].to_numpy()

        ipw, _ = hrs.response_ipw(d, primary_observed, response_predictors, "blood_weight")
        d["analysis_weight"] = d["blood_weight"] * ipw
        sustained = hrs.logistic_fit(d, primary_outcome, sustained_predictors, "analysis_weight")
        sustained_fits.append(sustained)
        contrasts.append(hrs.marginal_contrast(sustained, d, continuous, "plus_one", "blood_weight"))

        ipw, _ = hrs.response_ipw(
            d, "observed_f1_composite", transition_predictors, "blood_weight"
        )
        d["transition_weight"] = d["blood_weight"] * ipw
        multistate_fits.append(
            hrs.multinomial_fit(d, "f1_state", transition_predictors, "transition_weight")
        )

    sustained_index = sustained_fits[0]["names"].index(continuous)
    sustained_pool = hrs.pool_scalar(
        [fit["beta"][sustained_index] for fit in sustained_fits],
        [fit["cov"][sustained_index, sustained_index] for fit in sustained_fits],
        [fit["design_df"] for fit in sustained_fits],
    )
    q, _, df, p, lo, hi, within, between = sustained_pool

    transition_names = multistate_fits[0]["names"]
    transition_index = transition_names.index(continuous)
    disability_pool = hrs.pool_scalar(
        [fit["beta"][transition_index] for fit in multistate_fits],
        [fit["cov"][transition_index, transition_index] for fit in multistate_fits],
        [fit["design_df"] for fit in multistate_fits],
    )
    qd, _, dfd, pd_, lod, hid, withind, betweend = disability_pool

    contrast_pool = hrs.pool_scalar(
        [value[0] for value in contrasts],
        [value[1] for value in contrasts],
        [fit["design_df"] for fit in sustained_fits],
    )
    cq, _, cdf, cp, clo, chi, _, _ = contrast_pool

    return pd.DataFrame(
        [
            {
                "outcome": "Sustained independence",
                "variable": continuous,
                "contrast": "per 1 SD lower depressive symptom burden",
                "effect_metric": "OR",
                "effect": math.exp(q),
                "ci_low": math.exp(lo),
                "ci_high": math.exp(hi),
                "p_value": p,
                "df": df,
                "within_variance": within,
                "between_variance": between,
                "analysis_n": int(np.median([fit["N"] for fit in sustained_fits])),
                "events": int(np.median([fit["events"] for fit in sustained_fits])),
                "all_converged": bool(all(fit["converged"] for fit in sustained_fits)),
                "adjusted_probability_difference": cq,
                "probability_difference_ci_low": clo,
                "probability_difference_ci_high": chi,
                "probability_difference_p": cp,
                "probability_difference_df": cdf,
            },
            {
                "outcome": "ADL disability vs independent",
                "variable": continuous,
                "contrast": "per 1 SD lower depressive symptom burden",
                "effect_metric": "RRR",
                "effect": math.exp(qd),
                "ci_low": math.exp(lod),
                "ci_high": math.exp(hid),
                "p_value": pd_,
                "df": dfd,
                "within_variance": withind,
                "between_variance": betweend,
                "analysis_n": int(np.median([fit["N"] for fit in multistate_fits])),
                "events": int(np.median([fit["counts"]["1"] for fit in multistate_fits])),
                "all_converged": bool(all(fit["converged"] for fit in multistate_fits)),
                "adjusted_probability_difference": np.nan,
                "probability_difference_ci_low": np.nan,
                "probability_difference_ci_high": np.nan,
                "probability_difference_p": np.nan,
                "probability_difference_df": np.nan,
            },
        ]
    )


def continuous_depression_comparison(sensitivity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in ["Sustained independence", "ADL disability vs independent"]:
        subset = sensitivity[sensitivity["outcome"] == outcome].set_index("cohort")
        c = subset.loc["CHARLS"]
        h = subset.loc["HRS"]
        rows.append(
            {
                "outcome": outcome,
                "contrast": c["contrast"],
                "effect_metric": c["effect_metric"],
                "charls_effect": c["effect"],
                "charls_ci_low": c["ci_low"],
                "charls_ci_high": c["ci_high"],
                "charls_p": c["p_value"],
                "hrs_effect": h["effect"],
                "hrs_ci_low": h["ci_low"],
                "hrs_ci_high": h["ci_high"],
                "hrs_p": h["p_value"],
                "effect_difference_p": effect_difference_p(
                    c["effect"], c["ci_low"], c["ci_high"],
                    h["effect"], h["ci_low"], h["ci_high"],
                ),
            }
        )
    return pd.DataFrame(rows)


def effect_difference_p(est1: float, lo1: float, hi1: float, est2: float, lo2: float, hi2: float) -> float:
    se1 = (math.log(hi1) - math.log(lo1)) / (2 * 1.96)
    se2 = (math.log(hi2) - math.log(lo2)) / (2 * 1.96)
    z = (math.log(est1) - math.log(est2)) / math.sqrt(se1**2 + se2**2)
    return float(2 * stats.norm.sf(abs(z)))


def comparison_table(results: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for outcome, table_name, effect_col in [
        ("Sustained independence", "models", "odds_ratio"),
        ("ADL disability vs independent", "multistate", "relative_risk_ratio"),
    ]:
        by_cohort = {}
        for cohort_name, bundle in results.items():
            models, _, multistate, _ = bundle
            if table_name == "models":
                table = models[models["model"] == "primary_two_marker_sustained"].set_index("variable")
            else:
                table = multistate[multistate["transition"] == outcome].set_index("variable")
            by_cohort[cohort_name] = table
        for variable in ["persistent_low_depression", "memory_z"]:
            c = by_cohort["CHARLS"].loc[variable]
            h = by_cohort["HRS"].loc[variable]
            charls_p = float(c["holm_two_target_p"])
            hrs_p = float(h["holm_two_target_p"])
            c_effect = float(c[effect_col])
            h_effect = float(h[effect_col])
            favorable_c = c_effect > 1 if effect_col == "odds_ratio" else c_effect < 1
            favorable_h = h_effect > 1 if effect_col == "odds_ratio" else h_effect < 1
            if favorable_c and favorable_h and charls_p < 0.05 and hrs_p < 0.05:
                judgment = "supported in both cohorts"
            elif favorable_c and favorable_h:
                judgment = "directionally consistent; incomplete replication"
            else:
                judgment = "not directionally replicated"
            rows.append(
                {
                    "outcome": outcome,
                    "variable": variable,
                    "charls_effect": c_effect,
                    "charls_ci_low": float(c["ci_low"]),
                    "charls_ci_high": float(c["ci_high"]),
                    "charls_holm_p": charls_p,
                    "hrs_effect": h_effect,
                    "hrs_ci_low": float(h["ci_low"]),
                    "hrs_ci_high": float(h["ci_high"]),
                    "hrs_holm_p": hrs_p,
                    "effect_difference_p": effect_difference_p(
                        c_effect,
                        float(c["ci_low"]),
                        float(c["ci_high"]),
                        h_effect,
                        float(h["ci_low"]),
                        float(h["ci_high"]),
                    ),
                    "judgment": judgment,
                }
            )
    return pd.DataFrame(rows)


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    cohorts = {
        "CHARLS": charls_bridge_cohort(),
        "HRS": hrs_bridge_cohort(),
    }
    flow = pd.DataFrame([flow_row(cohort) for cohort in cohorts.values()])
    results = {
        name: result_bundle(cohort, cohort.attrs["cesd_threshold"])
        for name, cohort in cohorts.items()
    }

    model_parts = []
    multistate_parts = []
    contrast_parts = []
    ipw_parts = []
    for name, (models, contrasts, multistate, ipw) in results.items():
        for frame, destination in [
            (models, model_parts),
            (multistate, multistate_parts),
            (contrasts, contrast_parts),
            (ipw, ipw_parts),
        ]:
            part = frame.copy()
            part.insert(0, "cohort", name)
            destination.append(part)
    models = pd.concat(model_parts, ignore_index=True)
    multistate = pd.concat(multistate_parts, ignore_index=True)
    contrasts = pd.concat(contrast_parts, ignore_index=True)
    ipw = pd.concat(ipw_parts, ignore_index=True)
    comparison = comparison_table(results)
    continuous_parts = []
    for name, cohort in cohorts.items():
        part = continuous_depression_sensitivity(cohort)
        part.insert(0, "cohort", name)
        continuous_parts.append(part)
    continuous = pd.concat(continuous_parts, ignore_index=True)
    continuous_comparison = continuous_depression_comparison(continuous)

    flow.to_csv(OUT / "charls_hrs_harmonized_cohort_flow.csv", index=False, encoding="utf-8-sig")
    models.to_csv(OUT / "charls_hrs_harmonized_models.csv", index=False, encoding="utf-8-sig")
    multistate.to_csv(OUT / "charls_hrs_harmonized_multistate.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(OUT / "charls_hrs_harmonized_probability_differences.csv", index=False, encoding="utf-8-sig")
    ipw.to_csv(OUT / "charls_hrs_harmonized_ipw_diagnostics.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "charls_hrs_harmonized_comparison.csv", index=False, encoding="utf-8-sig")
    continuous.to_csv(
        OUT / "charls_hrs_continuous_depression_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    continuous_comparison.to_csv(
        OUT / "charls_hrs_continuous_depression_comparison.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "design": {
            "cohorts": ["CHARLS", "HRS"],
            "common_components": ["waist", "blood pressure/hypertension", "HbA1c/diabetes", "HDL"],
            "persistent_threshold": ">=3/4 at both exposure waves",
            "functional_followups": [2018, 2020],
            "death_in_primary": "counted as failure to sustain independence",
            "imputations": hrs.M,
            "seed": SEED,
        },
        "flow": records(flow),
        "comparison": records(comparison),
        "continuous_depression_sensitivity": records(continuous_comparison),
        "primary_models": records(
            models[(models["model"] == "primary_joint_sustained") & models["variable"].isin(hrs.MARKERS)]
        ),
        "multistate_targets": records(
            multistate[multistate["variable"].isin(["persistent_low_depression", "memory_z"])]
        ),
    }
    (OUT / "charls_hrs_harmonized_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("COHORT FLOW")
    print(flow.to_string(index=False))
    print("\nCOMPARISON")
    print(comparison.to_string(index=False))
    print("\nPRIMARY SIX-MARKER MODELS")
    print(
        models[(models["model"] == "primary_joint_sustained") & models["variable"].isin(hrs.MARKERS)]
        .to_string(index=False)
    )
    print("\nCONTINUOUS DEPRESSION SENSITIVITY")
    print(continuous_comparison.to_string(index=False))


if __name__ == "__main__":
    main()

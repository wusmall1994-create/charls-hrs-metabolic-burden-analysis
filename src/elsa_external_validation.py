from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.imputation.mice import MICEData


ELSA = Path(os.environ["ELSA_DATA_DIR"])
OUT = Path(__file__).resolve().parent
SEED = 20260903
M = 20


H_COLS = [
    "idauniq", "r2cesd", "r4cesd", "r2tr20", "r4tr20", "r4agey", "ragender",
    "raeducl", "r4mstat", "r4smoken", "r4drink", "r4shlt",
    "r4cancre", "r4lunge", "r4hearte", "r4stroke", "r4psyche", "r4arthre", "r4asthmae",
    "r2hibpe", "r4hibpe", "r2rxhibp", "r4rxhibp",
    "r2diabe", "r4diabe", "r2rxdiab", "r4rxdiab",
    "r4adltot6", "r4iadlza", "r6adltot6", "r8adltot6", "r6iwstat", "r8iwstat",
]


def clean_numeric(x: pd.Series, valid_min: float | None = None) -> pd.Series:
    y = pd.to_numeric(x, errors="coerce").astype(float)
    if valid_min is not None:
        y = y.where(y >= valid_min)
    return y


def binary(x: pd.Series) -> pd.Series:
    y = pd.to_numeric(x, errors="coerce")
    return y.where(y.isin([0, 1])).astype(float)


def zscore(x: pd.Series) -> pd.Series:
    y = pd.to_numeric(x, errors="coerce").astype(float)
    sd = y.std(ddof=1)
    return (y - y.mean()) / sd if pd.notna(sd) and sd > 0 else y * np.nan


def all_binary_sum(frame: pd.DataFrame) -> pd.Series:
    return frame.sum(axis=1, min_count=frame.shape[1]).astype(float)


def metabolic_score(
    df: pd.DataFrame,
    wave: int,
    male_waist: float = 90,
    female_waist: float = 80,
) -> tuple[pd.Series, pd.DataFrame]:
    suffix = f"_w{wave}"
    waist = df[f"waist{suffix}"]
    sbp = df[f"sbp{suffix}"]
    dbp = df[f"dbp{suffix}"]
    hdl = df[f"hdl{suffix}"]
    tg = df[f"tg{suffix}"]
    glucose = df[f"glucose{suffix}"]
    hba1c = df[f"hba1c{suffix}"]
    fasting = df[f"fasting{suffix}"]
    hbp = binary(df[f"hibp{suffix}"])
    bpmed = binary(df[f"rxhibp{suffix}"])
    diabetes = binary(df[f"diabetes{suffix}"])
    dmmed = binary(df[f"rxdiab{suffix}"])
    female = df["female"] == 1

    waist_threshold = pd.Series(np.where(female, female_waist, male_waist), index=df.index)
    hdl_threshold = pd.Series(np.where(female, 1.29, 1.03), index=df.index)
    c = pd.DataFrame(index=df.index)
    c["waist"] = np.where(waist.notna(), (waist >= waist_threshold).astype(float), np.nan)

    c["bp"] = np.nan
    bp_positive = (sbp >= 130) | (dbp >= 85) | (hbp == 1) | (bpmed == 1)
    bp_negative = sbp.notna() & dbp.notna() & (sbp < 130) & (dbp < 85) & (hbp == 0) & (bpmed == 0)
    c.loc[bp_positive, "bp"] = 1
    c.loc[bp_negative, "bp"] = 0

    c["glycemia"] = np.nan
    gly_positive = (hba1c >= 5.7) | ((fasting == 1) & (glucose >= 5.56)) | (diabetes == 1) | (dmmed == 1)
    gly_negative = (
        hba1c.notna() & (hba1c < 5.7) & (diabetes == 0) & (dmmed == 0)
        & ((((fasting == 1) & glucose.notna() & (glucose < 5.56))) | (fasting != 1))
    )
    c.loc[gly_positive, "glycemia"] = 1
    c.loc[gly_negative, "glycemia"] = 0

    c["tg"] = np.where(tg.notna(), (tg >= 1.70).astype(float), np.nan)
    c["hdl"] = np.where(hdl.notna(), (hdl < hdl_threshold).astype(float), np.nan)
    return c.sum(axis=1, min_count=5), c


def read_source_data() -> pd.DataFrame:
    h = pd.read_stata(ELSA / "gh_elsa_h.dta", columns=H_COLS, convert_categoricals=False)
    n2_cols = [
        "idauniq", "sex", "sysval", "diaval", "hdl", "trig", "fglu", "hba1c",
        "wstval", "fasteli", "w2wtbld",
    ]
    n4_cols = [
        "idauniq", "dhsex", "sysval", "diaval", "hdl", "trig", "fglu", "hba1c",
        "wstval", "fastelig", "w4bldwt", "hscrp",
    ]
    n2 = pd.read_stata(ELSA / "wave_2_nurse_data_v2.dta", columns=n2_cols, convert_categoricals=False)
    n4 = pd.read_stata(ELSA / "wave_4_nurse_data.dta", columns=n4_cols, convert_categoricals=False)
    design = pd.read_stata(
        ELSA / "wave_4_elsa_data_eul.dta", columns=["idauniq", "idahhw4", "gor"],
        convert_categoricals=False,
    )
    rural = pd.read_stata(
        ELSA / "elsa_geog_urindewr_2001_eul.dta", columns=["idauniq", "w4_urindewr_2001"],
        convert_categoricals=False,
    )

    n2 = n2.rename(columns={
        "sex": "sex_w2", "sysval": "sbp_w2", "diaval": "dbp_w2", "hdl": "hdl_w2",
        "trig": "tg_w2", "fglu": "glucose_w2", "hba1c": "hba1c_w2",
        "wstval": "waist_w2", "fasteli": "fasting_w2", "w2wtbld": "blood_weight_w2",
    })
    n4 = n4.rename(columns={
        "dhsex": "sex_w4", "sysval": "sbp_w4", "diaval": "dbp_w4", "hdl": "hdl_w4",
        "trig": "tg_w4", "fglu": "glucose_w4", "hba1c": "hba1c_w4",
        "wstval": "waist_w4", "fastelig": "fasting_w4", "w4bldwt": "blood_weight_w4",
        "hscrp": "crp_w4",
    })
    for d, wave in [(n2, 2), (n4, 4)]:
        for stem in ["sbp", "dbp", "hdl", "tg", "glucose", "hba1c", "waist"]:
            d[f"{stem}_w{wave}"] = clean_numeric(d[f"{stem}_w{wave}"], 0)
        d[f"fasting_w{wave}"] = pd.to_numeric(d[f"fasting_w{wave}"], errors="coerce").where(
            pd.to_numeric(d[f"fasting_w{wave}"], errors="coerce").isin([1, 2])
        )
    n2["blood_weight_w2"] = clean_numeric(n2["blood_weight_w2"], 0)
    n4["blood_weight_w4"] = clean_numeric(n4["blood_weight_w4"], 0)
    n4["crp_w4"] = clean_numeric(n4["crp_w4"], 0)

    df = h.merge(n2, on="idauniq", how="left", validate="one_to_one")
    df = df.merge(n4, on="idauniq", how="left", validate="one_to_one")
    df = df.merge(design, on="idauniq", how="left", validate="one_to_one")
    df = df.merge(rural, on="idauniq", how="left", validate="one_to_one")

    df["female"] = np.where(df["ragender"].isin([1, 2]), (df["ragender"] == 2).astype(float), np.nan)
    for wave in [2, 4]:
        for stem in ["hibp", "rxhibp", "diabetes", "rxdiab"]:
            source = {
                "hibp": f"r{wave}hibpe",
                "rxhibp": f"r{wave}rxhibp",
                "diabetes": f"r{wave}diabe",
                "rxdiab": f"r{wave}rxdiab",
            }[stem]
            df[f"{stem}_w{wave}"] = binary(df[source])
    return df


def derive_analysis_variables(df: pd.DataFrame, male_waist: float = 90, female_waist: float = 80) -> pd.DataFrame:
    out = df.copy()
    out["met_w2"], components_w2 = metabolic_score(out, 2, male_waist, female_waist)
    out["met_w4"], components_w4 = metabolic_score(out, 4, male_waist, female_waist)
    for c in components_w2:
        out[f"{c}_w2_component"] = components_w2[c]
        out[f"{c}_w4_component"] = components_w4[c]

    out["age"] = clean_numeric(out["r4agey"], 0)
    out["education"] = pd.to_numeric(out["raeducl"], errors="coerce").where(
        pd.to_numeric(out["raeducl"], errors="coerce").isin([1, 2, 3])
    )
    mstat = pd.to_numeric(out["r4mstat"], errors="coerce")
    out["partnered"] = np.where(mstat.isin([1, 3]), 1.0, np.where(mstat.isin([4, 5, 7, 8]), 0.0, np.nan))
    out["smoking"] = binary(out["r4smoken"])
    out["drinking"] = binary(out["r4drink"])
    out["self_health"] = pd.to_numeric(out["r4shlt"], errors="coerce").where(
        pd.to_numeric(out["r4shlt"], errors="coerce").between(1, 5)
    )
    rural_raw = pd.to_numeric(out["w4_urindewr_2001"], errors="coerce")
    out["rural"] = np.where(rural_raw == 2, 1.0, np.where(rural_raw == 1, 0.0, np.nan))
    disease_cols = ["r4cancre", "r4lunge", "r4hearte", "r4stroke", "r4psyche", "r4arthre", "r4asthmae"]
    disease = pd.concat([binary(out[c]) for c in disease_cols], axis=1)
    out["comorbidity"] = all_binary_sum(disease)

    out["memory_w2"] = clean_numeric(out["r2tr20"], 0).where(lambda s: s <= 20)
    out["memory_w4"] = clean_numeric(out["r4tr20"], 0).where(lambda s: s <= 20)
    out["cesd_w2"] = clean_numeric(out["r2cesd"], 0).where(lambda s: s <= 8)
    out["cesd_w4"] = clean_numeric(out["r4cesd"], 0).where(lambda s: s <= 8)
    out["persistent_low_depression"] = np.where(
        out["cesd_w2"].notna() & out["cesd_w4"].notna(),
        ((out["cesd_w2"] < 4) & (out["cesd_w4"] < 4)).astype(float), np.nan,
    )

    adl4 = clean_numeric(out["r4adltot6"], 0).where(lambda s: s <= 6)
    iadl4 = clean_numeric(out["r4iadlza"], 0).where(lambda s: s <= 5)
    adl6 = clean_numeric(out["r6adltot6"], 0).where(lambda s: s <= 6)
    adl8 = clean_numeric(out["r8adltot6"], 0).where(lambda s: s <= 6)
    stat6 = pd.to_numeric(out["r6iwstat"], errors="coerce")
    stat8 = pd.to_numeric(out["r8iwstat"], errors="coerce")
    out["baseline_adl_independent"] = np.where(adl4.notna(), (adl4 == 0).astype(float), np.nan)
    out["baseline_independent"] = np.where(adl4.notna() & iadl4.notna(), ((adl4 == 0) & (iadl4 == 0)).astype(float), np.nan)

    out["observed_w6_function"] = ((stat6 == 1) & adl6.notna()).astype(float)
    out["observed_w8_function"] = ((stat8 == 1) & adl8.notna()).astype(float)
    out["observed_durable"] = ((out["observed_w6_function"] == 1) & (out["observed_w8_function"] == 1)).astype(float)
    out["w6_independent"] = np.where(out["observed_w6_function"] == 1, (adl6 == 0).astype(float), np.nan)
    out["w8_independent"] = np.where(out["observed_w8_function"] == 1, (adl8 == 0).astype(float), np.nan)
    out["durable_independent"] = np.where(
        out["observed_durable"] == 1, ((adl6 == 0) & (adl8 == 0)).astype(float), np.nan,
    )

    died_by_w6 = stat6.isin([5, 6])
    known_w6_composite = (out["observed_w6_function"] == 1) | died_by_w6
    out["observed_w6_composite"] = known_w6_composite.astype(float)
    out["w6_composite_independent"] = np.where(
        known_w6_composite, np.where(died_by_w6, 0.0, (adl6 == 0).astype(float)), np.nan,
    )
    out["w6_state"] = np.where(
        died_by_w6, 2.0,
        np.where(out["observed_w6_function"] == 1, np.where(adl6 > 0, 1.0, 0.0), np.nan),
    )

    w6_disabled = (out["observed_w6_function"] == 1) & (adl6 > 0)
    out["w6_disabled"] = w6_disabled.astype(float)
    out["observed_recovery"] = np.where(w6_disabled, out["observed_w8_function"], 0).astype(float)
    out["recovered_w8"] = np.where(w6_disabled & (out["observed_w8_function"] == 1), (adl8 == 0).astype(float), np.nan)

    out["blood_weight_w4"] = clean_numeric(out["blood_weight_w4"], 0)
    gor = out["gor"].astype("string").str.strip()
    out["design_stratum_raw"] = gor.where(~gor.isin(["", "-1", "-2", "-8", "-9", ".m"]))
    out["design_psu_raw"] = pd.to_numeric(out["idahhw4"], errors="coerce").where(
        pd.to_numeric(out["idahhw4"], errors="coerce") >= 0
    )
    return out


def select_cohort(df: pd.DataFrame, threshold: int = 3, baseline: str = "adl_iadl") -> pd.DataFrame:
    independent_col = "baseline_independent" if baseline == "adl_iadl" else "baseline_adl_independent"
    eligible = (
        (df["age"] >= 50) & df["met_w2"].notna() & df["met_w4"].notna()
        & (df["met_w2"] >= threshold) & (df["met_w4"] >= threshold)
        & (df[independent_col] == 1) & (df["blood_weight_w4"] > 0)
        & df["design_stratum_raw"].notna() & df["design_psu_raw"].notna()
    )
    c = df.loc[eligible].copy().reset_index(drop=True)
    c["design_stratum"] = pd.factorize(c["design_stratum_raw"], sort=True)[0]
    c["design_psu"] = pd.factorize(c["design_psu_raw"], sort=True)[0]
    return c


def cohort_flow(df: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    dual_nurse = df["sex_w2"].notna() & df["sex_w4"].notna()
    complete_met = df["met_w2"].notna() & df["met_w4"].notna()
    persistent = complete_met & (df["met_w2"] >= 3) & (df["met_w4"] >= 3)
    age_weight = persistent & (df["age"] >= 50) & (df["blood_weight_w4"] > 0)
    baseline = age_weight & (df["baseline_independent"] == 1)
    design = baseline & df["design_stratum_raw"].notna() & df["design_psu_raw"].notna()
    rows = [
        ("Harmonized ELSA sample", len(df)),
        ("Wave 2 and Wave 4 nurse records", int(dual_nurse.sum())),
        ("All five metabolic components ascertainable at both waves", int(complete_met.sum())),
        ("Metabolic score >=3 at both waves", int(persistent.sum())),
        ("Age >=50 and positive Wave 4 blood weight", int(age_weight.sum())),
        ("Wave 4 ADL and IADL independent", int(baseline.sum())),
        ("Valid Wave 4 stratum and household cluster", int(design.sum())),
        ("Primary durable functional outcome observed", int(cohort["durable_independent"].notna().sum())),
    ]
    result = pd.DataFrame(rows, columns=["step", "n"])
    result["retained_from_previous"] = result["n"] / result["n"].shift(1)
    result.loc[0, "retained_from_previous"] = 1.0
    return result


def survey_meat(scores: np.ndarray, strata: pd.Series, psu: pd.Series):
    scores = np.asarray(scores, float)
    strata = np.asarray(strata)
    psu = np.asarray(psu)
    contributions = []
    singleton = 0
    clusters = 0
    for h in pd.unique(strata):
        idx = np.where(strata == h)[0]
        groups = pd.unique(psu[idx])
        clusters += len(groups)
        totals = np.vstack([scores[idx[psu[idx] == g]].sum(axis=0) for g in groups])
        if len(groups) > 1:
            centered = totals - totals.mean(axis=0)
            contributions.append(len(groups) / (len(groups) - 1) * centered.T @ centered)
        else:
            singleton += 1
    if contributions:
        average = sum(contributions) / len(contributions)
        meat = sum(contributions) + singleton * average
    else:
        groups = pd.unique(psu)
        totals = np.vstack([scores[psu == g].sum(axis=0) for g in groups])
        centered = totals - totals.mean(axis=0)
        meat = len(groups) / max(len(groups) - 1, 1) * centered.T @ centered
    return meat, max(clusters - len(pd.unique(strata)), 1), clusters, singleton


def logistic_fit(data: pd.DataFrame, outcome: str, predictors: list[str], weight: str):
    cols = [outcome, weight, "design_stratum", "design_psu"] + predictors
    d = data[cols].dropna().copy()
    x = np.column_stack([np.ones(len(d)), d[predictors].astype(float).to_numpy()])
    y = d[outcome].astype(float).to_numpy()
    w = d[weight].astype(float).to_numpy(copy=True)
    w = w / np.mean(w)
    beta = np.zeros(x.shape[1])
    converged = False
    for _ in range(300):
        p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
        v = np.clip(p * (1 - p), 1e-8, None)
        info = x.T @ (x * (w * v)[:, None])
        step = np.linalg.pinv(info) @ (x.T @ (w * (y - p)))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            converged = True
            break
    p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
    v = np.clip(p * (1 - p), 1e-8, None)
    bread = np.linalg.pinv(x.T @ (x * (w * v)[:, None]))
    scores = x * (w * (y - p))[:, None]
    meat, design_df, clusters, singleton = survey_meat(scores, d["design_stratum"], d["design_psu"])
    cov = bread @ meat @ bread
    return {
        "beta": beta, "cov": cov, "names": ["const"] + predictors, "N": len(d),
        "events": int(y.sum()), "design_df": design_df, "clusters": clusters,
        "singleton_strata": singleton, "converged": converged,
    }


def response_ipw(data: pd.DataFrame, observed: str, predictors: list[str], base_weight: str):
    cols = [observed, base_weight] + predictors
    d = data[cols].dropna().copy()
    x = np.column_stack([np.ones(len(d)), d[predictors].astype(float).to_numpy()])
    y = d[observed].astype(float).to_numpy()
    w = d[base_weight].astype(float).to_numpy(copy=True)
    w /= np.mean(w)
    beta = np.zeros(x.shape[1])
    for _ in range(300):
        p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
        v = np.clip(p * (1 - p), 1e-8, None)
        step = np.linalg.pinv(x.T @ (x * (w * v)[:, None])) @ (x.T @ (w * (y - p)))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
    stabilized = np.average(y, weights=w) / np.clip(p, 0.05, 0.95)
    lo, hi = np.quantile(stabilized[y == 1], [0.01, 0.99])
    out = pd.Series(np.nan, index=data.index)
    out.loc[d.index] = np.clip(stabilized, lo, hi)
    return out, {
        "eligible_n": len(d), "observed_n": int(y.sum()), "observed_weighted_rate": float(np.average(y, weights=w)),
        "ipw_p1": float(lo), "ipw_p99": float(hi),
    }


def multinomial_fit(data: pd.DataFrame, outcome: str, predictors: list[str], weight: str):
    cols = [outcome, weight, "design_stratum", "design_psu"] + predictors
    d = data[cols].dropna().copy()
    y = d[outcome].astype(int).to_numpy()
    x = np.column_stack([np.ones(len(d)), d[predictors].astype(float).to_numpy()])
    n, pcols = x.shape
    w = d[weight].astype(float).to_numpy(copy=True)
    w /= np.mean(w)
    beta = np.zeros((2, pcols))
    converged = False
    for _ in range(300):
        eta = np.clip(x @ beta.T, -25, 25)
        exp_eta = np.exp(eta)
        denom = 1 + exp_eta.sum(axis=1)
        probs = np.column_stack([1 / denom, exp_eta[:, 0] / denom, exp_eta[:, 1] / denom])
        gradient = np.concatenate([x.T @ (w * ((y == j).astype(float) - probs[:, j])) for j in [1, 2]])
        info = np.zeros((2 * pcols, 2 * pcols))
        for a, ja in enumerate([1, 2]):
            for b, jb in enumerate([1, 2]):
                coefficient = probs[:, ja] * ((1 if ja == jb else 0) - probs[:, jb])
                info[a * pcols:(a + 1) * pcols, b * pcols:(b + 1) * pcols] = x.T @ (
                    x * (w * coefficient)[:, None]
                )
        step = np.linalg.pinv(info) @ gradient
        beta += step.reshape(2, pcols)
        if np.max(np.abs(step)) < 1e-8:
            converged = True
            break
    eta = np.clip(x @ beta.T, -25, 25)
    exp_eta = np.exp(eta)
    denom = 1 + exp_eta.sum(axis=1)
    probs = np.column_stack([1 / denom, exp_eta[:, 0] / denom, exp_eta[:, 1] / denom])
    info = np.zeros((2 * pcols, 2 * pcols))
    scores = np.zeros((n, 2 * pcols))
    for a, ja in enumerate([1, 2]):
        scores[:, a * pcols:(a + 1) * pcols] = x * (w * ((y == ja).astype(float) - probs[:, ja]))[:, None]
        for b, jb in enumerate([1, 2]):
            coefficient = probs[:, ja] * ((1 if ja == jb else 0) - probs[:, jb])
            info[a * pcols:(a + 1) * pcols, b * pcols:(b + 1) * pcols] = x.T @ (
                x * (w * coefficient)[:, None]
            )
    bread = np.linalg.pinv(info)
    meat, design_df, clusters, singleton = survey_meat(scores, d["design_stratum"], d["design_psu"])
    return {
        "beta": beta.reshape(-1), "cov": bread @ meat @ bread, "names": ["const"] + predictors,
        "N": n, "counts": {str(state): int(np.sum(y == state)) for state in [0, 1, 2]},
        "design_df": design_df, "clusters": clusters, "singleton_strata": singleton,
        "converged": converged,
    }


def prepare_completed(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    out["age_z"] = zscore(out["age"])
    out["education_z"] = zscore(out["education"])
    out["comorbidity_z"] = zscore(out["comorbidity"])
    out["self_health_z"] = zscore(out["self_health"])
    out["met_w4_z"] = zscore(out["met_w4"])
    out["memory_z"] = zscore(out["memory_w4"])
    out["memory_w2_z"] = zscore(out["memory_w2"])
    out["persistent_low_depression"] = ((out["cesd_w2"] < 4) & (out["cesd_w4"] < 4)).astype(float)
    return out


def pool_scalar(estimates, variances, dfs):
    q = np.asarray(estimates, float)
    u = np.asarray(variances, float)
    m = len(q)
    qbar = q.mean()
    ubar = u.mean()
    between = q.var(ddof=1) if m > 1 else 0.0
    total = ubar + (1 + 1 / m) * between
    if between > 1e-14:
        rubin_df = (m - 1) * (1 + ubar / ((1 + 1 / m) * between)) ** 2
    else:
        rubin_df = np.inf
    design_df = float(np.nanmedian(dfs))
    df = min(design_df, rubin_df) if np.isfinite(rubin_df) else design_df
    se = math.sqrt(max(total, 0))
    crit = stats.t.ppf(0.975, df)
    p = 2 * stats.t.sf(abs(qbar / se), df) if se > 0 else np.nan
    return qbar, se, df, p, qbar - crit * se, qbar + crit * se, ubar, between


def marginal_contrast(fit, data: pd.DataFrame, predictor: str, kind: str, standard_weight: str):
    names = fit["names"]
    predictors = names[1:]
    d = data[[standard_weight] + predictors].dropna().copy()
    x0 = np.column_stack([np.ones(len(d)), d[predictors].astype(float).to_numpy()])
    x1 = x0.copy()
    j = names.index(predictor)
    if kind == "binary":
        x0[:, j] = 0
        x1[:, j] = 1
    elif kind == "plus_one":
        x1[:, j] = x0[:, j] + 1
    else:
        raise ValueError(kind)
    beta = fit["beta"]
    p0 = 1 / (1 + np.exp(-np.clip(x0 @ beta, -30, 30)))
    p1 = 1 / (1 + np.exp(-np.clip(x1 @ beta, -30, 30)))
    w = d[standard_weight].to_numpy(float, copy=True)
    w /= w.sum()
    contrast = float(np.sum(w * (p1 - p0)))
    gradient = np.sum(
        w[:, None] * (p1 * (1 - p1))[:, None] * x1
        - w[:, None] * (p0 * (1 - p0))[:, None] * x0,
        axis=0,
    )
    variance = float(gradient @ fit["cov"] @ gradient)
    return contrast, variance


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.dropna().sort_values()
    running = 0.0
    m = len(valid)
    for rank, (idx, value) in enumerate(valid.items()):
        running = max(running, min(1.0, value * (m - rank)))
        result.loc[idx] = running
    return result


def frame_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def run_mi(cohort: pd.DataFrame):
    np.random.seed(SEED)
    base_vars = [
        "age", "female", "education", "rural", "partnered", "smoking", "drinking",
        "comorbidity", "self_health", "met_w4", "memory_w2", "memory_w4", "cesd_w2", "cesd_w4",
        "durable_independent", "w6_composite_independent", "w6_independent", "w8_independent", "recovered_w8", "w6_state",
        "observed_durable", "observed_w6_composite", "observed_w6_function", "observed_w8_function", "observed_recovery",
    ]
    original = cohort[[
        "durable_independent", "w6_composite_independent", "w6_independent", "w8_independent", "recovered_w8", "w6_state",
        "observed_durable", "observed_w6_composite", "observed_w6_function", "observed_w8_function", "observed_recovery",
        "blood_weight_w4", "design_stratum", "design_psu", "w6_disabled",
    ]].copy()
    mi_frame = cohort[base_vars].astype(float)
    mice = MICEData(mi_frame, perturbation_method="gaussian", k_pmm=20)
    mice.update_all(10)

    core = [
        "age_z", "female", "education_z", "rural", "partnered", "smoking", "drinking",
        "comorbidity_z", "self_health_z", "met_w4_z",
    ]
    model_specs = [
        ("attenuation_unadjusted_joint", "durable_independent", "observed_durable", ["persistent_low_depression", "memory_z"]),
        ("attenuation_age_sex_joint", "durable_independent", "observed_durable", ["age_z", "female", "persistent_low_depression", "memory_z"]),
        ("attenuation_sociodemographic_joint", "durable_independent", "observed_durable", ["age_z", "female", "education_z", "rural", "partnered", "persistent_low_depression", "memory_z"]),
        ("attenuation_without_health_status", "durable_independent", "observed_durable", ["age_z", "female", "education_z", "rural", "partnered", "smoking", "drinking", "met_w4_z", "persistent_low_depression", "memory_z"]),
        ("primary_joint_durable", "durable_independent", "observed_durable", core + ["persistent_low_depression", "memory_z"]),
        ("primary_memory_only", "durable_independent", "observed_durable", core + ["memory_z"]),
        ("primary_depression_only", "durable_independent", "observed_durable", core + ["persistent_low_depression"]),
        ("secondary_w6_disability_or_death", "w6_composite_independent", "observed_w6_composite", core + ["persistent_low_depression", "memory_z"]),
        ("secondary_w6_function_respondents", "w6_independent", "observed_w6_function", core + ["persistent_low_depression", "memory_z"]),
        ("secondary_w8_function_respondents", "w8_independent", "observed_w8_function", core + ["persistent_low_depression", "memory_z"]),
        ("sensitivity_prior_memory_adjusted", "durable_independent", "observed_durable", core + ["memory_w2_z", "persistent_low_depression", "memory_z"]),
    ]
    fits: dict[str, list[dict]] = {spec[0]: [] for spec in model_specs}
    fits["exploratory_recovery_w6_disabled"] = []
    multistate_fits = []
    contrasts = {"persistent_low_depression": [], "memory_z": []}
    ipw_diagnostics = []

    for imp in range(1, M + 1):
        mice.update_all(5)
        d = prepare_completed(mice.data.copy())
        for col in original:
            d[col] = original[col].to_numpy()
        response_predictors = core + ["persistent_low_depression", "memory_z"]
        for model_name, outcome, observed, predictors in model_specs:
            ipw, diag = response_ipw(d, observed, response_predictors, "blood_weight_w4")
            d["analysis_weight"] = d["blood_weight_w4"] * ipw
            fit = logistic_fit(d, outcome, predictors, "analysis_weight")
            fits[model_name].append(fit)
            ipw_diagnostics.append({"imputation": imp, "model": model_name, **diag})
            if model_name == "primary_joint_durable":
                for focal, kind in [("persistent_low_depression", "binary"), ("memory_z", "plus_one")]:
                    estimate, variance = marginal_contrast(fit, d, focal, kind, "blood_weight_w4")
                    contrasts[focal].append((estimate, variance, fit["design_df"]))

        recovery = d[d["w6_disabled"] == 1].copy()
        recovery_predictors = ["age_z", "female", "met_w4_z", "persistent_low_depression", "memory_z"]
        recovery_ipw, recovery_diag = response_ipw(
            recovery, "observed_recovery", recovery_predictors, "blood_weight_w4"
        )
        recovery["analysis_weight"] = recovery["blood_weight_w4"] * recovery_ipw
        recovery_fit = logistic_fit(recovery, "recovered_w8", recovery_predictors, "analysis_weight")
        fits["exploratory_recovery_w6_disabled"].append(recovery_fit)
        ipw_diagnostics.append({"imputation": imp, "model": "exploratory_recovery_w6_disabled", **recovery_diag})

        transition_predictors = ["age_z", "female", "met_w4_z", "persistent_low_depression", "memory_z"]
        transition_ipw, transition_diag = response_ipw(
            d, "observed_w6_composite", transition_predictors, "blood_weight_w4"
        )
        d["transition_weight"] = d["blood_weight_w4"] * transition_ipw
        multistate_fits.append(multinomial_fit(d, "w6_state", transition_predictors, "transition_weight"))
        ipw_diagnostics.append({"imputation": imp, "model": "multistate_w4_w6", **transition_diag})

    rows = []
    focal_names = {"persistent_low_depression", "memory_z", "memory_w2_z"}
    for model_name, flist in fits.items():
        for j, name in enumerate(flist[0]["names"]):
            if name not in focal_names:
                continue
            pooled = pool_scalar(
                [f["beta"][j] for f in flist], [f["cov"][j, j] for f in flist], [f["design_df"] for f in flist]
            )
            q, se, df, p, lo, hi, within, between = pooled
            rows.append({
                "model": model_name, "variable": name, "imputations": len(flist),
                "analysis_n": int(np.median([f["N"] for f in flist])),
                "independent_events": int(np.median([f["events"] for f in flist])),
                "odds_ratio": math.exp(q), "ci_low": math.exp(lo), "ci_high": math.exp(hi),
                "p_value": p, "df": df, "within_variance": within, "between_variance": between,
                "clusters": int(np.median([f["clusters"] for f in flist])),
                "singleton_strata": int(np.median([f["singleton_strata"] for f in flist])),
                "all_converged": bool(all(f["converged"] for f in flist)),
            })
    models = pd.DataFrame(rows)
    primary_mask = (models["model"] == "primary_joint_durable") & models["variable"].isin(["persistent_low_depression", "memory_z"])
    models.loc[primary_mask, "holm_p_value"] = holm_adjust(models.loc[primary_mask, "p_value"])

    contrast_rows = []
    for focal, values in contrasts.items():
        q, se, df, p, lo, hi, _, _ = pool_scalar(
            [x[0] for x in values], [x[1] for x in values], [x[2] for x in values]
        )
        contrast_rows.append({
            "variable": focal, "contrast": "low symptoms vs not" if focal == "persistent_low_depression" else "+1 SD memory",
            "adjusted_probability_difference": q, "ci_low": lo, "ci_high": hi, "p_value": p,
            "imputations": len(values),
        })
    multistate_rows = []
    multistate_names = multistate_fits[0]["names"]
    block = len(multistate_names)
    for state_index, transition_name in [(0, "ADL disability vs independent"), (1, "death vs independent")]:
        for focal in ["persistent_low_depression", "memory_z"]:
            j = state_index * block + multistate_names.index(focal)
            q, se, df, p, lo, hi, within, between = pool_scalar(
                [f["beta"][j] for f in multistate_fits], [f["cov"][j, j] for f in multistate_fits],
                [f["design_df"] for f in multistate_fits],
            )
            multistate_rows.append({
                "transition": transition_name, "variable": focal, "imputations": len(multistate_fits),
                "analysis_n": int(np.median([f["N"] for f in multistate_fits])),
                "independent_n": int(np.median([f["counts"]["0"] for f in multistate_fits])),
                "disability_n": int(np.median([f["counts"]["1"] for f in multistate_fits])),
                "death_n": int(np.median([f["counts"]["2"] for f in multistate_fits])),
                "relative_risk_ratio": math.exp(q), "ci_low": math.exp(lo), "ci_high": math.exp(hi),
                "p_value": p, "df": df, "within_variance": within, "between_variance": between,
                "clusters": int(np.median([f["clusters"] for f in multistate_fits])),
                "all_converged": bool(all(f["converged"] for f in multistate_fits)),
            })
    multistate = pd.DataFrame(multistate_rows)
    disability_mask = multistate["transition"] == "ADL disability vs independent"
    multistate.loc[disability_mask, "holm_p_value"] = holm_adjust(multistate.loc[disability_mask, "p_value"])
    return models, pd.DataFrame(contrast_rows), pd.DataFrame(ipw_diagnostics), multistate


def run_complete_case(cohort: pd.DataFrame, label: str) -> pd.DataFrame:
    d = prepare_completed(cohort.copy())
    core = [
        "age_z", "female", "education_z", "rural", "partnered", "smoking", "drinking",
        "comorbidity_z", "self_health_z", "met_w4_z",
    ]
    predictors = core + ["persistent_low_depression", "memory_z"]
    ipw, _ = response_ipw(d, "observed_durable", predictors, "blood_weight_w4")
    d["analysis_weight"] = d["blood_weight_w4"] * ipw
    fit = logistic_fit(d, "durable_independent", predictors, "analysis_weight")
    rows = []
    for focal in ["persistent_low_depression", "memory_z"]:
        j = fit["names"].index(focal)
        beta = fit["beta"][j]
        se = math.sqrt(max(fit["cov"][j, j], 0))
        crit = stats.t.ppf(0.975, fit["design_df"])
        p = 2 * stats.t.sf(abs(beta / se), fit["design_df"])
        rows.append({
            "analysis": label, "variable": focal, "analysis_n": fit["N"], "independent_events": fit["events"],
            "odds_ratio": math.exp(beta), "ci_low": math.exp(beta - crit * se), "ci_high": math.exp(beta + crit * se),
            "p_value": p, "clusters": fit["clusters"], "converged": fit["converged"],
        })
    return pd.DataFrame(rows)


def make_quality_table(cohort: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "memory_w2", "memory_w4", "cesd_w2", "cesd_w4", "education", "rural", "partnered",
        "smoking", "drinking", "comorbidity", "self_health", "durable_independent",
        "w6_composite_independent", "w8_independent", "recovered_w8",
    ]
    rows = []
    for name in variables:
        s = cohort[name]
        rows.append({
            "variable": name, "nonmissing_n": int(s.notna().sum()), "missing_n": int(s.isna().sum()),
            "complete_percent": float(100 * s.notna().mean()), "mean": float(s.mean()) if s.notna().any() else np.nan,
            "sd": float(s.std(ddof=1)) if s.notna().sum() > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    raw = read_source_data()
    derived = derive_analysis_variables(raw)
    cohort = select_cohort(derived)
    flow = cohort_flow(derived, cohort)
    quality = make_quality_table(cohort)
    models, contrasts, ipw_diag, multistate = run_mi(cohort)

    sensitivity_frames = [run_complete_case(cohort, "main_definition_complete_case")]
    derived_europid = derive_analysis_variables(raw, male_waist=94, female_waist=80)
    cohort_europid = select_cohort(derived_europid)
    sensitivity_frames.append(run_complete_case(cohort_europid, "europid_waist_94_80_complete_case"))
    cohort_strict = select_cohort(derived, threshold=4)
    sensitivity_frames.append(run_complete_case(cohort_strict, "persistent_score_ge4_complete_case"))
    cohort_adl_only = select_cohort(derived, baseline="adl_only")
    sensitivity_frames.append(run_complete_case(cohort_adl_only, "baseline_adl_only_complete_case"))
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)

    weight = cohort["blood_weight_w4"]
    outcomes = {
        "cohort_n": len(cohort),
        "primary_observed_n": int(cohort["durable_independent"].notna().sum()),
        "primary_independent_n": int((cohort["durable_independent"] == 1).sum()),
        "primary_failure_n": int((cohort["durable_independent"] == 0).sum()),
        "w6_composite_observed_n": int(cohort["w6_composite_independent"].notna().sum()),
        "w6_composite_independent_n": int((cohort["w6_composite_independent"] == 1).sum()),
        "w8_function_observed_n": int(cohort["w8_independent"].notna().sum()),
        "w8_independent_n": int((cohort["w8_independent"] == 1).sum()),
        "w6_disabled_n": int((cohort["w6_disabled"] == 1).sum()),
        "recovery_observed_n": int(cohort["recovered_w8"].notna().sum()),
        "recovered_n": int((cohort["recovered_w8"] == 1).sum()),
        "persistent_low_depression_nonmissing_n": int(cohort["persistent_low_depression"].notna().sum()),
        "persistent_low_depression_n": int((cohort["persistent_low_depression"] == 1).sum()),
        "memory_w4_nonmissing_n": int(cohort["memory_w4"].notna().sum()),
        "design_strata": int(cohort["design_stratum"].nunique()),
        "household_clusters": int(cohort["design_psu"].nunique()),
        "blood_weight_kish_n": float(weight.sum() ** 2 / (weight.pow(2).sum())),
        "europid_waist_cohort_n": len(cohort_europid),
        "persistent_score_ge4_cohort_n": len(cohort_strict),
        "baseline_adl_only_cohort_n": len(cohort_adl_only),
    }

    variable_mapping = pd.DataFrame([
        ["Persistent metabolic high risk", "W2 and W4 nurse biomarkers + harmonized diagnoses/medication", "All 5 components known; score >=3 at both waves", "Exact threshold translation; glucose/lipids converted to mmol/L"],
        ["Baseline functional independence", "r4adltot6 and r4iadlza", "Both equal 0", "Exact conceptual match"],
        ["Memory", "r4tr20", "Immediate + delayed 10-word recall, 0-20; modeled per SD", "Direct harmonized match"],
        ["Persistent low depressive symptoms", "r2cesd and r4cesd", "CES-D-8 <4 at both waves", "Scale-specific validated analogue of CHARLS CES-D-10 <10"],
        ["Primary outcome", "r6adltot6 and r8adltot6", "ADL=0 at both observed waves", "Functional-only analogue; Wave 8 mortality is not fully identified locally"],
        ["Key secondary outcome", "r6adltot6 and r6iwstat", "ADL=0 vs ADL disability/death by Wave 6", "Closer composite replication; death status identifiable through Wave 6"],
        ["Survey weight", "w4bldwt", "Wave 4 blood-sample weight", "Official cross-sectional blood weight; follow-up response IPW added"],
        ["Design", "gor and idahhw4", "Region strata and household clusters", "Recommended for Wave 3 onward in ELSA nurse guide"],
    ], columns=["construct", "ELSA_source", "operational_definition", "comparability_note"])

    flow.to_csv(OUT / "elsa_cohort_flow.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(OUT / "elsa_data_quality.csv", index=False, encoding="utf-8-sig")
    models.to_csv(OUT / "elsa_validation_models.csv", index=False, encoding="utf-8-sig")
    multistate.to_csv(OUT / "elsa_multistate_models.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(OUT / "elsa_adjusted_probability_differences.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(OUT / "elsa_sensitivity_models.csv", index=False, encoding="utf-8-sig")
    ipw_diag.to_csv(OUT / "elsa_ipw_diagnostics.csv", index=False, encoding="utf-8-sig")
    variable_mapping.to_csv(OUT / "elsa_variable_mapping.csv", index=False, encoding="utf-8-sig")

    primary = models[(models["model"] == "primary_joint_durable") & models["variable"].isin(["persistent_low_depression", "memory_z"])].copy()
    charls_reference = pd.DataFrame([
        ["persistent_low_depression", 1.4063311751, 0.9734367715, 2.0317368647, 0.0691571752],
        ["memory_z", 1.3052296973, 1.0813709885, 1.5754302463, 0.0056802955],
    ], columns=["variable", "charls_or", "charls_ci_low", "charls_ci_high", "charls_p_value"])
    comparison = charls_reference.merge(
        primary[["variable", "odds_ratio", "ci_low", "ci_high", "p_value", "holm_p_value"]],
        on="variable", how="left",
    ).rename(columns={
        "odds_ratio": "elsa_or", "ci_low": "elsa_ci_low", "ci_high": "elsa_ci_high",
        "p_value": "elsa_p_value", "holm_p_value": "elsa_holm_p_value",
    })
    comparison["same_point_direction"] = comparison["elsa_or"] > 1
    comparison["elsa_replication_status"] = np.where(
        (comparison["elsa_or"] > 1) & (comparison["elsa_holm_p_value"] < 0.05),
        "replicated",
        np.where(comparison["elsa_or"] > 1, "directionally_consistent_but_inconclusive", "not_replicated_opposite_point_direction"),
    )
    comparison.to_csv(OUT / "charls_elsa_comparison.csv", index=False, encoding="utf-8-sig")

    replication = {}
    for _, row in primary.iterrows():
        direction_ok = row["odds_ratio"] > 1
        adjusted_sig = pd.notna(row["holm_p_value"]) and row["holm_p_value"] < 0.05
        replication[row["variable"]] = "replicated" if direction_ok and adjusted_sig else ("directionally_consistent_but_inconclusive" if direction_ok else "not_replicated_opposite_point_direction")
    summary = {
        "seed": SEED,
        "imputations": M,
        "outcomes": outcomes,
        "primary_results": frame_records(primary),
        "adjusted_probability_differences": frame_records(contrasts),
        "multistate_results": frame_records(multistate),
        "charls_elsa_comparison": frame_records(comparison),
        "replication_classification": replication,
        "mortality_limitation": "Local harmonized interview status does not newly classify deaths after Wave 6, and local end-of-life releases do not fill 2013-2015; the Wave 6 composite is valid, whereas the Wave 8 primary is function-only among observed respondents with response IPW.",
        "participant_level_data_exported": False,
    }
    (OUT / "elsa_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


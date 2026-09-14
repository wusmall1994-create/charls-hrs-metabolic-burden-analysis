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


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", ROOT / "results"))
HERE = OUT / "transition_upgrade"
ELSA = Path(os.environ["ELSA_DATA_DIR"])
sys.path.insert(0, str(Path(__file__).resolve().parent))

import charls_hrs_harmonized_analysis as bridge
import hrs_external_validation as hrs


def clean(x: pd.Series, low: float | None = None, high: float | None = None) -> pd.Series:
    y = pd.to_numeric(x, errors="coerce").astype(float)
    if low is not None:
        y = y.where(y >= low)
    if high is not None:
        y = y.where(y <= high)
    return y


def binary(x: pd.Series) -> pd.Series:
    y = clean(x)
    return y.where(y.isin([0, 1]))


def zscore(x: pd.Series) -> pd.Series:
    y = clean(x)
    sd = y.std(ddof=1)
    return (y - y.mean()) / sd if pd.notna(sd) and sd > 0 else y * np.nan


def ternary(positive: pd.Series, known: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=positive.index, dtype=float)
    out.loc[known & ~positive] = 0.0
    out.loc[positive] = 1.0
    return out


def elsa_cohort() -> pd.DataFrame:
    hcols = [
        "idauniq", "ragender", "raeducl",
        "r4cesd", "r6cesd", "r4tr20", "r6tr20", "r6agey", "r6mstat",
        "r6smoken", "r6drink", "r6shlt", "r6adltot6", "r6iadlza",
        "r4hibpe", "r6hibpe", "r4rxhibp", "r6rxhibp",
        "r4diabe", "r6diabe", "r4rxdiab", "r6rxdiab",
    ]
    for w in [7, 8, 9]:
        hcols += [f"r{w}adltot6", f"r{w}iwstat"]
    h = pd.read_stata(ELSA / "gh_elsa_h.dta", columns=hcols, convert_categoricals=False)
    n4 = pd.read_stata(
        ELSA / "wave_4_nurse_data.dta",
        columns=["idauniq", "sysval", "diaval", "hdl", "hba1c", "wstval", "w4bldwt"],
        convert_categoricals=False,
    ).rename(columns={
        "sysval": "sbp4", "diaval": "dbp4", "hdl": "hdl4", "hba1c": "a1c4",
        "wstval": "waist4", "w4bldwt": "weight4",
    })
    n6 = pd.read_stata(
        ELSA / "wave_6_elsa_nurse_data_v2.dta",
        columns=["idauniq", "SYSVAL", "DIAVAL", "hdl", "hba1c", "WSTVAL", "w6bldwt"],
        convert_categoricals=False,
    ).rename(columns={
        "SYSVAL": "sbp6", "DIAVAL": "dbp6", "hdl": "hdl6", "hba1c": "a1c6",
        "WSTVAL": "waist6", "w6bldwt": "weight6",
    })
    design = pd.read_stata(
        ELSA / "wave_6_elsa_data_eul.dta",
        columns=["idauniq", "idahhw6", "gor"], convert_categoricals=False,
    )
    d = h.merge(n4, on="idauniq", how="inner", validate="one_to_one")
    d = d.merge(n6, on="idauniq", how="inner", validate="one_to_one")
    d = d.merge(design, on="idauniq", how="left", validate="one_to_one")
    female = clean(d["ragender"]).eq(2)
    sex_known = clean(d["ragender"]).isin([1, 2])
    d["female"] = np.where(sex_known, female.astype(float), np.nan)
    for wave in [4, 6]:
        waist = clean(d[f"waist{wave}"], 50, 200)
        sbp = clean(d[f"sbp{wave}"], 60, 260)
        dbp = clean(d[f"dbp{wave}"], 30, 160)
        hdl = clean(d[f"hdl{wave}"], 0.1, 10)
        a1c = clean(d[f"a1c{wave}"], 2, 20) if wave == 4 else clean(d[f"a1c{wave}"], 15, 200)
        hibp = binary(d[f"r{wave}hibpe"])
        diab = binary(d[f"r{wave}diabe"])
        comps = pd.DataFrame(index=d.index)
        comps["waist"] = ternary(
            ((~female) & (waist >= 90)) | (female & (waist >= 80)), waist.notna() & sex_known
        )
        comps["bp"] = ternary(
            (sbp >= 130) | (dbp >= 85) | (hibp == 1),
            sbp.notna() & dbp.notna() & hibp.notna(),
        )
        gly_cut = 5.7 if wave == 4 else 39.0
        comps["glycemia"] = ternary(
            (a1c >= gly_cut) | (diab == 1),
            a1c.notna() & diab.notna(),
        )
        comps["hdl"] = ternary(
            ((~female) & (hdl < 1.03)) | (female & (hdl < 1.29)), hdl.notna() & sex_known
        )
        d[f"score4_{wave}"] = comps.sum(axis=1, min_count=4)

    d["age"] = clean(d["r6agey"], 50, 110)
    d["met_anchor"] = d["score4_6"]
    d["memory_first"] = clean(d["r4tr20"], 0, 20)
    d["memory_anchor"] = clean(d["r6tr20"], 0, 20)
    d["cesd_first"] = clean(d["r4cesd"], 0, 8)
    d["cesd_anchor"] = clean(d["r6cesd"], 0, 8)
    d["persistent_low_depression"] = np.where(
        d["cesd_first"].notna() & d["cesd_anchor"].notna(),
        ((d["cesd_first"] < 4) & (d["cesd_anchor"] < 4)).astype(float), np.nan,
    )
    d["memory_z"] = zscore(d["memory_anchor"])
    adl6 = clean(d["r6adltot6"], 0, 6)
    iadl6 = clean(d["r6iadlza"], 0, 5)
    d["baseline_independent"] = np.where(
        adl6.notna() & iadl6.notna(), ((adl6 == 0) & (iadl6 == 0)).astype(float), np.nan
    )
    d["blood_weight"] = clean(d["weight6"], 0)
    d["design_stratum"] = pd.factorize(d["gor"].astype("string"), sort=True)[0]
    d["design_psu"] = pd.factorize(clean(d["idahhw6"]), sort=True)[0]
    for w in [7, 8, 9]:
        adl = clean(d[f"r{w}adltot6"], 0, 6)
        status = clean(d[f"r{w}iwstat"])
        d[f"state_{w}"] = np.where((status == 1) & adl.notna(), (adl > 0).astype(float), np.nan)
    eligible = (
        (d["score4_4"] >= 3) & (d["score4_6"] >= 3) & (d["baseline_independent"] == 1)
        & (d["blood_weight"] > 0) & clean(d["idahhw6"]).notna() & d["gor"].notna()
    )
    return d.loc[eligible].copy().reset_index(drop=True)


def common_charls() -> pd.DataFrame:
    c = bridge.charls_bridge_cohort().copy()
    source = bridge.charls.build_cohort("persistent3_of_4_no_tg").copy()
    source = source.loc[
        (pd.to_numeric(source["weight2015"], errors="coerce") > 0)
        & source["stratum_prov_urban"].notna() & source["psu_code"].notna()
    ].reset_index(drop=True)
    c["person_id"] = source["ID"].astype(str).to_numpy()
    c["memory_z"] = zscore(c["memory_anchor"])
    c["persistent_low_depression"] = np.where(
        c["cesd_first"].notna() & c["cesd_anchor"].notna(),
        ((c["cesd_first"] < 10) & (c["cesd_anchor"] < 10)).astype(float), np.nan,
    )
    c["state_2015"] = 0.0
    c["state_2018"] = c["f1_state"]
    c["state_2020"] = c["f2_state"]
    return c


def common_hrs() -> pd.DataFrame:
    biomarkers = hrs.load_biomarkers()
    longitudinal = hrs.load_longitudinal()
    source = hrs.build_window(longitudinal, biomarkers, 10, 12, followup_offsets=(1, 2, 3, 4))
    c = hrs.select_cohort(source, 10, 12, threshold=3).copy()
    c["person_id"] = c["hhidpn"].astype(str)
    c["memory_z"] = zscore(c["memory_anchor"])
    c["state_2014"] = 0.0
    for i, year in enumerate([2016, 2018, 2020, 2022], start=1):
        c[f"state_{year}"] = c[f"f{i}_state"]
    return c


def person_period(cohort: str, d: pd.DataFrame, years: list[int], id_col: str) -> pd.DataFrame:
    rows = []
    for interval, (a, b) in enumerate(zip(years[:-1], years[1:]), start=1):
        keep = [id_col, "age", "female", "met_anchor", "persistent_low_depression", "memory_z",
                "blood_weight", "design_stratum", "design_psu", f"state_{a}", f"state_{b}"]
        if "lower_depression_burden_z" in d:
            keep.insert(keep.index("memory_z"), "lower_depression_burden_z")
        x = d[keep].copy()
        x = x.rename(columns={id_col: "person_id", f"state_{a}": "origin", f"state_{b}": "destination"})
        x["cohort"] = cohort
        x["interval"] = interval
        x["from_year"] = a
        x["to_year"] = b
        rows.append(x)
    return pd.concat(rows, ignore_index=True)


def survey_logistic(d: pd.DataFrame, outcome: str, predictors: list[str]) -> dict:
    cols = [outcome, "blood_weight", "design_stratum", "design_psu"] + predictors
    q = d[cols].dropna().copy()
    x = np.column_stack([np.ones(len(q)), q[predictors].to_numpy(float)])
    y = q[outcome].to_numpy(float)
    w = q["blood_weight"].to_numpy(float).copy()
    w /= w.mean()
    beta = np.zeros(x.shape[1])
    for _ in range(300):
        p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
        v = np.clip(p * (1 - p), 1e-9, None)
        info = x.T @ (x * (w * v)[:, None])
        step = np.linalg.pinv(info) @ (x.T @ (w * (y - p)))
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
    bread = np.linalg.pinv(x.T @ (x * (w * np.clip(p * (1 - p), 1e-9, None))[:, None]))
    scores = x * (w * (y - p))[:, None]
    meat, design_df, clusters, singleton = hrs.survey_meat(scores, q["design_stratum"], q["design_psu"])
    cov = bread @ meat @ bread
    names = ["const"] + predictors
    return {"beta": beta, "cov": cov, "names": names, "n": len(q), "events": int(y.sum()),
            "design_df": design_df, "clusters": clusters, "singletons": singleton}


def fit_transitions(pp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in pp["cohort"].unique():
        d0 = pp[(pp["cohort"] == cohort) & (pp["origin"] == 0) & pp["destination"].isin([0, 1])].copy()
        d0["event"] = (d0["destination"] == 1).astype(float)
        max_int = int(d0["interval"].max()) if len(d0) else 0
        interval_terms = []
        for k in range(2, max_int + 1):
            name = f"interval_{k}"
            d0[name] = (d0["interval"] == k).astype(float)
            interval_terms.append(name)
        predictors = ["persistent_low_depression", "memory_z", "age", "female", "met_anchor"] + interval_terms
        fit = survey_logistic(d0, "event", predictors)
        for focal in ["persistent_low_depression", "memory_z"]:
            j = fit["names"].index(focal)
            se = math.sqrt(max(fit["cov"][j, j], 0))
            crit = stats.t.ppf(0.975, fit["design_df"])
            rows.append({
                "cohort": cohort, "transition": "independence_to_disability", "variable": focal,
                "records": fit["n"], "events": fit["events"], "estimate": math.exp(fit["beta"][j]),
                "ci_low": math.exp(fit["beta"][j] - crit * se),
                "ci_high": math.exp(fit["beta"][j] + crit * se),
                "p_value": 2 * stats.t.sf(abs(fit["beta"][j] / se), fit["design_df"]),
                "measure": "OR", "clusters": fit["clusters"], "singleton_strata": fit["singletons"],
            })
        d1 = pp[(pp["cohort"] == cohort) & (pp["origin"] == 1) & pp["destination"].isin([0, 1])].copy()
        d1["event"] = (d1["destination"] == 0).astype(float)
        max_int = int(d1["interval"].max()) if len(d1) else 0
        interval_terms = []
        for k in range(2, max_int + 1):
            name = f"interval_{k}"
            d1[name] = (d1["interval"] == k).astype(float)
            interval_terms.append(name)
        if len(d1) and d1["event"].sum() >= 10:
            predictors = ["persistent_low_depression", "memory_z", "age", "female", "met_anchor"] + interval_terms
            fit = survey_logistic(d1, "event", predictors)
            for focal in ["persistent_low_depression", "memory_z"]:
                j = fit["names"].index(focal)
                se = math.sqrt(max(fit["cov"][j, j], 0))
                crit = stats.t.ppf(0.975, fit["design_df"])
                rows.append({
                    "cohort": cohort, "transition": "disability_to_independence", "variable": focal,
                    "records": fit["n"], "events": fit["events"], "estimate": math.exp(fit["beta"][j]),
                    "ci_low": math.exp(fit["beta"][j] - crit * se),
                    "ci_high": math.exp(fit["beta"][j] + crit * se),
                    "p_value": 2 * stats.t.sf(abs(fit["beta"][j] / se), fit["design_df"]),
                    "measure": "OR", "clusters": fit["clusters"], "singleton_strata": fit["singletons"],
                })
    return pd.DataFrame(rows)


def impute_wide(d: pd.DataFrame, m: int = 20, seed: int = 20260914) -> list[pd.DataFrame]:
    """Impute baseline predictors only; functional states are never imputed."""
    np.random.seed(seed)
    cols = ["age", "female", "met_anchor", "memory_first", "memory_anchor", "cesd_first", "cesd_anchor"]
    aux = d[[c for c in d.columns if c.startswith("state_")]].copy()
    aux["ever_disabled"] = (aux == 1).any(axis=1).astype(float)
    x = d[cols].astype(float).copy()
    x["ever_disabled"] = aux["ever_disabled"]
    mice = MICEData(x, perturbation_method="gaussian", k_pmm=20)
    mice.update_all(10)
    out = []
    for _ in range(m):
        mice.update_all(5)
        z = d.copy()
        completed = mice.data
        for col in cols:
            z[col] = completed[col].to_numpy()
        z["persistent_low_depression"] = (
            (z["cesd_first"] < z.attrs["cesd_threshold"])
            & (z["cesd_anchor"] < z.attrs["cesd_threshold"])
        ).astype(float)
        z["memory_z"] = zscore(z["memory_anchor"])
        z["lower_depression_burden_z"] = -zscore(
            (zscore(z["cesd_first"]) + zscore(z["cesd_anchor"])) / 2
        )
        out.append(z)
    return out


def fit_transitions_mi(cohorts: dict[str, pd.DataFrame], years: dict[str, list[int]], m: int = 20) -> pd.DataFrame:
    fitted: dict[tuple[str, str, str], list[dict]] = {}
    for cohort, wide in cohorts.items():
        for completed in impute_wide(wide, m=m):
            pp = person_period(cohort, completed, years[cohort], "person_id")
            specs = [
                ("independence_to_disability", 0, 1, [0, 1]),
                ("disability_to_independence", 1, 0, [0, 1]),
            ]
            if cohort in {"CHARLS", "HRS"}:
                specs += [
                    ("independence_to_death", 0, 2, [0, 1, 2]),
                    ("disability_to_death", 1, 2, [0, 1, 2]),
                ]
            for transition, origin, destination_event, allowed in specs:
                q = pp[(pp["origin"] == origin) & pp["destination"].isin(allowed)].copy()
                q["event"] = (q["destination"] == destination_event).astype(float)
                if len(q) == 0 or q["event"].sum() < 10:
                    continue
                interval_terms = []
                for k in range(2, int(q["interval"].max()) + 1):
                    term = f"interval_{k}"
                    q[term] = (q["interval"] == k).astype(float)
                    interval_terms.append(term)
                joint_predictors = ["persistent_low_depression", "memory_z", "age", "female", "met_anchor"] + interval_terms
                joint_fit = survey_logistic(q, "event", joint_predictors)
                for focal in ["persistent_low_depression", "memory_z"]:
                    fit = joint_fit
                    fitted.setdefault((cohort, transition, focal), []).append(fit)
                continuous_predictors = ["lower_depression_burden_z", "memory_z", "age", "female", "met_anchor"] + interval_terms
                continuous_fit = survey_logistic(q, "event", continuous_predictors)
                fitted.setdefault((cohort, transition, "lower_depression_burden_z"), []).append(continuous_fit)
    rows = []
    for (cohort, transition, focal), fits in fitted.items():
        j = fits[0]["names"].index(focal)
        pooled = hrs.pool_scalar(
            [f["beta"][j] for f in fits], [f["cov"][j, j] for f in fits], [f["design_df"] for f in fits]
        )
        q, se, df, p, lo, hi, within, between = pooled
        rows.append({
            "cohort": cohort, "transition": transition, "variable": focal,
            "imputations": len(fits), "records": int(np.median([f["n"] for f in fits])),
            "events": int(np.median([f["events"] for f in fits])), "estimate": math.exp(q),
            "ci_low": math.exp(lo), "ci_high": math.exp(hi), "p_value": p, "measure": "OR",
            "df": df, "within_variance": within, "between_variance": between,
            "clusters": int(np.median([f["clusters"] for f in fits])),
        })
    return pd.DataFrame(rows)


def fixed_effect_meta(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (transition, variable), q in results.groupby(["transition", "variable"]):
        y = np.log(q["estimate"].to_numpy(float))
        se = (np.log(q["ci_high"].to_numpy(float)) - np.log(q["ci_low"].to_numpy(float))) / (2 * 1.96)
        w = 1 / np.square(se)
        pooled = np.sum(w * y) / np.sum(w)
        pooled_se = math.sqrt(1 / np.sum(w))
        qstat = np.sum(w * np.square(y - pooled))
        qdf = max(len(y) - 1, 0)
        i2 = max(0.0, (qstat - qdf) / qstat) * 100 if qstat > 0 else 0.0
        rows.append({
            "transition": transition, "variable": variable, "cohorts": len(q),
            "estimate": math.exp(pooled), "ci_low": math.exp(pooled - 1.96 * pooled_se),
            "ci_high": math.exp(pooled + 1.96 * pooled_se),
            "p_value": 2 * stats.norm.sf(abs(pooled / pooled_se)),
            "q": qstat, "q_df": qdf, "heterogeneity_p": stats.chi2.sf(qstat, qdf) if qdf else np.nan,
            "i2_percent": i2, "measure": "fixed-effect pooled OR",
        })
    return pd.DataFrame(rows)


def event_audit(pp: pd.DataFrame) -> pd.DataFrame:
    q = pp.dropna(subset=["origin", "destination"]).copy()
    q["transition"] = q["origin"].astype(int).astype(str) + "->" + q["destination"].astype(int).astype(str)
    return q.groupby(["cohort", "from_year", "to_year", "transition"], observed=True).size().rename("n").reset_index()


def interval_specific_models(pp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, a, b), q0 in pp.groupby(["cohort", "from_year", "to_year"]):
        q = q0[(q0["origin"] == 0) & q0["destination"].isin([0, 1])].copy()
        q["event"] = (q["destination"] == 1).astype(float)
        if q["event"].sum() < 10:
            continue
        fit = survey_logistic(q, "event", ["persistent_low_depression", "memory_z", "age", "female", "met_anchor"])
        for focal in ["persistent_low_depression", "memory_z"]:
            j = fit["names"].index(focal)
            se = math.sqrt(max(fit["cov"][j, j], 0))
            crit = stats.t.ppf(0.975, fit["design_df"])
            rows.append({
                "cohort": cohort, "from_year": a, "to_year": b, "variable": focal,
                "records": fit["n"], "events": fit["events"], "estimate": math.exp(fit["beta"][j]),
                "ci_low": math.exp(fit["beta"][j] - crit * se),
                "ci_high": math.exp(fit["beta"][j] + crit * se),
                "p_value": 2 * stats.t.sf(abs(fit["beta"][j] / se), fit["design_df"]),
            })
    return pd.DataFrame(rows)


def weighted_absolute_risks(pp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    q = pp[(pp["origin"] == 0) & pp["destination"].isin([0, 1]) & pp["persistent_low_depression"].notna()].copy()
    q["event"] = (q["destination"] == 1).astype(float)
    for (cohort, low), d in q.groupby(["cohort", "persistent_low_depression"]):
        risk = np.average(d["event"], weights=d["blood_weight"])
        rows.append({"cohort": cohort, "persistent_low_depression": int(low), "records": len(d),
                     "events": int(d["event"].sum()), "weighted_risk": risk})
    return pd.DataFrame(rows)


def baseline_summary(cohorts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cohort, d in cohorts.items():
        w = d["blood_weight"]
        row = {"cohort": cohort, "n": len(d)}
        for name in ["age", "female", "met_anchor", "memory_anchor", "cesd_anchor", "persistent_low_depression"]:
            ok = d[name].notna() & w.notna() & (w > 0)
            row[f"{name}_observed_n"] = int(ok.sum())
            row[f"{name}_weighted_mean"] = float(np.average(d.loc[ok, name], weights=w.loc[ok]))
            mean = row[f"{name}_weighted_mean"]
            row[f"{name}_weighted_sd"] = float(np.sqrt(np.average(np.square(d.loc[ok, name] - mean), weights=w.loc[ok])))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    c = common_charls()
    h = common_hrs()
    e = elsa_cohort()
    e["person_id"] = e["idauniq"].astype(str)
    e["state_2012"] = 0.0
    e = e.rename(columns={"state_7": "state_2014", "state_8": "state_2016", "state_9": "state_2018"})
    c.attrs["cesd_threshold"] = 10
    h.attrs["cesd_threshold"] = 4
    e.attrs["cesd_threshold"] = 4
    cohorts = {"CHARLS": c, "HRS": h, "ELSA": e}
    years = {"CHARLS": [2015, 2018, 2020], "HRS": [2014, 2016, 2018, 2020, 2022], "ELSA": [2012, 2014, 2016, 2018]}
    pp = pd.concat([
        person_period("CHARLS", c, [2015, 2018, 2020], "person_id"),
        person_period("HRS", h, [2014, 2016, 2018, 2020, 2022], "person_id"),
        person_period("ELSA", e, [2012, 2014, 2016, 2018], "person_id"),
    ], ignore_index=True)
    audit = event_audit(pp)
    results = fit_transitions(pp)
    mi_results = fit_transitions_mi(cohorts, years)
    meta_results = fixed_effect_meta(mi_results)
    interval_results = interval_specific_models(pp)
    absolute_risks = weighted_absolute_risks(pp)
    characteristics = baseline_summary(cohorts)
    cohort_summary = pd.DataFrame([
        {"cohort": "CHARLS", "eligible_n": len(c), "low_depression_n": int((c["persistent_low_depression"] == 1).sum())},
        {"cohort": "HRS", "eligible_n": len(h), "low_depression_n": int((h["persistent_low_depression"] == 1).sum())},
        {"cohort": "ELSA", "eligible_n": len(e), "low_depression_n": int((e["persistent_low_depression"] == 1).sum())},
    ])
    audit.to_csv(HERE / "transition_event_audit.csv", index=False)
    results.to_csv(HERE / "transition_complete_case_feasibility.csv", index=False)
    mi_results.to_csv(HERE / "transition_mi_primary.csv", index=False)
    meta_results.to_csv(HERE / "transition_fixed_effect_meta.csv", index=False)
    interval_results.to_csv(HERE / "transition_interval_specific.csv", index=False)
    absolute_risks.to_csv(HERE / "transition_absolute_risks.csv", index=False)
    characteristics.to_csv(HERE / "baseline_characteristics.csv", index=False)
    # Internal analysis bridge for R only: replace source identifiers with sequential IDs.
    rdata = pp.copy()
    rdata["person_id"] = rdata.groupby("cohort")["person_id"].transform(
        lambda x: pd.factorize(x, sort=True)[0] + 1
    )
    rdata.to_csv(HERE / "person_period_internal.csv", index=False)
    cohort_summary.to_csv(HERE / "cohort_summary.csv", index=False)
    (HERE / "feasibility_summary.json").write_text(json.dumps({
        "cohort_summary": cohort_summary.to_dict("records"),
        "transition_results": results.to_dict("records"),
        "transition_mi_results": mi_results.to_dict("records"),
        "meta_results": meta_results.to_dict("records"),
    }, indent=2), encoding="utf-8")
    print(cohort_summary.to_string(index=False))
    print(audit.to_string(index=False))
    print(results.to_string(index=False))
    print(mi_results.to_string(index=False))
    print(meta_results.to_string(index=False))


if __name__ == "__main__":
    main()


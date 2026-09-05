from pathlib import Path
import json
import math
import os
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.imputation.mice import MICEData

import formal_analysis as fa

ROOT = Path(os.environ.get("CHARLS_DATA_DIR", PROJECT / "data" / "charls"))
EXPORT = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", PROJECT / "results"))
EXPORT.mkdir(parents=True, exist_ok=True)
SEED = 20260818
M = 20


def zscore(x):
    x = pd.to_numeric(x, errors="coerce")
    sd = x.std(ddof=1)
    return (x - x.mean()) / sd if pd.notna(sd) and sd > 0 else x * np.nan


def norm_id(x):
    y = x.astype("string").str.strip()
    return y.mask(y == "", pd.NA)


def survey_meat(scores, strata, psu):
    scores = np.asarray(scores, float)
    strata = np.asarray(strata)
    psu = np.asarray(psu)
    contributions, singleton = [], 0
    n_clusters = 0
    for h in pd.unique(strata):
        idx = np.where(strata == h)[0]
        groups = pd.unique(psu[idx])
        n_clusters += len(groups)
        totals = np.vstack([scores[idx[psu[idx] == g]].sum(axis=0) for g in groups])
        if len(groups) > 1:
            centered = totals - totals.mean(axis=0)
            contributions.append(len(groups) / (len(groups) - 1) * (centered.T @ centered))
        else:
            singleton += 1
    if contributions:
        avg = sum(contributions) / len(contributions)
        meat = sum(contributions) + singleton * avg
    else:
        totals = np.vstack([scores[psu == g].sum(axis=0) for g in pd.unique(psu)])
        centered = totals - totals.mean(axis=0)
        meat = len(totals) / max(len(totals) - 1, 1) * centered.T @ centered
    design_df = max(n_clusters - len(pd.unique(strata)), 1)
    return meat, design_df, n_clusters, singleton


def logistic_fit(data, outcome, predictors, weight, strata, psu):
    cols = [outcome, weight, strata, psu] + predictors
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
        a = x.T @ (x * (w * v)[:, None])
        step = np.linalg.pinv(a) @ (x.T @ (w * (y - p)))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            converged = True
            break
    p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
    v = np.clip(p * (1 - p), 1e-8, None)
    bread = np.linalg.pinv(x.T @ (x * (w * v)[:, None]))
    scores = x * (w * (y - p))[:, None]
    meat, ddf, clusters, lonely = survey_meat(scores, d[strata], d[psu])
    cov = bread @ meat @ bread
    return {
        "beta": beta, "cov": cov, "names": ["const"] + predictors, "N": len(d),
        "events": int(y.sum()), "design_df": ddf, "clusters": clusters,
        "lonely_strata": lonely, "converged": converged, "pred": p, "index": d.index
    }


def response_ipw(data, observed, predictors, survey_weight):
    d = data[[observed, survey_weight] + predictors].dropna().copy()
    x = np.column_stack([np.ones(len(d)), d[predictors].astype(float).to_numpy()])
    y = d[observed].astype(float).to_numpy()
    w = d[survey_weight].astype(float).to_numpy(copy=True); w /= np.mean(w)
    beta = np.zeros(x.shape[1])
    for _ in range(300):
        p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
        v = np.clip(p * (1 - p), 1e-8, None)
        step = np.linalg.pinv(x.T @ (x * (w * v)[:, None])) @ (x.T @ (w * (y - p)))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
    observed_rate = np.average(y, weights=w)
    ipw = observed_rate / np.clip(p, 0.05, 0.95)
    lo, hi = np.quantile(ipw[y == 1], [0.01, 0.99])
    out = pd.Series(np.nan, index=data.index)
    out.loc[d.index] = np.clip(ipw, lo, hi)
    return out, {"N": len(d), "observed": int(y.sum()), "min": float(lo), "max": float(hi)}


def multinomial_fit(data, outcome, predictors, weight, strata, psu):
    cols = [outcome, weight, strata, psu] + predictors
    d = data[cols].dropna().copy()
    y = d[outcome].astype(int).to_numpy()
    cats = [0, 1, 2]
    if any(np.sum(y == c) < 10 for c in cats):
        return None
    x = np.column_stack([np.ones(len(d)), d[predictors].astype(float).to_numpy()])
    n, pcols = x.shape; q = 2
    w = d[weight].astype(float).to_numpy(copy=True); w /= np.mean(w)
    beta = np.zeros((q, pcols)); converged = False
    for _ in range(300):
        eta = np.clip(x @ beta.T, -25, 25)
        denom = 1 + np.exp(eta).sum(axis=1)
        probs = np.column_stack([1 / denom, np.exp(eta[:, 0]) / denom, np.exp(eta[:, 1]) / denom])
        grad = np.concatenate([x.T @ (w * ((y == j).astype(float) - probs[:, j])) for j in [1, 2]])
        info = np.zeros((q * pcols, q * pcols))
        for a, ja in enumerate([1, 2]):
            for b, jb in enumerate([1, 2]):
                coef = probs[:, ja] * ((1 if ja == jb else 0) - probs[:, jb])
                block = x.T @ (x * (w * coef)[:, None])
                info[a*pcols:(a+1)*pcols, b*pcols:(b+1)*pcols] = block
        step = np.linalg.pinv(info) @ grad
        beta += step.reshape(q, pcols)
        if np.max(np.abs(step)) < 1e-8:
            converged = True
            break
    eta = np.clip(x @ beta.T, -25, 25)
    denom = 1 + np.exp(eta).sum(axis=1)
    probs = np.column_stack([1 / denom, np.exp(eta[:, 0]) / denom, np.exp(eta[:, 1]) / denom])
    info = np.zeros((q * pcols, q * pcols))
    scores = np.zeros((n, q * pcols))
    for a, ja in enumerate([1, 2]):
        scores[:, a*pcols:(a+1)*pcols] = x * (w * ((y == ja).astype(float) - probs[:, ja]))[:, None]
        for b, jb in enumerate([1, 2]):
            coef = probs[:, ja] * ((1 if ja == jb else 0) - probs[:, jb])
            info[a*pcols:(a+1)*pcols, b*pcols:(b+1)*pcols] = x.T @ (x * (w * coef)[:, None])
    bread = np.linalg.pinv(info)
    meat, ddf, clusters, lonely = survey_meat(scores, d[strata], d[psu])
    cov = bread @ meat @ bread
    names = ["const"] + predictors
    return {"beta": beta.reshape(-1), "cov": cov, "names": names, "N": n,
            "counts": {str(c): int(np.sum(y == c)) for c in cats}, "design_df": ddf,
            "clusters": clusters, "lonely_strata": lonely, "converged": converged}


def pool_scalar(estimates, variances, dfs):
    q = np.asarray(estimates, float); u = np.asarray(variances, float); m = len(q)
    qbar = q.mean(); ubar = u.mean(); b = q.var(ddof=1) if m > 1 else 0
    total = ubar + (1 + 1/m) * b
    if b > 1e-14:
        rdf = (m - 1) * (1 + ubar / ((1 + 1/m) * b)) ** 2
    else:
        rdf = np.inf
    df = min(float(np.nanmedian(dfs)), rdf) if np.isfinite(rdf) else float(np.nanmedian(dfs))
    se = math.sqrt(max(total, 0)); crit = stats.t.ppf(0.975, df)
    p = 2 * stats.t.sf(abs(qbar / se), df) if se > 0 else np.nan
    return qbar, se, df, p, qbar - crit*se, qbar + crit*se, ubar, b


def build_cohort(eligibility_variant="persistent3", baseline_independence="adl_iadl"):
    h = fa.read(fa.H_PATH, fa.H_COLS); h["ID"] = norm_id(h.ID); h["ID_w1"] = norm_id(h.ID_w1)
    b1 = fa.read(fa.B1_PATH, ["ID", "newglu", "newtg", "newhdl", "newhba1c", "qc1_va003"])
    b1["ID_w1"] = norm_id(b1.pop("ID"))
    b3 = fa.read(fa.B3_PATH, ["ID", "bl_glu", "bl_tg", "bl_hdl", "bl_hbalc", "bl_fasting", "bl_crp", "bl_cysc"])
    b3["ID"] = norm_id(b3.ID)
    h20 = fa.read(fa.H20_PATH, ["ID", "db001", "db003", "db005", "db007", "db009", "db011"])
    h20["ID"] = norm_id(h20.ID); h20["disability_2020"] = fa.adl2020(h20)
    e20 = fa.read(fa.E20_PATH, ["ID", "exb001_1"]); e20["ID"] = norm_id(e20.ID)
    e20 = e20.rename(columns={"exb001_1": "exit_death_year"})
    s18 = pd.read_stata(ROOT / "2018" / "CHARLS2018r" / "Sample_Infor.dta",
                        columns=["ID", "died"], convert_categoricals=False)
    s18["ID"] = norm_id(s18.ID); s18 = s18.rename(columns={"died": "died_2018_sample"})
    s20 = pd.read_stata(ROOT / "2020" / "CHARLS2020r" / "Sample_Infor.dta",
                        columns=["ID", "died"], convert_categoricals=False)
    s20["ID"] = norm_id(s20.ID); s20 = s20.rename(columns={"died": "died_2020_sample"})
    w1 = pd.read_stata(ROOT / "2011" / "weight" / "weight.dta", convert_categoricals=False)
    w1["ID_w1"] = norm_id(w1.pop("ID")); w1["communityID_w1"] = norm_id(w1.communityID)
    w1 = w1[["ID_w1", "bio_weight2", "communityID_w1"]]
    w3 = pd.read_stata(ROOT / "2015" / "Weights" / "Weights.dta", convert_categoricals=False)
    w3["ID"] = norm_id(w3.ID); w3["communityID"] = norm_id(w3.communityID)
    w3 = w3[["ID", "Biomarker_weight", "INDV_weight_ad2", "communityID"]]
    psu = pd.read_stata(ROOT / "2011" / "PSU" / "PSU.dta", convert_categoricals=False)
    psu["communityID_w1"] = norm_id(psu.communityID)
    psu = psu[["communityID_w1", "province_eng", "urban_nbs"]]

    df = h.merge(b1, on="ID_w1", how="left", validate="many_to_one")
    df = df.merge(b3, on="ID", how="left", validate="one_to_one")
    df = df.merge(h20[["ID", "disability_2020"]], on="ID", how="left", validate="one_to_one")
    df = df.merge(e20, on="ID", how="left", validate="one_to_one")
    df = df.merge(s18, on="ID", how="left", validate="one_to_one")
    df = df.merge(s20, on="ID", how="left", validate="one_to_one")
    df = df.merge(w1, on="ID_w1", how="left", validate="many_to_one")
    df = df.merge(w3, on="ID", how="left", validate="one_to_one")
    df = df.merge(psu, on="communityID_w1", how="left", validate="many_to_one")

    df["met1"] = fa.met_score(df, 1); df["met3"] = fa.met_score(df, 3)
    df["met1_4c"] = fa.met_score_without_triglycerides(df, 1)
    df["met3_4c"] = fa.met_score_without_triglycerides(df, 3)
    df["met1_waist85"] = fa.met_score(df, 1, female_waist=85)
    df["met3_waist85"] = fa.met_score(df, 3, female_waist=85)
    df["persistent_high"] = ((df.met1 >= 3) & (df.met3 >= 3) & df.met1.notna() & df.met3.notna()).astype(float)
    df["independent_2015"] = np.where(df.r3adlab_c.notna() & df.r3iadlza.notna(), ((df.r3adlab_c == 0) & (df.r3iadlza == 0)).astype(float), np.nan)
    df["adl_disability_2018"] = np.where(df.r4adlab_c.notna(), (df.r4adlab_c > 0).astype(float), np.nan)
    death_year = pd.to_numeric(df.exit_death_year, errors="coerce").combine_first(pd.to_numeric(df.radyear, errors="coerce"))
    death_by_2018 = (pd.to_numeric(df.died_2018_sample, errors="coerce") == 1) | ((death_year >= 2016) & (death_year <= 2018))
    death_2018_2020 = (pd.to_numeric(df.died_2020_sample, errors="coerce") == 1) | ((death_year >= 2019) & (death_year <= 2020))
    df["death_2018"] = death_by_2018.astype(float)
    df["death_2020"] = (death_by_2018 | death_2018_2020).astype(float)
    df["bad_adl_2018"] = np.where(df.adl_disability_2018.notna() | (df.death_2018 == 1), ((df.adl_disability_2018 == 1) | (df.death_2018 == 1)).astype(float), np.nan)
    df["bad_2020"] = np.where(df.disability_2020.notna() | (df.death_2020 == 1), ((df.disability_2020 == 1) | (df.death_2020 == 1)).astype(float), np.nan)
    df["maintain_adl_2018"] = 1 - df.bad_adl_2018; df["maintain_2020"] = 1 - df.bad_2020
    df["durable_maintain"] = np.where(df.bad_adl_2018.notna() & df.bad_2020.notna(), ((df.bad_adl_2018 == 0) & (df.bad_2020 == 0)).astype(float), np.nan)
    df["observed_durable"] = df.durable_maintain.notna().astype(float)

    df["state_2018"] = np.nan
    df.loc[df.death_2018 == 1, "state_2018"] = 2
    df.loc[(df.death_2018 == 0) & (df.adl_disability_2018 == 0), "state_2018"] = 0
    df.loc[(df.death_2018 == 0) & (df.adl_disability_2018 == 1), "state_2018"] = 1
    df["state_2020"] = np.nan
    df.loc[df.death_2020 == 1, "state_2020"] = 2
    df.loc[(df.death_2020 == 0) & (df.disability_2020 == 0), "state_2020"] = 0
    df.loc[(df.death_2020 == 0) & (df.disability_2020 == 1), "state_2020"] = 1

    df["female"] = (df.ragender == 2).astype(float)
    df["education"] = pd.to_numeric(df.raeducl, errors="coerce")
    df["rural"] = fa.binary(df.h3rural)
    df["partnered"] = np.where(df.r3mstat.notna(), (df.r3mstat == 1).astype(float), np.nan)
    df["smoking"] = fa.binary(df.r3smoken)
    df["drinking"] = np.where(df.r3drinkn_c.notna(), (df.r3drinkn_c > 0).astype(float), np.nan)
    comorb = pd.concat([fa.binary(df[c]) for c in fa.COMORB], axis=1)
    df["comorbidity"] = comorb.sum(axis=1, min_count=len(fa.COMORB))
    df["self_health"] = pd.to_numeric(df.r3shlt, errors="coerce")
    df["social_reserve"] = fa.binary(df.r3socwk)
    df["log_crp"] = np.log(pd.to_numeric(df.bl_crp, errors="coerce").where(df.bl_crp > 0))
    df["log_cysc"] = np.log(pd.to_numeric(df.bl_cysc, errors="coerce").where(df.bl_cysc > 0))

    if baseline_independence == "adl_iadl":
        baseline_independent = df.independent_2015.eq(1)
    elif baseline_independence == "adl_only":
        baseline_independent = df.r3adlab_c.notna() & df.r3adlab_c.eq(0)
    else:
        raise ValueError(f"Unknown baseline_independence: {baseline_independence}")
    common = (df.r1agey >= 45) & baseline_independent
    if eligibility_variant == "persistent3":
        eligible = common & df.met1.notna() & df.met3.notna() & (df.met1 >= 3) & (df.met3 >= 3)
    elif eligibility_variant == "persistent4":
        eligible = common & df.met1.notna() & df.met3.notna() & (df.met1 >= 4) & (df.met3 >= 4)
    elif eligibility_variant == "female_waist85":
        eligible = common & df.met1_waist85.notna() & df.met3_waist85.notna() & (df.met1_waist85 >= 3) & (df.met3_waist85 >= 3)
    elif eligibility_variant == "met2015_only":
        eligible = common & df.met3.notna() & (df.met3 >= 3)
    elif eligibility_variant == "persistent3_no_diabetes_cvd":
        no_dm = fa.binary(df.r3diabe).eq(0)
        no_cvd = fa.binary(df.r3hearte).eq(0) & fa.binary(df.r3stroke).eq(0)
        eligible = common & df.met1.notna() & df.met3.notna() & (df.met1 >= 3) & (df.met3 >= 3) & no_dm & no_cvd
    elif eligibility_variant == "persistent3_of_4_no_tg":
        eligible = (
            common
            & df.met1_4c.notna()
            & df.met3_4c.notna()
            & (df.met1_4c >= 3)
            & (df.met3_4c >= 3)
        )
    else:
        raise ValueError(f"Unknown eligibility_variant: {eligibility_variant}")
    c = df.loc[eligible].copy().reset_index(drop=True)
    c["stratum_prov_urban"] = pd.factorize(c.province_eng.astype("string") + "_" + c.urban_nbs.astype("string"))[0]
    c["stratum_province"] = pd.factorize(c.province_eng.astype("string"))[0]
    c["stratum_one"] = 0
    c["psu_code"] = pd.factorize(c.communityID_w1.astype("string"))[0]
    c["weight2015"] = pd.to_numeric(c.Biomarker_weight, errors="coerce")
    c["weight2011"] = pd.to_numeric(c.bio_weight2, errors="coerce")
    c["weight_individual2015"] = pd.to_numeric(c.INDV_weight_ad2, errors="coerce")
    return c


def prepare_completed(d):
    out = d.copy()
    out["stratum_one"] = 0
    out["age_z"] = zscore(out.r1agey); out["education_z"] = zscore(out.education)
    out["comorbidity_z"] = zscore(out.comorbidity); out["self_health_z"] = zscore(out.self_health)
    out["met3_z"] = zscore(out.met3); out["grip_reserve"] = zscore(out.r3gripsum)
    out["cognitive_reserve"] = zscore(out.r3tr20); out["psychological_reserve"] = -zscore(out.r3cesd10)
    out["persistent_psychological_health"] = ((out.r1cesd10 < 10) & (out.r3cesd10 < 10)).astype(float)
    out["inflammatory_reserve"] = -zscore(out.log_crp); out["renal_reserve"] = -zscore(out.log_cysc)
    return out


def make_transition_rows(d, weight_col):
    core = ["age_z", "female", "education_z", "rural", "partnered", "smoking", "drinking", "comorbidity_z", "self_health_z", "met3_z"]
    d = d.copy()
    d["obs1"] = d.state_2018.notna().astype(float)
    ipw1, _ = response_ipw(d, "obs1", core, weight_col)
    at2 = d[d.state_2018.isin([0, 1])].copy()
    at2["obs2"] = at2.state_2020.notna().astype(float)
    ipw2, _ = response_ipw(at2, "obs2", core + ["state_2018"], weight_col)
    rows = []
    for i, r in d.iterrows():
        if pd.notna(r.state_2018):
            row = r.to_dict(); row.update({"start_state": 0, "next_state": int(r.state_2018), "interval2": 0, "transition_weight": r[weight_col] * ipw1.loc[i]})
            rows.append(row)
        if r.state_2018 in [0, 1] and pd.notna(r.state_2020):
            row = r.to_dict(); row.update({"start_state": int(r.state_2018), "next_state": int(r.state_2020), "interval2": 1, "transition_weight": r[weight_col] * ipw2.loc[i]})
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    np.random.seed(SEED)
    cohort = build_cohort()
    # Outcomes and survey-design variables are locked to observed values.  In
    # particular, missing survey weights must never be generated by MICE.
    original = cohort[["durable_maintain", "maintain_adl_2018", "maintain_2020", "state_2018", "state_2020", "observed_durable",
                       "weight2015", "weight2011", "weight_individual2015",
                       "stratum_prov_urban", "stratum_province", "psu_code"]].copy()
    mi_cols = [
        "durable_maintain", "maintain_adl_2018", "maintain_2020", "state_2018", "state_2020", "observed_durable",
        "r1agey", "female", "education", "rural", "partnered", "smoking", "drinking", "comorbidity", "self_health", "met3",
        "r1cesd10", "r3cesd10", "r1tr20", "r3tr20", "r3gripsum", "social_reserve", "log_crp", "log_cysc",
        "weight2015", "weight2011", "weight_individual2015", "stratum_prov_urban", "stratum_province", "psu_code"
    ]
    mi_frame = cohort[mi_cols].astype(float)
    mice = MICEData(mi_frame, perturbation_method="gaussian", k_pmm=20)
    mice.update_all(10)

    core = ["age_z", "female", "education_z", "rural", "partnered", "smoking", "drinking", "comorbidity_z", "self_health_z", "met3_z"]
    resources = ["persistent_psychological_health", "cognitive_reserve", "grip_reserve", "social_reserve", "inflammatory_reserve", "renal_reserve"]
    predictors = core + resources
    schemes = [
        ("MI_2015生物标志物权重_省城乡分层_PSU", "weight2015", "stratum_prov_urban"),
        ("MI_2015生物标志物权重_仅PSU", "weight2015", "stratum_one"),
        ("MI_2011生物标志物权重_省城乡分层_PSU", "weight2011", "stratum_prov_urban"),
    ]
    fits = {s[0]: [] for s in schemes}; ms_fits = []; transition_tables = []
    ipw_diagnostics = []
    completed_sets = []

    for m in range(M):
        mice.update_all(5)
        d = prepare_completed(mice.data.copy())
        for col in original.columns:
            d[col] = original[col].to_numpy()
        completed_sets.append(d)
        for scheme, weight_col, stratum_col in schemes:
            ipw, info = response_ipw(d, "observed_durable", core, weight_col)
            d[f"analysis_weight_{scheme}"] = d[weight_col] * ipw
            ipw_diagnostics.append({"插补": m+1, "方案": scheme, **info})
            fit = logistic_fit(d, "durable_maintain", predictors, f"analysis_weight_{scheme}", stratum_col, "psu_code")
            fits[scheme].append(fit)

        trans = make_transition_rows(d, "weight2015")
        indep = trans[trans.start_state == 0].copy()
        # Mortality is sparse, so the multistate model is deliberately parsimonious.
        ms_predictors = ["interval2", "age_z", "female", "met3_z",
                         "persistent_psychological_health", "cognitive_reserve"]
        mf = multinomial_fit(indep, "next_state", ms_predictors, "transition_weight", "stratum_prov_urban", "psu_code")
        if mf is not None: ms_fits.append(mf)
        for (interval, start), g in trans.groupby(["interval2", "start_state"]):
            w = g.transition_weight / g.transition_weight.mean()
            total = w.sum()
            for state in [0, 1, 2]:
                transition_tables.append({"插补": m+1, "区间": "2015-2018" if interval == 0 else "2018-2020", "起始状态": int(start),
                                          "终止状态": state, "未加权N": int((g.next_state == state).sum()),
                                          "设计加权比例": float(w[g.next_state == state].sum() / total)})

    pooled_rows = []
    for scheme, flist in fits.items():
        names = flist[0]["names"]
        for j, name in enumerate(names):
            qbar, se, dfv, pv, lo, hi, ubar, b = pool_scalar([f["beta"][j] for f in flist], [f["cov"][j, j] for f in flist], [f["design_df"] for f in flist])
            pooled_rows.append({"模型": scheme, "变量": name, "插补数": len(flist), "平均N": int(np.mean([f["N"] for f in flist])),
                                "OR": math.exp(qbar), "95%CI下限": math.exp(lo), "95%CI上限": math.exp(hi), "P值": pv,
                                "自由度": dfv, "插补内方差": ubar, "插补间方差": b,
                                "PSU数": int(np.median([f["clusters"] for f in flist])), "孤立分层数": int(np.median([f["lonely_strata"] for f in flist]))})
    pooled_df = pd.DataFrame(pooled_rows)

    ms_rows = []
    if ms_fits:
        pnames = ms_fits[0]["names"]; pcols = len(pnames)
        for state_idx, state_name in [(0, "转为ADL残疾_vs保持独立"), (1, "死亡_vs保持独立")]:
            for j, name in enumerate(pnames):
                k = state_idx * pcols + j
                qbar, se, dfv, pv, lo, hi, ubar, b = pool_scalar([f["beta"][k] for f in ms_fits], [f["cov"][k, k] for f in ms_fits], [f["design_df"] for f in ms_fits])
                ms_rows.append({"转移": state_name, "变量": name, "插补数": len(ms_fits), "平均人时记录": int(np.mean([f["N"] for f in ms_fits])),
                                "RRR": math.exp(qbar), "95%CI下限": math.exp(lo), "95%CI上限": math.exp(hi), "P值": pv, "自由度": dfv,
                                "PSU数": int(np.median([f["clusters"] for f in ms_fits])), "孤立分层数": int(np.median([f["lonely_strata"] for f in ms_fits]))})
    ms_df = pd.DataFrame(ms_rows)

    trans_df = pd.DataFrame(transition_tables)
    trans_summary = trans_df.groupby(["区间", "起始状态", "终止状态"], as_index=False).agg(
        未加权N=("未加权N", "median"), 设计加权比例=("设计加权比例", "mean"), 插补间标准差=("设计加权比例", "std"))

    convergence_rows = []
    for scheme, flist in fits.items():
        for imp, fit in enumerate(flist, 1):
            convergence_rows.append({"模型": scheme, "插补": imp, "收敛": bool(fit["converged"]),
                                     "N": fit["N"], "PSU数": fit["clusters"]})
    for imp, fit in enumerate(ms_fits, 1):
        convergence_rows.append({"模型": "多状态多项Logit", "插补": imp, "收敛": bool(fit["converged"]),
                                 "N": fit["N"], "PSU数": fit["clusters"]})
    convergence_df = pd.DataFrame(convergence_rows)

    weight_diag = []
    for name, col in [("2015生物标志物", "weight2015"), ("2011生物标志物", "weight2011"), ("2015个体", "weight_individual2015")]:
        x = cohort[col].dropna().astype(float)
        weight_diag.append({"权重": name, "非缺失N": len(x), "最小值": x.min(), "P1": x.quantile(.01), "中位数": x.median(), "P99": x.quantile(.99), "最大值": x.max(),
                            "变异系数": x.std()/x.mean(), "Kish有效样本量": x.sum()**2/(x.pow(2).sum())})
    weight_df = pd.DataFrame(weight_diag)

    pooled_df.to_csv(EXPORT / "mi_complex_survey_models.csv", index=False, encoding="utf-8-sig")
    ms_df.to_csv(EXPORT / "multistate_models.csv", index=False, encoding="utf-8-sig")
    trans_summary.to_csv(EXPORT / "multistate_transition_probabilities.csv", index=False, encoding="utf-8-sig")
    weight_df.to_csv(EXPORT / "survey_weight_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ipw_diagnostics).to_csv(EXPORT / "mi_ipw_diagnostics.csv", index=False, encoding="utf-8-sig")
    convergence_df.to_csv(EXPORT / "model_convergence_diagnostics.csv", index=False, encoding="utf-8-sig")

    key = pooled_df[(pooled_df.模型 == schemes[0][0]) & pooled_df.变量.isin(["persistent_psychological_health", "cognitive_reserve"])]
    key_ms = ms_df[ms_df.变量.isin(["persistent_psychological_health", "cognitive_reserve"])] if len(ms_df) else ms_df
    summary = {
        "seed": SEED, "imputations": M, "cohort_n": len(cohort), "durable_outcome_observed": int(cohort.durable_maintain.notna().sum()),
        "official_fine_strata_available": False,
        "operational_design": "2015 biomarker weight + community PSU + province-by-urban/rural reconstructed strata; lonely strata use average adjustment",
        "convergence": convergence_df.groupby("模型")["收敛"].agg(["sum", "count"]).reset_index().to_dict(orient="records"),
        "transition_counts": trans_summary.to_dict(orient="records"),
        "key_mi_results": key.to_dict(orient="records"), "key_multistate_results": key_ms.to_dict(orient="records"),
        "no_participant_level_exports": True
    }
    (EXPORT / "advanced_inference_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

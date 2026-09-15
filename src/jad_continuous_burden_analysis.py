"""JAD-focused analyses with continuous depressive symptom burden as the principal exposure."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
HERE = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", ROOT / "results")) / "transition_upgrade"
HERE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.imputation.mice import MICEData

import three_cohort_transition_analysis as base
import fully_adjusted_transition_analysis as enh

SEED = 20260915
M = 20
DISEASE_CHARLS = ["cancre", "lunge", "hearte", "stroke", "psyche", "arthre", "livere", "kidneye", "asthmae"]
DISEASE_HRS = ["cancre", "lunge", "hearte", "stroke", "psyche", "arthre"]
DISEASE_ELSA = ["cancre", "lunge", "hearte", "stroke", "psyche", "arthre", "asthmae"]


def binary_sum(frame: pd.DataFrame) -> pd.Series:
    vals = pd.concat([base.binary(frame[c]) for c in frame], axis=1)
    return vals.sum(axis=1, min_count=vals.shape[1])


def add_first_wave_health(d: pd.DataFrame, cohort: str) -> pd.DataFrame:
    x = d.copy()
    if cohort == "CHARLS":
        cols = ["ID", "r1shlt"] + [f"r1{s}" for s in DISEASE_CHARLS]
        h = pd.read_stata(base.bridge.charls.fa.H_PATH, columns=cols, convert_categoricals=False)
        h["person_id"] = base.bridge.charls.norm_id(h["ID"])
        h["self_health_first"] = base.clean(h["r1shlt"], 1, 5)
        h["comorbidity_first"] = binary_sum(h[[f"r1{s}" for s in DISEASE_CHARLS]])
        x = x.merge(h[["person_id", "self_health_first", "comorbidity_first"]],
                    on="person_id", how="left", validate="one_to_one")
    elif cohort == "HRS":
        x["self_health_first"] = base.clean(x["r10shlt"], 1, 5)
        x["comorbidity_first"] = binary_sum(x[[f"r10{s}" for s in DISEASE_HRS]])
    else:
        cols = ["idauniq", "r4shlt"] + [f"r4{s}" for s in DISEASE_ELSA]
        h = pd.read_stata(base.ELSA / "gh_elsa_h.dta", columns=cols, convert_categoricals=False)
        h["self_health_first"] = base.clean(h["r4shlt"], 1, 5)
        h["comorbidity_first"] = binary_sum(h[[f"r4{s}" for s in DISEASE_ELSA]])
        x = x.merge(h[["idauniq", "self_health_first", "comorbidity_first"]],
                    on="idauniq", how="left", validate="one_to_one")
    return x


def load_cohorts() -> tuple[dict[str, pd.DataFrame], dict[str, list[int]]]:
    cohorts = {
        "CHARLS": enh.add_covariates(base.common_charls(), "CHARLS"),
        "HRS": enh.add_covariates(base.common_hrs(), "HRS"),
        "ELSA": enh.add_covariates(base.elsa_cohort(), "ELSA"),
    }
    for name in cohorts:
        cohorts[name] = add_first_wave_health(cohorts[name], name)
        cohorts[name]["cohort"] = name
    years = {"CHARLS":[2015,2018,2020], "HRS":[2014,2016,2018,2020,2022], "ELSA":[6,7,8,9]}
    return cohorts, years


RAW_IMPUTE = ["age", "female", "education", "rural", "partnered", "smoking", "drinking",
              "comorbidity", "self_health", "comorbidity_first", "self_health_first",
              "met_anchor", "memory_first", "memory_anchor", "cesd_first", "cesd_anchor"]


def imputed_sets(d: pd.DataFrame):
    np.random.seed(SEED)
    imp = MICEData(d[RAW_IMPUTE].astype(float), perturbation_method="gaussian", k_pmm=20)
    for _ in range(10):
        imp.update_all()
    for _ in range(M):
        imp.update_all()
        z = d.copy()
        z[RAW_IMPUTE] = imp.data[RAW_IMPUTE].to_numpy()
        first_z = base.zscore(z["cesd_first"])
        anchor_z = base.zscore(z["cesd_anchor"])
        z["depression_burden_z"] = base.zscore((first_z + anchor_z) / 2)
        threshold = 10 if z["cohort"].iloc[0] == "CHARLS" else 4
        z["persistent_low_depression"] = ((z.cesd_first < threshold) & (z.cesd_anchor < threshold)).astype(float)
        z["memory_z"] = base.zscore(z["memory_anchor"])
        for v in ["age", "education", "comorbidity", "self_health", "comorbidity_first", "self_health_first", "met_anchor"]:
            z[v + "_z"] = base.zscore(z[v])
        yield z


BASE_COLS = ["person_id", "blood_weight", "design_stratum", "design_psu", "depression_burden_z",
             "persistent_low_depression", "memory_z", "age_z", "female", "education_z", "rural",
             "partnered", "smoking", "drinking", "comorbidity_z", "self_health_z",
             "comorbidity_first_z", "self_health_first_z", "met_anchor_z"]


def person_period(d: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    rows = []
    for interval, (a, b) in enumerate(zip(years[:-1], years[1:]), 1):
        q = d[BASE_COLS + [f"state_{a}", f"state_{b}"]].copy()
        q = q.rename(columns={f"state_{a}":"origin", f"state_{b}":"destination"})
        q["interval"] = interval
        rows.append(q)
    return pd.concat(rows, ignore_index=True)


DEMOGRAPHIC = ["age_z", "female", "education_z", "rural", "partnered"]
BEHAVIOR = ["smoking", "drinking"]
HEALTH_ANCHOR = ["comorbidity_z", "self_health_z"]
HEALTH_FIRST = ["comorbidity_first_z", "self_health_first_z"]
METABOLIC = ["met_anchor_z"]


def add_intervals(q: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    z = q.copy(); terms=[]
    observed_intervals = sorted(int(k) for k in z.interval.dropna().unique())
    for k in observed_intervals[1:]:
        nm=f"interval_{k}"; z[nm]=z.interval.eq(k).astype(float); terms.append(nm)
    return z, terms


def add_response_weight(q: pd.DataFrame, health_covars: list[str]) -> pd.DataFrame:
    z = q.copy()
    z["observed"] = z.destination.notna().astype(float)
    predictors = ["depression_burden_z", "memory_z"] + DEMOGRAPHIC + BEHAVIOR + health_covars + METABOLIC
    ipw, _ = base.bridge.charls.response_ipw(z, "observed", predictors, "blood_weight")
    z["response_weight"] = ipw
    z["analysis_weight"] = z.blood_weight * ipw
    return z


def fit(q: pd.DataFrame, exposure: str, covars: list[str], weighted: bool = True,
        quartile: int | None = None, response_health: list[str] = HEALTH_FIRST) -> dict | None:
    z = q[q.origin.eq(0)].copy()
    z = add_response_weight(z, response_health)
    z = z[z.destination.isin([0,1])].copy()
    z["event"] = z.destination.eq(1).astype(float)
    z, intervals = add_intervals(z)
    if quartile is not None:
        cuts = np.quantile(z["depression_burden_z"], [.25,.50,.75])
        qq = np.digitize(z["depression_burden_z"], cuts) + 1
        for k in [2,3,4]: z[f"burden_q{k}"] = (qq == k).astype(float)
        focal=f"burden_q{quartile}"; exposure_terms=["burden_q2","burden_q3","burden_q4"]
    else:
        focal=exposure; exposure_terms=[exposure]
    secondary_marker = ["depression_burden_z"] if exposure == "memory_z" else ["memory_z"]
    predictors = exposure_terms + secondary_marker + covars + intervals
    z["blood_weight"] = z["analysis_weight"] if weighted else z["blood_weight"]
    ans=base.survey_logistic(z,"event",predictors)
    j=ans["names"].index(focal)
    return {"b":ans["beta"][j],"v":ans["cov"][j,j],"df":ans["design_df"],
            "records":ans["n"],"events":ans["events"]}


def pool(fits: list[dict], **meta) -> dict:
    q=np.array([f["b"] for f in fits]); u=np.array([f["v"] for f in fits]); m=len(q)
    qb=q.mean(); ub=u.mean(); b=q.var(ddof=1) if m>1 else 0.; total=ub+(1+1/m)*b
    rdf=(m-1)*(1+ub/((1+1/m)*b))**2 if b>1e-14 else np.inf
    df=min(np.nanmedian([f["df"] for f in fits]),rdf) if np.isfinite(rdf) else np.nanmedian([f["df"] for f in fits])
    se=math.sqrt(max(total,0)); crit=stats.t.ppf(.975,df)
    return {**meta,"records":int(np.median([f["records"] for f in fits])),"events":int(np.median([f["events"] for f in fits])),
            "estimate":math.exp(qb),"ci_low":math.exp(qb-crit*se),"ci_high":math.exp(qb+crit*se),
            "p_value":2*stats.t.sf(abs(qb/se),df),"imputations":m}


def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline basis following Harrell's truncated-power form."""
    x=np.asarray(x,float); k=np.asarray(knots,float); scale=(k[-1]-k[0])**2
    cols=[x]
    for j in range(len(k)-2):
        term=np.maximum(x-k[j],0)**3
        term-=np.maximum(x-k[-2],0)**3*(k[-1]-k[j])/(k[-1]-k[-2])
        term+=np.maximum(x-k[-1],0)**3*(k[-2]-k[j])/(k[-1]-k[-2])
        cols.append(term/scale)
    return np.column_stack(cols)


def fit_spline(q: pd.DataFrame, knots: np.ndarray, grid: np.ndarray) -> tuple[list[dict], dict]:
    z=q[q.origin.eq(0)].copy(); z=add_response_weight(z,HEALTH_FIRST)
    z=z[z.destination.isin([0,1])].copy(); z["event"]=z.destination.eq(1).astype(float)
    z,intervals=add_intervals(z)
    basis=rcs_basis(z.depression_burden_z.to_numpy(),knots)
    spline_names=[f"burden_rcs_{j}" for j in range(basis.shape[1])]
    for j,nm in enumerate(spline_names): z[nm]=basis[:,j]
    predictors=spline_names+["memory_z"]+DEMOGRAPHIC+BEHAVIOR+HEALTH_FIRST+METABOLIC+intervals
    z["blood_weight"]=z.analysis_weight
    ans=base.survey_logistic(z,"event",predictors)
    idx=[ans["names"].index(nm) for nm in spline_names]
    beta=ans["beta"][idx]; cov=ans["cov"][np.ix_(idx,idx)]
    ref=rcs_basis(np.array([0.0]),knots)[0]
    out=[]
    for x,row in zip(grid,rcs_basis(grid,knots)):
        contrast=row-ref; b=float(contrast@beta); v=float(contrast@cov@contrast)
        out.append({"grid":float(x),"b":b,"v":v,"df":ans["design_df"],
                    "records":ans["n"],"events":ans["events"]})
    nonlinear_idx = idx[1:]
    nonlinear = {
        "q": ans["beta"][nonlinear_idx],
        "u": ans["cov"][np.ix_(nonlinear_idx, nonlinear_idx)],
        "df": ans["design_df"],
        "records": ans["n"],
        "events": ans["events"],
    }
    return out, nonlinear


def pool_multivariate_wald(fits: list[dict], **meta) -> dict:
    """Rubin-pool a multivariate coefficient vector and test that it equals zero."""
    q = np.stack([f["q"] for f in fits])
    ubar = np.mean(np.stack([f["u"] for f in fits]), axis=0)
    m = len(fits)
    between = np.cov(q, rowvar=False, ddof=1) if m > 1 else np.zeros_like(ubar)
    total = ubar + (1 + 1 / m) * between
    qbar = q.mean(axis=0)
    wald = float(qbar @ np.linalg.pinv(total) @ qbar)
    test_df = int(qbar.size)
    return {
        **meta,
        "records": int(np.median([f["records"] for f in fits])),
        "events": int(np.median([f["events"] for f in fits])),
        "wald_chi2": wald,
        "test_df": test_df,
        "p_nonlinearity": float(stats.chi2.sf(wald, test_df)),
        "imputations": m,
    }


def main():
    cohorts, years = load_cohorts()
    primary=[]; timing=[]; quartiles=[]; attenuation=[]; splines=[]; spline_tests=[]; interval_sensitivity=[]; missing=[]
    for cohort,d in cohorts.items():
        missing.append({"cohort":cohort,"n":len(d),**{v:int(d[v].notna().sum()) for v in
                       ["comorbidity_first","self_health_first","comorbidity","self_health"]}})
        stores={
            "continuous_anchor":[], "continuous_first":[], "continuous_nohealth":[],
            "binary_first":[], "memory_firsthealth":[],
        }
        qstore={k:[] for k in [2,3,4]}
        stages={
            "marker_interval":[],
            "demographic":DEMOGRAPHIC,
            "behavior":DEMOGRAPHIC+BEHAVIOR,
            "health":DEMOGRAPHIC+BEHAVIOR+HEALTH_FIRST,
            "full":DEMOGRAPHIC+BEHAVIOR+HEALTH_FIRST+METABOLIC,
        }
        att={(ex,stage,w):[] for ex in ["depression_burden_z","persistent_low_depression"]
             for stage in stages for w in [False,True]}
        observed_burden=(base.zscore(d.cesd_first)+base.zscore(d.cesd_anchor))/2
        observed=base.zscore(observed_burden).dropna().to_numpy()
        knots=np.quantile(observed,[.05,.35,.65,.95])
        grid=np.linspace(np.quantile(observed,.05),np.quantile(observed,.95),41)
        spline_store={float(x):[] for x in grid}
        spline_nonlinear=[]
        interval_store={f"interval_{k}":[] for k in range(1,len(years[cohort]))}
        interval_store["exclude_first_interval"]=[]
        for z in imputed_sets(d):
            pp=person_period(z,years[cohort])
            specs={
                "continuous_anchor":("depression_burden_z",DEMOGRAPHIC+BEHAVIOR+HEALTH_ANCHOR+METABOLIC),
                "continuous_first":("depression_burden_z",DEMOGRAPHIC+BEHAVIOR+HEALTH_FIRST+METABOLIC),
                "continuous_nohealth":("depression_burden_z",DEMOGRAPHIC+BEHAVIOR+METABOLIC),
                "binary_first":("persistent_low_depression",DEMOGRAPHIC+BEHAVIOR+HEALTH_FIRST+METABOLIC),
                "memory_firsthealth":("memory_z",DEMOGRAPHIC+BEHAVIOR+HEALTH_FIRST+METABOLIC),
            }
            for name,(ex,covs) in specs.items():
                response_health = HEALTH_ANCHOR if name == "continuous_anchor" else HEALTH_FIRST
                f=fit(pp,ex,covs,True,response_health=response_health)
                if f: stores[name].append(f)
            for k in qstore:
                f=fit(pp,"depression_burden_z",DEMOGRAPHIC+BEHAVIOR+HEALTH_FIRST+METABOLIC,True,k)
                if f: qstore[k].append(f)
            for (ex,stage,w) in att:
                f=fit(pp,ex,stages[stage],w)
                if f: att[(ex,stage,w)].append(f)
            spline_rows, nonlinear_fit = fit_spline(pp,knots,grid)
            for row in spline_rows: spline_store[row["grid"]].append(row)
            spline_nonlinear.append(nonlinear_fit)
            for k in range(1,len(years[cohort])):
                f=fit(pp[pp.interval.eq(k)],"depression_burden_z",DEMOGRAPHIC+BEHAVIOR+HEALTH_FIRST+METABOLIC,True)
                if f: interval_store[f"interval_{k}"].append(f)
            f=fit(pp[pp.interval.gt(1)],"depression_burden_z",DEMOGRAPHIC+BEHAVIOR+HEALTH_FIRST+METABOLIC,True)
            if f: interval_store["exclude_first_interval"].append(f)
        for name,vals in stores.items():
            exposure="memory_z" if name.startswith("memory") else ("persistent_low_depression" if name.startswith("binary") else "depression_burden_z")
            row=pool(vals,cohort=cohort,analysis=name,exposure=exposure,measure="OR")
            (timing if name.startswith("continuous_") else primary).append(row)
        primary.append(pool(stores["continuous_first"],cohort=cohort,analysis="principal_continuous_first_health",exposure="depression_burden_z",measure="OR"))
        for k,vals in qstore.items(): quartiles.append(pool(vals,cohort=cohort,quartile=k,reference=1,measure="OR"))
        for (ex,stage,w),vals in att.items():
            attenuation.append(pool(vals,cohort=cohort,exposure=ex,stage=stage,response_weighted=w,measure="OR"))
        for x,vals in spline_store.items():
            splines.append(pool(vals,cohort=cohort,burden_z=x,reference_z=0,measure="OR"))
        spline_tests.append(pool_multivariate_wald(spline_nonlinear,cohort=cohort,test="nonlinear_spline_terms"))
        for analysis,vals in interval_store.items():
            if vals:
                interval_sensitivity.append(pool(vals,cohort=cohort,analysis=analysis,
                                                 exposure="depression_burden_z",measure="OR"))
    pd.DataFrame(primary).to_csv(HERE/"jad_primary_models.csv",index=False)
    pd.DataFrame(timing).to_csv(HERE/"jad_covariate_timing.csv",index=False)
    pd.DataFrame(quartiles).to_csv(HERE/"jad_burden_quartiles.csv",index=False)
    pd.DataFrame(attenuation).to_csv(HERE/"jad_attenuation_matrix.csv",index=False)
    pd.DataFrame(splines).to_csv(HERE/"jad_spline_curves.csv",index=False)
    pd.DataFrame(spline_tests).to_csv(HERE/"jad_spline_nonlinearity.csv",index=False)
    pd.DataFrame(interval_sensitivity).to_csv(HERE/"jad_interval_sensitivity.csv",index=False)
    pd.DataFrame(missing).to_csv(HERE/"jad_first_wave_covariate_audit.csv",index=False)


if __name__ == "__main__":
    main()

"""Fully adjusted transition analyses for the three-cohort manuscript.

No row-level data are exported. Outputs contain only aggregate estimates/counts.
"""
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
import elsa_external_validation as elsa_early

SEED = 20260914
M = 20
FULL = ["persistent_low_depression", "memory_z", "age_z", "female", "education_z",
        "rural", "partnered", "smoking", "drinking", "comorbidity_z",
        "self_health_z", "met_anchor_z"]


def add_covariates(d: pd.DataFrame, cohort: str) -> pd.DataFrame:
    x = d.copy()
    if cohort == "ELSA":
        # The principal ELSA loader is augmented from harmonized Wave 6 measures.
        hcols = ["idauniq", "r6cancre", "r6lunge", "r6hearte", "r6stroke", "r6psyche",
                 "r6arthre", "r6asthmae"]
        h = pd.read_stata(base.ELSA / "gh_elsa_h.dta", columns=hcols,
                          convert_categoricals=False)
        rural = pd.read_stata(base.ELSA / "elsa_geog_urindewr_2001_eul.dta",
                              columns=["idauniq", "w4_urindewr_2001"],
                              convert_categoricals=False)
        x = x.merge(h, on="idauniq", how="left", validate="one_to_one")
        x = x.merge(rural, on="idauniq", how="left", validate="one_to_one")
        x["education"] = pd.to_numeric(x["raeducl"], errors="coerce").where(lambda s: s.isin([1,2,3]))
        m = pd.to_numeric(x["r6mstat"], errors="coerce")
        x["partnered"] = np.where(m.isin([1,3]), 1., np.where(m.isin([4,5,7,8]), 0., np.nan))
        x["smoking"] = base.binary(x["r6smoken"])
        x["drinking"] = base.binary(x["r6drink"])
        x["self_health"] = base.clean(x["r6shlt"], 1, 5)
        rr = pd.to_numeric(x["w4_urindewr_2001"], errors="coerce")
        x["rural"] = np.where(rr.eq(2), 1., np.where(rr.eq(1), 0., np.nan))
        diseases = pd.concat([base.binary(x[c]) for c in
                              ["r6cancre","r6lunge","r6hearte","r6stroke","r6psyche","r6arthre","r6asthmae"]], axis=1)
        x["comorbidity"] = diseases.sum(axis=1, min_count=7)
        x["person_id"] = x["idauniq"].astype(str)
        x["state_6"] = 0.0
    # CHARLS and HRS loaders already carry these harmonized fields.
    for v in ["education", "rural", "partnered", "smoking", "drinking", "comorbidity", "self_health"]:
        if v not in x:
            raise KeyError(f"{cohort}: missing covariate {v}")
        x[v] = pd.to_numeric(x[v], errors="coerce")
    for v in ["age", "education", "comorbidity", "self_health", "met_anchor"]:
        x[v + "_z"] = base.zscore(x[v])
    x["cohort"] = cohort
    return x


def add_strict_states(d: pd.DataFrame, cohort: str) -> pd.DataFrame:
    x = d.copy()
    if cohort == "CHARLS":
        root = Path(os.environ["CHARLS_DATA_DIR"])
        specs = {
            2018: (root / "2018" / "CHARLS2018r" / "Health_Status_and_Functioning.dta",
                   ["db010","db011","db012","db013","db014","db015"]),
            2020: (root / "2020" / "CHARLS2020r" / "Health_Status_and_Functioning.dta",
                   ["db001","db003","db005","db007","db009","db011"]),
        }
        for year, (path, items) in specs.items():
            z = pd.read_stata(path, columns=["ID"] + items, convert_categoricals=False)
            z["person_id"] = base.bridge.charls.norm_id(z["ID"])
            vals = z[items].apply(pd.to_numeric, errors="coerce")
            z["strict"] = np.where(vals.notna().all(axis=1), (vals.gt(1).sum(axis=1) >= 2).astype(float), np.nan)
            x = x.merge(z[["person_id","strict"]].rename(columns={"strict":f"strict_{year}"}),
                        on="person_id", how="left", validate="one_to_one")
            death = x[f"state_{year}"].eq(2)
            x[f"strict_state_{year}"] = np.where(death, 2., x[f"strict_{year}"])
        x["strict_state_2015"] = 0.
    elif cohort == "HRS":
        for year, wave in zip([2016,2018,2020,2022], [13,14,15,16]):
            cnt = pd.to_numeric(x[f"r{wave}adl6a"], errors="coerce").where(lambda s: s.between(0,6))
            x[f"strict_state_{year}"] = np.where(x[f"state_{year}"].eq(2), 2.,
                                                   np.where(cnt.notna(), (cnt >= 2).astype(float), np.nan))
        x["strict_state_2014"] = 0.
    else:
        for year, wave in zip([2014,2016,2018], [7,8,9]):
            cnt = pd.to_numeric(x[f"r{wave}adltot6"], errors="coerce").where(lambda s: s.between(0,6))
            x[f"strict_state_{wave}"] = np.where(cnt.notna(), (cnt >= 2).astype(float), np.nan)
        x["strict_state_6"] = 0.
    return x


def pp_make(d: pd.DataFrame, years: list[int], state_prefix: str = "state_") -> pd.DataFrame:
    rows = []
    covars = ["person_id","blood_weight","design_stratum","design_psu"] + FULL
    if "lower_depression_burden_z" in d:
        covars.append("lower_depression_burden_z")
    covars = list(dict.fromkeys(covars))
    for k, (a,b) in enumerate(zip(years[:-1], years[1:]), 1):
        q = d[covars + [f"{state_prefix}{a}", f"{state_prefix}{b}"]].copy()
        q = q.rename(columns={f"{state_prefix}{a}":"origin", f"{state_prefix}{b}":"destination"})
        q["interval"] = k
        q["from_year"], q["to_year"] = a, b
        rows.append(q)
    return pd.concat(rows, ignore_index=True)


def imputed_sets(d: pd.DataFrame):
    cols = ["age","female","education","rural","partnered","smoking","drinking",
            "comorbidity","self_health","met_anchor","memory_first","memory_anchor",
            "cesd_first","cesd_anchor"]
    imp = MICEData(d[cols].astype(float), perturbation_method="gaussian", k_pmm=20)
    np.random.seed(SEED)
    for _ in range(10): imp.update_all()
    for _ in range(M):
        imp.update_all()
        z = d.copy()
        z[cols] = imp.data[cols].to_numpy()
        z["persistent_low_depression"] = ((z["cesd_first"] < (10 if z["cohort"].iloc[0]=="CHARLS" else 4)) &
                                             (z["cesd_anchor"] < (10 if z["cohort"].iloc[0]=="CHARLS" else 4))).astype(float)
        z["memory_z"] = base.zscore(z["memory_anchor"])
        z["lower_depression_burden_z"] = -base.zscore((base.zscore(z["cesd_first"]) + base.zscore(z["cesd_anchor"])) / 2)
        for v in ["age","education","comorbidity","self_health","met_anchor"]:
            z[v+"_z"] = base.zscore(z[v])
        yield z


def response_weights(q: pd.DataFrame) -> pd.DataFrame:
    z = q.copy()
    z["observed"] = z["destination"].notna().astype(float)
    predictors = [v for v in FULL if v in z]
    try:
        ipw, _ = base.bridge.charls.response_ipw(z, "observed", predictors, "blood_weight")
        z["analysis_weight"] = z["blood_weight"] * ipw
    except Exception:
        z["analysis_weight"] = z["blood_weight"]
    return z


def fit_one(q: pd.DataFrame, transition: str, focal: str, model: str) -> dict | None:
    z = q.copy()
    if transition == "incidence":
        z = z[z.origin.eq(0)].copy(); z = response_weights(z)
        if model == "live": z = z[z.destination.isin([0,1])].copy(); z["event"] = z.destination.eq(1).astype(float)
        else: z = z[z.destination.isin([0,1,2])].copy(); z["event"] = z.destination.ne(0).astype(float)
    else:
        z = z[z.origin.eq(1)].copy(); z = response_weights(z)
        z = z[z.destination.isin([0,1])].copy(); z["event"] = z.destination.eq(0).astype(float)
    if len(z) < 30 or z.event.sum() < 8 or z.event.nunique() < 2: return None
    for k in range(2, int(z.interval.max())+1): z[f"interval_{k}"] = z.interval.eq(k).astype(float)
    terms = FULL.copy()
    if focal == "lower_depression_burden_z":
        terms = ["lower_depression_burden_z" if v == "persistent_low_depression" else v for v in terms]
    terms += [c for c in z if c.startswith("interval_")]
    z["blood_weight"] = z["analysis_weight"]
    fit = base.survey_logistic(z, "event", terms)
    j = fit["names"].index(focal)
    return {"b":fit["beta"][j], "v":fit["cov"][j,j], "df":fit["design_df"],
            "records":fit["n"], "events":fit["events"]}


def pool(fits: list[dict], **meta) -> dict:
    q=np.array([f["b"] for f in fits]); u=np.array([f["v"] for f in fits]); m=len(q)
    qb=q.mean(); ub=u.mean(); b=q.var(ddof=1); total=ub+(1+1/m)*b
    rdf=(m-1)*(1+ub/((1+1/m)*b))**2 if b>1e-14 else np.inf
    df=min(np.median([f["df"] for f in fits]), rdf) if np.isfinite(rdf) else np.median([f["df"] for f in fits])
    se=math.sqrt(max(total,0)); crit=stats.t.ppf(.975,df)
    return {**meta,"records":int(np.median([f["records"] for f in fits])),
            "events":int(np.median([f["events"] for f in fits])),"estimate":math.exp(qb),
            "ci_low":math.exp(qb-crit*se),"ci_high":math.exp(qb+crit*se),
            "p_value":2*stats.t.sf(abs(qb/se),df),"imputations":m}


def early_elsa() -> pd.DataFrame:
    d=elsa_early.derive_analysis_variables(elsa_early.read_source_data())
    d["score4_2"]=d[[f"{c}_w2_component" for c in ["waist","bp","glycemia","hdl"]]].sum(axis=1,min_count=4)
    d["score4_4"]=d[[f"{c}_w4_component" for c in ["waist","bp","glycemia","hdl"]]].sum(axis=1,min_count=4)
    elig=(d.score4_2.ge(3)&d.score4_4.ge(3)&d.baseline_independent.eq(1)&d.blood_weight_w4.gt(0)&d.idahhw4.notna()&d.gor.notna())
    d=d.loc[elig].copy().reset_index(drop=True)
    d=d.rename(columns={"idauniq":"person_id","blood_weight_w4":"blood_weight","met_w4":"met_anchor",
                        "memory_w2":"memory_first","memory_w4":"memory_anchor","cesd_w2":"cesd_first","cesd_w4":"cesd_anchor"})
    d["met_anchor"]=d["score4_4"]; d["state_4"]=0.; d["state_6"]=d["w6_state"]
    d["design_stratum"]=pd.factorize(d.gor.astype("string"),sort=True)[0]
    d["design_psu"]=pd.factorize(pd.to_numeric(d.idahhw4,errors="coerce"),sort=True)[0]
    d["cohort"]="ELSA_early"
    for v in ["age","education","comorbidity","self_health","met_anchor"]: d[v+"_z"]=base.zscore(d[v])
    d["memory_z"]=base.zscore(d.memory_anchor)
    return d


def retention(pp: pd.DataFrame, cohort: str) -> pd.DataFrame:
    rows=[]
    for (k,a,b),q in pp.groupby(["interval","from_year","to_year"]):
        r=q[q.origin.isin([0,1])]
        rows.append({"cohort":cohort,"interval":k,"from_year":a,"to_year":b,"risk_records":len(r),
                     "observed_state":int(r.destination.notna().sum()),"independent":int(r.destination.eq(0).sum()),
                     "disabled":int(r.destination.eq(1).sum()),"death":int(r.destination.eq(2).sum()),
                     "missing":int(r.destination.isna().sum()),"retention_percent":100*r.destination.notna().mean()})
    return pd.DataFrame(rows)


def main():
    cohorts={"CHARLS":add_covariates(base.common_charls(),"CHARLS"),
             "HRS":add_covariates(base.common_hrs(),"HRS"),
             "ELSA":add_covariates(base.elsa_cohort(),"ELSA")}
    years={"CHARLS":[2015,2018,2020],"HRS":[2014,2016,2018,2020,2022],"ELSA":[6,7,8,9]}
    rows=[]; ret=[]
    for cohort,d in cohorts.items():
        d=add_strict_states(d,cohort); cohorts[cohort]=d
        rawpp=pp_make(d,years[cohort]); ret.append(retention(rawpp,cohort))
        fits={(t,m,f):[] for t,m in [("incidence","live"),("incidence","composite"),("recovery","live")] for f in ["persistent_low_depression","memory_z"]}
        fits.update({("incidence","live","lower_depression_burden_z"):[]})
        strictfits={f:[] for f in ["persistent_low_depression","memory_z"]}
        for z in imputed_sets(d):
            pp=pp_make(z,years[cohort])
            for t,m,f in fits:
                ans=fit_one(pp,t,f,m)
                if ans: fits[(t,m,f)].append(ans)
            spp=pp_make(z,years[cohort],"strict_state_")
            for f in strictfits:
                ans=fit_one(spp,"incidence",f,"live")
                if ans: strictfits[f].append(ans)
        for (t,m,f),vals in fits.items():
            if vals: rows.append(pool(vals,cohort=cohort,transition=t,model=m,threshold="any_ADL",variable=f,measure="OR"))
        for f,vals in strictfits.items():
            if vals: rows.append(pool(vals,cohort=cohort,transition="incidence",model="live",threshold="two_or_more_ADL",variable=f,measure="OR"))
    # Same estimand/model for the alternative ELSA exposure window.
    d=early_elsa(); pp0=pp_make(d,[4,6]); ret.append(retention(pp0,"ELSA_early"))
    fs={f:[] for f in ["persistent_low_depression","memory_z"]}
    for z in imputed_sets(d):
        pp=pp_make(z,[4,6])
        for f in fs:
            ans=fit_one(pp,"incidence",f,"live")
            if ans: fs[f].append(ans)
    for f,vals in fs.items():
        if vals: rows.append(pool(vals,cohort="ELSA_early",transition="incidence",model="live",threshold="any_ADL",variable=f,measure="OR"))
    pd.DataFrame(rows).to_csv(HERE/"enhanced_transition_models.csv",index=False)
    pd.concat(ret,ignore_index=True).to_csv(HERE/"interval_retention.csv",index=False)
    # Aggregate baseline table, with no row-level exports.
    br=[]
    variables=["age","female","education","rural","partnered","smoking","drinking","comorbidity","self_health","met_anchor","persistent_low_depression","memory_anchor"]
    for c,d in cohorts.items():
        for v in variables:
            s=pd.to_numeric(d[v],errors="coerce"); w=pd.to_numeric(d.blood_weight,errors="coerce")
            ok=s.notna()&w.gt(0)
            mean=np.average(s[ok],weights=w[ok]); var=np.average((s[ok]-mean)**2,weights=w[ok])
            br.append({"cohort":c,"variable":v,"n_nonmissing":int(ok.sum()),"weighted_mean":mean,"weighted_sd":math.sqrt(var)})
    pd.DataFrame(br).to_csv(HERE/"expanded_baseline_characteristics.csv",index=False)


if __name__ == "__main__": main()

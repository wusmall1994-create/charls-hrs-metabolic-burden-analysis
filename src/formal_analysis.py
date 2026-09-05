from pathlib import Path
import json
import math
import os
import sys
import warnings

PROJECT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("CHARLS_DATA_DIR", PROJECT / "data" / "charls"))
EXPORT = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", PROJECT / "results"))
EXPORT.mkdir(parents=True, exist_ok=True)

H_PATH = ROOT / "Harmonized CHARLS" / "H_CHARLS_D_Data" / "H_CHARLS_D_Data.dta"
B1_PATH = ROOT / "2011" / "Blood_20140429" / "Blood_20140429.dta"
B3_PATH = ROOT / "2015" / "Blood" / "Blood.dta"
H20_PATH = ROOT / "2020" / "CHARLS2020r" / "Health_Status_and_Functioning.dta"
E20_PATH = ROOT / "2020" / "CHARLS2020r" / "Exit_Module.dta"

COMORB = ["r3cancre", "r3lunge", "r3hearte", "r3stroke", "r3psyche", "r3arthre", "r3livere", "r3kidneye", "r3asthmae"]
H_COLS = [
    "ID", "ID_w1", "r1agey", "ragender", "raeducl", "h3rural", "r3mstat",
    "r1mwaist", "r3mwaist", "r1systo", "r3systo", "r1diasto", "r3diasto",
    "r1hibpe", "r3hibpe", "r1rxhibp_c", "r3rxhibp_c", "r1diabe", "r3diabe",
    "r1rxdiab_c", "r3rxdiab_c", "r1rxdyslip_c", "r3rxdyslip_c",
    "r3adlab_c", "r4adlab_c", "r3iadlza", "r4iadlza", "r3vgact_c", "r3mdact_c",
    "r3socwk", "r3smoken", "r3drinkn_c", "r1cesd10", "r3cesd10", "r1tr20", "r3tr20", "r3gripsum",
    "r3shlt", "radyear"
] + COMORB


def read(path, columns):
    return pd.read_stata(path, columns=columns, convert_categoricals=False)


def norm_id(x):
    out = x.astype("string").str.strip()
    return out.mask(out == "", pd.NA)


def binary(x):
    out = pd.Series(np.nan, index=x.index, dtype=float)
    out.loc[x == 0] = 0
    out.loc[x == 1] = 1
    return out


def any_yes(*args):
    f = pd.concat(args, axis=1)
    out = pd.Series(np.nan, index=f.index, dtype=float)
    out.loc[(f == 1).any(axis=1)] = 1
    out.loc[f.notna().all(axis=1) & (f == 0).all(axis=1)] = 0
    return out


def met_score(df, wave, female_waist=80, female_hdl=50):
    if wave == 1:
        waist, sbp, dbp = df.r1mwaist, df.r1systo, df.r1diasto
        hbp, bpmed = binary(df.r1hibpe), binary(df.r1rxhibp_c)
        diab, dmmed = binary(df.r1diabe), binary(df.r1rxdiab_c)
        glu, a1c, tg, hdl, fasting = df.newglu, df.newhba1c, df.newtg, df.newhdl, df.qc1_va003
    else:
        waist, sbp, dbp = df.r3mwaist, df.r3systo, df.r3diasto
        hbp, bpmed = binary(df.r3hibpe), binary(df.r3rxhibp_c)
        diab, dmmed = binary(df.r3diabe), binary(df.r3rxdiab_c)
        glu, a1c, tg, hdl, fasting = df.bl_glu, df.bl_hbalc, df.bl_tg, df.bl_hdl, df.bl_fasting
    female = df.ragender == 2
    wthr = pd.Series(np.where(female, female_waist, 90), index=df.index)
    hthr = pd.Series(np.where(female, female_hdl, 40), index=df.index)
    c = pd.DataFrame(index=df.index)
    c["waist"] = np.where(waist.notna(), (waist >= wthr).astype(float), np.nan)
    c["bp"] = np.nan
    c.loc[(sbp >= 130) | (dbp >= 85) | (hbp == 1) | (bpmed == 1), "bp"] = 1
    c.loc[sbp.notna() & dbp.notna() & (sbp < 130) & (dbp < 85) & (hbp == 0) & (bpmed == 0), "bp"] = 0
    c["glycemia"] = np.nan
    positive = (a1c >= 5.7) | ((fasting == 1) & (glu >= 100)) | (diab == 1) | (dmmed == 1)
    negative = a1c.notna() & (a1c < 5.7) & (diab == 0) & (dmmed == 0) & (((fasting == 1) & glu.notna() & (glu < 100)) | (fasting != 1))
    c.loc[positive, "glycemia"] = 1
    c.loc[negative, "glycemia"] = 0
    c["tg"] = np.where(tg.notna(), (tg >= 150).astype(float), np.nan)
    c["hdl"] = np.where(hdl.notna(), (hdl < hthr).astype(float), np.nan)
    return c.sum(axis=1, min_count=5)


def met_score_without_triglycerides(df, wave, female_waist=80, female_hdl=50):
    """Four-component score using the same observable rules as the HRS bridge."""
    if wave == 1:
        waist, sbp, dbp = df.r1mwaist, df.r1systo, df.r1diasto
        hbp, diab = binary(df.r1hibpe), binary(df.r1diabe)
        a1c, hdl = df.newhba1c, df.newhdl
    else:
        waist, sbp, dbp = df.r3mwaist, df.r3systo, df.r3diasto
        hbp, diab = binary(df.r3hibpe), binary(df.r3diabe)
        a1c, hdl = df.bl_hbalc, df.bl_hdl
    female = df.ragender == 2
    wthr = pd.Series(np.where(female, female_waist, 90), index=df.index)
    hthr = pd.Series(np.where(female, female_hdl, 40), index=df.index)
    c = pd.DataFrame(index=df.index)
    c["waist"] = np.where(waist.notna(), (waist >= wthr).astype(float), np.nan)
    c["bp"] = np.nan
    c.loc[(sbp >= 130) | (dbp >= 85) | (hbp == 1), "bp"] = 1
    c.loc[sbp.notna() & dbp.notna() & (sbp < 130) & (dbp < 85) & (hbp == 0), "bp"] = 0
    c["glycemia"] = np.nan
    positive = (a1c >= 5.7) | (diab == 1)
    negative = a1c.notna() & (a1c < 5.7) & (diab == 0)
    c.loc[positive, "glycemia"] = 1
    c.loc[negative, "glycemia"] = 0
    c["hdl"] = np.where(hdl.notna(), (hdl < hthr).astype(float), np.nan)
    return c.sum(axis=1, min_count=4)


def adl2020(raw):
    cols = ["db001", "db003", "db005", "db007", "db009", "db011"]
    x = raw[cols]
    valid = x.isin([1, 2, 3, 4]).all(axis=1)
    out = pd.Series(np.nan, index=raw.index, dtype=float)
    out.loc[valid] = (x.loc[valid] >= 2).any(axis=1).astype(float)
    return out


def zscore(x):
    x = pd.to_numeric(x, errors="coerce")
    sd = x.std()
    return (x - x.mean()) / sd if sd and np.isfinite(sd) else x * np.nan


def fit_logit_robust(data, outcome, predictors, model, focal, weight_col=None):
    cols = [outcome] + predictors + ([weight_col] if weight_col else [])
    d = data[cols].dropna().copy()
    if len(d) < 150 or d[outcome].nunique() < 2:
        return {"模型": model, "结局": outcome, "候选资源": focal, "N": len(d), "状态": "样本不足"}
    x = np.column_stack([np.ones(len(d)), d[predictors].astype(float).to_numpy()])
    y = d[outcome].astype(float).to_numpy()
    sw = d[weight_col].astype(float).to_numpy() if weight_col else np.ones(len(d))
    beta = np.zeros(x.shape[1]); converged = False
    for _ in range(300):
        p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
        w = np.clip(p * (1 - p), 1e-8, None)
        info = x.T @ (x * (sw * w)[:, None])
        step = np.linalg.pinv(info) @ (x.T @ (sw * (y - p)))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            converged = True; break
    p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
    bread = np.linalg.pinv(x.T @ (x * (sw * np.clip(p * (1 - p), 1e-8, None))[:, None]))
    scores = x * (sw * (y - p))[:, None]
    cov = bread @ (scores.T @ scores) @ bread * len(d) / max(len(d) - x.shape[1], 1)
    j = predictors.index(focal) + 1
    se = math.sqrt(max(cov[j, j], 0)); coef = beta[j]
    z = coef / se if se > 0 else np.nan
    pv = math.erfc(abs(z) / math.sqrt(2)) if np.isfinite(z) else np.nan
    return {
        "模型": model, "结局": outcome, "候选资源": focal, "N": int(len(d)), "事件数_保持功能": int(y.sum()),
        "OR": float(np.exp(coef)), "95%CI下限": float(np.exp(coef - 1.96 * se)),
        "95%CI上限": float(np.exp(coef + 1.96 * se)), "P值": float(pv),
        "加权": "IPW" if weight_col else "否", "状态": "收敛" if converged else "达到迭代上限"
    }


def response_weights(data, observed_col, predictors, output_col):
    d = data[predictors + [observed_col]].dropna().copy()
    x = np.column_stack([np.ones(len(d)), d[predictors].astype(float).to_numpy()])
    y = d[observed_col].astype(float).to_numpy(); beta = np.zeros(x.shape[1])
    for _ in range(300):
        p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
        w = np.clip(p * (1 - p), 1e-8, None)
        step = np.linalg.pinv(x.T @ (x * w[:, None])) @ (x.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-8: break
    p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
    weight = y.mean() / np.clip(p, 0.05, 0.95)
    lo, hi = np.quantile(weight[y == 1], [0.01, 0.99])
    data[output_col] = np.nan
    data.loc[d.index, output_col] = np.clip(weight, lo, hi)
    return {"N": int(len(d)), "observed": int(y.sum()), "weight_min": float(lo), "weight_max": float(hi)}


def rescale_for_models(data, strict=False):
    out = data.copy()
    raw_map = {
        "age_z": out.r1agey, "education_z": out.education, "comorbidity_z": out.comorbidity,
        "self_health_z": out.self_health, "met3_z": out.met3_strict if strict else out.met3,
        "grip_reserve": out.r3gripsum, "cognitive_reserve": out.r3tr20,
        "psychological_reserve": -out.r3cesd10,
        "cognitive_change_reserve": pd.to_numeric(out.r3tr20, errors="coerce") - pd.to_numeric(out.r1tr20, errors="coerce"),
        "inflammatory_reserve": -np.log(pd.to_numeric(out.bl_crp, errors="coerce").where(out.bl_crp > 0)),
        "renal_reserve": -np.log(pd.to_numeric(out.bl_cysc, errors="coerce").where(out.bl_cysc > 0)),
    }
    for name, raw in raw_map.items(): out[name] = zscore(raw)
    return out


def bh_adjust(pvalues):
    p = pd.to_numeric(pvalues, errors="coerce").to_numpy(); q = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    if not len(ok): return q
    order = ok[np.argsort(p[ok])]; ranked = p[order] * len(ok) / np.arange(1, len(ok) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1)
    return q


def smd(x, group):
    a, b = x[group == 1].dropna(), x[group == 0].dropna()
    if not len(a) or not len(b): return np.nan
    denom = math.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / denom) if denom else 0.0


def main():
    h = read(H_PATH, H_COLS); h["ID"] = norm_id(h.ID); h["ID_w1"] = norm_id(h.ID_w1)
    b1 = read(B1_PATH, ["ID", "newglu", "newtg", "newhdl", "newhba1c", "qc1_va003"])
    b1["ID_w1"] = norm_id(b1.pop("ID"))
    b3 = read(B3_PATH, ["ID", "bl_glu", "bl_tg", "bl_hdl", "bl_hbalc", "bl_fasting", "bl_crp", "bl_cysc"])
    b3["ID"] = norm_id(b3.ID)
    h20 = read(H20_PATH, ["ID", "db001", "db003", "db005", "db007", "db009", "db011"])
    h20["ID"] = norm_id(h20.ID); h20["disability_2020"] = adl2020(h20)
    e20 = read(E20_PATH, ["ID", "exb001_1"]); e20["ID"] = norm_id(e20.ID)
    e20 = e20.rename(columns={"exb001_1": "exit_death_year"})
    df = h.merge(b1, on="ID_w1", how="left", validate="many_to_one")
    df = df.merge(b3, on="ID", how="left", validate="one_to_one")
    df = df.merge(h20[["ID", "disability_2020"]], on="ID", how="left", validate="one_to_one")
    df = df.merge(e20, on="ID", how="left", validate="one_to_one")

    df["met1"] = met_score(df, 1); df["met3"] = met_score(df, 3)
    df["met1_strict"] = met_score(df, 1, 85, 40); df["met3_strict"] = met_score(df, 3, 85, 40)
    df["persistent_high"] = ((df.met1 >= 3) & (df.met3 >= 3) & df.met1.notna() & df.met3.notna()).astype(float)
    df["persistent_high_strict"] = ((df.met1_strict >= 3) & (df.met3_strict >= 3) & df.met1_strict.notna() & df.met3_strict.notna()).astype(float)
    df["independent_2015"] = np.where(df.r3adlab_c.notna() & df.r3iadlza.notna(), ((df.r3adlab_c == 0) & (df.r3iadlza == 0)).astype(float), np.nan)
    df["disability_2018"] = np.where(df.r4adlab_c.notna() & df.r4iadlza.notna(), ((df.r4adlab_c > 0) | (df.r4iadlza > 0)).astype(float), np.nan)
    df["adl_disability_2018"] = np.where(df.r4adlab_c.notna(), (df.r4adlab_c > 0).astype(float), np.nan)
    death_year = pd.to_numeric(df.exit_death_year, errors="coerce").combine_first(pd.to_numeric(df.radyear, errors="coerce"))
    df["death_2018"] = ((death_year >= 2016) & (death_year <= 2018)).astype(float)
    df["death_2020"] = ((death_year >= 2016) & (death_year <= 2020)).astype(float)
    df["bad_2018"] = np.where(df.disability_2018.notna() | (df.death_2018 == 1), ((df.disability_2018 == 1) | (df.death_2018 == 1)).astype(float), np.nan)
    df["bad_adl_2018"] = np.where(df.adl_disability_2018.notna() | (df.death_2018 == 1), ((df.adl_disability_2018 == 1) | (df.death_2018 == 1)).astype(float), np.nan)
    df["bad_2020"] = np.where(df.disability_2020.notna() | (df.death_2020 == 1), ((df.disability_2020 == 1) | (df.death_2020 == 1)).astype(float), np.nan)
    df["maintain_2018"] = 1 - df.bad_2018
    df["maintain_adl_2018"] = 1 - df.bad_adl_2018
    df["maintain_2020"] = 1 - df.bad_2020
    df["durable_maintain"] = np.where(df.bad_adl_2018.notna() & df.bad_2020.notna(), ((df.bad_adl_2018 == 0) & (df.bad_2020 == 0)).astype(float), np.nan)

    df["female"] = (df.ragender == 2).astype(float)
    df["education"] = pd.to_numeric(df.raeducl, errors="coerce")
    df["rural"] = binary(df.h3rural)
    df["partnered"] = np.where(df.r3mstat.notna(), (df.r3mstat == 1).astype(float), np.nan)
    df["smoking"] = binary(df.r3smoken)
    df["drinking"] = np.where(df.r3drinkn_c.notna(), (df.r3drinkn_c > 0).astype(float), np.nan)
    comorb_bin = pd.concat([binary(df[c]) for c in COMORB], axis=1)
    df["comorbidity"] = comorb_bin.sum(axis=1, min_count=len(COMORB))
    df["self_health"] = pd.to_numeric(df.r3shlt, errors="coerce")
    df["age_z"] = zscore(df.r1agey); df["education_z"] = zscore(df.education)
    df["comorbidity_z"] = zscore(df.comorbidity); df["self_health_z"] = zscore(df.self_health)
    df["met3_z"] = zscore(df.met3)
    df["grip_reserve"] = zscore(df.r3gripsum)
    df["cognitive_reserve"] = zscore(df.r3tr20)
    df["psychological_reserve"] = -zscore(df.r3cesd10)
    df["social_reserve"] = binary(df.r3socwk)
    df["activity_reserve"] = any_yes(binary(df.r3vgact_c), binary(df.r3mdact_c))
    df["inflammatory_reserve"] = -zscore(np.log(pd.to_numeric(df.bl_crp, errors="coerce").where(df.bl_crp > 0)))
    df["renal_reserve"] = -zscore(np.log(pd.to_numeric(df.bl_cysc, errors="coerce").where(df.bl_cysc > 0)))
    df["persistent_psychological_health"] = np.where(df.r1cesd10.notna() & df.r3cesd10.notna(), ((df.r1cesd10 < 10) & (df.r3cesd10 < 10)).astype(float), np.nan)
    df["cognitive_change_reserve"] = zscore(pd.to_numeric(df.r3tr20, errors="coerce") - pd.to_numeric(df.r1tr20, errors="coerce"))

    eligible = (df.r1agey >= 45) & df.met1.notna() & df.met3.notna() & (df.independent_2015 == 1)
    cohort = df[eligible & (df.persistent_high == 1)].copy()
    strict = df[(df.r1agey >= 45) & df.met1_strict.notna() & df.met3_strict.notna() & (df.independent_2015 == 1) & (df.persistent_high_strict == 1)]
    analysis_all = df[eligible].copy()
    cohort = rescale_for_models(cohort, strict=False)
    strict = rescale_for_models(strict, strict=True)
    analysis_all = rescale_for_models(analysis_all, strict=False)
    analysis_all["reserve_index"] = zscore(analysis_all[["grip_reserve", "cognitive_reserve", "psychological_reserve"]].mean(axis=1, skipna=True))
    analysis_all["high_x_reserve"] = analysis_all.persistent_high * analysis_all.reserve_index

    flow = pd.DataFrame([
        ["总样本", len(df)], ["年龄≥45且两波代谢完整", int(((df.r1agey >= 45) & df.met1.notna() & df.met3.notna()).sum())],
        ["加2015功能独立", int(eligible.sum())], ["持续代谢高风险主队列", len(cohort)],
        ["2018复合结局可判定", int(cohort.bad_2018.notna().sum())], ["2020复合结局可判定", int(cohort.bad_2020.notna().sum())]
    ], columns=["步骤", "人数"])
    flow["占上一步比例"] = flow.人数 / flow.人数.shift(1); flow.loc[0, "占上一步比例"] = 1

    candidates = ["grip_reserve", "cognitive_reserve", "psychological_reserve", "persistent_psychological_health", "cognitive_change_reserve", "social_reserve", "activity_reserve", "inflammatory_reserve", "renal_reserve"]
    missing = []
    for c in candidates:
        missing.append([c, int(cohort[c].notna().sum()), float(cohort[c].notna().mean()), int((cohort[c].notna() & cohort.bad_2018.notna()).sum())])
    missing_df = pd.DataFrame(missing, columns=["候选资源", "高风险队列非缺失", "完整率", "且2018结局可判定"])

    responder = cohort.bad_2018.notna().astype(int)
    attrs = ["r1agey", "female", "education", "rural", "partnered", "smoking", "drinking", "comorbidity", "self_health", "met3"]
    attrition = []
    for v in attrs:
        attrition.append([v, cohort.loc[responder == 1, v].mean(), cohort.loc[responder == 0, v].mean(), smd(cohort[v], responder), int(cohort[v].notna().sum())])
    attrition_df = pd.DataFrame(attrition, columns=["变量", "2018可判定均值", "2018不可判定均值", "标准化差异SMD", "非缺失N"])

    core = ["age_z", "female", "education_z", "rural", "partnered", "smoking", "drinking", "comorbidity_z", "self_health_z", "met3_z"]
    cohort["observed_2018"] = cohort.bad_2018.notna().astype(float)
    cohort["observed_adl_2018"] = cohort.bad_adl_2018.notna().astype(float)
    cohort["observed_2020"] = cohort.bad_2020.notna().astype(float)
    cohort["observed_durable"] = cohort.durable_maintain.notna().astype(float)
    ipw_info = {
        "2018": response_weights(cohort, "observed_2018", core, "ipw_2018"),
        "ADL2018": response_weights(cohort, "observed_adl_2018", core, "ipw_adl_2018"),
        "2020": response_weights(cohort, "observed_2020", core, "ipw_2020")
        ,"durable": response_weights(cohort, "observed_durable", core, "ipw_durable")
    }
    models = []
    for outcome, weight_col in [("maintain_2018", "ipw_2018"), ("maintain_adl_2018", "ipw_adl_2018"), ("maintain_2020", "ipw_2020"), ("durable_maintain", "ipw_durable")]:
        for c in candidates:
            models.append(fit_logit_robust(cohort, outcome, core + [c], f"主队列_{outcome}_{c}", c))
            models.append(fit_logit_robust(cohort, outcome, core + [c], f"主队列IPW_{outcome}_{c}", c, weight_col))
    for outcome in ["maintain_2018", "maintain_adl_2018", "maintain_2020", "durable_maintain"]:
        for c in candidates:
            models.append(fit_logit_robust(strict, outcome, core + [c], f"严格定义_{outcome}_{c}", c))
    joint_candidates = ["grip_reserve", "cognitive_reserve", "psychological_reserve", "social_reserve", "inflammatory_reserve", "renal_reserve"]
    for outcome in ["maintain_2018", "maintain_adl_2018", "maintain_2020", "durable_maintain"]:
        for c in joint_candidates:
            models.append(fit_logit_robust(cohort, outcome, core + joint_candidates, f"联合领域_{outcome}", c))
        interaction_predictors = core + ["persistent_high", "reserve_index", "high_x_reserve"]
        models.append(fit_logit_robust(analysis_all, outcome, interaction_predictors, f"全队列储备交互_{outcome}", "high_x_reserve"))
    landmark = cohort[cohort.bad_adl_2018 == 0].copy()
    landmark_strict = strict[strict.bad_adl_2018 == 0].copy()
    for c in candidates:
        models.append(fit_logit_robust(landmark, "maintain_2020", core + [c], f"2018功能保持者landmark_{c}", c))
        models.append(fit_logit_robust(landmark_strict, "maintain_2020", core + [c], f"严格定义landmark_{c}", c))
    model_df = pd.DataFrame(models)
    model_df["BH校正P"] = bh_adjust(model_df["P值"])

    compare = pd.DataFrame([
        ["主定义", len(cohort), int(cohort.bad_2018.notna().sum()), int(cohort.bad_2018.sum()), int(cohort.bad_adl_2018.notna().sum()), int(cohort.bad_adl_2018.sum()), int(cohort.bad_2020.notna().sum()), int(cohort.bad_2020.sum())],
        ["严格替代定义", len(strict), int(strict.bad_2018.notna().sum()), int(strict.bad_2018.sum()), int(strict.bad_adl_2018.notna().sum()), int(strict.bad_adl_2018.sum()), int(strict.bad_2020.notna().sum()), int(strict.bad_2020.sum())]
    ], columns=["定义", "高风险队列N", "2018_ADL或IADL可判定", "2018_ADL或IADL事件", "2018_ADL可判定", "2018_ADL或死亡事件", "2020可判定", "2020不良事件"])

    flow.to_csv(EXPORT / "formal_sample_flow.csv", index=False, encoding="utf-8-sig")
    missing_df.to_csv(EXPORT / "candidate_missingness.csv", index=False, encoding="utf-8-sig")
    attrition_df.to_csv(EXPORT / "attrition_balance.csv", index=False, encoding="utf-8-sig")
    model_df.to_csv(EXPORT / "domain_models.csv", index=False, encoding="utf-8-sig")
    compare.to_csv(EXPORT / "definition_outcome_sensitivity.csv", index=False, encoding="utf-8-sig")
    summary = {
        "formal_high_risk_n": int(len(cohort)), "strict_high_risk_n": int(len(strict)),
        "events_2018": int(cohort.bad_2018.sum()), "events_adl_2018": int(cohort.bad_adl_2018.sum()), "events_2020": int(cohort.bad_2020.sum()),
        "durable_outcome_n": int(cohort.durable_maintain.notna().sum()), "durable_maintainers": int(cohort.durable_maintain.sum()),
        "max_attrition_abs_smd": float(attrition_df["标准化差异SMD"].abs().max()),
        "candidate_complete_rates": dict(zip(missing_df["候选资源"], missing_df["完整率"])),
        "ipw": ipw_info,
        "interpretation_rule": "OR>1表示候选资源与保持功能相关；本阶段仅作关联和模型筛选。"
    }
    (EXPORT / "formal_analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

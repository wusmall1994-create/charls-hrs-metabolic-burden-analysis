from __future__ import annotations

import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.imputation.mice import MICEData

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


PROJECT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", PROJECT / "results"))
HRS_ROOT = Path(os.environ.get("HRS_DATA_DIR", PROJECT / "data" / "hrs"))
BIOM_DIR = Path(os.environ.get("HRS_BIOMARKER_DIR", HRS_ROOT / "biomarkers"))
RAND_FILE = (
    HRS_ROOT
    / "01_rand_longitudinal"
    / "extracted"
    / "randhrs1992_2022v1_STATA"
    / "randhrs1992_2022v1.dta"
)
TRACKER_FILE = HRS_ROOT / "04_tracker" / "extracted" / "trk2022tr_r.dta"
HARMONIZED_FILE = HRS_ROOT / "06_harmonized_hrs" / "extracted" / "H_HRS_d.dta"
CRP_FILE = HRS_ROOT / "DBS_CRP" / "HRS_CRP_XWAVE.dta"

SEED = 20260903
M = 20

BIOM_INFO = {
    8: (2006, "biomk06bl_r.dta", "KA1C_ADJ", "KHDL_ADJ", "KCRP_ADJ", "KCYSC_ADJ", "KBIOWGTR", "KXCRP_ADJ"),
    9: (2008, "biomk08bl_r.dta", "LA1C_ADJ", "LHDL_ADJ", "LCRP_ADJ", "LCYSC_ADJ", "LBIOWGTR", "LXCRP_ADJ"),
    10: (2010, "biomk10bl_r.dta", "MA1C_ADJ", "MHDL_ADJ", "MCRP_ADJ", "MCYSC_ADJ", "MBIOWGTR", "MXCRP_ADJ"),
    11: (2012, "biomk12bl_r.dta", "NA1C_ADJ", "NHDL_ADJ", "NCRP_ADJ", "NCYSC_ADJ", "NBIOWGTR", "NXCRP_ADJ"),
    12: (2014, "biomk14bl.dta", "OA1C_ADJ", "OHDL_ADJ", "OCRP_ADJ", "OCYSC_ADJ", "OBIOWGTR", "OXCRP_ADJ"),
    13: (2016, "BIOMK16BL_R.dta", "PA1C_ADJ", "PHDL_ADJ", "PCRP_ADJ", "PCYSC_ADJ", None, "PXCRP_ADJ"),
}

WINDOWS = [(8, 10), (9, 11), (10, 12), (11, 13)]
MARKERS = [
    "persistent_low_depression",
    "memory_z",
    "grip_z",
    "social_participation",
    "favorable_crp_z",
    "favorable_cystatin_z",
]


def clean_numeric(x: pd.Series, lower: float | None = None, upper: float | None = None) -> pd.Series:
    y = pd.to_numeric(x, errors="coerce").astype(float)
    y = y.mask(y >= 2_000_000_000)
    if lower is not None:
        y = y.mask(y < lower)
    if upper is not None:
        y = y.mask(y > upper)
    return y


def binary(x: pd.Series) -> pd.Series:
    y = clean_numeric(x)
    return y.where(y.isin([0, 1]))


def zscore(x: pd.Series) -> pd.Series:
    y = clean_numeric(x)
    sd = y.std(ddof=1)
    return (y - y.mean()) / sd if pd.notna(sd) and sd > 0 else y * np.nan


def all_binary_sum(frame: pd.DataFrame) -> pd.Series:
    return frame.sum(axis=1, min_count=frame.shape[1]).astype(float)


def hhidpn_from_parts(hhid: pd.Series, pn: pd.Series) -> pd.Series:
    return hhid.astype(str).str.zfill(6) + pn.astype(str).str.zfill(3)


def load_biomarkers(requested_waves: tuple[int, ...] = (10, 12)) -> dict[int, pd.DataFrame]:
    """Load the HRS biomarker waves required by the harmonized analysis."""
    unknown = set(requested_waves) - set(BIOM_INFO)
    if unknown:
        raise ValueError(f"Unknown HRS biomarker waves: {sorted(unknown)}")

    tracker = None
    if any(BIOM_INFO[wave][6] is None for wave in requested_waves):
        tracker = pd.read_stata(
            TRACKER_FILE,
            columns=["HHID", "PN", "PBIOWGTR"],
            convert_categoricals=False,
        )
        tracker["hhidpn"] = hhidpn_from_parts(tracker["HHID"], tracker["PN"])
        tracker = tracker[["hhidpn", "PBIOWGTR"]].drop_duplicates("hhidpn")

    crp = pd.read_stata(CRP_FILE, convert_categoricals=False)
    crp["hhidpn"] = hhidpn_from_parts(crp["HHID"], crp["PN"])

    loaded: dict[int, pd.DataFrame] = {}
    for wave in requested_waves:
        info = BIOM_INFO[wave]
        _, filename, a1c_col, hdl_col, crp_col, cys_col, weight_col, updated_crp_col = info
        raw = pd.read_stata(BIOM_DIR / filename, convert_categoricals=False)
        raw["hhidpn"] = hhidpn_from_parts(raw["HHID"], raw["PN"])
        if raw["hhidpn"].duplicated().any():
            raise ValueError(f"Duplicate biomarker IDs in {filename}")
        d = pd.DataFrame({"hhidpn": raw["hhidpn"]})
        d["a1c"] = clean_numeric(raw[a1c_col], 3, 20)
        d["hdl"] = clean_numeric(raw[hdl_col], 5, 250)
        d["crp_release"] = clean_numeric(raw[crp_col], 0)
        d["cystatin_c"] = clean_numeric(raw[cys_col], 0)
        if weight_col is None:
            assert tracker is not None
            d = d.merge(tracker, on="hhidpn", how="left", validate="one_to_one")
            d["biowgt"] = clean_numeric(d.pop("PBIOWGTR"), 0)
        else:
            d["biowgt"] = clean_numeric(raw[weight_col], 0)
        d = d.merge(crp[["hhidpn", updated_crp_col]], on="hhidpn", how="left", validate="one_to_one")
        d["crp"] = clean_numeric(d.pop(updated_crp_col), 0)
        d["crp"] = d["crp"].fillna(d["crp_release"])
        loaded[wave] = d
    return loaded


def load_longitudinal() -> pd.DataFrame:
    reader = pd.read_stata(RAND_FILE, iterator=True)
    available = set(reader.variable_labels())
    wanted = ["hhidpn", "hhid", "pn", "ragender", "raedyrs", "raestrat", "raehsamp"]
    stems = [
        "pmwaist", "bpsys", "bpdia", "hibpe", "diabe", "adl6a", "iadl5a", "tr20", "cesd",
        "iwstat", "agey_e", "grp", "urbrur", "mstat", "smoken", "drink", "shlt", "cancre",
        "lunge", "hearte", "stroke", "psyche", "arthre", "wtresp",
    ]
    for wave in range(8, 17):
        wanted.extend(f"r{wave}{stem}" for stem in stems)
    wanted = [c for c in wanted if c in available]
    data = pd.read_stata(RAND_FILE, columns=wanted, convert_categoricals=False)
    data["hhidpn"] = hhidpn_from_parts(data["hhid"], data["pn"])
    if data["hhidpn"].duplicated().any():
        raise ValueError("Duplicate IDs in RAND HRS")
    for col in data.columns:
        if col not in {"hhidpn", "hhid", "pn"}:
            data[col] = clean_numeric(data[col])

    social_cols = ["hhidpn"] + [f"r{wave}socwk" for wave in range(9, 14)]
    social = pd.read_stata(HARMONIZED_FILE, columns=social_cols, convert_categoricals=False)
    social["hhidpn"] = (
        pd.to_numeric(social["hhidpn"], errors="coerce").astype("Int64").astype(str).str.zfill(9)
    )
    for col in social_cols[1:]:
        social[col] = clean_numeric(social[col])
    return data.merge(social, on="hhidpn", how="left", validate="one_to_one")


def ternary_component(any_abnormal: pd.Series, all_known: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=any_abnormal.index, dtype=float)
    result.loc[all_known & ~any_abnormal] = 0.0
    result.loc[any_abnormal] = 1.0
    return result


def add_metabolic_components(
    data: pd.DataFrame,
    wave: int,
    waist_male_cm: float = 90,
    waist_female_cm: float = 80,
) -> pd.DataFrame:
    female = data["ragender"] == 2
    sex_known = data["ragender"].isin([1, 2])
    waist = clean_numeric(data[f"r{wave}pmwaist"] * 2.54, 50, 200)
    sbp = clean_numeric(data[f"r{wave}bpsys"], 60, 260)
    dbp = clean_numeric(data[f"r{wave}bpdia"], 30, 160)
    hibp = binary(data[f"r{wave}hibpe"])
    diabetes = binary(data[f"r{wave}diabe"])
    a1c = clean_numeric(data[f"a1c_{wave}"], 3, 20)
    hdl = clean_numeric(data[f"hdl_{wave}"], 5, 250)

    waist_high = ((~female) & (waist >= waist_male_cm)) | (female & (waist >= waist_female_cm))
    data[f"waist_abn_{wave}"] = ternary_component(waist_high, waist.notna() & sex_known)

    bp_high = (sbp >= 130) | (dbp >= 85) | (hibp == 1)
    bp_known = sbp.notna() & dbp.notna() & hibp.notna()
    data[f"bp_abn_{wave}"] = ternary_component(bp_high, bp_known)

    gly_high = (a1c >= 5.7) | (diabetes == 1)
    gly_known = a1c.notna() & diabetes.notna()
    data[f"gly_abn_{wave}"] = ternary_component(gly_high, gly_known)

    hdl_low = ((~female) & (hdl < 40)) | (female & (hdl < 50))
    data[f"hdl_abn_{wave}"] = ternary_component(hdl_low, hdl.notna() & sex_known)

    cols = [f"waist_abn_{wave}", f"bp_abn_{wave}", f"gly_abn_{wave}", f"hdl_abn_{wave}"]
    data[f"components_complete_{wave}"] = data[cols].notna().all(axis=1)
    data[f"score4_{wave}"] = data[cols].sum(axis=1, min_count=4)
    return data


def build_window(
    longitudinal: pd.DataFrame,
    biomarkers: dict[int, pd.DataFrame],
    first: int,
    anchor: int,
    waist_male_cm: float = 90,
    waist_female_cm: float = 80,
    followup_offsets: tuple[int, ...] = (2, 4),
) -> pd.DataFrame:
    b1 = biomarkers[first].add_suffix(f"_{first}").rename(columns={f"hhidpn_{first}": "hhidpn"})
    b2 = biomarkers[anchor].add_suffix(f"_{anchor}").rename(columns={f"hhidpn_{anchor}": "hhidpn"})
    out = b1.merge(b2, on="hhidpn", how="inner").merge(
        longitudinal, on="hhidpn", how="left", validate="one_to_one"
    )
    out = add_metabolic_components(out, first, waist_male_cm, waist_female_cm)
    out = add_metabolic_components(out, anchor, waist_male_cm, waist_female_cm)

    out["female"] = np.where(
        out["ragender"].isin([1, 2]), (out["ragender"] == 2).astype(float), np.nan
    )
    out["age"] = clean_numeric(out[f"r{anchor}agey_e"], 40, 110)
    out["education"] = clean_numeric(out["raedyrs"], 0, 25)

    residence = clean_numeric(out[f"r{anchor}urbrur"])
    out["rural"] = np.where(residence.isin([1, 2, 3]), (residence == 3).astype(float), np.nan)
    mstat = clean_numeric(out[f"r{anchor}mstat"])
    out["partnered"] = np.where(
        mstat.isin([1, 2, 3]), 1.0, np.where(mstat.isin([4, 5, 6, 7, 8]), 0.0, np.nan)
    )
    out["smoking"] = binary(out[f"r{anchor}smoken"])
    out["drinking"] = binary(out[f"r{anchor}drink"])
    out["self_health"] = clean_numeric(out[f"r{anchor}shlt"], 1, 5)

    disease_stems = ["cancre", "lunge", "hearte", "stroke", "psyche", "arthre"]
    disease = pd.concat([binary(out[f"r{anchor}{stem}"]) for stem in disease_stems], axis=1)
    disease.columns = disease_stems
    out["comorbidity"] = all_binary_sum(disease)

    out["memory_first"] = clean_numeric(out[f"r{first}tr20"], 0, 20)
    out["memory_anchor"] = clean_numeric(out[f"r{anchor}tr20"], 0, 20)
    out["cesd_first"] = clean_numeric(out[f"r{first}cesd"], 0, 8)
    out["cesd_anchor"] = clean_numeric(out[f"r{anchor}cesd"], 0, 8)
    both_cesd = out["cesd_first"].notna() & out["cesd_anchor"].notna()
    out["persistent_low_depression"] = np.where(
        both_cesd,
        ((out["cesd_first"] < 4) & (out["cesd_anchor"] < 4)).astype(float),
        np.nan,
    )
    out["grip"] = clean_numeric(out[f"r{anchor}grp"], 1, 100)
    social_col = f"r{anchor}socwk"
    out["social_participation"] = binary(out[social_col]) if social_col in out else np.nan
    out["log_crp"] = np.log(clean_numeric(out[f"crp_{anchor}"], 0.001))
    out["log_cystatin"] = np.log(clean_numeric(out[f"cystatin_c_{anchor}"], 0.001))
    out["met_anchor"] = out[f"score4_{anchor}"]

    adl_anchor = clean_numeric(out[f"r{anchor}adl6a"], 0, 6)
    iadl_anchor = clean_numeric(out[f"r{anchor}iadl5a"], 0, 5)
    out["baseline_independent"] = np.where(
        adl_anchor.notna() & iadl_anchor.notna(),
        ((adl_anchor == 0) & (iadl_anchor == 0)).astype(float),
        np.nan,
    )
    out["blood_weight"] = clean_numeric(out[f"biowgt_{anchor}"], 0)
    out["design_stratum_raw"] = clean_numeric(out["raestrat"])
    out["design_psu_raw"] = clean_numeric(out["raehsamp"])

    followups = [anchor + offset for offset in followup_offsets if anchor + offset <= 16]
    for i, wave in enumerate(followups, start=1):
        adl = clean_numeric(out[f"r{wave}adl6a"], 0, 6)
        status = clean_numeric(out[f"r{wave}iwstat"])
        observed_function = (status == 1) & adl.notna()
        died = status.isin([5, 6])
        out[f"observed_f{i}_function"] = observed_function.astype(float)
        out[f"f{i}_independent"] = np.where(observed_function, (adl == 0).astype(float), np.nan)
        known = observed_function | died
        out[f"observed_f{i}_composite"] = known.astype(float)
        out[f"f{i}_composite_independent"] = np.where(
            known, np.where(died, 0.0, (adl == 0).astype(float)), np.nan
        )
        out[f"f{i}_state"] = np.where(
            died, 2.0, np.where(observed_function, np.where(adl > 0, 1.0, 0.0), np.nan)
        )

    if len(followups) == 2:
        out["observed_durable"] = (
            (out["observed_f1_function"] == 1) & (out["observed_f2_function"] == 1)
        ).astype(float)
        out["durable_independent"] = np.where(
            out["observed_durable"] == 1,
            ((out["f1_independent"] == 1) & (out["f2_independent"] == 1)).astype(float),
            np.nan,
        )
        out["observed_durable_composite"] = (
            (out["observed_f1_composite"] == 1) & (out["observed_f2_composite"] == 1)
        ).astype(float)
        out["durable_composite_independent"] = np.where(
            out["observed_durable_composite"] == 1,
            ((out["f1_state"] == 0) & (out["f2_state"] == 0)).astype(float),
            np.nan,
        )
        f1_disabled = (out["observed_f1_function"] == 1) & (out["f1_independent"] == 0)
        out["f1_disabled"] = f1_disabled.astype(float)
        out["observed_recovery"] = np.where(
            f1_disabled, out["observed_f2_function"], 0
        ).astype(float)
        out["recovered_f2"] = np.where(
            f1_disabled & (out["observed_f2_function"] == 1), out["f2_independent"], np.nan
        )
    else:
        out["observed_durable"] = 0.0
        out["durable_independent"] = np.nan
        out["observed_durable_composite"] = 0.0
        out["durable_composite_independent"] = np.nan
        out["f1_disabled"] = (
            (out["observed_f1_function"] == 1) & (out["f1_independent"] == 0)
        ).astype(float)
        out["observed_recovery"] = 0.0
        out["recovered_f2"] = np.nan

    out.attrs["first_wave"] = first
    out.attrs["anchor_wave"] = anchor
    out.attrs["followups"] = followups
    return out


def select_cohort(data: pd.DataFrame, first: int, anchor: int, threshold: int) -> pd.DataFrame:
    complete = data[f"components_complete_{first}"] & data[f"components_complete_{anchor}"]
    persistent = (
        complete
        & (data[f"score4_{first}"] >= threshold)
        & (data[f"score4_{anchor}"] >= threshold)
    )
    eligible = (
        persistent
        & (data["age"] >= 50)
        & (data["blood_weight"] > 0)
        & (data["baseline_independent"] == 1)
        & data["design_stratum_raw"].notna()
        & data["design_psu_raw"].notna()
    )
    cohort = data.loc[eligible].copy().reset_index(drop=True)
    cohort["design_stratum"] = pd.factorize(cohort["design_stratum_raw"], sort=True)[0]
    cohort["design_psu"] = cohort["design_psu_raw"].astype(int)
    cohort.attrs.update(data.attrs)
    cohort.attrs["threshold"] = threshold
    return cohort


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
        totals = np.vstack([scores[idx[psu[idx] == group]].sum(axis=0) for group in groups])
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
        totals = np.vstack([scores[psu == group].sum(axis=0) for group in groups])
        centered = totals - totals.mean(axis=0)
        meat = len(groups) / max(len(groups) - 1, 1) * centered.T @ centered
    return meat, max(clusters - len(pd.unique(strata)), 1), clusters, singleton


def logistic_fit(data: pd.DataFrame, outcome: str, predictors: list[str], weight: str):
    cols = [outcome, weight, "design_stratum", "design_psu"] + predictors
    d = data[cols].dropna().copy()
    x = np.column_stack([np.ones(len(d)), d[predictors].astype(float).to_numpy()])
    y = d[outcome].astype(float).to_numpy()
    w = d[weight].astype(float).to_numpy(copy=True)
    w /= np.mean(w)
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
    return {
        "beta": beta,
        "cov": bread @ meat @ bread,
        "names": ["const"] + predictors,
        "N": len(d),
        "events": int(y.sum()),
        "design_df": design_df,
        "clusters": clusters,
        "singleton_strata": singleton,
        "converged": converged,
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
        gradient = np.concatenate(
            [x.T @ (w * ((y == state).astype(float) - probs[:, state])) for state in [1, 2]]
        )
        info = np.zeros((2 * pcols, 2 * pcols))
        for a, state_a in enumerate([1, 2]):
            for b, state_b in enumerate([1, 2]):
                coefficient = probs[:, state_a] * ((1 if state_a == state_b else 0) - probs[:, state_b])
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
    for a, state_a in enumerate([1, 2]):
        scores[:, a * pcols:(a + 1) * pcols] = x * (
            w * ((y == state_a).astype(float) - probs[:, state_a])
        )[:, None]
        for b, state_b in enumerate([1, 2]):
            coefficient = probs[:, state_a] * ((1 if state_a == state_b else 0) - probs[:, state_b])
            info[a * pcols:(a + 1) * pcols, b * pcols:(b + 1) * pcols] = x.T @ (
                x * (w * coefficient)[:, None]
            )
    bread = np.linalg.pinv(info)
    meat, design_df, clusters, singleton = survey_meat(scores, d["design_stratum"], d["design_psu"])
    return {
        "beta": beta.reshape(-1),
        "cov": bread @ meat @ bread,
        "names": ["const"] + predictors,
        "N": n,
        "counts": {str(state): int(np.sum(y == state)) for state in [0, 1, 2]},
        "design_df": design_df,
        "clusters": clusters,
        "singleton_strata": singleton,
        "converged": converged,
    }


def response_ipw(data: pd.DataFrame, observed: str, predictors: list[str], base_weight: str):
    cols = [observed, base_weight] + predictors
    d = data[cols].dropna().copy()
    x = np.column_stack([np.ones(len(d)), d[predictors].astype(float).to_numpy()])
    y = d[observed].astype(float).to_numpy()
    w = d[base_weight].astype(float).to_numpy(copy=True)
    w /= np.mean(w)
    beta = np.zeros(x.shape[1])
    converged = False
    for _ in range(300):
        p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
        v = np.clip(p * (1 - p), 1e-8, None)
        step = np.linalg.pinv(x.T @ (x * (w * v)[:, None])) @ (x.T @ (w * (y - p)))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            converged = True
            break
    p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
    stabilized = np.average(y, weights=w) / np.clip(p, 0.05, 0.95)
    observed_weights = stabilized[y == 1]
    lo, hi = np.quantile(observed_weights, [0.01, 0.99])
    result = pd.Series(np.nan, index=data.index)
    result.loc[d.index] = np.clip(stabilized, lo, hi)
    return result, {
        "eligible_n": len(d),
        "observed_n": int(y.sum()),
        "observed_weighted_rate": float(np.average(y, weights=w)),
        "ipw_p1": float(lo),
        "ipw_p99": float(hi),
        "converged": converged,
    }


def prepare_completed(data: pd.DataFrame, cesd_threshold: float = 4) -> pd.DataFrame:
    out = data.copy()
    for name in ["female", "rural", "partnered", "smoking", "drinking", "social_participation"]:
        out[name] = np.where(out[name].notna(), (out[name] >= 0.5).astype(float), np.nan)
    out["persistent_low_depression"] = (
        (out["cesd_first"] < cesd_threshold) & (out["cesd_anchor"] < cesd_threshold)
    ).astype(float)
    out["age_z"] = zscore(out["age"])
    out["education_z"] = zscore(out["education"])
    out["comorbidity_z"] = zscore(out["comorbidity"])
    out["self_health_z"] = zscore(out["self_health"])
    out["met_anchor_z"] = zscore(out["met_anchor"])
    out["memory_first_z"] = zscore(out["memory_first"])
    out["memory_z"] = zscore(out["memory_anchor"])
    out["grip_z"] = zscore(out["grip"])
    out["favorable_crp_z"] = -zscore(out["log_crp"])
    out["favorable_cystatin_z"] = -zscore(out["log_cystatin"])
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
    return contrast, float(gradient @ fit["cov"] @ gradient)


def adjust_pvalues(pvalues: pd.Series, method: str) -> pd.Series:
    result = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.dropna().sort_values()
    m = len(valid)
    if not m:
        return result
    if method == "holm":
        running = 0.0
        for rank, (idx, value) in enumerate(valid.items()):
            running = max(running, min(1.0, value * (m - rank)))
            result.loc[idx] = running
    elif method == "bh":
        adjusted = np.empty(m)
        values = valid.to_numpy()
        running = 1.0
        for i in range(m - 1, -1, -1):
            running = min(running, values[i] * m / (i + 1))
            adjusted[i] = min(1.0, running)
        result.loc[valid.index] = adjusted
    else:
        raise ValueError(method)
    return result


def run_mi(
    cohort: pd.DataFrame,
    primary_outcome: str = "durable_independent",
    primary_observed: str = "observed_durable",
    cesd_threshold: float = 4,
):
    np.random.seed(SEED)
    raw_predictors = [
        "age", "female", "education", "rural", "partnered", "smoking", "drinking",
        "comorbidity", "self_health", "met_anchor", "memory_first", "memory_anchor",
        "cesd_first", "cesd_anchor", "grip", "social_participation", "log_crp", "log_cystatin",
    ]
    outcome_vars = [
        "durable_independent", "f1_composite_independent", "f1_independent", "f2_independent",
        "recovered_f2", "f1_state", "observed_durable", "observed_f1_composite",
        "observed_f1_function", "observed_f2_function", "observed_recovery",
        "durable_composite_independent", "observed_durable_composite",
    ]
    # Include one nonredundant outcome representation per follow-up as auxiliary
    # information for covariate imputation.  The broader outcome/observation set
    # contains deterministic relationships (for example, observed_durable and
    # durable_independent); putting all of them in MICE makes the parameter
    # covariance singular.  Outcomes are restored to their observed values below
    # and are never imputed for an analysis model.
    auxiliary_outcomes = list(dict.fromkeys([primary_outcome, "f1_state", "f2_independent"]))
    original_cols = outcome_vars + ["blood_weight", "design_stratum", "design_psu", "f1_disabled"]
    original = cohort[original_cols].copy()
    mice = MICEData(
        cohort[raw_predictors + auxiliary_outcomes].astype(float),
        perturbation_method="gaussian",
        k_pmm=20,
    )
    mice.update_all(10)

    core = [
        "age_z", "female", "education_z", "rural", "partnered", "smoking", "drinking",
        "comorbidity_z", "self_health_z", "met_anchor_z",
    ]
    model_specs = [
        ("primary_joint_sustained", primary_outcome, primary_observed, core + MARKERS),
        ("primary_two_marker_sustained", primary_outcome, primary_observed, core + ["persistent_low_depression", "memory_z"]),
        ("prior_memory_adjusted_sustained", primary_outcome, primary_observed, core + MARKERS + ["memory_first_z"]),
        ("secondary_f1_disability_or_death", "f1_composite_independent", "observed_f1_composite", core + MARKERS),
        ("secondary_f1_function_respondents", "f1_independent", "observed_f1_function", core + MARKERS),
        ("secondary_f2_function_respondents", "f2_independent", "observed_f2_function", core + MARKERS),
    ]
    fits: dict[str, list[dict]] = {name: [] for name, *_ in model_specs}
    fits["exploratory_recovery"] = []
    multistate_fits = []
    contrasts = {"persistent_low_depression": [], "memory_z": []}
    ipw_rows = []

    for imp in range(1, M + 1):
        mice.update_all(5)
        d = prepare_completed(mice.data.copy(), cesd_threshold=cesd_threshold)
        for col in original:
            d[col] = original[col].to_numpy()
        response_predictors = core + MARKERS

        for name, outcome, observed, predictors in model_specs:
            ipw, diag = response_ipw(d, observed, response_predictors, "blood_weight")
            d["analysis_weight"] = d["blood_weight"] * ipw
            fit = logistic_fit(d, outcome, predictors, "analysis_weight")
            fits[name].append(fit)
            ipw_rows.append({"imputation": imp, "model": name, **diag})
            if name == "primary_joint_sustained":
                for focal, kind in [("persistent_low_depression", "binary"), ("memory_z", "plus_one")]:
                    q, u = marginal_contrast(fit, d, focal, kind, "blood_weight")
                    contrasts[focal].append((q, u, fit["design_df"]))

        transition_predictors = ["age_z", "female", "met_anchor_z", "persistent_low_depression", "memory_z"]
        ipw, diag = response_ipw(d, "observed_f1_composite", transition_predictors, "blood_weight")
        d["transition_weight"] = d["blood_weight"] * ipw
        multistate_fits.append(multinomial_fit(d, "f1_state", transition_predictors, "transition_weight"))
        ipw_rows.append({"imputation": imp, "model": "multistate_anchor_to_f1", **diag})

        recovery = d[d["f1_disabled"] == 1].copy()
        ipw, diag = response_ipw(recovery, "observed_recovery", transition_predictors, "blood_weight")
        recovery["recovery_weight"] = recovery["blood_weight"] * ipw
        fits["exploratory_recovery"].append(
            logistic_fit(recovery, "recovered_f2", transition_predictors, "recovery_weight")
        )
        ipw_rows.append({"imputation": imp, "model": "exploratory_recovery", **diag})

    model_rows = []
    for model_name, flist in fits.items():
        report_names = set(MARKERS + ["memory_first_z"])
        for j, variable in enumerate(flist[0]["names"]):
            if variable not in report_names:
                continue
            pooled = pool_scalar(
                [fit["beta"][j] for fit in flist],
                [fit["cov"][j, j] for fit in flist],
                [fit["design_df"] for fit in flist],
            )
            q, se, df, p, lo, hi, within, between = pooled
            model_rows.append(
                {
                    "model": model_name,
                    "variable": variable,
                    "imputations": len(flist),
                    "analysis_n": int(np.median([fit["N"] for fit in flist])),
                    "independent_events": int(np.median([fit["events"] for fit in flist])),
                    "odds_ratio": math.exp(q),
                    "ci_low": math.exp(lo),
                    "ci_high": math.exp(hi),
                    "p_value": p,
                    "df": df,
                    "within_variance": within,
                    "between_variance": between,
                    "clusters": int(np.median([fit["clusters"] for fit in flist])),
                    "singleton_strata": int(np.median([fit["singleton_strata"] for fit in flist])),
                    "all_converged": bool(all(fit["converged"] for fit in flist)),
                }
            )
    models = pd.DataFrame(model_rows)
    primary = models["model"] == "primary_joint_sustained"
    models.loc[primary, "bh_six_marker_p"] = adjust_pvalues(models.loc[primary, "p_value"], "bh")
    focal = primary & models["variable"].isin(["persistent_low_depression", "memory_z"])
    models.loc[focal, "holm_two_target_p"] = adjust_pvalues(models.loc[focal, "p_value"], "holm")
    targeted = (models["model"] == "primary_two_marker_sustained") & models["variable"].isin(
        ["persistent_low_depression", "memory_z"]
    )
    models.loc[targeted, "holm_two_target_p"] = adjust_pvalues(
        models.loc[targeted, "p_value"], "holm"
    )

    contrast_rows = []
    for variable, values in contrasts.items():
        q, se, df, p, lo, hi, _, _ = pool_scalar(
            [value[0] for value in values], [value[1] for value in values], [value[2] for value in values]
        )
        contrast_rows.append(
            {
                "variable": variable,
                "contrast": "low symptoms vs not" if variable == "persistent_low_depression" else "+1 SD memory",
                "adjusted_probability_difference": q,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": p,
                "imputations": len(values),
            }
        )

    multistate_rows = []
    names = multistate_fits[0]["names"]
    block = len(names)
    for state_index, transition in [(0, "ADL disability vs independent"), (1, "death vs independent")]:
        for variable in ["persistent_low_depression", "memory_z"]:
            j = state_index * block + names.index(variable)
            q, se, df, p, lo, hi, within, between = pool_scalar(
                [fit["beta"][j] for fit in multistate_fits],
                [fit["cov"][j, j] for fit in multistate_fits],
                [fit["design_df"] for fit in multistate_fits],
            )
            multistate_rows.append(
                {
                    "transition": transition,
                    "variable": variable,
                    "imputations": len(multistate_fits),
                    "analysis_n": int(np.median([fit["N"] for fit in multistate_fits])),
                    "independent_n": int(np.median([fit["counts"]["0"] for fit in multistate_fits])),
                    "disability_n": int(np.median([fit["counts"]["1"] for fit in multistate_fits])),
                    "death_n": int(np.median([fit["counts"]["2"] for fit in multistate_fits])),
                    "relative_risk_ratio": math.exp(q),
                    "ci_low": math.exp(lo),
                    "ci_high": math.exp(hi),
                    "p_value": p,
                    "df": df,
                    "within_variance": within,
                    "between_variance": between,
                    "clusters": int(np.median([fit["clusters"] for fit in multistate_fits])),
                    "all_converged": bool(all(fit["converged"] for fit in multistate_fits)),
                }
            )
    multistate = pd.DataFrame(multistate_rows)
    for transition in multistate["transition"].unique():
        mask = multistate["transition"] == transition
        multistate.loc[mask, "holm_two_target_p"] = adjust_pvalues(
            multistate.loc[mask, "p_value"], "holm"
        )
    return models, pd.DataFrame(contrast_rows), multistate, pd.DataFrame(ipw_rows)

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", PROJECT / "results"))
SUBMISSION = Path(os.environ.get("TABLE_OUTPUT_DIR", OUT / "tables"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import charls_hrs_harmonized_analysis as bridge


def weighted_summary(x: pd.Series, w: pd.Series, binary: bool = False) -> tuple[int, str]:
    valid = pd.to_numeric(x, errors="coerce").notna() & pd.to_numeric(w, errors="coerce").gt(0)
    xv = pd.to_numeric(x.loc[valid], errors="coerce").astype(float).to_numpy()
    wv = pd.to_numeric(w.loc[valid], errors="coerce").astype(float).to_numpy()
    if not len(xv):
        return 0, "--"
    mean = float(np.average(xv, weights=wv))
    if binary:
        return len(xv), f"{100 * mean:.1f}%"
    variance = float(np.average((xv - mean) ** 2, weights=wv))
    return len(xv), f"{mean:.2f} ({np.sqrt(variance):.2f})"


def describe(cohort: pd.DataFrame) -> dict[str, tuple[int, str]]:
    d = cohort.copy()
    threshold = cohort.attrs["cesd_threshold"]
    d["persistent_low_depression"] = np.where(
        d["cesd_first"].notna() & d["cesd_anchor"].notna(),
        ((d["cesd_first"] < threshold) & (d["cesd_anchor"] < threshold)).astype(float),
        np.nan,
    )
    w = d["blood_weight"]
    output = {
        "Age, years": weighted_summary(d["age"], w),
        "Women": weighted_summary(d["female"], w, True),
        "Rural residence": weighted_summary(d["rural"], w, True),
        "Partnered": weighted_summary(d["partnered"], w, True),
        "Current smoking": weighted_summary(d["smoking"], w, True),
        "Current drinking": weighted_summary(d["drinking"], w, True),
        "Comorbidity count": weighted_summary(d["comorbidity"], w),
        "Self-rated health score": weighted_summary(d["self_health"], w),
        "Metabolic abnormalities at anchor, count": weighted_summary(d["met_anchor"], w),
        "CES-D score at anchor": weighted_summary(d["cesd_anchor"], w),
        "Persistently low depressive symptoms": weighted_summary(d["persistent_low_depression"], w, True),
        "Episodic-memory recall score at anchor": weighted_summary(d["memory_anchor"], w),
        "Grip strength at anchor": weighted_summary(d["grip"], w),
        "Social participation": weighted_summary(d["social_participation"], w, True),
    }
    observed = d["durable_composite_independent"].notna()
    sustained = int((d.loc[observed, "durable_composite_independent"] == 1).sum())
    output["Sustained independence, unweighted n/N"] = (
        int(observed.sum()), f"{sustained}/{int(observed.sum())} ({100 * sustained / observed.sum():.1f}%)"
    )
    return output


def main() -> None:
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    cohorts = {"CHARLS": bridge.charls_bridge_cohort(), "HRS": bridge.hrs_bridge_cohort()}
    descriptions = {name: describe(cohort) for name, cohort in cohorts.items()}
    characteristics = list(descriptions["CHARLS"])
    rows = []
    for characteristic in characteristics:
        row = {"characteristic": characteristic}
        for name in ["CHARLS", "HRS"]:
            available_n, value = descriptions[name][characteristic]
            row[f"{name.lower()}_available_n"] = available_n
            row[name.lower()] = value
        rows.append(row)
    table1 = pd.DataFrame(rows)
    table1.to_csv(SUBMISSION / "table1_cohort_characteristics.csv", index=False, encoding="utf-8-sig")

    comparison = pd.read_csv(OUT / "charls_hrs_harmonized_comparison.csv")
    continuous = pd.read_csv(OUT / "charls_hrs_continuous_depression_comparison.csv")
    primary_rows = []
    label = {
        "persistent_low_depression": "Persistently low depressive symptoms",
        "memory_z": "Memory performance, per 1 SD higher",
    }
    for _, row in comparison.iterrows():
        primary_rows.append(
            {
                "analysis": "Confirmatory binary/continuous marker model",
                "outcome": row["outcome"],
                "marker": label[row["variable"]],
                "metric": "OR" if row["outcome"] == "Sustained independence" else "RRR",
                "charls_effect_95ci": f'{row["charls_effect"]:.2f} ({row["charls_ci_low"]:.2f}-{row["charls_ci_high"]:.2f})',
                "charls_p": f'{row["charls_holm_p"]:.3f}',
                "hrs_effect_95ci": f'{row["hrs_effect"]:.2f} ({row["hrs_ci_low"]:.2f}-{row["hrs_ci_high"]:.2f})',
                "hrs_p": f'{row["hrs_holm_p"]:.3f}',
                "between_cohort_p": f'{row["effect_difference_p"]:.3f}',
            }
        )
    for _, row in continuous.iterrows():
        primary_rows.append(
            {
                "analysis": "Continuous depressive-burden sensitivity analysis",
                "outcome": row["outcome"],
                "marker": "Depressive symptom burden, per 1 SD lower",
                "metric": row["effect_metric"],
                "charls_effect_95ci": f'{row["charls_effect"]:.2f} ({row["charls_ci_low"]:.2f}-{row["charls_ci_high"]:.2f})',
                "charls_p": f'{row["charls_p"]:.3f}',
                "hrs_effect_95ci": f'{row["hrs_effect"]:.2f} ({row["hrs_ci_low"]:.2f}-{row["hrs_ci_high"]:.2f})',
                "hrs_p": f'{row["hrs_p"]:.3f}',
                "between_cohort_p": f'{row["effect_difference_p"]:.3f}',
            }
        )
    pd.DataFrame(primary_rows).to_csv(
        SUBMISSION / "table2_cross_cohort_results.csv", index=False, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()

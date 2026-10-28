"""Deployment guardrail for the 60+ AWGS-2019 calculator.

Two checks, both must PASS before pushing:
  1. Artifact identity: the bundled pipeline reproduces the analysis-directory
     deployment model (models/awgs60_deploy_logreg.joblib) value by value on
     the full analytic sample (N=7,976; 2,965 events).
  2. Form-to-model parity: every participant's raw inputs (height, weight,
     waist, interview items ...) are pushed through the same helper functions
     the web form uses (calc_core.py), and the resulting predictions are
     compared with the model's predictions on the analysis features. Residual
     differences come only from float32 storage of anthropometry in the Stata file.

2026-10-28: the final model has 30 predictors; the four physical-performance and
muscle measures are no longer inputs (analysis_plan.md, 修改记录).

Run with the analysis environment (scikit-learn 1.7.2), e.g. `py -3.10 verify_calculator.py`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from calc_core import (EXPECTED_FEATURES, assemble_row, derive_anthropometry,
                       load_model)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "papers/charls_falls_ml/models/awgs60_deploy_logreg.joblib"
SC = ROOT / "CHARLS/derived/charls_2015to2020_screen_clean.dta"
GC = ROOT / "CHARLS/derived/gait_chair_wave3.csv"


def analytic_sample():
    """Replicates code/18_awgs60_final_holdout.py sections 1-2."""
    df = pd.read_stata(SC, convert_categoricals=False)
    df["ID"] = df["ID"].astype(str).str.strip()
    h = pd.read_csv(GC, dtype={"ID": str})
    h = h[h["inw3"] == 1].copy(); h["ID"] = h["ID"].astype(str).str.strip()
    df = df.merge(h, on="ID", how="left")
    df["gait_speed_ms"] = 2.5 / df["r3wspeed"]
    df.loc[(df["gait_speed_ms"] <= 0) | (df["gait_speed_ms"] > 3.0), "gait_speed_ms"] = np.nan
    df["chair_stand_s"] = df["r3chr5sec"]
    df.loc[(df["chair_stand_s"] <= 0) | (df["chair_stand_s"] > 120), "chair_stand_s"] = np.nan
    df.loc[df["chair_stand_s"] < 3.0, "chair_stand_s"] = np.nan
    df.loc[(df["gait_speed_ms"] < 0.1) | (df["gait_speed_ms"] > 2.0), "gait_speed_ms"] = np.nan
    df.loc[df["r3walktryu"] == 1, "gait_speed_ms"] = 0.0
    df.loc[df["r3chr5tryu"] == 1, "chair_stand_s"] = 120.0
    return df[(df["r3agey"] >= 60) & (df["fall_2015_2020"].notna())].copy()


def form_inputs(r):
    """What a user would type for participant r, mapped through calc_core."""
    val = lambda v: None if pd.isna(v) else float(v)
    inp = {k: val(r[k]) for k in EXPECTED_FEATURES if k != "bmi"}
    # BMI from raw height/weight; a derived value the cleaning step set to
    # missing (implausible BMI) is left blank, as a user would
    if pd.notna(r["r3mheight"]) and pd.notna(r["r3mweight"]):
        inp.update(derive_anthropometry(r["r3agey"], r["female"] == 1,
                                        float(r["r3mheight"]) * 100, float(r["r3mweight"])))
        if pd.isna(r["bmi"]):
            inp.pop("bmi", None)
    return inp


def main():
    src, feats = load_model(SOURCE)
    dep, feats_dep = load_model()
    assert feats == feats_dep == EXPECTED_FEATURES, "feature order differs"

    d = analytic_sample()
    y = d["fall_2015_2020"].astype(int)
    assert len(d) == 7976 and int(y.sum()) == 2965, f"sample drift: N={len(d)}, events={int(y.sum())}"

    X = d[feats]
    p_src = src.predict_proba(X)[:, 1]
    p_dep = dep.predict_proba(X)[:, 1]
    d1 = float(np.max(np.abs(p_src - p_dep)))
    assert d1 < 1e-12, f"bundled artifact differs from analysis artifact (max|Δp|={d1:.2e})"
    print(f"PASS 1 artifact identity: N={len(d)}, events={int(y.sum())}, max|Δp|={d1:.2e}")

    rows = pd.concat([assemble_row(form_inputs(r), feats) for _, r in d.iterrows()],
                     ignore_index=True)
    p_form = dep.predict_proba(rows)[:, 1]
    d2 = np.abs(p_form - p_dep)
    assert float(d2.max()) < 1e-4, f"form path diverges (max|Δp|={d2.max():.2e})"
    print(f"PASS 2 form-to-model parity: max|Δp|={d2.max():.2e}, "
          f"mean|Δp|={d2.mean():.2e} over {len(d)} participants")

    print(f"Info: full-data apparent AUROC={roc_auc_score(y, p_dep):.4f} "
          "(never report this; manuscript performance is the hold-out AUROC in Table 2).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

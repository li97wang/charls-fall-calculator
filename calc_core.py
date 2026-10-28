"""
Pure, UI-free helpers shared by app.py and verify_calculator.py.

Every transformation applied to a user's raw inputs lives here, so the
verification script can push the analysis dataset through exactly the same
code path as the web form and compare the result with the fitted pipeline.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "awgs60_deploy_logreg.joblib"

# 2026-10-28: final model = 30 predictors. The four physical-performance and muscle
# measures (grip strength, ASMI, gait speed, chair-stand time) were evaluated in the
# pre-specified comparison (code/68) and added no detectable increment, so they are
# no longer model inputs (code/18_awgs60_final_holdout.py).
EXPECTED_FEATURES = [
    "r3agey", "female", "raeducl", "married", "rural",
    "bmi", "r3mwaist",
    "r3shlta", "r3iadla", "r3adlfive",
    "r3imrc", "r3dlrc", "r3draw", "r3cesd10", "r3drinkev", "r3smokev",
    "baseline_fall_2015",
    "r3hibpe", "r3diabe", "r3cancre", "r3lunge", "r3hearte", "r3stroke", "r3psyche",
    "r3arthre", "r3livere", "r3kidneye", "r3digeste", "r3asthmae", "r3memrye",
]


def load_model(path=MODEL_PATH):
    model = joblib.load(path)
    features = list(model.feature_names_in_)
    if features != EXPECTED_FEATURES:
        raise ValueError("bundled model feature order differs from the manuscript specification")
    return model, features


def derive_anthropometry(age, female, height_cm, weight_kg):
    """Body mass index from height and weight, replicating
    code/13_build_analytic_standard.py (age and sex are kept in the signature
    for call-site compatibility; the muscle-mass index is no longer a model input)."""
    if height_cm is None or weight_kg is None:
        return {}
    h_m = height_cm / 100.0
    return {"bmi": weight_kg / h_m ** 2}


def assemble_row(inputs, features):
    """Map named inputs to the 30-predictor vector; anything left as None
    becomes missing and is handled by the fitted median imputer with
    missingness indicators, exactly as in the deployment fit."""
    row = {f: np.nan for f in features}
    for k, v in inputs.items():
        if k in row and v is not None and not (isinstance(v, float) and np.isnan(v)):
            row[k] = float(v)
    return pd.DataFrame([row])[features]

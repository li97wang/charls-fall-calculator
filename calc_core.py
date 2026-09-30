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

# ---- fixed analysis constants (code/18_awgs60_final_holdout.py) ----
COURSE_M = 2.5            # CHARLS timed walk course length (metres)
GAIT_UNABLE = 0.0         # pre-specified worst value: attempted but unable to walk
CHAIR_UNABLE = 120.0      # pre-specified worst value: attempted but unable to stand 5 times

EXPECTED_FEATURES = [
    "r3agey", "female", "raeducl", "married", "rural",
    "grip_dom", "bmi", "asmi", "r3mwaist", "gait_speed_ms", "chair_stand_s",
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
    """BMI and appendicular skeletal muscle mass index (Wen et al. 2011),
    replicating code/13_build_analytic_standard.py."""
    if height_cm is None or weight_kg is None:
        return {}
    h_m = height_cm / 100.0
    sex_n = 2 if female else 1                          # Wen 2011: male 1, female 2
    asm = 0.193 * weight_kg + 0.107 * height_cm - 4.157 * sex_n - 0.037 * age - 2.631
    return {"bmi": weight_kg / h_m ** 2, "asmi": asm / h_m ** 2}


def gait_speed(walk_time_s, status):
    """status: 'measured' | 'unable' | 'not_measured'. Time is the average
    time in seconds for the 2.5-metre walk."""
    if status == "unable":
        return GAIT_UNABLE
    if status == "measured" and walk_time_s:
        return COURSE_M / walk_time_s
    return None


def chair_stand(seconds, status):
    """status: 'measured' | 'unable' | 'not_measured'. Seconds for five
    repeated chair stands."""
    if status == "unable":
        return CHAIR_UNABLE
    if status == "measured" and seconds:
        return float(seconds)
    return None


def assemble_row(inputs, features):
    """Map named inputs to the 34-predictor vector; anything left as None
    becomes missing and is handled by the fitted median imputer with
    missingness indicators, exactly as in the deployment fit."""
    row = {f: np.nan for f in features}
    for k, v in inputs.items():
        if k in row and v is not None and not (isinstance(v, float) and np.isnan(v)):
            row[k] = float(v)
    return pd.DataFrame([row])[features]

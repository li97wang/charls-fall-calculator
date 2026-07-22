"""
CHARLS 5-Year Fall-Risk Screening Calculator
================================================================
A web-based screening tool that returns an individual's predicted
probability of a fall over ~5 years, using the penalized logistic-
regression model developed on the China Health and Retirement
Longitudinal Study (CHARLS), baseline 2015 -> cumulative 2018/2020 intervals.

The app wraps the corrected fitted pipeline reported after the 2026-07-21
outcome audit. Faithfulness to the analysis artifact is checked by
verify_calculator.py.

RESEARCH / EDUCATIONAL TOOL — NOT a diagnostic device.
================================================================
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "screen_LogReg.joblib"

# --- model facts (from outputs/screen_model_metrics.csv & cleaning_report.md) ---
DEV_N = 16242
PREVALENCE = 0.2977
AUROC = 0.685
AUROC_CI = (0.676, 0.694)
BRIER = 0.188
CALIB_SLOPE, CALIB_INT = 0.942, 0.001
TIER_LOW, TIER_HIGH = 0.20, 0.40

st.set_page_config(page_title="CHARLS Fall-Risk Screening Calculator",
                   page_icon="🦴", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = list(model.feature_names_in_)
    return model, features


model, FEATURES = load_model()


def bmi_category(bmi):
    """Asian WHO cut points, right-closed to match the training pd.cut."""
    if bmi <= 18.5:
        return 0
    if bmi <= 24:
        return 1
    if bmi <= 28:
        return 2
    return 3


def assemble_row(inp: dict) -> pd.DataFrame:
    """Map raw inputs to the prespecified 32-feature vector.

    Missing-value imputation, indicators and scaling are performed inside the
    fitted sklearn pipeline, exactly as during model development.
    """
    row = {f: np.nan for f in FEATURES}
    for k, v in inp.items():
        if k in row and v is not None and not (isinstance(v, float) and np.isnan(v)):
            row[k] = float(v)
    return pd.DataFrame([row])[FEATURES]


def derive_features(age, sex_female, height_cm, weight_kg, grip):
    """Compute BMI, BMI category, ASMI, and AWGS low-grip / low-muscle flags
    from raw anthropometrics, replicating 13_build_analytic_standard.py."""
    d = {}
    h_m = height_cm / 100.0 if height_cm else None
    if h_m and weight_kg:
        bmi = weight_kg / (h_m ** 2)
        d["bmi"] = bmi
        d["bmi_cat"] = bmi_category(bmi)
        sex_n = 2 if sex_female else 1                       # Wen 2011: male 1 / female 2
        asm = 0.193 * weight_kg + 0.107 * height_cm - 4.157 * sex_n - 0.037 * age - 2.631
        asmi = asm / (h_m ** 2)
        d["asmi"] = asmi
        d["low_asm"] = 1.0 if (asmi < 5.7 if sex_female else asmi < 7.0) else 0.0
    if grip is not None:
        d["low_grip"] = 1.0 if (grip < 18 if sex_female else grip < 28) else 0.0
    return d


# ============================ UI ============================
st.title("CHARLS 5-Year Fall-Risk Screening Calculator")
st.caption(
    "Estimates the probability of a self-reported fall over ~5 years in "
    "community-dwelling Chinese adults aged 45+, from the model developed on "
    "CHARLS (baseline 2015 → cumulative 2018 and 2020 interval outcome).")

st.warning(
    "**Research and educational use only — not a diagnostic device and not a "
    "substitute for clinical judgement.** Discrimination is modest "
    f"(AUROC {AUROC}, 95% CI {AUROC_CI[0]}–{AUROC_CI[1]}); the tool is intended "
    "to help *prioritise* who receives a fuller falls assessment, not to rule "
    "falls in or out for any individual.", icon="⚠️")

with st.sidebar:
    st.header("Model card")
    st.markdown(
        f"""
**Outcome** Any self-reported fall in the 2015–2018 or 2018–2020 interval
**Development cohort** CHARLS, N = {DEV_N:,} community-dwelling adults ≥45 y
**Cumulative fall prevalence** {PREVALENCE*100:.1f}%
**Primary model** Logistic regression (32 prespecified predictors)

**Internal validation (5-fold out-of-fold)**
- AUROC {AUROC} (95% CI {AUROC_CI[0]}–{AUROC_CI[1]})
- Brier score {BRIER}
- Calibration slope {CALIB_SLOPE}, intercept {CALIB_INT}

**Intended use** Point-of-care *screening* prioritisation in a comparable
population (Chinese, community-dwelling, ≥45 y).

**Do not use** for individual diagnosis, for populations outside the
development setting, or as the sole basis for any clinical decision.

_The bundled full-data pipeline is verified against the corrected analysis artifact._
""")
    st.caption("Citation: [manuscript reference to be inserted on acceptance].")

st.subheader("Enter baseline information")
st.caption("Leave a numeric field blank if a measurement is unavailable — it "
           "will be handled by the model's missing-data rule. ⭐ marks the "
           "muscle/anthropometric inputs that drive the sarcopenia signal.")

with st.form("calc"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Demographics**")
        age = st.number_input("Age (years)", 45, 110, 65, 1)
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
        sex_female = (sex == "Female")
        edu = st.selectbox("Education", ["Low (≤ primary)", "Middle", "High (≥ tertiary)"])
        edu_code = {"Low (≤ primary)": 1, "Middle": 2, "High (≥ tertiary)": 3}[edu]
        married = 1.0 if st.radio("Married / partnered", ["Yes", "No"],
                                  horizontal=True) == "Yes" else 0.0
        rural = 1.0 if st.radio("Residence", ["Rural", "Urban"],
                                horizontal=True) == "Rural" else 0.0

        st.markdown("**⭐ Anthropometry & muscle**")
        height = st.number_input("Height (cm)", 120.0, 210.0, value=None,
                                 step=0.5, format="%.1f")
        weight = st.number_input("Weight (kg)", 25.0, 180.0, value=None,
                                 step=0.5, format="%.1f")
        grip = st.number_input("⭐ Grip strength, dominant hand (kg)", 5.0, 90.0,
                               value=None, step=0.5, format="%.1f",
                               help="AWGS 2019 low grip: <28 kg men, <18 kg women.")
        waist = st.number_input("Waist circumference (cm)", 40.0, 150.0,
                                value=None, step=0.5, format="%.1f")

    with c2:
        st.markdown("**Functional status**")
        srh = st.selectbox("Self-rated health",
                           ["1 Excellent", "2 Very good", "3 Good",
                            "4 Fair", "5 Poor"], index=2)
        srh_code = int(srh[0])
        adl = st.number_input("ADL limitations (count)", 0, 6, 0, 1,
                              help="Basic activities of daily living.")
        iadl = st.number_input("IADL limitations (count)", 0, 5, 0, 1,
                               help="Instrumental activities of daily living.")
        adl5 = st.number_input("Five-item ADL limitations (count)", 0, 5, 0, 1)

        st.markdown("**Mood & behaviours**")
        cesd = st.number_input("CES-D-10 depression score (0–30)", 0, 30, 6, 1)
        drink = 1.0 if st.radio("Ever drinks alcohol", ["Yes", "No"],
                                horizontal=True, index=1) == "Yes" else 0.0
        smoke = 1.0 if st.radio("Ever smoked", ["Yes", "No"],
                                horizontal=True, index=1) == "Yes" else 0.0

    with c3:
        st.markdown("**Cognition**")
        imrc = st.number_input("Immediate word recall (0–10)", 0, 10, value=None, step=1)
        dlrc = st.number_input("Delayed word recall (0–10)", 0, 10, value=None, step=1)
        draw = st.radio("Figure drawing correct", ["Yes", "No", "Unknown"],
                        horizontal=True, index=2)
        draw_val = {"Yes": 1.0, "No": 0.0, "Unknown": None}[draw]

        st.markdown("**History**")
        prior_fall = 1.0 if st.radio("Fell in the past (baseline)",
                                     ["Yes", "No"], horizontal=True,
                                     index=1) == "Yes" else 0.0

        st.markdown("**Physician-diagnosed chronic conditions**")
        chronic_labels = {
            "r3hibpe": "Hypertension", "r3diabe": "Diabetes",
            "r3cancre": "Cancer", "r3lunge": "Chronic lung disease",
            "r3hearte": "Heart disease", "r3stroke": "Stroke",
            "r3psyche": "Psychiatric problem", "r3arthre": "Arthritis",
            "r3livere": "Liver disease", "r3kidneye": "Kidney disease",
            "r3digeste": "Digestive disease", "r3asthmae": "Asthma",
            "r3memrye": "Memory-related disease"}
        chronic_vals = {}
        for key, lab in chronic_labels.items():
            chronic_vals[key] = 1.0 if st.checkbox(lab) else 0.0

    submitted = st.form_submit_button("Estimate fall risk", type="primary",
                                      use_container_width=True)

if submitted:
    inp = {
        "r3agey": age, "female": 1.0 if sex_female else 0.0,
        "raeducl": edu_code, "married": married, "rural": rural,
        "grip_dom": grip, "r3mwaist": waist,
        "r3shlta": srh_code, "r3adla_c": adl, "r3iadla": iadl, "r3adlfive": adl5,
        "r3cesd10": cesd, "r3drinkev": drink, "r3smokev": smoke,
        "r3imrc": imrc, "r3dlrc": dlrc, "r3draw": draw_val,
        "baseline_fall_2015": prior_fall,
    }
    inp.update(chronic_vals)
    inp.update(derive_features(age, sex_female, height, weight, grip))

    X = assemble_row(inp)
    prob = float(model.predict_proba(X)[0, 1])
    pct_pop = prob / PREVALENCE

    st.divider()
    r1, r2 = st.columns([1, 1.4])
    with r1:
        st.metric("Predicted 5-year fall probability", f"{prob*100:.0f}%",
                  f"{(prob-PREVALENCE)*100:+.0f} pts vs population ({PREVALENCE*100:.0f}%)",
                  delta_color="inverse")
        st.progress(min(prob / 0.6, 1.0))
    with r2:
        if prob < TIER_LOW:
            st.success(f"**Lower than average risk** (below the population's "
                       f"bottom third, < {TIER_LOW*100:.0f}%). Routine care; "
                       "re-screen periodically.", icon="🟢")
        elif prob < TIER_HIGH:
            st.warning("**Around average risk.** Consider standard falls advice "
                       "and review modifiable factors (strength, balance, "
                       "vision, medications).", icon="🟡")
        else:
            st.error(f"**Elevated risk** (top ~20% of the population, "
                     f"≥ {TIER_HIGH*100:.0f}%). Prioritise for a fuller "
                     "multifactorial falls assessment.", icon="🔴")
        st.caption(
            f"This estimate is about **{pct_pop:.1f}×** the average 5-year fall "
            f"risk in the development cohort. The displayed risk bands are "
            "descriptive and are not validated treatment thresholds — useful for "
            "prioritisation, not for diagnosis.")

    with st.expander("How to read this / limitations"):
        st.markdown(
            f"""
- **Probability, not certainty.** A {prob*100:.0f}% estimate means that among
  people with this profile, roughly {prob*100:.0f} in 100 reported a fall over
  ~5 years — most of the *risk* difference between people is not captured
  (AUROC {AUROC}).
- **Calibration.** Predicted risks track observed risk reasonably well overall
  (slope {CALIB_SLOPE}, intercept {CALIB_INT}) but individual estimates carry
  uncertainty.
- **Scope.** Developed only in Chinese community-dwelling adults ≥45 y; not
  validated elsewhere or for clinical/inpatient populations.
- **Missing inputs** are handled by the fitted fold-safe preprocessing pipeline;
  entering the muscle/anthropometric values
  (⭐) gives the most individualised estimate.
""")

st.divider()
st.caption("Model: logistic regression, CHARLS 2015→2020 cumulative outcome. "
           "Performance is internally validated with out-of-fold predictions; no external validation. "
           "Source: papers/charls_falls_ml. © Research use only.")

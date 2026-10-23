"""
CHARLS 5-Year Fall-Risk Research Calculator (community-dwelling adults aged 60+)
================================================================
Returns the predicted probability of any self-reported fall over 2015-2020,
from the logistic-regression model with the complete AWGS-2019 construct
(grip strength, muscle-mass index, gait speed, five-repetition chair stand)
developed on the China Health and Retirement Longitudinal Study (CHARLS).

The bundled pipeline is the deployment refit on all 7,976 participants
(median imputation with missingness indicators, scaling, logistic regression).
Artifact identity and form-to-model parity are checked by verify_calculator.py.

RESEARCH / EDUCATIONAL TOOL - NOT a diagnostic device.
================================================================
"""
import streamlit as st

from calc_core import assemble_row, chair_stand, derive_anthropometry, gait_speed, load_model

# ---- model facts: hold-out test set of the development pipeline (manuscript Table 2) ----
DEV_N = 7976
EVENTS = 2965
PREVALENCE = 0.3717
TEST_N = 2393
AUROC, AUROC_CI = 0.686, (0.662, 0.706)
BRIER = 0.210
CALIB_SLOPE, CALIB_SLOPE_CI, CALIB_INT = 0.84, (0.73, 0.96), -0.01
# descriptive probability bands and observed fall frequency in the test set
BANDS = [(0.20, "below 20%", 0.167), (0.30, "20% to below 30%", 0.297),
         (0.40, "30% to below 40%", 0.322), (1.01, "40% or above", 0.552)]

st.set_page_config(page_title="CHARLS Fall-Risk Research Calculator",
                   page_icon="🦴", layout="wide")


@st.cache_resource
def get_model():
    return load_model()


model, FEATURES = get_model()

st.title("CHARLS 5-Year Fall-Risk Research Calculator")
st.caption(
    "Estimates the probability of any self-reported fall over five years in "
    "community-dwelling Chinese adults aged 60 years and older, using the model "
    "developed on CHARLS (2015 baseline; cumulative 2015–2018 and 2018–2020 "
    "interval outcome).")

st.warning(
    "**Research and educational use only — not a diagnostic device and not a "
    "substitute for clinical judgement.** Discrimination is moderate "
    f"(hold-out AUROC {AUROC}, 95% CI {AUROC_CI[0]}–{AUROC_CI[1]}) and the model "
    "has not been externally or prospectively validated. It may help decide who "
    "is prioritised for a fuller falls assessment; it cannot rule falls in or out "
    "for any individual.", icon="⚠️")

with st.sidebar:
    st.header("Model card")
    st.markdown(
        f"""
**Population** Community-dwelling adults aged ≥60 years, CHARLS 2015 baseline
**Outcome** Any self-reported fall in 2015–2018 or 2018–2020
**Development data** N = {DEV_N:,}; {EVENTS:,} fallers ({PREVALENCE*100:.1f}%)
**Model** Logistic regression, 34 pre-specified predictors including the
complete AWGS-2019 construct

**Hold-out test set** (n = {TEST_N:,}; 70/30 split stratified on outcome)
- AUROC {AUROC} (95% CI {AUROC_CI[0]}–{AUROC_CI[1]})
- Brier score {BRIER}
- Calibration slope {CALIB_SLOPE} ({CALIB_SLOPE_CI[0]}–{CALIB_SLOPE_CI[1]}), intercept {CALIB_INT}

The calculator runs the same model refitted on all {DEV_N:,} participants
with median imputation and missingness indicators; the figures above come
from the hold-out evaluation, not from this full-data refit.

**Do not use** for diagnosis, for people younger than 60, for populations
outside the development setting, or as the sole basis for any intervention.
""")
    st.caption("Citation: Zhang J, Wang H. Manuscript under review; the reference "
               "will be added on publication.")

st.subheader("Enter baseline information")
st.caption("Leave a field blank if the measurement is unavailable; it is then "
           "handled by the model's missing-data rule. Entering the physical "
           "measures gives the most individualised estimate.")

with st.form("calc"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Demographics**")
        age = st.number_input("Age (years)", 60, 110, 68, 1)
        sex_female = st.radio("Sex", ["Female", "Male"], horizontal=True) == "Female"
        edu = st.selectbox("Education", ["Below lower secondary",
                                         "Upper secondary or vocational",
                                         "Tertiary"])
        edu_code = {"Below lower secondary": 1, "Upper secondary or vocational": 2,
                    "Tertiary": 3}[edu]
        married = 1.0 if st.radio("Married or cohabiting", ["Yes", "No"],
                                  horizontal=True) == "Yes" else 0.0
        rural = 1.0 if st.radio("Residence", ["Rural", "Urban"],
                                horizontal=True) == "Rural" else 0.0

        st.markdown("**Anthropometry and muscle strength**")
        height = st.number_input("Height (cm)", 120.0, 210.0, value=None,
                                 step=0.5, format="%.1f")
        weight = st.number_input("Weight (kg)", 25.0, 180.0, value=None,
                                 step=0.5, format="%.1f")
        waist = st.number_input("Waist circumference (cm)", 40.0, 150.0,
                                value=None, step=0.5, format="%.1f")
        grip = st.number_input("Grip strength, dominant hand (kg)", 5.0, 90.0,
                               value=None, step=0.5, format="%.1f",
                               help="AWGS-2019 low grip strength: <28 kg in men, <18 kg in women.")

    with c2:
        st.markdown("**Physical performance**")
        gait_status = st.radio("Timed 2.5-metre walk",
                               ["Measured", "Attempted but unable", "Not measured"],
                               horizontal=True, index=2)
        walk_time = st.number_input("Walk time for 2.5 m (seconds, average of two trials)",
                                    1.25, 25.0, value=None, step=0.1, format="%.2f",
                                    help="Gait speed = 2.5 / time. AWGS-2019 low gait speed: <1.0 m/s.")
        chair_status = st.radio("Five repeated chair stands",
                                ["Measured", "Attempted but unable", "Not measured"],
                                horizontal=True, index=2)
        chair_time = st.number_input("Time for five chair stands (seconds)",
                                     3.0, 60.0, value=None, step=0.1, format="%.1f",
                                     help="AWGS-2019 low physical performance: ≥12 s.")

        st.markdown("**Functional status**")
        srh = st.selectbox("Self-rated health",
                           ["1 Very good", "2 Good", "3 Fair", "4 Poor", "5 Very poor"],
                           index=2)
        adl5 = st.number_input("Five-item ADL: number of activities with difficulty (0–5)",
                               0, 5, 0, 1,
                               help="Bathing, dressing, eating, getting in or out of bed, toileting.")
        iadl = st.number_input("IADL: number of activities with difficulty (0–3)",
                               0, 3, 0, 1,
                               help="Harmonized CHARLS three-item IADL summary.")

    with c3:
        st.markdown("**Mood, cognition and behaviours**")
        cesd = st.number_input("CES-D-10 depressive symptoms score (0–30)", 0, 30,
                               value=None, step=1)
        imrc = st.number_input("Immediate word recall (0–10)", 0, 10, value=None, step=1)
        dlrc = st.number_input("Delayed word recall (0–10)", 0, 10, value=None, step=1)
        draw = st.radio("Figure drawing correct", ["Yes", "No", "Unknown"],
                        horizontal=True, index=2)
        drink = 1.0 if st.radio("Ever drunk alcohol", ["Yes", "No"],
                                horizontal=True, index=1) == "Yes" else 0.0
        smoke = 1.0 if st.radio("Ever smoked", ["Yes", "No"],
                                horizontal=True, index=1) == "Yes" else 0.0

        st.markdown("**Fall history**")
        prior_fall = 1.0 if st.radio("Previous fall (baseline fall history)",
                                     ["Yes", "No"], horizontal=True,
                                     index=1) == "Yes" else 0.0

        st.markdown("**Physician-diagnosed chronic conditions**")
        chronic_labels = {
            "r3hibpe": "Hypertension", "r3diabe": "Diabetes",
            "r3cancre": "Cancer", "r3lunge": "Chronic lung disease",
            "r3hearte": "Heart disease", "r3stroke": "Stroke",
            "r3psyche": "Emotional or psychiatric problem", "r3arthre": "Arthritis or rheumatism",
            "r3livere": "Liver disease", "r3kidneye": "Kidney disease",
            "r3digeste": "Digestive disease", "r3asthmae": "Asthma",
            "r3memrye": "Memory-related disease"}
        chronic_vals = {k: (1.0 if st.checkbox(lab) else 0.0)
                        for k, lab in chronic_labels.items()}

    submitted = st.form_submit_button("Estimate fall risk", type="primary",
                                      use_container_width=True)

if submitted:
    status = {"Measured": "measured", "Attempted but unable": "unable",
              "Not measured": "not_measured"}
    inputs = {
        "r3agey": age, "female": 1.0 if sex_female else 0.0,
        "raeducl": edu_code, "married": married, "rural": rural,
        "grip_dom": grip, "r3mwaist": waist,
        "gait_speed_ms": gait_speed(walk_time, status[gait_status]),
        "chair_stand_s": chair_stand(chair_time, status[chair_status]),
        "r3shlta": int(srh[0]), "r3iadla": iadl, "r3adlfive": adl5,
        "r3imrc": imrc, "r3dlrc": dlrc,
        "r3draw": {"Yes": 1.0, "No": 0.0, "Unknown": None}[draw],
        "r3cesd10": cesd, "r3drinkev": drink, "r3smokev": smoke,
        "baseline_fall_2015": prior_fall,
        **chronic_vals,
        **derive_anthropometry(age, sex_female, height, weight),
    }
    prob = float(model.predict_proba(assemble_row(inputs, FEATURES))[0, 1])
    band = next(b for b in BANDS if prob < b[0])

    st.divider()
    r1, r2 = st.columns([1, 1.4])
    with r1:
        st.metric("Predicted 5-year fall probability", f"{prob*100:.0f}%",
                  f"{(prob-PREVALENCE)*100:+.0f} points vs development cohort "
                  f"({PREVALENCE*100:.0f}%)", delta_color="inverse")
        st.progress(min(prob / 0.8, 1.0))
    with r2:
        msg = (f"**Predicted-risk band: {band[1]}.** In the hold-out test set, "
               f"{band[2]*100:.1f}% of people in this band reported a fall over five years.")
        if prob < 0.20:
            st.success(msg, icon="🟢")
        elif prob < 0.40:
            st.warning(msg, icon="🟡")
        else:
            st.error(msg + " People in this band may be prioritised for a fuller "
                     "multifactorial falls assessment.", icon="🔴")
        st.caption("The four bands are descriptive, not validated intervention "
                   "thresholds; the two middle bands were poorly separated in the "
                   "test set (29.7% and 32.2%).")

    with st.expander("How to read this estimate"):
        st.markdown(
            f"""
- **Probability, not certainty.** A {prob*100:.0f}% estimate means that roughly
  {prob*100:.0f} in 100 people with a similar profile reported a fall over five
  years. With an AUROC of {AUROC}, much of the difference in risk between
  individuals is not captured.
- **Previous falls dominate.** Fall history is by far the strongest input; the
  model is suited to prioritising screening in the general older population,
  not to precise prediction of a first fall.
- **Calibration.** Predicted and observed risks agreed well overall in the
  hold-out test set (slope {CALIB_SLOPE}, intercept {CALIB_INT}), but the model
  separated people poorly in the middle of the risk range.
- **Scope.** Developed only in Chinese community-dwelling adults aged ≥60 years;
  not validated elsewhere or in clinical or inpatient populations.
""")

st.divider()
st.caption("Model: logistic regression, CHARLS 2015→2020 cumulative outcome, adults "
           "aged ≥60 years. Performance from an untouched hold-out test set; no "
           "external validation. Research use only. No entered data are stored.")

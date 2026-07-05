"""
test_app.py — validate the calculator's feature derivation and end-to-end path.
Run: py -3.12 test_app.py
"""
import sys, warnings, importlib
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

# import the app module (loads the model at import time)
sys.path.insert(0, str(Path(__file__).resolve().parent))
app = importlib.import_module("app")

DATA = Path(r"D:\academic\xy\CHARLS\derived\charls_2015to2020_screen_clean.dta")


def test_derivations():
    """App-side BMI/ASMI/AWGS flags must match the dataset build."""
    df = pd.read_stata(DATA, convert_categoricals=False)
    df = df[df["fall_2020"].notna()].copy()
    m = df[["r3mheight", "r3mweight", "grip_dom", "r3agey", "female",
            "bmi", "bmi_cat", "asmi", "low_asm", "low_grip"]].dropna(
            subset=["r3mheight", "r3mweight", "grip_dom"])
    sample = m.sample(400, random_state=1)
    err = {"bmi": 0.0, "asmi": 0.0}; flag_bad = 0
    for _, r in sample.iterrows():
        d = app.derive_features(r["r3agey"], bool(r["female"]),
                                r["r3mheight"] * 100.0, r["r3mweight"], r["grip_dom"])
        err["bmi"] = max(err["bmi"], abs(d["bmi"] - r["bmi"]))
        err["asmi"] = max(err["asmi"], abs(d["asmi"] - r["asmi"]))
        if d["bmi_cat"] != r["bmi_cat"]: flag_bad += 1
        if d["low_asm"] != r["low_asm"]: flag_bad += 1
        if d["low_grip"] != r["low_grip"]: flag_bad += 1
    print(f"[derive] n=400  max|Δbmi|={err['bmi']:.2e}  max|Δasmi|={err['asmi']:.2e}  "
          f"flag mismatches={flag_bad}")
    # reference bmi/asmi are stored as Stata float32 (~7 sig figs), so agreement
    # is limited to float32 rounding (~1e-5 at bmi≈25); flags must match exactly.
    assert err["bmi"] < 1e-4 and err["asmi"] < 1e-4 and flag_bad == 0
    return True


def predict(persona):
    base = {"r3agey": 65, "female": 0.0, "raeducl": 1, "married": 1.0, "rural": 1.0,
            "grip_dom": None, "r3mwaist": None, "pa_ipaq_cat_2015": 0,
            "pa_met_total_2015": app.IPAQ_MET[0], "r3shlta": 3, "r3adla_c": 0,
            "r3iadla": 0, "r3adlfive": 0, "r3cesd10": 6, "r3drinkev": 0.0,
            "r3smokev": 0.0, "r3imrc": None, "r3dlrc": None, "r3draw": None,
            "baseline_fall_2015": 0.0}
    for k in app.MODEL_PATH and ["r3hibpe","r3diabe","r3cancre","r3lunge","r3hearte",
              "r3stroke","r3psyche","r3arthre","r3livere","r3kidneye","r3digeste",
              "r3asthmae","r3memrye"]:
        base[k] = 0.0
    base.update(persona)
    # derive anthropometrics if height/weight present
    if persona.get("_height") and persona.get("_weight"):
        base.update(app.derive_features(base["r3agey"], bool(base["female"]),
                    persona["_height"], persona["_weight"], base.get("grip_dom")))
    X = app.assemble_row(base)
    return float(app.model.predict_proba(X)[0, 1])


def test_end_to_end():
    low = predict({"r3agey": 50, "female": 0.0, "grip_dom": 45.0,
                   "_height": 172, "_weight": 68, "pa_ipaq_cat_2015": 2,
                   "pa_met_total_2015": app.IPAQ_MET[2], "r3shlta": 1})
    high = predict({"r3agey": 82, "female": 1.0, "grip_dom": 14.0,
                    "_height": 150, "_weight": 40, "baseline_fall_2015": 1.0,
                    "r3shlta": 5, "r3adla_c": 4, "r3iadla": 3, "r3cesd10": 20,
                    "r3arthre": 1.0, "r3stroke": 1.0})
    print(f"[e2e] fit 50yo active man = {low*100:.0f}%   "
          f"frail 82yo faller woman = {high*100:.0f}%")
    assert 0.0 < low < high < 1.0 and low < app.PREVALENCE < high
    return True


if __name__ == "__main__":
    ok = test_derivations() and test_end_to_end()
    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

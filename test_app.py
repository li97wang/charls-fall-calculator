"""Smoke tests for corrected deployment artifact without executing Streamlit UI."""
from pathlib import Path
import joblib, numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent
model=joblib.load(HERE/"models/screen_LogReg.joblib"); FEATURES=list(model.feature_names_in_)
def row(values):
    d={f:np.nan for f in FEATURES}; d.update({k:v for k,v in values.items() if k in d})
    return pd.DataFrame([d])[FEATURES]
base={f:0 for f in FEATURES}
base.update({"r3agey":60,"raeducl":2,"married":1,"grip_dom":32,"bmi":24,"asmi":6.8,"r3mwaist":86,
             "r3shlta":3,"r3imrc":4,"r3dlrc":3,"r3draw":1,"r3cesd10":6})
low_values=base|{"r3agey":50,"female":0,"baseline_fall_2015":0,"grip_dom":45,"r3adlfive":0,"r3cesd10":3}
high_values=base|{"r3agey":82,"female":1,"baseline_fall_2015":1,"grip_dom":14,"r3adlfive":4,"r3iadla":3,"r3cesd10":20,"r3arthre":1,"r3stroke":1}
low=row(low_values); high=row(high_values)
p_low=float(model.predict_proba(low)[0,1]); p_high=float(model.predict_proba(high)[0,1])
assert 0<p_low<p_high<1 and len(FEATURES)==32
print(f"PASS: features=32, low-risk persona={p_low:.3f}, high-risk persona={p_high:.3f}")

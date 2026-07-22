"""Deployment guardrail for the corrected post-audit full-data pipeline."""
import sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
SOURCE=ROOT/"papers/charls_falls_ml/models/screen_LogReg.joblib"
BUNDLE=HERE/"models/screen_LogReg.joblib"
DATA=ROOT/"CHARLS/derived/charls_2015to2020_screen_clean.dta"

def main():
    src=joblib.load(SOURCE); dep=joblib.load(BUNDLE)
    features=list(src.feature_names_in_)
    assert features==list(dep.feature_names_in_),"bundle feature order differs"
    assert features==["r3agey","female","raeducl","married","rural","grip_dom","bmi","asmi","r3mwaist",
      "r3shlta","r3iadla","r3adlfive","r3imrc","r3dlrc","r3draw","r3cesd10","r3drinkev","r3smokev",
      "baseline_fall_2015","r3hibpe","r3diabe","r3cancre","r3lunge","r3hearte","r3stroke","r3psyche",
      "r3arthre","r3livere","r3kidneye","r3digeste","r3asthmae","r3memrye"]
    df=pd.read_stata(DATA,convert_categoricals=False); df=df[df.fall_2015_2020.notna()].copy()
    X=df[features]; y=df.fall_2015_2020.astype(int)
    p1=src.predict_proba(X)[:,1]; p2=dep.predict_proba(X)[:,1]
    delta=float(np.max(np.abs(p1-p2))); auc=roc_auc_score(y,p2)
    assert delta<1e-12 and len(df)==16242 and int(y.sum())==4835
    print(f"PASS: N={len(df)}, events={int(y.sum())}, artifact max|Δp|={delta:.2e}, full-fit apparent AUROC={auc:.4f}")
    print("Note: reported manuscript performance must use OOF AUROC 0.6852, not this apparent full-fit value.")
    return 0
if __name__=="__main__": sys.exit(main())

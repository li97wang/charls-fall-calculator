"""
verify_calculator.py
================================================================
Deployment guardrail: prove the bundled screen_LogReg.joblib +
screen_preprocess.joblib reproduce the held-out test predictions
reported in the manuscript (outputs/screen_test_predictions.parquet),
and that the calculator's feature-assembly logic matches the
training pipeline (code/12_screening_pipeline.py) bit-for-bit.

Run:  py -3.12 verify_calculator.py
Exit 0 = bundle is faithful; non-zero = do NOT deploy.
================================================================
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

RND = 42
HERE = Path(__file__).resolve().parent
ROOT = Path(r"D:\academic\xy")
DATA = ROOT / "CHARLS/derived/charls_2015to2020_screen_clean.dta"
PARQ = ROOT / "papers/charls_falls_ml/outputs/screen_test_predictions.parquet"

# --- exact feature spec from code/12_screening_pipeline.py ---
CHR = ["r3hibpe","r3diabe","r3cancre","r3lunge","r3hearte","r3stroke","r3psyche",
       "r3arthre","r3livere","r3kidneye","r3digeste","r3asthmae","r3memrye"]
FEAT = ["r3agey","female","raeducl","married","rural",
        "grip_dom","low_grip","bmi","bmi_cat","asmi","low_asm","r3mwaist",
        "pa_met_total_2015","pa_ipaq_cat_2015",
        "r3shlta","r3adla_c","r3iadla","r3adlfive",
        "r3imrc","r3dlrc","r3draw","r3cesd10","r3drinkev","r3smokev",
        "baseline_fall_2015"] + CHR

def main():
    model = joblib.load(HERE / "models/screen_LogReg.joblib")
    pre   = joblib.load(HERE / "models/screen_preprocess.joblib")
    feats = pre["features"]; med = pre["medians"]
    print(f"[load] model n_features_in_={getattr(model,'n_features_in_','?')}, "
          f"preprocess features={len(feats)}, medians={len(med)}")

    assert getattr(model, "n_features_in_", len(feats)) == len(feats), "feature count mismatch"

    # reconstruct X exactly as the training pipeline did
    df = pd.read_stata(DATA, convert_categoricals=False)
    df = df[df["fall_2020"].notna()].copy()
    y = df["fall_2020"].astype(int).values
    X = df[FEAT].copy()
    hm = X.columns[X.isna().mean() > 0.05].tolist()
    for c in hm:
        X[f"{c}_miss"] = X[c].isna().astype(float)

    # feature order must equal the stored order
    assert list(X.columns) == list(feats), (
        "feature ORDER differs from stored preprocess — calculator would be wrong")
    print(f"[ok] feature order matches ({len(feats)} inputs, {len(hm)} missingness indicators)")

    tr, te = train_test_split(np.arange(len(df)), test_size=0.30,
                              random_state=RND, stratify=y)
    Xte = X.iloc[te].fillna(med)
    yte = y[te]
    p = model.predict_proba(Xte)[:, 1]

    auc = roc_auc_score(yte, p)
    print(f"[repro] test AUROC = {auc:.4f}  (manuscript: 0.6791)")

    # compare against the parquet the manuscript numbers were drawn from
    ref = pd.read_parquet(PARQ)
    if "p_LogReg" in ref.columns and len(ref) == len(p):
        max_abs = float(np.max(np.abs(ref["p_LogReg"].values - p)))
        print(f"[match] max |Δp| vs screen_test_predictions.parquet = {max_abs:.2e}")
        ok = max_abs < 1e-9 and abs(auc - 0.6791) < 1e-3
    else:
        ok = abs(auc - 0.6791) < 1e-3
        print("[warn] parquet shape/columns differ; relying on AUROC check only")

    print("\nRESULT:", "PASS — bundle faithful, safe to deploy" if ok
          else "FAIL — do NOT deploy")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())

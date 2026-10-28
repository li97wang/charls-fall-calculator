# Deployment steps

1. Build the environment from `requirements.txt` (Python 3.12; scikit-learn is pinned
   to 1.7.2, the version the model was serialised with).
2. Run `py -3.10 verify_calculator.py` in the analysis environment and `python test_app.py`
   in the requirements environment. Deploy only if every line reports `PASS`.
3. Start locally with `streamlit run app.py` and confirm that the model card reports
   adults aged ≥60, N=7,976, prevalence 37.2%, hold-out AUROC 0.688, 30 predictors
   with no physical-performance test, and no claim of external validation.
4. Push to `main`; Streamlit Community Cloud redeploys automatically. Open the public
   URL and confirm the model card again.

The deployment must contain `app.py`, `calc_core.py`, and `models/awgs60_deploy_logreg.joblib`.
Do not restore the superseded 45+ artifact `screen_LogReg.joblib` (N=16,242) or the
34-predictor artifact archived on 2026-10-28 (`outputs/_archive/pre30var_20261028/`),
which still required grip strength, gait speed and chair-stand time as inputs.

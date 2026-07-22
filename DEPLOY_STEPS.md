# Deployment steps

Use Python 3.12, install packages from `requirements.txt`, and run `py -3.12 verify_calculator.py` followed by `py -3.12 test_app.py`. Deployment is allowed only if both commands return `PASS`. Start the local application with `streamlit run app.py` and confirm that the model card reports N=16,242, prevalence 29.77%, five-fold out-of-fold AUROC 0.685, and no claim of held-out or external validation.

The deployment must contain `app.py` and the corrected `models/screen_LogReg.joblib`. Do not restore the pre-audit `screen_preprocess.joblib` or the archived single-wave model. If the application is already hosted, redeploy this directory or take the old instance offline; changing local files does not update an existing remote deployment automatically.

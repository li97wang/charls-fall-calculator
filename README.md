# CHARLS cumulative-fall research calculator

This directory contains the corrected post-audit logistic-regression pipeline for the 2015–2020 cumulative fall outcome. The development dataset included 16,242 participants and 4,835 events (29.77%). Internal five-fold out-of-fold performance was AUROC 0.685 (95% CI 0.676–0.694), AUPRC 0.495, Brier score 0.188, calibration slope 0.942, and calibration intercept 0.001.

The single bundled artifact, `models/screen_LogReg.joblib`, contains median imputation, missingness indicators, scaling, and logistic regression. It expects the 32 source predictors listed in the fitted pipeline. Physical activity and threshold-derived grip, muscle-mass, and BMI categories are not model inputs.

Run `py -3.12 verify_calculator.py` before deployment. The guardrail verifies feature order, N=16,242, 4,835 events, and exact prediction agreement with the corrected analysis-directory artifact. Run `py -3.12 test_app.py` for a basic low-risk/high-risk persona smoke test. The full-data apparent AUROC must never be reported as model performance; manuscript performance comes from `outputs/screen_oof_predictions.parquet`.

The calculator is for research and educational screening only. It has no independent external or prospective validation and is not a diagnostic device or a sufficient basis for clinical intervention.

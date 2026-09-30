# CHARLS 5-year fall-risk research calculator (adults aged 60+)

Streamlit calculator for the logistic-regression model reported in the manuscript
*Interpretable prediction of fall risk among Chinese community-dwelling older adults*
(Zhang J, Wang H; under review). The model predicts any self-reported fall over
2015–2020 from 34 predictors measured at the 2015 CHARLS baseline, including the
complete AWGS-2019 construct: dominant-hand grip strength, appendicular skeletal
muscle mass index (Wen equation), gait speed from the timed 2.5-metre walk, and
five-repetition chair-stand time. Participants who attempted but could not complete
a physical test are coded at pre-specified worst values (0 m/s; 120 s).

Development data: 7,976 community-dwelling adults aged 60 years and older, 2,965
fallers (37.17%). Performance comes from an untouched hold-out test set (n=2,393)
of the development pipeline: AUROC 0.686 (95% CI 0.662–0.706), Brier score 0.210,
calibration slope 0.84 (0.73–0.96), intercept −0.01. The bundled artifact
`models/awgs60_deploy_logreg.joblib` is the deployment refit on all 7,976
participants (median imputation with missingness indicators, scaling, logistic
regression); its apparent full-data performance is never reported.

`calc_core.py` holds every input transformation used by the web form, so the
verification script can run the analysis data through the same code path.

## Verification before every deployment

    py -3.10 verify_calculator.py   # analysis environment, scikit-learn 1.7.2
    python test_app.py              # environment built from requirements.txt

`verify_calculator.py` checks (1) value-by-value identity between the bundled
artifact and the analysis-directory model on N=7,976 with 2,965 events, and
(2) form-to-model parity: each participant's raw inputs pushed through the form
helpers reproduce the model's predictions (max |Δp| < 1e-4; float32 storage of
anthropometry is the only source of difference). `test_app.py` checks persona
monotonicity and runs the Streamlit form headlessly.

The calculator is for research and educational screening only. It has no external
or prospective validation, is not a diagnostic device, and is not a sufficient
basis for clinical intervention. No entered data are stored.

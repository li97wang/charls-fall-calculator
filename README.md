# CHARLS 5-year fall-risk research calculator (adults aged 60+)

Streamlit calculator for the logistic-regression model reported in the manuscript
*Key predictors of five-year fall risk in community-dwelling older Chinese adults:
development and internal validation of a prediction model using CHARLS*
(Zhang J, Wang H; under review). The model predicts any self-reported fall over
2015–2020 from 30 predictors measured at the 2015 CHARLS baseline: interview items
(fall history, demographics, function, self-rated health, cognition, depressive
symptoms, health behaviours, 13 physician-diagnosed conditions) plus body mass index
and waist circumference. Dominant-hand grip strength, appendicular skeletal muscle
mass index, gait speed and five-repetition chair-stand time were evaluated as
candidates; a pre-specified comparison showed no detectable increment, so they are
not model inputs (2026-10-28).
Every form field is required: leaving an item blank would switch on its missingness
indicator, which encodes non-response in CHARLS rather than the user's own risk and
shifts the estimate (for example, 9.4 percentage points lower on average when delayed
word recall is blank; Supplementary Table S10 of the manuscript).

Development data: 7,976 community-dwelling adults aged 60 years and older, 2,965
fallers (37.17%). Performance comes from a hold-out test set reserved for evaluation (n=2,393)
of the development pipeline: AUROC 0.688 (95% CI 0.664–0.708), Brier score 0.210,
calibration slope 0.86 (0.74–0.98), intercept −0.01. The bundled artifact
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
height and weight is the only source of difference). `test_app.py` checks persona
monotonicity and runs the Streamlit form headlessly.

The calculator is for research and educational screening only. It has no external
or prospective validation, is not a diagnostic device, and is not a sufficient
basis for clinical intervention. No entered data are stored.

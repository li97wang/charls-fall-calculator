---
title: CHARLS Fall-Risk Screening Calculator
emoji: 🦴
colorFrom: indigo
colorTo: green
sdk: streamlit
sdk_version: 1.54.0
app_file: app.py
python_version: "3.12"
pinned: false
license: cc-by-4.0
---

# CHARLS 5-Year Fall-Risk Screening Calculator

A web-based screening tool that returns an individual's predicted probability of
a fall over ~5 years, using the penalised logistic-regression model developed on
the **China Health and Retirement Longitudinal Study (CHARLS)**, baseline 2015 →
outcome 2020 (N = 16,263 community-dwelling adults aged ≥45).

> **Research and educational use only.** Not a diagnostic device. Discrimination
> is modest (AUROC 0.679, 95% CI 0.658–0.700); the tool is meant to help
> *prioritise* who receives a fuller falls assessment, not to rule falls in or
> out for any individual.

## What this app does

It wraps the **exact** fitted primary model and preprocessing pipeline reported
in the manuscript (`models/screen_LogReg.joblib` + `models/screen_preprocess.joblib`)
and exposes it as a form. Raw clinical inputs are mapped to the 62 model inputs
— including the AWGS-2019 sarcopenia flags and ASMI derived from the Wen-2011
anthropometric equation, the Asian-specific BMI category, and the IPAQ activity
category — replicating `code/12_screening_pipeline.py` and
`code/13_build_analytic_standard.py`.

## Faithfulness check

```bash
py -3.12 verify_calculator.py
```

Confirms the bundled model reproduces the manuscript's held-out test predictions
bit-for-bit (max |Δp| = 0, AUROC 0.6791) before any deployment.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (choose one — both free, both need your own account)

### A. Streamlit Community Cloud  → `https://<app>.streamlit.app`
1. Put this `deploy/` folder at the root of a **public GitHub repo**
   (e.g. `charls-fall-calculator`).
2. Sign in at <https://share.streamlit.io> with that GitHub account.
3. **New app** → pick the repo/branch → main file `app.py` →
   Advanced settings → Python **3.12** → **Deploy**.
4. Copy the resulting `*.streamlit.app` URL into Methods §2.8.

### B. Hugging Face Spaces  → `https://huggingface.co/spaces/<user>/<app>`
1. Create a **Streamlit** Space at <https://huggingface.co/new-space>.
2. Push this folder (the YAML front-matter above configures the Space):
   ```bash
   git init && git add . && git commit -m "Fall-risk screening calculator"
   git remote add origin https://huggingface.co/spaces/<user>/<app>
   git push origin main
   ```
3. Copy the Space URL into Methods §2.8.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit calculator UI + faithful feature assembly |
| `models/screen_LogReg.joblib` | Fitted primary model (penalised logistic regression) |
| `models/screen_preprocess.joblib` | Training-set medians + feature order |
| `requirements.txt` | Pinned runtime dependencies |
| `verify_calculator.py` | Reproduction guardrail (run before deploying) |
| `runtime.txt` | Python version pin for platforms that read it |

## Citation

Wang H. *[manuscript title]*. [journal], [year]. — to be completed on acceptance.

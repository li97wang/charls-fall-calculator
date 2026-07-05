# One-time deploy to Streamlit Community Cloud

The `deploy/` folder is already a committed git repo (`git log` shows one commit,
faithful bundle verified: `py -3.12 verify_calculator.py` → PASS, max |Δp| = 0).
Only the browser OAuth step is left — it can only be done under your own GitHub
account, so the three clicks below are yours to make. Budget ~5 minutes.

## Step 1 — put this folder in a PUBLIC GitHub repo

You need a GitHub account. Then, from inside `deploy/`:

```bash
# create an EMPTY public repo on github.com first, e.g. "charls-fall-calculator"
# (no README/license — this folder already has them)
git remote add origin https://github.com/<your-user>/charls-fall-calculator.git
git push -u origin main
```

If you prefer not to use the CLI: on github.com create the empty repo, then
"uploading an existing file" and drag in every file listed by `git ls-files`
(keep the `models/` subfolder).

## Step 2 — deploy on Streamlit Community Cloud

1. Sign in at <https://share.streamlit.io> with that same GitHub account.
2. **Create app → Deploy a public app from GitHub**.
3. Repository = `<your-user>/charls-fall-calculator`, Branch = `main`,
   Main file path = `app.py`.
4. **Advanced settings → Python version = 3.12** (matches `runtime.txt`).
5. **Deploy**. First build takes ~2–3 min while it installs `requirements.txt`.

You get a URL of the form `https://<something>.streamlit.app`.

## Step 3 — send me the URL

Paste it back and I will replace the `⟨STREAMLIT_URL⟩` token already placed in
`chapters/methods.md` §2.8 and `chapters/methods_zh.md`, and clear the matching
`[DATA REQUIRED]` marker in the SOURCE MAP. Methods 2.8 is otherwise finalised.

## Troubleshooting

- **Build fails on a package**: the pins in `requirements.txt` are the exact
  versions the bundle was verified against; do not loosen them.
- **App loads but errors on predict**: run `py -3.12 verify_calculator.py`
  locally — if it still says PASS, the deployed copy is identical and the issue
  is environment-only; then use `/export-diagnose`.
- **"models/ not found"**: confirm `git ls-files` on the pushed repo lists
  `models/screen_LogReg.joblib` and `models/screen_preprocess.joblib` (2 MB total).

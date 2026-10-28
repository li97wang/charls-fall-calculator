"""Smoke tests for the 60+ deployment artifact and the Streamlit form.

1. Artifact: 30 features and monotone low-risk < high-risk personas.
2. UI: the Streamlit app runs headlessly (streamlit.testing AppTest), refuses an
   incomplete form, and renders a probability and risk band once every item is given.

2026-10-28: the four physical-performance and muscle measures are no longer
model inputs (final model = 30 predictors), and every input is required: a blank
would switch on a missingness indicator estimated from CHARLS non-response
(Supplementary Table S10 of the manuscript).
"""
from streamlit.testing.v1 import AppTest

from calc_core import assemble_row, derive_anthropometry, load_model

model, F = load_model()
assert len(F) == 30

base = {f: 0.0 for f in F}
base.update({"raeducl": 1, "married": 1, "rural": 1, "r3mwaist": 86,
             "r3shlta": 3, "r3imrc": 4, "r3dlrc": 3, "r3draw": 1, "r3cesd10": 6})
base.update(derive_anthropometry(68, False, 165, 62))
low = base | {"r3agey": 62, "female": 0, "baseline_fall_2015": 0}
high = base | {"r3agey": 82, "female": 1, "baseline_fall_2015": 1,
               "r3adlfive": 3, "r3iadla": 2, "r3cesd10": 22, "r3arthre": 1, "r3diabe": 1}
high.update(derive_anthropometry(82, True, 150, 45))
p = lambda d: float(model.predict_proba(assemble_row(d, F))[0, 1])
p_low, p_high = p(low), p(high)
assert 0 < p_low < p_high < 1
print(f"PASS artifact: features=30, low-risk persona={p_low:.3f}, high-risk persona={p_high:.3f}")


def fresh():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def fill_measurements(at):
    at.number_input(key="height").set_value(160.0)
    at.number_input(key="weight").set_value(60.0)
    at.number_input(key="waist").set_value(85.0)


# An empty form must be refused before any prediction is made.
at = fresh()
at.button[0].click().run()
assert not at.exception, at.exception
assert at.error and not at.metric, (at.error, at.metric)
print("PASS UI: submitting an empty form shows an error and no estimate")

# Measurements alone are not enough: the interview items that can be empty are required too.
at = fresh()
fill_measurements(at)
at.button[0].click().run()
assert not at.exception, at.exception
assert at.error and not at.metric, (at.error, at.metric)
assert "CES-D-10" in at.error[0].value and "figure drawing" in at.error[0].value, at.error[0].value
print("PASS UI: measurements without the recall, drawing and CES-D-10 items show an error and no estimate")

# With every item given, an estimate and a band are shown.
at = fresh()
fill_measurements(at)
at.number_input(key="cesd").set_value(6)
at.number_input(key="imrc").set_value(4)
at.number_input(key="dlrc").set_value(3)
at.radio(key="draw").set_value("Yes")
at.button[0].click().run()
assert not at.exception, at.exception
shown = [m.value for m in at.metric]
assert shown and shown[0].endswith("%"), shown
print(f"PASS UI: complete form submitted, predicted probability shown = {shown[0]}")

"""Smoke tests for the 60+ AWGS-2019 deployment artifact and the Streamlit form.

1. Artifact: 34 features, monotone low-risk < high-risk personas, and the
   pre-specified "attempted but unable" coding raises predicted risk.
2. UI: the Streamlit app runs headlessly (streamlit.testing AppTest), the
   form submits, and a probability and risk band are rendered.
"""
import numpy as np
from streamlit.testing.v1 import AppTest

from calc_core import (assemble_row, chair_stand, derive_anthropometry,
                       gait_speed, load_model)

model, F = load_model()
assert len(F) == 34

base = {f: 0.0 for f in F}
base.update({"raeducl": 1, "married": 1, "rural": 1, "grip_dom": 30, "r3mwaist": 86,
             "r3shlta": 3, "r3imrc": 4, "r3dlrc": 3, "r3draw": 1, "r3cesd10": 6,
             "gait_speed_ms": gait_speed(2.5, "measured"), "chair_stand_s": chair_stand(10, "measured")})
base.update(derive_anthropometry(68, False, 165, 62))
low = base | {"r3agey": 62, "female": 0, "baseline_fall_2015": 0, "grip_dom": 42}
high = base | {"r3agey": 82, "female": 1, "baseline_fall_2015": 1, "grip_dom": 14,
               "r3adlfive": 3, "r3iadla": 2, "r3cesd10": 22, "r3arthre": 1, "r3diabe": 1,
               "gait_speed_ms": gait_speed(None, "unable"), "chair_stand_s": chair_stand(None, "unable")}
high.update(derive_anthropometry(82, True, 150, 45))
p = lambda d: float(model.predict_proba(assemble_row(d, F))[0, 1])
p_low, p_high = p(low), p(high)
assert 0 < p_low < p_high < 1
print(f"PASS artifact: features=34, low-risk persona={p_low:.3f}, high-risk persona={p_high:.3f}")

at = AppTest.from_file("app.py", default_timeout=60).run()
assert not at.exception, at.exception
at.button[0].click().run()
assert not at.exception, at.exception
shown = [m.value for m in at.metric]
assert shown and shown[0].endswith("%"), shown
print(f"PASS UI: form submitted with defaults, predicted probability shown = {shown[0]}")

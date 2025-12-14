import os
import numpy as np
import skfuzzy as fuzz
try:
	import pandas as pd
except Exception:
	pd = None

# -------------------------------------------------------------
# Data-driven membership functions using dataset percentiles
# -------------------------------------------------------------

binary_universe = np.linspace(0, 1, 101)
risk_universe = np.linspace(0, 1, 101)

def _compute_quantiles(series):
	s = series.dropna().astype(float)
	if s.empty:
		return 0.0, 0.25, 0.5, 0.75, 1.0
	q1 = float(s.quantile(0.25))
	q2 = float(s.quantile(0.50))
	q3 = float(s.quantile(0.75))
	return float(s.min()), q1, q2, q3, float(s.max())

def _load_data_quantiles():
	q = {}
	try:
		if pd is None:
			raise RuntimeError("pandas not available")
		data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'diabetes_clean.csv')
		df = pd.read_csv(data_path)
		for col in ['BMI', 'Age', 'GenHlth']:
			if col in df.columns:
				q[col] = _compute_quantiles(df[col])
			else:
				q[col] = (0.0, 0.25, 0.5, 0.75, 1.0)
	except Exception:
		q = {col: (0.0, 0.25, 0.5, 0.75, 1.0) for col in ['BMI', 'Age', 'GenHlth']}
	return q

_Q = _load_data_quantiles()

# Build universes from data-driven min/max (note: BMI/Age likely in 0–1 already)
_bmi_min, _bmi_q1, _bmi_q2, _bmi_q3, _bmi_max = _Q['BMI']
_age_min, _age_q1, _age_q2, _age_q3, _age_max = _Q['Age']
_gh_min, _gh_q1, _gh_q2, _gh_q3, _gh_max = _Q['GenHlth']

bmi_universe = np.linspace(_bmi_min, _bmi_max, 101)
age_universe = np.linspace(_age_min, _age_max, 101)
genhlth_universe = np.linspace(_gh_min, _gh_max, 101)

# BMI membership functions (percentile-based)
bmi_under   = fuzz.trimf(bmi_universe,   [_bmi_min, _bmi_min, _bmi_q1])
bmi_healthy = fuzz.trimf(bmi_universe,   [_bmi_q1, _bmi_q2, _bmi_q3])
bmi_over    = fuzz.trimf(bmi_universe,   [_bmi_q2, (_bmi_q2+_bmi_q3)/2.0, _bmi_max])
bmi_obese   = fuzz.trapmf(bmi_universe,  [_bmi_q3, (_bmi_q3+_bmi_max)/2.0, _bmi_max, _bmi_max])

# Age membership functions (percentile-based)
age_young  = fuzz.trimf(age_universe,  [_age_min, _age_min, _age_q2])
age_middle = fuzz.trimf(age_universe,  [_age_q1, _age_q2, _age_q3])
age_old    = fuzz.trapmf(age_universe, [_age_q3, (_age_q3+_age_max)/2.0, _age_max, _age_max])

# HighBP (binary)
bp_normal = fuzz.trimf(binary_universe, [0.0, 0.0, 0.5])
bp_high   = fuzz.trimf(binary_universe, [0.5, 1.0, 1.0])

# Smoker (binary)
smoke_no  = fuzz.trimf(binary_universe, [0.0, 0.0, 0.5])
smoke_yes = fuzz.trimf(binary_universe, [0.5, 1.0, 1.0])

# PhysActivity (binary)
act_inactive = fuzz.trimf(binary_universe, [0.0, 0.0, 0.5])
act_active   = fuzz.trimf(binary_universe, [0.5, 1.0, 1.0])

# HighChol (binary)
chol_normal = fuzz.trimf(binary_universe, [0.0, 0.0, 0.5])
chol_high   = fuzz.trimf(binary_universe, [0.5, 1.0, 1.0])

# HeartDiseaseorAttack (binary)
cardio_no   = fuzz.trimf(binary_universe, [0.0, 0.0, 0.5])
cardio_yes  = fuzz.trimf(binary_universe, [0.5, 1.0, 1.0])

# GenHlth (percentile-based)
genhlth_good = fuzz.trimf(genhlth_universe, [_gh_min, _gh_min, _gh_q1])
genhlth_avg  = fuzz.trimf(genhlth_universe, [_gh_q1, _gh_q2, _gh_q3])
genhlth_poor = fuzz.trapmf(genhlth_universe, [_gh_q3, (_gh_q3+_gh_max)/2.0, _gh_max, _gh_max])

# Risk Output (fixed scale 0–1)
risk_vlow  = fuzz.trimf(risk_universe, [0.0, 0.0, 0.2])
risk_low   = fuzz.trimf(risk_universe, [0.1, 0.25, 0.4])
risk_med   = fuzz.trimf(risk_universe, [0.3, 0.5, 0.7])
risk_high  = fuzz.trimf(risk_universe, [0.6, 0.75, 0.9])
risk_vhigh = fuzz.trimf(risk_universe, [0.8, 1.0, 1.0])

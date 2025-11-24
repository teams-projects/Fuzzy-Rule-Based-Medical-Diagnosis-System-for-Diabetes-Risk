import numpy as np
import skfuzzy as fuzz

# Define universes
bmi_universe = np.linspace(0, 1, 101)
age_universe = np.linspace(0, 1, 101)
binary_universe = np.linspace(0, 1, 101)
risk_universe = np.linspace(0, 1, 101)

# BMI membership functions
bmi_under = fuzz.trimf(bmi_universe, [0.0, 0.0, 0.25])
bmi_healthy = fuzz.trimf(bmi_universe, [0.15, 0.35, 0.55])
bmi_over = fuzz.trimf(bmi_universe, [0.45, 0.60, 0.75])
bmi_obese = fuzz.trimf(bmi_universe, [0.65, 1.0, 1.0])

# Age membership functions
age_young  = fuzz.trimf(age_universe, [0.0, 0.0, 0.4])
age_middle = fuzz.trimf(age_universe, [0.2, 0.5, 0.8])
age_old    = fuzz.trimf(age_universe, [0.6, 1.0, 1.0])

# HighBP
bp_normal = fuzz.trimf(binary_universe, [0.0, 0.0, 0.4])
bp_high   = fuzz.trimf(binary_universe, [0.6, 1.0, 1.0])

# Smoker
smoke_no  = fuzz.trimf(binary_universe, [0.0, 0.0, 0.4])
smoke_yes = fuzz.trimf(binary_universe, [0.6, 1.0, 1.0])

# PhysActivity
act_inactive = fuzz.trimf(binary_universe, [0.0, 0.0, 0.4])
act_active   = fuzz.trimf(binary_universe, [0.6, 1.0, 1.0])

# Risk Output
risk_vlow  = fuzz.trimf(risk_universe, [0.0, 0.0, 0.2])
risk_low   = fuzz.trimf(risk_universe, [0.1, 0.25, 0.4])
risk_med   = fuzz.trimf(risk_universe, [0.3, 0.5, 0.7])
risk_high  = fuzz.trimf(risk_universe, [0.6, 0.75, 0.9])
risk_vhigh = fuzz.trimf(risk_universe, [0.8, 1.0, 1.0])

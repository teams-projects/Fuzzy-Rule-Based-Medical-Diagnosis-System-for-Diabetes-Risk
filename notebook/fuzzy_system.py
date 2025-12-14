"""
Fuzzy Rule-Based Medical Diagnosis System for Diabetes Risk
"""

import argparse
import time
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import skfuzzy.control as ctrl
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------
# Import fuzzy membership definitions
# ------------------------------------------------------------
from fuzzy_memberships import (
    bmi_universe, age_universe, binary_universe,
    risk_universe, genhlth_universe,

    bmi_under, bmi_healthy, bmi_over, bmi_obese,
    age_young, age_middle, age_old,

    bp_normal, bp_high,
    smoke_no, smoke_yes,
    act_inactive, act_active,

    chol_normal, chol_high,
    cardio_no, cardio_yes,

    genhlth_good, genhlth_avg, genhlth_poor,

    risk_vlow, risk_low, risk_med, risk_high, risk_vhigh,
)


# ------------------------------------------------------------
# Build fuzzy system
# ------------------------------------------------------------
def build_system() -> Tuple[ctrl.ControlSystem, ctrl.ControlSystemSimulation]:

    # ---------- Antecedents ----------
    bmi = ctrl.Antecedent(bmi_universe, 'BMI')
    age = ctrl.Antecedent(age_universe, 'Age')
    highbp = ctrl.Antecedent(binary_universe, 'HighBP')
    smoker = ctrl.Antecedent(binary_universe, 'Smoker')
    physact = ctrl.Antecedent(binary_universe, 'PhysActivity')
    chol = ctrl.Antecedent(binary_universe, 'HighChol')
    heart = ctrl.Antecedent(binary_universe, 'HeartDiseaseorAttack')
    genhlth = ctrl.Antecedent(genhlth_universe, 'GenHlth')

    # ---------- Consequent ----------
    risk = ctrl.Consequent(risk_universe, 'Risk', defuzzify_method='centroid')

    # ---------- Assign membership functions ----------
    bmi['under'] = bmi_under
    bmi['healthy'] = bmi_healthy
    bmi['over'] = bmi_over
    bmi['obese'] = bmi_obese

    age['young'] = age_young
    age['middle'] = age_middle
    age['old'] = age_old

    highbp['normal'] = bp_normal
    highbp['high'] = bp_high

    smoker['no'] = smoke_no
    smoker['yes'] = smoke_yes

    physact['inactive'] = act_inactive
    physact['active'] = act_active

    chol['normal'] = chol_normal
    chol['high'] = chol_high

    heart['no'] = cardio_no
    heart['yes'] = cardio_yes

    genhlth['good'] = genhlth_good
    genhlth['avg'] = genhlth_avg
    genhlth['poor'] = genhlth_poor

    risk['vlow'] = risk_vlow
    risk['low'] = risk_low
    risk['med'] = risk_med
    risk['high'] = risk_high
    risk['vhigh'] = risk_vhigh

    # ---------- Rule base ----------
    rules = []

    # VLOW / LOW
    rules += [
        ctrl.Rule(bmi['healthy'] & age['young'] & highbp['normal'] & smoker['no'] & physact['active'], risk['vlow']),
        ctrl.Rule(bmi['under'] & physact['active'] & smoker['no'], risk['vlow']),
        ctrl.Rule(bmi['healthy'] & age['middle'] & physact['active'], risk['low']),
        ctrl.Rule(bmi['healthy'] & smoker['yes'], risk['low']),
        ctrl.Rule(bmi['over'] & age['young'] & physact['active'], risk['low']),
    ]

    # MEDIUM
    rules += [
        ctrl.Rule(bmi['over'] & physact['inactive'], risk['med']),
        ctrl.Rule(bmi['obese'] & age['young'] & physact['active'], risk['med']),
        ctrl.Rule(highbp['high'] & age['young'] & physact['active'], risk['med']),
        ctrl.Rule(bmi['healthy'] & age['old'] & physact['inactive'], risk['med']),
        ctrl.Rule(smoker['yes'] & physact['inactive'], risk['med']),
        ctrl.Rule(bmi['over'] & age['middle'], risk['med']),
    ]

    # HIGH
    rules += [
        ctrl.Rule(bmi['obese'] & physact['inactive'], risk['high']),
        ctrl.Rule(bmi['over'] & age['old'] & physact['inactive'], risk['high']),
        ctrl.Rule(highbp['high'] & smoker['yes'], risk['high']),
        ctrl.Rule(highbp['high'] & age['old'], risk['high']),
    ]

    # VERY HIGH
    rules += [
        ctrl.Rule(bmi['obese'] & highbp['high'] & age['old'], risk['vhigh']),
        ctrl.Rule(bmi['obese'] & highbp['high'] & smoker['yes'], risk['vhigh']),
        ctrl.Rule(highbp['high'] & age['old'] & physact['inactive'], risk['vhigh']),
    ]

    # ---------- Comorbidity amplifiers (FIXED) ----------
    rules += [
        ctrl.Rule(heart['yes'] & bmi['over'], risk['high']),
        ctrl.Rule(heart['yes'] & highbp['high'], risk['high']),
        ctrl.Rule(heart['yes'] & age['old'], risk['high']),

        ctrl.Rule(chol['high'] & bmi['over'], risk['high']),
        ctrl.Rule(chol['high'] & highbp['high'], risk['high']),

        ctrl.Rule(genhlth['poor'] & bmi['over'], risk['high']),
        ctrl.Rule(genhlth['poor'] & age['old'], risk['high']),
        ctrl.Rule(genhlth['good'] & physact['active'] & bmi['healthy'], risk['low']),
    ]

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    return system, sim


# ------------------------------------------------------------
# Inference on dataset
# ------------------------------------------------------------
def compute_scores(sim: ctrl.ControlSystemSimulation, df: pd.DataFrame) -> np.ndarray:
    scores = []

    for _, row in df.iterrows():
        try:
            sim.reset()

            sim.input['BMI'] = float(row['BMI'])
            sim.input['Age'] = float(row['Age'])
            sim.input['HighBP'] = float(row['HighBP'])
            sim.input['Smoker'] = float(row['Smoker'])
            sim.input['PhysActivity'] = float(row['PhysActivity'])

            sim.input['HighChol'] = float(row['HighChol'])
            sim.input['HeartDiseaseorAttack'] = float(row['HeartDiseaseorAttack'])
            sim.input['GenHlth'] = float(row['GenHlth'])

            sim.compute()
            scores.append(float(sim.output['Risk']))

        except Exception:
            scores.append(0.5)

    return np.asarray(scores)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../data/diabetes_clean.csv')
    parser.add_argument('--sample-size', type=int, default=30000)
    parser.add_argument('--metric', choices=['f1', 'accuracy'], default='f1')
    parser.add_argument('--threshold-grid', nargs=3, type=float, default=[0.2, 0.8, 0.02])
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = df.sample(n=min(args.sample_size, len(df)), random_state=42)

    _, sim = build_system()

    print("Running fuzzy inference...")
    t0 = time.time()
    scores = compute_scores(sim, df)
    print(f"Done in {time.time()-t0:.2f}s")

    y_true = df['Diabetes_binary'].astype(int).values

    best_thr, best_f1 = 0.5, -1
    for thr in np.arange(*args.threshold_grid):
        f1 = f1_score(y_true, (scores >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    y_pred = (scores >= best_thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    print("\nCONFUSION MATRIX")
    print(cm)
    print("\nFINAL METRICS")
    print(f"Threshold: {best_thr:.2f}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    np.save("fuzzy_scores.npy", scores)
    np.save("y_true.npy", y_true)



if __name__ == '__main__':
    main()

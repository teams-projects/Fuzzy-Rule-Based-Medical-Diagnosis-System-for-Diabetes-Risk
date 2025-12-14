import streamlit as st
from fuzzy_system import build_system

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Fuzzy Expert System",
    layout="centered"
)

st.title("🩺 Diabetes Risk Assessment System")
st.markdown(
    """
    This application estimates **diabetes risk** using a  
    **Mamdani-type fuzzy rule-based expert system**.
    
    The system integrates medical, demographic, and lifestyle factors
    and provides **personalized, interpretable recommendations**.
    """
)

# -------------------------------------------------
# Build fuzzy system once
# -------------------------------------------------
_, sim = build_system()

# -------------------------------------------------
# Recommendation function
# -------------------------------------------------
def generate_recommendations(
    risk_score,
    bmi_real,
    age_real,
    highbp,
    smoker,
    physact,
    highchol,
    heart,
    genhlth_label
):
    recs = []

    # --- Risk-level advice ---
    if risk_score < 0.2:
        recs.append("🟢 Maintain your current healthy lifestyle.")
    elif risk_score < 0.4:
        recs.append("🟡 Monitor your health regularly and maintain healthy habits.")
    elif risk_score < 0.6:
        recs.append("🟠 Lifestyle improvements are recommended to reduce diabetes risk.")
    elif risk_score < 0.8:
        recs.append("🔴 High risk detected. Medical consultation is strongly recommended.")
    else:
        recs.append("⚫ Very high risk detected. Immediate medical follow-up is advised.")

    # --- Factor-specific recommendations ---
    if bmi_real >= 30:
        recs.append("⚠️ BMI indicates obesity. Weight management through diet and exercise is recommended.")
    elif bmi_real >= 25:
        recs.append("⚠️ BMI indicates overweight. Gradual weight reduction is advised.")

    if highbp == 1:
        recs.append("⚠️ High blood pressure detected. Reduce salt intake and consult a healthcare provider.")

    if smoker == 1:
        recs.append("⚠️ Smoking increases insulin resistance. Smoking cessation is strongly advised.")

    if physact == 0:
        recs.append("⚠️ Low physical activity detected. At least 150 minutes of moderate exercise per week is recommended.")

    if highchol == 1:
        recs.append("⚠️ High cholesterol detected. Dietary fat reduction and medical advice are recommended.")

    if heart == 1:
        recs.append("⚠️ History of heart disease significantly increases diabetes risk. Regular medical monitoring is essential.")

    if genhlth_label in ["Fair", "Poor"]:
        recs.append("⚠️ Poor general health reported. A comprehensive health evaluation is recommended.")

    # --- Protective feedback ---
    if smoker == 0 and physact == 1 and bmi_real < 25:
        recs.append("✅ Protective factors detected: non-smoker, active lifestyle, and healthy BMI.")

    return recs


# -------------------------------------------------
# Sidebar inputs (REAL VALUES)
# -------------------------------------------------
st.sidebar.header("Patient Information")

# Age
age_real = st.sidebar.slider(
    "Age (years)",
    min_value=18,
    max_value=90,
    value=40
)

# BMI
bmi_real = st.sidebar.slider(
    "Body Mass Index (BMI)",
    min_value=15.0,
    max_value=45.0,
    value=25.0,
    step=0.1
)

# General Health
genhlth_map = {
    "Excellent": 1,
    "Very Good": 2,
    "Good": 3,
    "Fair": 4,
    "Poor": 5
}
genhlth_label = st.sidebar.selectbox(
    "General Health",
    list(genhlth_map.keys())
)
genhlth_real = genhlth_map[genhlth_label]

# Binary inputs
HighBP = st.sidebar.selectbox(
    "High Blood Pressure",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

Smoker = st.sidebar.selectbox(
    "Smoker",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

PhysActivity = st.sidebar.selectbox(
    "Physically Active",
    [0, 1],
    format_func=lambda x: "Active" if x == 1 else "Inactive"
)

HighChol = st.sidebar.selectbox(
    "High Cholesterol",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

HeartDisease = st.sidebar.selectbox(
    "Heart Disease / Attack",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# -------------------------------------------------
# Normalize inputs (INTERNAL USE)
# -------------------------------------------------
age_norm = (age_real - 18) / (90 - 18)
bmi_norm = (bmi_real - 15) / (45 - 15)
genhlth_norm = (genhlth_real - 1) / 4


# -------------------------------------------------
# Run inference
# -------------------------------------------------
if st.button("🔍 Evaluate Diabetes Risk"):
    try:
        sim.reset()

        sim.input['Age'] = age_norm
        sim.input['BMI'] = bmi_norm
        sim.input['GenHlth'] = genhlth_norm
        sim.input['HighBP'] = HighBP
        sim.input['Smoker'] = Smoker
        sim.input['PhysActivity'] = PhysActivity
        sim.input['HighChol'] = HighChol
        sim.input['HeartDiseaseorAttack'] = HeartDisease

        sim.compute()
        risk_score = float(sim.output['Risk'])

        # Risk level
        if risk_score < 0.2:
            risk_level = "🟢 Very Low"
        elif risk_score < 0.4:
            risk_level = "🟡 Low"
        elif risk_score < 0.6:
            risk_level = "🟠 Medium"
        elif risk_score < 0.8:
            risk_level = "🔴 High"
        else:
            risk_level = "⚫ Very High"

        # Display results
        st.success("Assessment Completed")
        st.metric("Diabetes Risk Score", f"{risk_score:.3f}")
        st.markdown(f"### Risk Level: **{risk_level}**")

        # Personalized recommendations
        st.subheader("📌 Personalized Recommendations")
        recommendations = generate_recommendations(
            risk_score,
            bmi_real,
            age_real,
            HighBP,
            Smoker,
            PhysActivity,
            HighChol,
            HeartDisease,
            genhlth_label
        )

        for rec in recommendations:
            st.write("- " + rec)

    except Exception as e:
        st.error(f"Error during inference: {e}")

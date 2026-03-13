import streamlit as st
import numpy as np
import joblib

st.set_page_config(layout="wide")

model = joblib.load("Model/Linear_Regression_model.joblib")

st.markdown(
    "<h1 style='text-align:center;'>⚡Electricity Demand Predictor  </h1>",
    unsafe_allow_html=True
)
# Feature definitions
features = {
    "SolarEnergy_aspect1": ["SolarPowerEquipped","NoSolarPower"],
    "SolarEnergy_aspect2": ["AddnlSolarPower","NoAddnlPower"],
    "SolarEnergy_aspect3": ["MinPowerEnabled","MinPowerNotEnabled"],
    "SolarEnergy_aspect4": ["BatteriesEquipped","BatteriesNotEquipped"],
    "SolarEnergy_aspect5": ["DCtoACEquipped","DCtoACnotEquipped"],

    "Behavioural_aspect1": ["Awareness","NoAwareness"],
    # "Behavioural_aspect2": ["ACsOnNeed","ACsAlwaysOn"],
    "Behavioural_aspect2": ["ACsAlwaysOn","ACsOnNeed"],
    "Behavioural_aspect3": ["Slabs","NoSlabs"],
    "Behavioural_aspect4": ["Auto-Off","NoAuto-Off"],
    "Behavioural_aspect5": ["StreetLightsEquipped","StreetLightsNotEquipped"]
}

user_inputs = []

col1, col2 = st.columns(2)

# Solar features
with col1:
    st.subheader("Solar Energy Features")
    for i, (feature, options) in enumerate(list(features.items())[:5]):
        choice = st.selectbox(feature, options)
        user_inputs.append(choice == options[1])

# Behaviour features
with col2:
    st.subheader("Behavioural Features")
    for i, (feature, options) in enumerate(list(features.items())[5:]):
        choice = st.selectbox(feature, options)
        user_inputs.append(choice == options[1])
        # user_inputs.append(choice == options[0] if "ACsOnNeed" in options else choice == options[1])

if st.button("Predict Consumption State"):
    input_data = np.array([user_inputs])
    prediction = model.predict(input_data)[0]

    prob = max(0, min(1, prediction))
    prob_percent = prob * 100

    if prob >= 0.5:
        result = "Uncontrolled"
        st.error(f"Result: {result}")
    else:
        result = "Controlled"
        st.success(f"Result: {result}")

    st.write("Probability of Uncontrolled")
    st.markdown(f"## {prob_percent:.1f}%")
    st.subheader("Confidence Meter")
    st.progress(round(prob_percent))
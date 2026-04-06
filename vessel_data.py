import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------- SETTINGS ----------
# MODEL_PATH = "vessel_orientation_gbm.pkl"  # change if different

# ---------- LOAD MODEL ----------

with open("vessel_orientation_gbm.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Vessel Orientation Predictor")

st.markdown(
    "Provide the stream and equipment data below. "
    "The model will predict the **VesselOrientation**."
)

# ---------- USER INPUTS ----------
# Numeric features
BubblePointPressure = st.number_input("BubblePointPressure", value=0.0, step=0.1)
BubblePointTemperature = st.number_input("BubblePointTemperature", value=0.0, step=0.1)
Density = st.number_input("Density", value=0.0, step=0.1)
DewPointPressure = st.number_input("DewPointPressure", value=0.0, step=0.1)
DewPointTemperature = st.number_input("DewPointTemperature", value=0.0, step=0.1)
Enthalpy = st.number_input("Enthalpy", value=0.0, step=0.1)
HigherHeatingValue = st.number_input("HigherHeatingValue", value=0.0, step=0.1)
LowerHeatingValue = st.number_input("LowerHeatingValue", value=0.0, step=0.1)
MassFlow = st.number_input("MassFlow", value=0.0, step=0.1)
MassFlowWater = st.number_input("MassFlowWater", value=0.0, step=0.1)
MoleFlow = st.number_input("MoleFlow", value=0.0, step=0.1)
MoleWeight = st.number_input("MoleWeight", value=0.0, step=0.1)
Pressure = st.number_input("Pressure", value=0.0, step=0.1)
SpecificEnthalpy = st.number_input("SpecificEnthalpy", value=0.0, step=0.1)
SpecificEntropy = st.number_input("SpecificEntropy", value=0.0, step=0.1)
StreamTemperature = st.number_input("StreamTemperature", value=0.0, step=0.1)
VapourFraction = st.number_input("VapourFraction", value=0.0, min_value=0.0, max_value=1.0, step=0.01)
VapourFractionMass = st.number_input("VapourFractionMass", value=0.0, min_value=0.0, max_value=1.0, step=0.01)
VolumeFlow = st.number_input("VolumeFlow", value=0.0, step=0.1)

# Integer features
EquipmentType = st.number_input("EquipmentType (int)", value=0, step=1)
StreamName = st.number_input("StreamName (int)", value=0, step=1)

# Categorical / object features
# For free-text inputs, you can later replace these with selectboxes using
# the categories from your training data.
CaseName = st.text_input("CaseName", value="case_1")
EquipmentName = st.text_input("EquipmentName", value="equip_1")
JobNumber = st.text_input("JobNumber", value="job_1")
PortType = st.text_input("PortType", value="Inlet")
StreamId = st.text_input("StreamId", value="stream_1")
StreamRole = st.text_input("StreamRole", value="Process")
TechnologyType = st.text_input("TechnologyType", value="Tech_A")

# ---------- BUILD INPUT ROW ----------
# IMPORTANT: this column order MUST match what you used in training.
input_data = pd.DataFrame([{
    "BubblePointPressure": BubblePointPressure,
    "BubblePointTemperature": BubblePointTemperature,
    "CaseName": CaseName,
    "Density": Density,
    "DewPointPressure": DewPointPressure,
    "DewPointTemperature": DewPointTemperature,
    "Enthalpy": Enthalpy,
    "EquipmentName": EquipmentName,
    "EquipmentType": EquipmentType,
    "HigherHeatingValue": HigherHeatingValue,
    "JobNumber": JobNumber,
    "LowerHeatingValue": LowerHeatingValue,
    "MassFlow": MassFlow,
    "MassFlowWater": MassFlowWater,
    "MoleFlow": MoleFlow,
    "MoleWeight": MoleWeight,
    "PortType": PortType,
    "Pressure": Pressure,
    "SpecificEnthalpy": SpecificEnthalpy,
    "SpecificEntropy": SpecificEntropy,
    "StreamId": StreamId,
    "StreamName": StreamName,
    "StreamRole": StreamRole,
    "StreamTemperature": StreamTemperature,
    "TechnologyType": TechnologyType,
    "VapourFraction": VapourFraction,
    "VapourFractionMass": VapourFractionMass,
    # VesselOrientation is the target, so NOT included here
    "VolumeFlow": VolumeFlow,
}])

st.subheader("Input data preview")
st.dataframe(input_data)

# ---------- PREDICTION ----------
if st.button("Predict Vessel Orientation"):
    try:
        pred = model.predict(input_data)
        # If model outputs labels directly (e.g. "Horizontal", "Vertical"), show as-is.
        # If it outputs encoded integers, map them here.
        predicted_orientation = pred[0]

        st.subheader("Prediction")
        st.write(f"Predicted VesselOrientation: **{predicted_orientation}**")

        # Optional: show probabilities if available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            # If the model is a classifier with classes_ attribute
            classes = getattr(model, "classes_", None)
            if classes is not None:
                proba_df = pd.DataFrame({
                    "Class": classes,
                    "Probability": proba
                })
                st.write("Prediction probabilities:")
                st.dataframe(proba_df)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.write(
            "Check that the model was trained on these exact columns "
            "and that any preprocessing (e.g. encoders, pipelines) "
            "is included inside the saved model."
        )
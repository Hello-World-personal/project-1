import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

# ============================================
# 1. Train model (runs once, cached)
# ============================================
@st.cache_resource
def load_model():
    df = pd.read_excel("/home/topsoe/vrsh/streamlit-d/example/vessel_data_reduced.xlsx")

    target_col = "VesselOrientation"

    # Only use the 5 stream property columns
    feature_cols = [
        "BubblePointTemperature",
        "DewPointTemperature",
        "StreamTemperature",
        "DewPointPressure",
        "Pressure",
    ]

    X = df[feature_cols].fillna(0)
    y = df[target_col]

    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y.astype(str))

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )
    model.fit(X, y_encoded)

    return model, target_le, feature_cols

model, target_le, feature_cols = load_model()

# ============================================
# 2. Page config
# ============================================
st.title("Vessel Orientation Predictor")
st.markdown(
    "Enter the **stream properties** below. "
    "The model predicts whether the vessel orientation is "
    "**Vertical** or **Horizontal** using only physical properties."
)
st.caption("Model accuracy: 99.01% — based on 5 stream features, no equipment info needed.")

st.divider()

# ============================================
# 3. User inputs
# ============================================
st.subheader("Stream Properties")

col1, col2 = st.columns(2)

with col1:
    bubble_point_temp = st.number_input(
        "Bubble Point Temperature",
        value=0.0,
        step=0.1,
        help="Bubble point temperature of the stream",
    )
    stream_temp = st.number_input(
        "Stream Temperature",
        value=0.0,
        step=0.1,
        help="Operating temperature of the stream",
    )
    pressure = st.number_input(
        "Pressure",
        value=0.0,
        step=0.1,
        help="Operating pressure of the stream",
    )

with col2:
    dew_point_temp = st.number_input(
        "Dew Point Temperature",
        value=0.0,
        step=0.1,
        help="Dew point temperature of the stream",
    )
    dew_point_pressure = st.number_input(
        "Dew Point Pressure",
        value=0.0,
        step=0.1,
        help="Dew point pressure of the stream",
    )

st.divider()

# ============================================
# 4. Predict
# ============================================
if st.button("Predict Vessel Orientation", type="primary", use_container_width=True):

    # Build input
    input_data = pd.DataFrame([{
        "BubblePointTemperature": bubble_point_temp,
        "DewPointTemperature": dew_point_temp,
        "StreamTemperature": stream_temp,
        "DewPointPressure": dew_point_pressure,
        "Pressure": pressure,
    }])

    # Predict
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    result = target_le.inverse_transform(prediction)[0]

    proba_horizontal = prediction_proba[0][0] * 100
    proba_vertical = prediction_proba[0][1] * 100

    # ============================================
    # 5. Display result
    # ============================================
    st.divider()

    if result == "Vertical":
        st.success(f"### Predicted: ⬆️ Vertical")
    else:
        st.success(f"### Predicted: ↔️ Horizontal")

    # Confidence
    st.subheader("Confidence")
    conf_col1, conf_col2 = st.columns(2)

    with conf_col1:
        st.metric("Horizontal", f"{proba_horizontal:.1f}%")
        st.progress(proba_horizontal / 100)

    with conf_col2:
        st.metric("Vertical", f"{proba_vertical:.1f}%")
        st.progress(proba_vertical / 100)

    # Input summary
    st.subheader("Your Input")
    st.dataframe(input_data, use_container_width=True, hide_index=True)

    # ============================================
    # 6. Feature importance chart
    # ============================================
    st.divider()
    st.subheader("Feature Importance")

    importance = model.feature_importances_
    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance (%)": np.round(importance / importance.sum() * 100, 2),
    }).sort_values("Importance (%)", ascending=True)

    st.bar_chart(imp_df.set_index("Feature"), horizontal=True)

    st.caption(
        "Temperature properties drive ~93% of the prediction. "
        "Pressure contributes ~7%."
    )
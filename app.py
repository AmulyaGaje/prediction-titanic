import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------
# PAGE CONFIGURATION
# ------------------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# ------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------

model = tf.keras.models.load_model(
    "titanic_ann_model.keras"
)

# ------------------------------------
# HEADER SECTION
# ------------------------------------

st.title("🚢 Titanic Survival Prediction System")

st.subheader(
    "Deep Learning Based Passenger Survival Prediction"
)



st.markdown("---")

# ------------------------------------
# PROJECT DESCRIPTION
# ------------------------------------


st.markdown("---")

# ------------------------------------
# INPUT SECTION
# ------------------------------------

st.header("🧾 Passenger Information")

col1, col2, col3 = st.columns(3)

with col1:

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

with col2:

    age = st.slider(
        "Age",
        1,
        80,
        25
    )

with col3:

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=50.0
    )

st.markdown("---")

# ------------------------------------
# PREPROCESSING
# ------------------------------------

def preprocess_input(
    pclass,
    age,
    fare
):

    pclass = (pclass - 1) / 2

    age = age / 80

    fare = fare / 600

    return np.array([
        [pclass, age, fare]
    ])

# ------------------------------------
# PREDICTION BUTTON
# ------------------------------------

if st.button("🔍 Predict Survival"):

    input_data = preprocess_input(
        pclass,
        age,
        fare
    )

    prediction = model.predict(
        input_data
    )

    probability = prediction[0][0]

    # ------------------------------------
    # PREDICTION LOGIC
    # ------------------------------------

    if probability > 0.5:

        result = "✅ Passenger Survived"

    else:

        result = "❌ Passenger Did Not Survive"

    confidence = probability * 100

    # ------------------------------------
    # OUTPUT SECTION
    # ------------------------------------

    st.markdown("---")

    st.header("📊 Prediction Output")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            label="Prediction Result",
            value=result
        )

    with c2:

        st.metric(
            label="Survival Probability",
            value=f"{probability:.2f}"
        )

    with c3:

        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

    st.markdown("---")

    # ------------------------------------
    # VISUALIZATION
    # ------------------------------------

    st.header("📈 Prediction Visualization")

    survive_prob = probability

    not_survive_prob = 1 - probability

    labels = [
        "Survive",
        "Not Survive"
    ]

    values = [
        survive_prob,
        not_survive_prob
    ]

    fig, ax = plt.subplots()

    ax.bar(labels, values)

    ax.set_ylabel("Probability")

    ax.set_title("Survival Probability")

    st.pyplot(fig)
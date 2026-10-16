import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from predictor import TrajectoryModel


st.set_page_config(
    page_title="Depressive Symptom Trajectory Predictor",
    layout="centered",
)


# Load the original random forest model
@st.cache_resource
def load_model():
    path = Path(__file__).absolute().parent / "artifacts" / "model.json.gz"
    return TrajectoryModel.load(path)


model = load_model()

variable_names = [spec["name"] for spec in model.inputs]


# English variable labels
english_labels = {
    "gender": "Gender",
    "marry": "Marital status",
    "edu": "Education level",
    "retire": "Retirement status",
    "systo": "Systolic blood pressure",
    "bmi": "BMI",
    "drinkl": "Alcohol consumption",
    "social": "Social participation",
    "chronic": "Chronic disease",
    "arther": "Arthritis",
    "disability": "Disability",
    "hear": "Hearing",
    "eyesight_close": "Near vision",
    "sleep": "Sleep duration",
    "satlife": "Life satisfaction",
}


# Preserve the original category codes
category_labels = {
    "gender": {
        "1": "Female",
        "2": "Male",
    },
    "marry": {
        "1": "Other",
        "2": "Married",
    },
    "edu": {
        "1": "Less than primary school",
        "2": "Primary school",
        "3": "Middle school",
        "4": "High school or above",
    },
    "retire": {
        "1": "Not retired",
        "2": "Retired",
    },
    "hear": {
        "1": "Good",
        "2": "Fair",
        "3": "Poor",
    },
    "eyesight_close": {
        "1": "Good",
        "2": "Fair",
        "3": "Poor",
    },
    "satlife": {
        "1": "Poor",
        "2": "Fair",
        "3": "Good",
    },
}

binary_labels = {
    "1": "No",
    "2": "Yes",
}

units = {
    "systo": "mmHg",
    "bmi": "kg/m²",
    "sleep": "hours/night",
}

feature_names = [
    english_labels.get(name, name)
    for name in variable_names
]


st.title("Depressive Symptom Trajectory Predictor")


# Generate input fields
inputs = {}

for spec in model.inputs:
    name = spec["name"]
    label = english_labels.get(name, name)

    if spec["type"] == "category":
        options = [item["value"] for item in spec["options"]]
        labels = category_labels.get(name, binary_labels)

        inputs[name] = st.selectbox(
            label,
            options=options,
            index=options.index(str(spec["default"])),
            format_func=lambda value, mapping=labels: mapping[value],
        )

    else:
        unit = units.get(name, "")
        display_label = f"{label} ({unit})" if unit else label

        inputs[name] = st.number_input(
            display_label,
            value=float(spec["default"]),
            step=0.1,
            format="%.2f",
        )


# Convert SHAP inputs back to the original model inputs
def predict_proba(values):
    values = np.asarray(values, dtype=float)
    probabilities = []

    for row in values:
        raw = {}

        for i, spec in enumerate(model.inputs):
            if spec["type"] == "category":
                raw[spec["name"]] = str(int(row[i]))
            else:
                raw[spec["name"]] = float(row[i])

        probabilities.append(model.predict(raw)["probabilities"])

    return np.asarray(probabilities)


if st.button("Predict"):
    prediction = model.predict(inputs)

    st.write("**Predicted class:**", prediction["class"])

    features = pd.DataFrame(
        [[float(inputs[name]) for name in variable_names]],
        columns=feature_names,
    )

    # Reference profile: development-set medians and modes
    background = pd.DataFrame(
        [[float(spec["default"]) for spec in model.inputs]],
        columns=feature_names,
    )

    with st.spinner("Calculating SHAP values..."):
        explainer = shap.PermutationExplainer(
            predict_proba,
            background,
            feature_names=feature_names,
            seed=2026,
        )

        explanation = explainer(
            features,
            max_evals=32 * (2 * len(feature_names) + 1),
        )

    # Explain the predicted class
    class_index = model.classes.index(prediction["class"])
    shap_values = explanation.values[0, :, class_index]
    base_value = float(explanation.base_values[0, class_index])

    st.subheader("Individual SHAP Analysis")

    if np.allclose(shap_values, 0):
        st.write(
            "All SHAP values are zero relative to the reference profile."
        )

    else:
        fig = shap.force_plot(
            base_value,
            shap_values,
            features.iloc[0],
            feature_names=feature_names,
            out_names=prediction["class"],
            matplotlib=True,
            show=False,
            figsize=(16, 4),
        )

        st.pyplot(fig)
        plt.close(fig)

    st.caption(
        "Reference: development-set medians and modes. "
        "Positive SHAP values increase the predicted-class score; "
        "negative values decrease it."
    )

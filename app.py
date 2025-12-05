import streamlit as st
import pandas as pd
import numpy as np
import joblib

from typing import List


# -----------------------------
# 1. Load models & metadata
# -----------------------------
@st.cache_resource
def load_models():
    reg_model = joblib.load("reg_model.joblib")
    clf_model = joblib.load("clf_model.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    feature_cols: List[str] = joblib.load("feature_cols.joblib")

    # Optional schema sample for inferring dtypes & dropdown values
    try:
        schema_df = pd.read_csv("schema_sample.csv")
    except Exception:
        schema_df = None

    return reg_model, clf_model, label_encoder, feature_cols, schema_df


# -----------------------------
# 2. Build user input form
# -----------------------------
def build_input_form(feature_cols, schema_df):
    st.subheader("Enter Project Features")

    input_data = {}

    # If we have a schema sample, use it to infer types and categories
    if schema_df is not None:
        dtypes = schema_df.dtypes
    else:
        # Fallback: assume strings for safety
        dtypes = pd.Series('object', index=pd.Index(feature_cols))

    for col in feature_cols:
        col_type = dtypes.get(col, 'object')

        st.markdown(f"**{col}**")

        if schema_df is not None and col_type == 'object':
            # Categorical: use unique values as options
            options = sorted(schema_df[col].dropna().unique().tolist())
            if len(options) == 0:
                value = st.text_input(col, "")
            else:
                value = st.selectbox(col, options, key=col)
        elif np.issubdtype(col_type, np.number):
            # Numeric: use median as default if available
            default = float(schema_df[col].median()) if schema_df is not None else 0.0
            value = st.number_input(col, value=default, key=col)
        else:
            # Fallback for unknown type
            value = st.text_input(col, "", key=col)

        input_data[col] = value

    # Single-row DataFrame
    input_df = pd.DataFrame([input_data], columns=feature_cols)
    return input_df


# -----------------------------
# 3. Main Streamlit app
# -----------------------------
def main():
    st.set_page_config(page_title="IFC Model Deployment", layout="wide")
    st.title("IFC Investment – ML Model Deployment (Regression + Classification)")

    reg_model, clf_model, label_encoder, feature_cols, schema_df = load_models()

    st.write("This app loads pre-trained XGBoost models (via joblib) and allows you to manually input project features to get:")
    st.markdown("- **Regression**: Predicted Total IFC investment (Board, Million USD)")
    st.markdown("- **Classification**: Predicted Environmental Category with class probabilities")

    with st.form("prediction_form"):
        input_df = build_input_form(feature_cols, schema_df)
        submitted = st.form_submit_button("Predict")

    if submitted:
        st.subheader("Model Input")
        st.dataframe(input_df)

        # -------------------------
        # Regression prediction
        # -------------------------
        reg_pred = reg_model.predict(input_df)[0]

        st.subheader("Regression Result")
        st.write(f"**Predicted Total IFC investment (Board, Million USD):** {reg_pred:,.4f}")

        # -------------------------
        # Classification prediction
        # -------------------------
        clf_pred_int = clf_model.predict(input_df)[0]
        clf_proba = clf_model.predict_proba(input_df)[0]

        pred_class = label_encoder.inverse_transform([clf_pred_int])[0]
        class_labels = label_encoder.classes_

        st.subheader("Classification Result")
        st.write(f"**Predicted Environmental Category:** {pred_class}")

        proba_df = pd.DataFrame({
            "Category": class_labels,
            "Probability": clf_proba
        }).sort_values("Probability", ascending=False)

        st.write("**Class probabilities:**")
        st.dataframe(proba_df.reset_index(drop=True))


if __name__ == "__main__":
    main()

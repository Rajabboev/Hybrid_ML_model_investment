import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="IFC Investment ML App", layout="wide")

# -------------------------------------------------
# Load models and data
# -------------------------------------------------
@st.cache_resource
def load_models():
    reg_model = joblib.load("models/reg_model.joblib")
    clf_model = joblib.load("models/clf_model.joblib")
    label_encoder = joblib.load("models/label_encoder.joblib")
    reg_feature_cols = joblib.load("models/reg_feature_cols.joblib")
    clf_feature_cols = joblib.load("models/clf_feature_cols.joblib")
    return reg_model, clf_model, label_encoder, reg_feature_cols, clf_feature_cols


@st.cache_data
def load_data():
    df = pd.read_csv("data/ifc_clean.csv")
    return df


reg_model, clf_model, label_encoder, reg_feature_cols, clf_feature_cols = load_models()
df = load_data()

# Defaults for filling non-asked columns in Tab 2
default_values = {}
for col in df.columns:
    if df[col].dtype == "object":
        default_values[col] = df[col].mode(dropna=True)[0]
    else:
        default_values[col] = float(df[col].median())

# Map from full Env category to short code (used by regression features)
envcat_to_short = {
    "A - Significant": "A",
    "B - Limited": "B",
    "C - No Impact": "C",
    "FI": "FI",
    "FI-1 - Significant": "FI-1",
    "FI-2 - Limited": "FI-2",
    "FI-3 - No Impact": "FI-3",
    "Other": "O",
}

st.title("IFC Investment ML System – Best XGBoost Models")

tab1, tab2 = st.tabs(["Existing Projects (Dataset)", "New Project Entry"])


# =================================================
# TAB 1 – EXISTING PROJECTS
# =================================================
with tab1:
    st.subheader("🔍 Predict for a Project from the Dataset")

    st.write("Below is the cleaned dataset used to train the models (first 10 rows).")
    st.dataframe(df.head(10))
    st.write(f"Total rows: **{len(df)}**, columns: **{len(df.columns)}**")

    row_idx = st.number_input(
        "Choose a row index to test:",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1,
    )

    row_full = df.iloc[[row_idx]]

    st.write("**Selected project (raw data):**")
    st.dataframe(row_full)

    # Build model inputs using the exact training feature sets
    X_reg = row_full[reg_feature_cols]
    X_clf = row_full[clf_feature_cols]

    if st.button("Run Predictions for This Row", key="tab1_btn"):
        # Regression
        reg_pred = reg_model.predict(X_reg)[0]

        # Classification
        clf_pred_int = clf_model.predict(X_clf)[0]
        clf_pred_label = label_encoder.inverse_transform([clf_pred_int])[0]
        clf_proba = clf_model.predict_proba(X_clf)[0]

        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                "Predicted Total IFC Investment (Million USD)",
                f"{reg_pred:,.2f}",
            )
        with c2:
            st.metric(
                "Predicted Environmental Category",
                clf_pred_label,
            )

        st.write("### Class probabilities")
        proba_df = (
            pd.DataFrame(
                {"Category": label_encoder.classes_, "Probability": clf_proba}
            )
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(proba_df)


# =================================================
# TAB 2 – NEW PROJECT ENTRY (MANUAL INPUT)
# =================================================
with tab2:
    st.subheader("🆕 New Project – Manual Input")

    st.write(
        "Provide key project characteristics below. "
        "The classifier will predict the **Environmental Category**, "
        "and then the regressor will estimate the **Total IFC Investment (Million USD)**."
    )

    # ---- Choices for categorical fields
    countries = sorted(df["Country"].dropna().unique().tolist())
    industries = sorted(df["Industry"].dropna().unique().tolist())
    product_lines = sorted(df["Product Line"].dropna().unique().tolist())
    doc_types = sorted(df["Document Type"].dropna().unique().tolist())
    departments = sorted(df["Department"].dropna().unique().tolist())
    statuses = sorted(df["Status"].dropna().unique().tolist())

    # ---- Numeric ranges for sliders
    loan_col = "IFC investment for Loan(Million - USD)"
    eq_col = "IFC investment for Equity(Million - USD)"
    year_col = "Year"
    appr_col = "Approval_Delay"
    sign_col = "Signing_Delay"
    inv_col = "Invest_Delay"

    loan_min, loan_max = float(df[loan_col].min()), float(df[loan_col].max())
    loan_default = float(df[loan_col].median())

    eq_min, eq_max = float(df[eq_col].min()), float(df[eq_col].max())
    eq_default = float(df[eq_col].median())

    year_min, year_max = int(df[year_col].min()), int(df[year_col].max())
    year_default = int(df[year_col].median())

    appr_min, appr_max = int(df[appr_col].min()), int(df[appr_col].max())
    appr_default = int(df[appr_col].median())

    sign_min, sign_max = int(df[sign_col].min()), int(df[sign_col].max())
    sign_default = int(df[sign_col].median())

    inv_min, inv_max = int(df[inv_col].min()), int(df[inv_col].max())
    inv_default = int(df[inv_col].median())

    # ---- Layout for inputs
    c1, c2 = st.columns(2)

    with c1:
        country = st.selectbox("Country", countries)
        industry = st.selectbox("Industry", industries)
        product_line = st.selectbox("Product Line", product_lines)

        loan_amt = st.slider(
            "Loan Amount (Million USD)",
            min_value=loan_min,
            max_value=loan_max,
            value=loan_default,
        )
        equity_amt = st.slider(
            "Equity Amount (Million USD)",
            min_value=eq_min,
            max_value=eq_max,
            value=eq_default,
        )
        year = st.slider(
            "Year",
            min_value=year_min,
            max_value=year_max,
            value=year_default,
            step=1,
        )

    with c2:
        document_type = st.selectbox("Document Type", doc_types)
        department = st.selectbox("Department", departments)
        status = st.selectbox("Project Status", statuses)

        approval_delay = st.slider(
            "Approval Delay (days)",
            min_value=appr_min,
            max_value=appr_max,
            value=appr_default,
            step=1,
        )
        signing_delay = st.slider(
            "Signing Delay (days)",
            min_value=sign_min,
            max_value=sign_max,
            value=sign_default,
            step=1,
        )
        invest_delay = st.slider(
            "Investment Delay (days)",
            min_value=inv_min,
            max_value=inv_max,
            value=inv_default,
            step=1,
        )

    # ---- Collect user input in a dict
    user_input = {
        "Document Type": document_type,
        "Product Line": product_line,
        "Country": country,
        "Industry": industry,
        "Department": department,
        "Status": status,
        loan_col: loan_amt,
        eq_col: equity_amt,
        year_col: year,
        appr_col: approval_delay,
        sign_col: signing_delay,
        inv_col: invest_delay,
    }

    if st.button("Predict for New Project", key="tab2_btn"):
        # ---------------------------------------------
        # 1) Build input for CLASSIFICATION model
        # ---------------------------------------------
        clf_row_dict = {}
        for col in clf_feature_cols:
            if col in user_input:
                clf_row_dict[col] = user_input[col]
            else:
                # fallback to dataset default if not asked from the user
                clf_row_dict[col] = default_values.get(col)

        X_clf_new = pd.DataFrame([clf_row_dict])

        # Predict category
        clf_pred_int = clf_model.predict(X_clf_new)[0]
        clf_pred_label = label_encoder.inverse_transform([clf_pred_int])[0]
        clf_proba = clf_model.predict_proba(X_clf_new)[0]

        # ---------------------------------------------
        # 2) Build input for REGRESSION model
        #    - use user inputs
        #    - fill Environmental Category + EnvCatShort from prediction
        #    - fill any other columns with defaults
        # ---------------------------------------------
        reg_row_dict = {}
        for col in reg_feature_cols:
            if col in user_input:
                reg_row_dict[col] = user_input[col]
            elif col == "Environmental Category":
                reg_row_dict[col] = clf_pred_label
            elif col == "EnvCatShort":
                reg_row_dict[col] = envcat_to_short.get(clf_pred_label, "O")
            else:
                reg_row_dict[col] = default_values.get(col)

        X_reg_new = pd.DataFrame([reg_row_dict])

        reg_pred = reg_model.predict(X_reg_new)[0]

        # ---------------------------------------------
        # 3) Show results (no table, just metrics)
        # ---------------------------------------------
        cc1, cc2 = st.columns(2)
        with cc1:
            st.metric(
                "Predicted Environmental Category (XGBoost Classifier)",
                clf_pred_label,
            )
        with cc2:
            st.metric(
                "Estimated Total IFC Investment (Million USD, XGBoost Regressor)",
                f"{reg_pred:,.2f}",
            )

        st.write("### Category probabilities for this new project")
        proba_df = (
            pd.DataFrame(
                {"Category": label_encoder.classes_, "Probability": clf_proba}
            )
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(proba_df)

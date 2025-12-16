# =====================================================
# Patient Adherence Prediction - Single Page Dashboard
# =====================================================

import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# -----------------------------------------------------
# Page Config
# -----------------------------------------------------
st.set_page_config(
    page_title="Patient Adherence Dashboard",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------------------------------
# Modern CSS Styling
# -----------------------------------------------------
st.markdown("""
<style>
.header {
    font-size: 36px;
    font-weight: 700;
    color: #111827;
}
.subheader {
    font-size: 16px;
    color: #6b7280;
}
.card {
    padding: 24px;
    border-radius: 18px;
    background-color: #ffffff;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
}
.good {
    background: linear-gradient(135deg,#22c55e,#16a34a);
    color: white;
    padding: 20px;
    border-radius: 16px;
}
.bad {
    background: linear-gradient(135deg,#ef4444,#b91c1c);
    color: white;
    padding: 20px;
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# Header
# -----------------------------------------------------
st.markdown('<div class="header">🧠 Patient Adherence Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Clinical Decision Support • Machine Learning Dashboard</div>', unsafe_allow_html=True)
st.divider()

# -----------------------------------------------------
# Load / Train Model
# -----------------------------------------------------
MODEL_PATH = "best_patient_adherence_model.pkl"
DATA_PATH = "patient_adherence_dataset.csv"

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    X = df.drop("Adherence", axis=1)
    y = df["Adherence"]

    num_cols = [
        "Age", "Dosage_mg", "Income",
        "Comorbidities_Count",
        "Previous_Adherence",
        "Insurance_Coverage"
    ]

    cat_cols = [
        "Gender", "Medication_Type", "Education_Level",
        "Social_Support_Level", "Condition_Severity",
        "Healthcare_Access", "Mental_Health_Status"
    ]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", SVC(probability=True, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline

model = load_or_train_model()

# -----------------------------------------------------
# Layout: Input (Left) | Output (Right)
# -----------------------------------------------------
left, right = st.columns([2, 1])

# ---------------- LEFT: INPUT FORM -------------------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Patient Information")

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", 18, 90, 45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            education = st.selectbox("Education", ["High School", "Graduate", "Postgraduate"])

        with col2:
            medication = st.selectbox("Medication Type", ["TypeA", "TypeB", "TypeC"])
            dosage = st.number_input("Dosage (mg)", 10, 500, 100)
            income = st.number_input("Annual Income", 50000, 2000000, 300000)

        with col3:
            social_support = st.selectbox("Social Support", ["Low", "Medium", "High"])
            severity = st.selectbox("Condition Severity", ["Mild", "Moderate", "Severe"])
            comorbidity = st.slider("Comorbidities", 0, 5, 1)

        healthcare = st.radio("Healthcare Access", ["Poor", "Good"], horizontal=True)
        mental_health = st.radio("Mental Health Status", ["Poor", "Moderate", "Good"], horizontal=True)
        insurance = st.checkbox("Insurance Coverage")
        previous = st.checkbox("Previously Adherent")

        submit = st.form_submit_button("🚀 Predict Adherence")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- RIGHT: RESULT -------------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    if submit:
        input_df = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Medication_Type": medication,
            "Dosage_mg": dosage,
            "Previous_Adherence": int(previous),
            "Education_Level": education,
            "Income": income,
            "Social_Support_Level": social_support,
            "Condition_Severity": severity,
            "Comorbidities_Count": comorbidity,
            "Healthcare_Access": healthcare,
            "Mental_Health_Status": mental_health,
            "Insurance_Coverage": int(insurance)
        }])

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.markdown('<div class="good">✅ Likely to Adhere</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="bad">❌ Not Likely to Adhere</div>', unsafe_allow_html=True)

        st.metric("Adherence Probability", f"{prob:.2%}")
        st.progress(int(prob * 100))

    else:
        st.info("Fill patient details and click **Predict Adherence**")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------
# Footer
# -----------------------------------------------------
st.divider()
st.caption("Single-Page Clinical Dashboard • Streamlit • Machine Learning")

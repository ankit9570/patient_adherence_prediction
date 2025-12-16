# =====================================================
# Patient Adherence Prediction - SAFE Streamlit Version
# =====================================================

import streamlit as st
import pandas as pd

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
# CSS
# -----------------------------------------------------
st.markdown("""
<style>
.header {font-size:36px;font-weight:700;}
.sub {color:#6b7280;}
.card {padding:24px;border-radius:18px;background:#fff;
       box-shadow:0 10px 25px rgba(0,0,0,.08);}
.good {background:#22c55e;color:white;padding:20px;border-radius:16px;}
.bad {background:#ef4444;color:white;padding:20px;border-radius:16px;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# Header
# -----------------------------------------------------
st.markdown('<div class="header">🧠 Patient Adherence Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Clinical Decision Support • ML Dashboard</div>', unsafe_allow_html=True)
st.divider()

# -----------------------------------------------------
# Train Model INSIDE Streamlit (SAFE)
# -----------------------------------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("patient_adherence_dataset (1).csv")

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

    model = Pipeline([
        ("preprocess", preprocessor),
        ("model", SVC(probability=True, random_state=42))
    ])

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    return model


model = train_model()

# -----------------------------------------------------
# Layout
# -----------------------------------------------------
left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Patient Information")

    with st.form("form"):
        age = st.slider("Age", 18, 90, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        medication = st.selectbox("Medication", ["TypeA", "TypeB", "TypeC"])
        dosage = st.number_input("Dosage (mg)", 10, 500, 100)
        education = st.selectbox("Education", ["High School", "Graduate", "Postgraduate"])
        income = st.number_input("Income", 50000, 2000000, 300000)
        social = st.selectbox("Social Support", ["Low", "Medium", "High"])
        severity = st.selectbox("Condition Severity", ["Mild", "Moderate", "Severe"])
        comorbidity = st.slider("Comorbidities", 0, 5, 1)
        healthcare = st.selectbox("Healthcare Access", ["Poor", "Good"])
        mental = st.selectbox("Mental Health", ["Poor", "Moderate", "Good"])
        insurance = st.checkbox("Insurance Coverage")
        previous = st.checkbox("Previous Adherence")

        submit = st.form_submit_button("🚀 Predict")

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction")

    if submit:
        data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Medication_Type": medication,
            "Dosage_mg": dosage,
            "Previous_Adherence": int(previous),
            "Education_Level": education,
            "Income": income,
            "Social_Support_Level": social,
            "Condition_Severity": severity,
            "Comorbidities_Count": comorbidity,
            "Healthcare_Access": healthcare,
            "Mental_Health_Status": mental,
            "Insurance_Coverage": int(insurance)
        }])

        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]

        if pred == 1:
            st.markdown('<div class="good">✅ Likely to Adhere</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="bad">❌ Not Likely to Adhere</div>', unsafe_allow_html=True)

        st.metric("Probability", f"{prob:.2%}")
        st.progress(int(prob * 100))

    else:
        st.info("Enter patient data and click Predict")

    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.caption("Streamlit Cloud Safe • No Pickle Errors • ML Pipeline")

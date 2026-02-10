import pickle
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

## config & styling
st.set_page_config(
    page_title="CardioCheck AI",
    layout="wide",
)

def local_css():
    st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
        [data-testid="stExpander"] { background: white; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

local_css()

## model loading
@st.cache_resource
def load_artifacts():
    """Load model and return. Added error handling for missing files."""
    model_path = Path(__file__).parent / "model" / "model.pkl"
    if not model_path.exists():
        st.error(f"Model file not found at {model_path}. Please ensure the model is trained.")
        st.stop()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Trying to unpickle estimator", category=UserWarning)
        with open(model_path, "rb") as f:
            return pickle.load(f)

## ui components
def draw_probability_gauge(proba):
    """Creates a clean Plotly gauge for the prediction probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score (%)", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#dc2626" if proba > 0.5 else "#15803d"},
            'steps': [
                {'range': [0, 40], 'color': "#dcfce7"},
                {'range': [40, 70], 'color': "#fef9c3"},
                {'range': [70, 100], 'color': "#fee2e2"}
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def main():
    st.title("CardioCheck AI")
    st.subheader("Clinical Decision Support Dashboard")
    
    model = load_artifacts()

    ## input section
    with st.form("clinical_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Patient Info")
            age = st.number_input("Age", 1, 120, 55)
            sex = st.selectbox("Biological Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            ca = st.slider("Major Vessels (0-3) Colored by Flourosopy", 0, 3, 0)
            
        with col2:
            st.markdown("### Vitals")
            trestbps = st.number_input("Resting BP (mmHg)", 80, 250, 120)
            chol = st.number_input("Serum Chol (mg/dl)", 100, 600, 200)
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)
            fbs = st.radio("Fasting Blood Sugar > 120mg/dl", [0, 1], format_func=lambda x: "True" if x == 1 else "False", horizontal=True)

        with col3:
            st.markdown("### Clinical Tests")
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                              format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x])
            restecg = st.selectbox("Resting ECG", [0, 1, 2], 
                                   format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "LV Hypertrophy"][x])
            exang = st.radio("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
            oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.2, 1.0)
            slope = st.selectbox("ST Slope", [0, 1, 2])
            thal = st.selectbox("Thalassemia", [0, 1, 2, 3], format_func=lambda x: ["Null", "Fixed Defect", "Normal", "Reversible Defect"][x])

        submitted = st.form_submit_button("Run Diagnostic Analysis", type="primary")

    ## prediction logic
    if submitted:
        input_data = pd.DataFrame([{
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }])

        prediction = model.predict(input_data)[0]
        try:
            probability = model.predict_proba(input_data)[0][1]
        except (AttributeError, TypeError):
            probability = None
            st.warning(
                "Probability unavailable (model was saved with a different scikit-learn version). "
                "Run `pip install 'scikit-learn>=1.8.0'` to fix."
            )

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if probability is not None:
                st.plotly_chart(draw_probability_gauge(probability), use_container_width=True)
            else:
                st.plotly_chart(draw_probability_gauge(0.5), use_container_width=True)
                st.caption("Gauge placeholder — upgrade scikit-learn for real probability.")

        with res_col2:
            if prediction == 1:
                st.error("### High Risk Detected")
                if probability is not None:
                    st.write(f"The model indicates a **{probability:.1%}** probability of heart disease. Urgent clinical consultation is advised.")
                else:
                    st.write("The model indicates **heart disease (positive)**. Urgent clinical consultation is advised.")
            else:
                st.success("### Low Risk Detected")
                if probability is not None:
                    st.write(f"The model indicates a **{probability:.1%}** probability of heart disease. Maintain regular checkups.")
                else:
                    st.write("The model indicates **no heart disease (negative)**. Maintain regular checkups.")

    ## footer
    st.markdown("---")
    st.caption("**Disclaimer:** This tool is for research purposes only. Decisions should be made by qualified medical professionals.")

if __name__ == "__main__":
    main()
import os
import requests
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(
    page_title="LLM Hallucination Detection",
    layout="wide"
)

st.title("LLM Hallucination Detection & Responsible AI Dashboard")

DATA_PATH = "data/llm_hallucination_dataset_v1.csv"
MODEL_PATH = "models/tfidf_logistic_model.pkl"
API_URL = os.getenv("API_URL", "http://host.docker.internal:8000/predict")

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Prediction",
    "Explainability",
    "Responsible AI"
])

with tab1:
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Models Evaluated", df["model_name"].nunique())
    col3.metric("Domains", df["domain"].nunique())

    st.dataframe(df.head(20), use_container_width=True)

    fig1 = px.histogram(
        df,
        x="model_name",
        color="hallucination_label",
        title="Hallucination Distribution by Model"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(
        df,
        x="domain",
        color="hallucination_label",
        title="Hallucination Distribution by Domain"
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Live Hallucination Prediction")

    prompt = st.text_area(
        "Enter Prompt",
        value="Who invented the telephone?",
        height=120
    )

    response = st.text_area(
        "Enter LLM Response",
        value="Nikola Tesla invented the telephone in 1876.",
        height=160
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Predict Using FastAPI"):
            payload = {
                "prompt": prompt,
                "response": response
            }

            try:
                api_response = requests.post(
                    API_URL,
                    json=payload,
                    timeout=20
                )

                api_response.raise_for_status()
                result = api_response.json()

                prediction = result["hallucination_prediction"]
                factual_prob = result["factual_probability"]
                hallucination_prob = result["hallucination_probability"]

                if prediction == 1:
                    st.error("Hallucination Detected")
                else:
                    st.success("Response Appears Factual")

                st.metric("Factual Probability", f"{factual_prob:.2%}")
                st.metric("Hallucination Probability", f"{hallucination_prob:.2%}")

            except Exception as e:
                st.error(f"API Error: {e}")
                st.info("Make sure FastAPI is running on port 8000.")

    with col2:
        if st.button("Predict Locally"):
            text = prompt + " " + response
            prediction = model.predict([text])[0]
            probability = model.predict_proba([text])[0]

            if prediction == 1:
                st.error("Hallucination Detected")
            else:
                st.success("Response Appears Factual")

            st.metric("Factual Probability", f"{probability[0]:.2%}")
            st.metric("Hallucination Probability", f"{probability[1]:.2%}")

with tab3:
    st.subheader("Explainable AI Reports")

    if os.path.exists("reports/confusion_matrix.png"):
        st.image("reports/confusion_matrix.png", caption="Confusion Matrix")

    if os.path.exists("reports/roc_curve.png"):
        st.image("reports/roc_curve.png", caption="ROC Curve")

    if os.path.exists("reports/shap_bar.png"):
        st.image("reports/shap_bar.png", caption="SHAP Feature Importance")

    if os.path.exists("reports/lime_explanation.html"):
        st.success("LIME explanation generated successfully.")
        st.code("reports/lime_explanation.html")
    else:
        st.warning("LIME explanation file not found.")

with tab4:
    st.subheader("Responsible AI / Fairness Analysis")

    if os.path.exists("reports/fairness_by_domain.csv"):
        fairness_df = pd.read_csv("reports/fairness_by_domain.csv")
        st.dataframe(fairness_df, use_container_width=True)

    if os.path.exists("reports/fairness_accuracy_by_domain.png"):
        st.image(
            "reports/fairness_accuracy_by_domain.png",
            caption="Fairness Accuracy by Domain"
        )
    else:
        st.warning("Fairness reports not found.")
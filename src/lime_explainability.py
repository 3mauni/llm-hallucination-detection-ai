import pandas as pd
import joblib

from lime.lime_text import LimeTextExplainer

print("=" * 60)
print("LIME Explainability Pipeline")
print("=" * 60)

test_df = pd.read_csv("data/test_processed.csv")

model = joblib.load("models/tfidf_logistic_model.pkl")

explainer = LimeTextExplainer(
    class_names=["Factual", "Hallucinated"]
)

sample_text = test_df["text"].iloc[0]

explanation = explainer.explain_instance(
    sample_text,
    model.predict_proba,
    num_features=10
)

explanation.save_to_file("reports/lime_explanation.html")

print("LIME explanation saved successfully")
print("=" * 60)
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

print("=" * 60)
print("SHAP Explainability Pipeline")
print("=" * 60)

train_df = pd.read_csv("data/train_processed.csv")

pipeline = joblib.load("models/tfidf_logistic_model.pkl")

vectorizer = pipeline.named_steps["tfidf"]
classifier = pipeline.named_steps["classifier"]

X_train_text = train_df["text"]

X_train_tfidf = vectorizer.transform(X_train_text)

feature_names = vectorizer.get_feature_names_out()

explainer = shap.LinearExplainer(
    classifier,
    X_train_tfidf,
    feature_names=feature_names
)

shap_values = explainer(X_train_tfidf[:50])

plt.figure()
shap.plots.bar(
    shap_values,
    max_display=15,
    show=False
)
plt.savefig("reports/shap_bar.png", bbox_inches="tight")
plt.close()

importance = abs(shap_values.values).mean(axis=0)
top_indices = importance.argsort()[-15:]

print("\nTop important words:")

for idx in reversed(top_indices):
    print(f"{feature_names[idx]} -> importance: {importance[idx]:.4f}")

print("\nSHAP explainability completed successfully")
print("=" * 60)
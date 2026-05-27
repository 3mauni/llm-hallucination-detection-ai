import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score
from fairlearn.metrics import MetricFrame, selection_rate

print("=" * 60)
print("Responsible AI Fairness Analysis")
print("=" * 60)

original_df = pd.read_csv("data/llm_hallucination_dataset_v1.csv")
test_df = pd.read_csv("data/test_processed.csv")

model = joblib.load("models/tfidf_logistic_model.pkl")

X_test = test_df["text"]
y_test = test_df["label"]

predictions = model.predict(X_test)

sensitive_feature = original_df.loc[test_df.index, "domain"]

metric_frame = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "selection_rate": selection_rate,
    },
    y_true=y_test,
    y_pred=predictions,
    sensitive_features=sensitive_feature,
)

fairness_report = metric_frame.by_group

print("\nFairness report by domain:")
print(fairness_report)

fairness_report.to_csv("reports/fairness_by_domain.csv")

fairness_report["accuracy"].plot(kind="bar", figsize=(8, 5))
plt.title("Model Accuracy by Domain")
plt.xlabel("Domain")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("reports/fairness_accuracy_by_domain.png")
plt.close()

print("\nFairness analysis completed successfully")
print("=" * 60)
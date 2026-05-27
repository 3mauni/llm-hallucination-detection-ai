import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
)

# Load datasets
test_df = pd.read_csv("data/test_processed.csv")

# Load model
model = joblib.load(
    "models/tfidf_logistic_model.pkl"
)

# Features and labels
X_test = test_df["text"]

y_test = test_df["label"]

# Predictions
y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.savefig(
    "reports/confusion_matrix.png"
)

plt.close()

# =========================
# ROC Curve
# =========================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob,
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.2f}",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig(
    "reports/roc_curve.png"
)

plt.close()

# =========================
# Label Distribution
# =========================

plt.figure(figsize=(6, 5))

sns.countplot(
    x=y_test
)

plt.title(
    "Hallucination Label Distribution"
)

plt.xlabel("Label")

plt.ylabel("Count")

plt.savefig(
    "reports/label_distribution.png"
)

plt.close()

print("=" * 60)

print("Visualizations generated successfully")

print("=" * 60)
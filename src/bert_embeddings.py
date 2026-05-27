import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 60)
print("Sentence Transformer Embedding Model")
print("=" * 60)

train_df = pd.read_csv("data/train_processed.csv")
test_df = pd.read_csv("data/test_processed.csv")

X_train = train_df["text"].tolist()
y_train = train_df["label"]

X_test = test_df["text"].tolist()
y_test = test_df["label"]

print("\nLoading sentence transformer model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating train embeddings...")
X_train_embeddings = embedding_model.encode(X_train, show_progress_bar=True)

print("Generating test embeddings...")
X_test_embeddings = embedding_model.encode(X_test, show_progress_bar=True)

classifier = LogisticRegression(max_iter=1000, random_state=42)

print("\nTraining embedding classifier...")
classifier.fit(X_train_embeddings, y_train)

predictions = classifier.predict(X_test_embeddings)

accuracy = accuracy_score(y_test, predictions)

print("\nEmbedding Model Accuracy:")
print(round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(
    classifier,
    "models/sentence_transformer_classifier.pkl"
)

print("\nSentence Transformer classifier saved successfully")
print("=" * 60)
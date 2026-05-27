import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split

DATA_PATH = "data/llm_hallucination_dataset_v1.csv"

print("=" * 60)
print("LLM Hallucination Preprocessing Pipeline")
print("=" * 60)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully")
print("Original shape:", df.shape)

# Keep only required columns
df = df[
    [
        "prompt_text",
        "response_text",
        "hallucination_label",
        "model_name",
        "domain",
        "task_type",
    ]
]

# Remove missing rows
df = df.dropna()

print("\nShape after removing missing values:")
print(df.shape)

# Combine prompt + response
df["combined_text"] = (
    df["prompt_text"].astype(str)
    + " "
    + df["response_text"].astype(str)
)

# Text cleaning function
def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\\S+", "", text)

    text = re.sub(r"\\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\\s+", " ", text).strip()

    return text

# Apply cleaning
df["clean_text"] = df["combined_text"].apply(clean_text)

print("\nSample cleaned text:")
print(df["clean_text"].iloc[0][:500])

# Features and labels
X = df["clean_text"]

y = df["hallucination_label"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Save processed datasets
train_df = pd.DataFrame({
    "text": X_train,
    "label": y_train
})

test_df = pd.DataFrame({
    "text": X_test,
    "label": y_test
})

train_df.to_csv(
    "data/train_processed.csv",
    index=False
)

test_df.to_csv(
    "data/test_processed.csv",
    index=False
)

print("\nProcessed datasets saved successfully")

print("=" * 60)
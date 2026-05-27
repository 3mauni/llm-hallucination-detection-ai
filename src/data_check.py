import pandas as pd

DATA_PATH = "data/llm_hallucination_dataset_v1.csv"

required_columns = [
    "prompt_text",
    "response_text",
    "model_name",
    "domain",
    "task_type",
    "hallucination_label",
    "hallucination_type",
    "severity",
    "domain_risk",
    "mitigation_strategy",
    "verified_source",
    "correction_text",
]

print("=" * 60)
print("LLM Hallucination Dataset Validation")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully")
print("\nDataset shape:", df.shape)

print("\nDataset columns:")
print(df.columns.tolist())

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print("\nMissing columns:")
    print(missing_columns)
else:
    print("\nAll required columns exist")

print("\nMissing values:")
print(df[required_columns].isnull().sum())

print("\nHallucination label distribution:")
print(df["hallucination_label"].value_counts())

print("\nHallucination type distribution:")
print(df["hallucination_type"].value_counts())

print("\nSeverity distribution:")
print(df["severity"].value_counts())

print("\nDomain distribution:")
print(df["domain"].value_counts())

print("\nModel distribution:")
print(df["model_name"].value_counts())

print("\nDataset validation completed successfully.")
print("=" * 60)
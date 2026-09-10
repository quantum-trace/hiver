import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

GOLDEN_PATH = "evaluation/golden_set.csv"

df = pd.read_csv(GOLDEN_PATH)

# Always predict the most common intent
majority_intent = df["intent"].value_counts().idxmax()

predictions = [majority_intent] * len(df)

accuracy = accuracy_score(df["intent"], predictions)
macro_f1 = f1_score(df["intent"], predictions, average="macro", zero_division=0)

print("Majority intent:", majority_intent)
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))

print("\nClassification report:")
print(classification_report(
    df["intent"],
    predictions,
    zero_division=0
))
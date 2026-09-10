import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report


GOLDEN_PATH = "evaluation/golden_set.csv"

df = pd.read_csv(GOLDEN_PATH)

X = df["text"].fillna("")
y = df["intent"]

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

predictions = cross_val_predict(
    model,
    X,
    y,
    cv=cv
)

accuracy = accuracy_score(y, predictions)
macro_f1 = f1_score(
    y,
    predictions,
    average="macro",
    zero_division=0
)

print("=== TF-IDF + LOGISTIC REGRESSION ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y,
        predictions,
        zero_division=0
    )
)
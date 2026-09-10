import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


GOLDEN_PATH = "evaluation/golden_set.csv"


df = pd.read_csv(GOLDEN_PATH)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

split = int(len(df) * 0.8)

train = df.iloc[:split]
test = df.iloc[split:]

print("Training examples:", len(train))
print("Test examples:", len(test))


vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1
)

X_train = vectorizer.fit_transform(train["text"])
X_test = vectorizer.transform(test["text"])


model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train, train["intent"])


predictions = model.predict(X_test)


accuracy = accuracy_score(
    test["intent"],
    predictions
)

macro_f1 = f1_score(
    test["intent"],
    predictions,
    average="macro",
    zero_division=0
)


print("\n=== CLASSIFIER RESULTS ===")

print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))

print("\nClassification report:")

print(
    classification_report(
        test["intent"],
        predictions,
        zero_division=0
    )
)

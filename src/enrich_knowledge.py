import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


GOLDEN_PATH = "evaluation/golden_set.csv"
KNOWLEDGE_PATH = "data/train_knowledge_clean.csv"
OUTPUT_PATH = "data/train_knowledge_intent.csv"


# -----------------------------
# Load human-labelled examples
# -----------------------------

golden = pd.read_csv(GOLDEN_PATH)

X_golden = golden["text"].fillna("")
y_golden = golden["intent"]


# -----------------------------
# Train intent classifier
# -----------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_golden)

classifier = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

classifier.fit(X_train, y_golden)


# -----------------------------
# Load historical knowledge
# -----------------------------

knowledge = pd.read_csv(KNOWLEDGE_PATH)

knowledge_text = knowledge[
    "customer_text"
].fillna("")


# -----------------------------
# Infer intent
# -----------------------------

X_knowledge = vectorizer.transform(
    knowledge_text
)

knowledge["intent"] = classifier.predict(
    X_knowledge
)


# -----------------------------
# Save
# -----------------------------

knowledge.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Historical knowledge pairs:", len(knowledge))
print("Saved to:", OUTPUT_PATH)

print("\nInferred intent distribution:")
print(
    knowledge["intent"]
    .value_counts()
)
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


GOLDEN_PATH = "evaluation/golden_set.csv"
TRAIN_PATH = "data/train_pairs.csv"


# Load data
train = pd.read_csv(TRAIN_PATH)
golden = pd.read_csv(GOLDEN_PATH)


# For now, create training labels using simple keyword rules.
def assign_intent(text):
    text = str(text).lower()

    if any(word in text for word in [
        "ios update", "ios 11", "ios 10", "ios 12",
        "update", "updating", "downgrade"
    ]):
        return "ios_update_issue"

    if any(word in text for word in [
        "battery", "charging", "charge", "drain"
    ]):
        return "battery_issue"

    if any(word in text for word in [
        "icloud", "sync", "photos disappeared", "data lost"
    ]):
        return "icloud_data_issue"

    if any(word in text for word in [
        "password", "login", "log in", "apple id",
        "activation lock", "locked account"
    ]):
        return "account_security"

    if any(word in text for word in [
        "apple music", "itunes", "airplay"
    ]):
        return "apple_music_media"

    if any(word in text for word in [
        "screen", "button", "headphones", "earphones",
        "cable", "speaker"
    ]):
        return "hardware_issue"

    if any(word in text for word in [
        "order", "purchase", "buy", "payment",
        "refund", "upgrade"
    ]):
        return "order_purchase_issue"

    if any(word in text for word in [
        "slow", "lag", "freeze", "freezing",
        "crash", "crashing"
    ]):
        return "device_performance"

    if any(word in text for word in [
        "app", "keyboard", "siri", "wifi",
        "settings", "notification"
    ]):
        return "app_software_issue"

    return "general_support"


# Generate training labels
train["intent"] = train["customer_text"].apply(assign_intent)


# TF-IDF converts text into numerical features
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

X_train = vectorizer.fit_transform(train["customer_text"])
X_test = vectorizer.transform(golden["text"])


# Train classifier
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, train["intent"])


# Predict golden set
predictions = model.predict(X_test)


# Evaluate
accuracy = accuracy_score(golden["intent"], predictions)
macro_f1 = f1_score(
    golden["intent"],
    predictions,
    average="macro",
    zero_division=0
)

print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))

print("\nClassification report:")
print(
    classification_report(
        golden["intent"],
        predictions,
        zero_division=0
    )
)
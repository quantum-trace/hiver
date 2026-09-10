import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

DATA_PATH = "evaluation/sample_200.csv"

df = pd.read_csv(DATA_PATH)

def assign_intent(text):
    text = str(text).lower()

    if any(x in text for x in ["battery", "battery life", "drain", "dying"]):
        return "battery_issue"

    if any(x in text for x in ["update", "ios 11", "ios 10", "ios 12", "ios 13"]):
        return "ios_update_issue"

    if any(x in text for x in ["freeze", "freezing", "frozen", "lag", "slow", "stuck"]):
        return "device_performance"

    if any(x in text for x in ["home button", "volume button", "screen", "headset", "cable"]):
        return "hardware_issue"

    if any(x in text for x in ["icloud", "photos", "pictures", "notes", "messages erased"]):
        return "icloud_data_issue"

    if any(x in text for x in ["apple music", "airplay", "itunes", "audio", "video"]):
        return "apple_music_media"

    if any(x in text for x in ["siri", "weather", "keyboard", "typing", "messages app"]):
        return "app_software_issue"

    if any(x in text for x in ["activation lock", "password", "verification", "account"]):
        return "account_security"

    if any(x in text for x in ["order", "preorder", "reserved", "payment", "delivery"]):
        return "order_purchase_issue"

    return "general_support"


df["intent"] = df["text"].apply(assign_intent)

print("Intent distribution:")
print(df["intent"].value_counts())

print("\nExamples:")
print(df[["text", "intent"]].head(20).to_string(index=False))
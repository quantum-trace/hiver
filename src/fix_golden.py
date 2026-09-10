import pandas as pd

PATH = "evaluation/golden_set.csv"

df = pd.read_csv(PATH)

corrections = {
    36647: "device_performance",
    487900: "general_support",
    1756336: "general_support",
    1412578: "general_support",
    2586432: "general_support",
}

for tweet_id, new_intent in corrections.items():
    df.loc[df["tweet_id"] == tweet_id, "intent"] = new_intent

df.to_csv(PATH, index=False)

print("Corrections applied.")
print("Rows:", len(df))
print("\nNew distribution:")
print(df["intent"].value_counts())
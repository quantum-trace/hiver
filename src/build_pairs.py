import pandas as pd

DATA_PATH = "data/apple_support.csv"
OUTPUT_PATH = "data/apple_pairs.csv"

df = pd.read_csv(DATA_PATH)

customers = df[df["inbound"] == True].copy()

replies = df[df["inbound"] == False].copy()

pairs = customers.merge(
    replies[["tweet_id", "in_response_to_tweet_id", "text"]],
    left_on="tweet_id",
    right_on="in_response_to_tweet_id",
    how="inner",
    suffixes=("_customer", "_support"),
)

pairs = pairs[
    [
        "tweet_id_customer",
        "text_customer",
        "tweet_id_support",
        "text_support",
    ]
]

pairs.columns = [
    "customer_tweet_id",
    "customer_text",
    "support_tweet_id",
    "support_text",
]

pairs.to_csv(OUTPUT_PATH, index=False)

print("Customer-resolution pairs:", len(pairs))
print(f"Saved to: {OUTPUT_PATH}")

print("\nSample pairs:")

for _, row in pairs.head(5).iterrows():
    print("\nCUSTOMER:")
    print(row["customer_text"])

    print("\nAPPLE SUPPORT:")
    print(row["support_text"])

    print("-" * 60)

import pandas as pd

DATA_PATH = "data/raw/twcs.csv"
OUTPUT_PATH = "data/apple_support.csv"

df = pd.read_csv(
    DATA_PATH,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ],
)

# Find tweets directly involving AppleSupport
apple = df[
    (df["author_id"] == "AppleSupport")
    | (df["text"].str.contains("@AppleSupport", case=False, na=False))
].copy()

apple.to_csv(OUTPUT_PATH, index=False)

print("AppleSupport tweets:", len(apple))
print("Customer tweets:", apple["inbound"].sum())
print("Brand tweets:", (~apple["inbound"]).sum())
print(f"Saved to: {OUTPUT_PATH}")
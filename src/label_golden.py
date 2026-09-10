import pandas as pd

INPUT_PATH = "evaluation/sample_200.csv"
OUTPUT_PATH = "evaluation/golden_set.csv"

INTENTS = {
    "1": "ios_update_issue",
    "2": "battery_issue",
    "3": "device_performance",
    "4": "hardware_issue",
    "5": "icloud_data_issue",
    "6": "apple_music_media",
    "7": "app_software_issue",
    "8": "account_security",
    "9": "order_purchase_issue",
    "10": "general_support",
}

df = pd.read_csv(INPUT_PATH)

try:
    existing = pd.read_csv(OUTPUT_PATH)
    labeled_ids = set(existing["tweet_id"])
    print(f"Already labeled: {len(labeled_ids)}")
except FileNotFoundError:
    existing = pd.DataFrame(columns=["tweet_id", "text", "intent"])
    labeled_ids = set()

for _, row in df.iterrows():

    if row["tweet_id"] in labeled_ids:
        continue

    print("\n" + "=" * 70)
    print("MESSAGE:")
    print(row["text"])
    print("\nChoose intent:")

    for key, value in INTENTS.items():
        print(f"{key}. {value}")

    while True:
        choice = input("\nEnter number (1-10), or q to quit: ").strip()

        if choice == "q":
            existing.to_csv(OUTPUT_PATH, index=False)
            print(f"\nSaved progress to {OUTPUT_PATH}")
            raise SystemExit

        if choice in INTENTS:
            break

        print("Invalid choice. Enter 1-10.")

    new_row = pd.DataFrame([{
        "tweet_id": row["tweet_id"],
        "text": row["text"],
        "intent": INTENTS[choice],
    }])

    existing = pd.concat([existing, new_row], ignore_index=True)
    existing.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {INTENTS[choice]}")

print("\nFinished labeling!")
print(f"Saved to {OUTPUT_PATH}")

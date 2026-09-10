import pandas as pd


RESULTS_PATH = "evaluation/agent_results.csv"
OUTPUT_PATH = "evaluation/agent_errors.csv"


df = pd.read_csv(RESULTS_PATH)

errors = df[
    df["true_intent"] != df["predicted_intent"]
].copy()

errors = errors[
    [
        "tweet_id",
        "text",
        "true_intent",
        "predicted_intent",
        "similarity",
        "escalate",
        "escalation_reason"
    ]
]

errors.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Total examples:", len(df))
print("Incorrect predictions:", len(errors))
print("Error rate:", round(len(errors) / len(df), 4))

print("\nMost common confusion pairs:")

confusions = (
    errors
    .groupby(["true_intent", "predicted_intent"])
    .size()
    .sort_values(ascending=False)
)

print(confusions.head(15))

print("\nErrors saved to:")
print(OUTPUT_PATH)
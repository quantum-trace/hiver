import pandas as pd

INPUT_PATH = "evaluation/agent_results.csv"
OUTPUT_PATH = "evaluation/evidence_human_sample.csv"

df = pd.read_csv(INPUT_PATH)

sample = df.sample(30, random_state=42).copy()

sample = sample[
    [
        "tweet_id",
        "text",
        "true_intent",
        "predicted_intent",
        "similarity",
        "top_evidence_customer",
        "top_evidence_response",
        "evidence_ids",
        "evidence_scores",
        "reply",
    ]
]

sample["evidence_relevance"] = ""
sample["evidence_notes"] = ""

sample.to_csv(OUTPUT_PATH, index=False)

print("Created human evidence sample:", len(sample))
print("Saved to:", OUTPUT_PATH)
import pandas as pd

from agent import run_agent
from sklearn.metrics import accuracy_score, f1_score, classification_report


GOLDEN_PATH = "evaluation/golden_set.csv"
OUTPUT_PATH = "evaluation/agent_results.csv"


# Load golden set
golden = pd.read_csv(GOLDEN_PATH)

results = []

print("Evaluating", len(golden), "messages...\n")

for i, row in golden.iterrows():

    result = run_agent(row["text"])

    results.append({
        "tweet_id": row["tweet_id"],
        "text": row["text"],
        "true_intent": row["intent"],
        "predicted_intent": result["intent"],
        "similarity": result["similarity"],
        "escalate": result["escalate"],
        "escalation_reason": result["reason"],

        "evidence_ids": [
           r["evidence_id"]
           for r in result["retrieved"]
        ],

       "evidence_scores": [
           round(r["similarity"], 4)
           for r in result["retrieved"]
        ],

        "top_evidence_customer": result["retrieved"][0]["customer_message"],
        "top_evidence_response": result["retrieved"][0]["support_response"],

        "reply": result["reply"]
    })

    if (i + 1) % 25 == 0:
        print("Processed:", i + 1)


results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# Intent evaluation
accuracy = accuracy_score(
    results_df["true_intent"],
    results_df["predicted_intent"]
)

macro_f1 = f1_score(
    results_df["true_intent"],
    results_df["predicted_intent"],
    average="macro",
    zero_division=0
)


print("\n=== AGENT INTENT RESULTS ===")

print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))

print("\nClassification report:")

print(
    classification_report(
        results_df["true_intent"],
        results_df["predicted_intent"],
        zero_division=0
    )
)


print("\nResults saved to:")
print(OUTPUT_PATH)
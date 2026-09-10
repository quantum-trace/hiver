import re
import subprocess
import pandas as pd


INPUT_PATH = "evaluation/evidence_human_sample.csv"
OUTPUT_PATH = "evaluation/llm_judge_results.csv"


def ask_llm(prompt):
    result = subprocess.run(
        ["ollama", "run", "qwen2.5:1.5b"],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def yes_no_judge(question):
    prompt = f"""
Answer the following customer-support evaluation question.

{question}

Reply with ONLY:
1
or
0

Do not explain anything.
"""

    response = ask_llm(prompt)

    match = re.search(r"(?<!\d)[01](?!\d)", response)

    if match:
        return int(match.group())

    return None


def judge_case(row):

    customer = row["text"]
    evidence_customer = row["top_evidence_customer"]
    evidence_response = row["top_evidence_response"]
    reply = row["reply"]

    evidence_relevant = yes_no_judge(f"""
Current customer message:
{customer}

Historical customer message:
{evidence_customer}

Historical support response:
{evidence_response}

Is the historical example about the same or a closely related problem
as the current customer message?

1 = yes
0 = no
""")

    reply_helpful = yes_no_judge(f"""
Current customer message:
{customer}

AI reply:
{reply}

Would this reply give the customer a useful next step for their problem?

1 = yes
0 = no
""")

    reply_grounded = yes_no_judge(f"""
Historical support response:
{evidence_response}

AI reply:
{reply}

Does the AI reply stay supported by the historical support response
rather than inventing unsupported information?

1 = yes
0 = no
""")

    reply_appropriate = yes_no_judge(f"""
Current customer message:
{customer}

AI reply:
{reply}

Is this reply appropriate and professional for customer support?

1 = yes
0 = no
""")

    return {
        "evidence_relevant_llm": evidence_relevant,
        "reply_helpful_llm": reply_helpful,
        "reply_grounded_llm": reply_grounded,
        "reply_appropriate_llm": reply_appropriate,
    }


df = pd.read_csv(INPUT_PATH)

results = []

for i, row in df.iterrows():

    print(f"Judging {i + 1}/{len(df)}...")

    judgment = judge_case(row)

    results.append({
        "tweet_id": row["tweet_id"],
        **judgment
    })


results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_PATH, index=False)

print()
print("Done!")
print(f"Saved to: {OUTPUT_PATH}")

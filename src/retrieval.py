import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TRAIN_PATH = "data/train_knowledge_intent.csv"

df = pd.read_csv(TRAIN_PATH)

df = df.dropna(
    subset=["customer_text", "support_text"]
).copy()

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

X = vectorizer.fit_transform(
    df["customer_text"]
)


def retrieve(message, top_k=3, intent=None):

    candidates = df

    
    if intent is not None and "intent" in df.columns:

        filtered = df[
            df["intent"] == intent
        ]

        if len(filtered) >= top_k:

            candidates = filtered

            candidate_indices = candidates.index

            candidate_X = X[
                [df.index.get_loc(i) for i in candidate_indices]
            ]

        else:
            candidate_X = X

    else:
        candidate_X = X

    query_vector = vectorizer.transform(
        [message]
    )

    similarities = cosine_similarity(
        query_vector,
        candidate_X
    ).flatten()

    top_indices = similarities.argsort()[
        -top_k:
    ][::-1]

    results = []

    for position in top_indices:

        row = candidates.iloc[position]

        results.append({
            "evidence_id": int(row["customer_tweet_id"]),
            "customer_message": row["customer_text"],
            "support_response": row["support_text"],
            "similarity": float(
                similarities[position]
            )
        })

    return results


if __name__ == "__main__":

    message = "My iPhone battery is draining very quickly"

    results = retrieve(
        message,
        top_k=3
    )

    for i, result in enumerate(results, 1):

        print(f"\nResult {i}")
        print(
            "Similarity:",
            round(result["similarity"], 4)
        )
        print(
            "Customer:",
            result["customer_message"]
        )
        print(
            "Support:",
            result["support_response"]
        )

# Decision Log

## 1. Brand selection — AppleSupport
Selected AppleSupport because it has a large number of customer-support interactions and contains recurring technical-support issues suitable for retrieval and intent classification.

## 2. Golden-set size — 200 examples
Used 200 manually labelled customer messages, which is within the required 150–250 range.

## 3. Intent taxonomy — 10 intents
Defined 10 support intents to balance coverage with enough examples per category for evaluation.

## 4. Primary-intent labeling
When a message contained multiple topics, labelled the main active customer problem rather than incidental context.

## 5. Historical pair construction
Linked customer messages to support responses using `in_response_to_tweet_id`, treating the support response as the resolution associated with the customer message.

## 6. Excluding direct-message responses
Removed historical responses containing DM/direct-message instructions because these are not useful as reusable public troubleshooting evidence.

## 7. Short-response filtering
Removed extremely short acknowledgement-style responses such as "Correct" and "No problem" because they provide little reusable resolution information.

## 8. Intent-aware retrieval
Restricted retrieval to historical examples assigned to the predicted intent when enough examples were available, reducing cross-intent retrieval errors.

## 9. TF-IDF retrieval
Used TF-IDF with unigram and bigram features as a lightweight, reproducible retrieval method that can run locally without external APIs.

## 10. Rule-based escalation
Escalated account/security and purchase-related issues deterministically because these categories can involve sensitive or transactional actions.

## 11. Retrieval-confidence escalation
Escalated cases with low top-retrieval similarity because weak evidence increases the risk of producing an irrelevant response.

## 12. Local LLM judge
Used Qwen2.5-1.5B through Ollama for LLM-based evaluation because no external OpenAI API key was available and local execution keeps the evaluation reproducible.

## 13. Human/LLM evidence comparison
Compared the local LLM judge with a 30-example human-reviewed evidence sample and reported agreement rather than treating the LLM judge as ground truth.

## 14. Raw dataset handling
Kept the original Kaggle dataset local and excluded raw CSV files from Git because the dataset contains user-generated support conversations and has non-commercial licensing constraints.

## 15. Headline metric selection
Reported both accuracy and macro F1 because accuracy alone can hide poor performance on minority intents.
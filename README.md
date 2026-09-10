# Hiver

## 1. Overview

This project implements a lightweight AI customer-support agent using the Customer Support on Twitter dataset.

The selected support brand is **AppleSupport**.

For each incoming customer message, the system:

1. Classifies the customer's primary intent.
2. Retrieves similar historical AppleSupport cases.
3. Uses the historical support response as evidence for a draft reply.
4. Decides whether the case should be automatically handled or escalated.
5. Evaluates intent classification, retrieval quality, and reply quality.

The system is designed to run locally without requiring an external API.

---

## 2. Dataset

Dataset:

**Customer Support on Twitter** — Kaggle

The original dataset is kept locally and is **not included in this repository**.

Relevant fields include:

- `tweet_id`
- `author_id`
- `inbound`
- `created_at`
- `text`
- `response_tweet_id`
- `in_response_to_tweet_id`

AppleSupport was selected because it has a large number of customer-support interactions and contains recurring technical-support problems suitable for historical-resolution retrieval.

---

## 3. System Architecture

```text
          Customer message
                 |
                 v
          Intent detection
                 |
                 v
  Intent-aware historical retrieval
                 |
                 v
   Top historical support evidence
                 |
       +--------------------+
       |                    |
       v                    v
Draft reply           Escalation policy
       |                    |
       +---------+----------+
                 |
                 v
             Final result

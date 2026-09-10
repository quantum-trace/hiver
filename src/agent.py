import re
from retrieval import retrieve


def detect_intent(message):
    text = message.lower()

    # 1. Purchase/order issues must come before "charge"
    if any(x in text for x in [
        "order", "purchase", "buy", "bought",
        "refund", "preorder", "pre-order",
        "payment", "invoice", "delivery",
        "charged for"
    ]):
        return "order_purchase_issue"

    # 2. Account/security
    if any(x in text for x in [
        "password", "forgot password",
        "apple id", "appleid",
        "login", "log in", "sign in",
        "activation lock", "locked out",
        "account security"
    ]):
        return "account_security"

    # 3. iCloud/data
    if any(x in text for x in [
        "icloud", "i cloud",
        "photos disappeared",
        "photos missing",
        "lost photos",
        "lost data",
        "data disappeared",
        "sync", "synchroniz"
    ]):
        return "icloud_data_issue"

    # 4. Battery / charging
    if any(x in text for x in [
        "battery", "battery life",
        "battery drain", "draining battery",
        "charging", "charger"
    ]):
        return "battery_issue"

    # 5. Hardware
    if any(x in text for x in [
        "screen", "display", "button",
        "volume button", "power button",
        "home button", "cable",
        "headphones", "earphones",
        "speaker", "broken", "cracked",
        "physical damage"
    ]):
        return "hardware_issue"

    # 6. Apple media services
    if any(x in text for x in [
        "apple music", "itunes",
        "airplay", "music", "song",
        "album", "playlist"
    ]):
        return "apple_music_media"

    # 7. iOS update-related problems
    update_words = [
        "ios", "ios update", "update",
        "updating", "downgrade",
        "updated", "upgrade"
    ]

    if any(x in text for x in update_words):
        return "ios_update_issue"

    # 8. Device performance
    if any(x in text for x in [
        "slow", "slower", "lag",
        "lagging", "freezing",
        "freeze", "crashing", "crash",
        "keeps crashing"
    ]):
        return "device_performance"

    # 9. Other software/app problems
    if any(x in text for x in [
        "app", "application",
        "keyboard", "siri",
        "wifi", "wi-fi",
        "settings", "notification",
        "bluetooth", "software"
    ]):
        return "app_software_issue"

    # 10. Generic / non-problem messages
    return "general_support"


def decide_escalation(intent, top_similarity):

    if intent in [
        "account_security",
        "order_purchase_issue"
    ]:
        return True, "Sensitive account or purchase issue"

    if top_similarity < 0.25:
        return True, "Low confidence in historical evidence"

    return False, "Sufficient historical evidence available"


def generate_reply(message, intent, retrieved):

    if not retrieved:
        return (
            "Thanks for reaching out. "
            "We need a little more information to help troubleshoot this issue."
        )

    best = retrieved[0]

    clean_response = best["support_response"]

  
    clean_response = re.sub(
        r"@\w+",
        "",
        clean_response
    )

    clean_response = re.sub(
        r"https?://\S+",
        "",
        clean_response
    )

    clean_response = " ".join(
        clean_response.split()
    )

    reply = (
        "Thanks for reaching out. Based on similar support cases, "
        "a helpful next step is:\n\n"
        f"{clean_response}\n\n"
        "If that doesn't resolve the issue, please let us know."
    )

    return reply


def run_agent(message):
     
    intent = detect_intent(message)

    retrieved = retrieve(
        message,
        top_k=3,
        intent=intent
    )

    top_similarity = (
        retrieved[0]["similarity"]
        if retrieved
        else 0
    )

    escalate, reason = decide_escalation(
        intent,
        top_similarity
    )

    reply = generate_reply(
        message,
        intent,
        retrieved
    )

    return {
        "message": message,
        "intent": intent,
        "similarity": top_similarity,
        "escalate": escalate,
        "reason": reason,
        "reply": reply,
        "retrieved": retrieved
    }


if __name__ == "__main__":

    message = "I was charged for an iPhone order but never received it"

    result = run_agent(message)

    print("Intent:", result["intent"])
    print("Similarity:", result["similarity"])
    print("Escalate:", result["escalate"])
    print("Reason:", result["reason"])
    print("\nDraft reply:")
    print(result["reply"])

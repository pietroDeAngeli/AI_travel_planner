import json
import re
from typing import Any, Dict, List, Optional

from schema import SLOTS, INTENTS, INTENT_SLOTS, JSON_SCHEMA_HINT


# --- Robust JSON extraction (non-greedy) ---
def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first JSON object from model output text.
    """
    m = re.search(r"\{[\s\S]*?\}", text)  # non-greedy
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _get_last_assistant(dialogue_history: Optional[List[Dict[str, str]]]) -> str:
    if not dialogue_history:
        return ""
    for t in reversed(dialogue_history):
        if t.get("role") == "assistant":
            return t.get("content", "")
    return ""


def _quick_intent_override(user_utterance: str, last_assistant: str) -> Optional[str]:
    """
    Cheap heuristics to stabilize:
    - yes/no answers to confirmation -> CONFIRM_DETAILS / CHANGE_DETAILS
    - explicit goodbye -> END_DIALOGUE
    - reply to 'What would you like to change your X to?' -> PROVIDE_CHANGE_VALUE
    - reply to 'Which detail would you like to change?' -> CHANGE_DETAILS
    """
    u = user_utterance.strip().lower()
    la = last_assistant.strip().lower()

    # Goodbye
    if re.search(r"\b(bye|goodbye|ciao|arrivederci|stop|exit|quit)\b", u):
        return "END_DIALOGUE"

    # If the bot just asked which slot to change
    if "which detail would you like to change" in la or "which of these details" in la:
        return "CHANGE_DETAILS"

    # If the bot just asked for the NEW value of a slot-to-change
    if "what would you like to change your" in la:
        return "PROVIDE_CHANGE_VALUE"

    # Confirmation yes/no only if last bot message is a confirmation request
    if "do you confirm" in la or "confirm the details" in la:
        if re.fullmatch(r"(yes|y|ok|okay|sure|certo|va bene|confermo|s[iì]|si)\.?", u):
            return "CONFIRM_DETAILS"
        if re.fullmatch(r"(no|n|nope|nah|non va bene|cambia|modifica)\.?", u):
            return "CHANGE_DETAILS"

    return None


def nlu_parse(
    pipe,
    user_utterance: str,
    dialogue_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Return: {intent, slots{...}, confidence}
    """

    # Keep a short context (your approach)
    history_text = ""
    if dialogue_history:
        last = dialogue_history[-2:]
        history_text = "\n".join([f"{t['role'].upper()}: {t['content']}" for t in last])

    last_assistant = _get_last_assistant(dialogue_history)
    forced_intent = _quick_intent_override(user_utterance, last_assistant)

    # Better example: show ALL slots (avoids bias from JSON_SCHEMA_HINT showing only 4 fields)
    example = {
        "intent": "REQUEST_PLAN",
        "slots": {k: None for k in SLOTS},
        "confidence": 0.0,
    }

    system = (
        "You are an NLU module for a travel-planner dialogue system.\n"
        "Task: classify the user's intent and extract slot values.\n"
        f"Valid intents: {INTENTS}\n"
        f"Valid slots: {SLOTS}\n"
        f"Valid intent->slot constraints: {INTENT_SLOTS}\n\n"
        "Output MUST be a single JSON object and nothing else.\n"
        "Rules:\n"
        "- If the user starts planning a trip (destination/going somewhere) -> START_TRIP.\n"
        "- If the user explicitly asks for the itinerary/plan -> REQUEST_PLAN.\n"
        "- If the user sets travel method (flight/train/car/...) -> TRAVEL_METHOD.\n"
        "- If the user sets accommodation type (hotel/hostel/airbnb/...) -> ACCOMMODATION_PREFERENCE.\n"
        "- If the user wants to change previously given details -> CHANGE_DETAILS.\n"
        "- If the user confirms details (yes/ok/correct) AFTER a confirmation question -> CONFIRM_DETAILS.\n"
        "- If the user provides the NEW value AFTER choosing what to change -> PROVIDE_CHANGE_VALUE.\n"
        "- If the user ends the chat -> END_DIALOGUE.\n"
        "- Put null for unknown slots.\n"
        "- Never invent details.\n"
        "- confidence is a float in [0,1].\n"
    )

    # Optional strong hint for the model (when heuristics detected a forced intent)
    hint_line = f"\nIntent hint (high priority): {forced_intent}\n" if forced_intent else ""

    user = (
        f"Dialogue context (may help):\n{history_text}\n\n"
        f"Last assistant message:\n{last_assistant}\n"
        f"{hint_line}\n"
        f"User utterance:\n{user_utterance}\n\n"
        "Return JSON with keys: intent, slots, confidence.\n"
        f"Example format:\n{json.dumps(example)}"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    out = pipe(messages, max_new_tokens=256)
    generated = out[0]["generated_text"]
    if isinstance(generated, list):
        text = generated[-1].get("content", "")
    else:
        text = str(generated)

    parsed = _extract_json(text)
    if not parsed:
        return {"intent": "FALLBACK", "slots": {k: None for k in SLOTS}, "confidence": 0.0}

    intent = parsed.get("intent", "FALLBACK")
    if intent not in INTENTS:
        intent = "FALLBACK"

    # If we have a forced intent from heuristics, override
    if forced_intent in INTENTS:
        intent = forced_intent

    slots = parsed.get("slots", {}) or {}
    clean_slots = {k: slots.get(k, None) for k in SLOTS}

    conf = parsed.get("confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    # Keep your threshold behavior (optional; you can relax it)
    if conf < 0.6 and intent not in ("CONFIRM_DETAILS", "END_DIALOGUE"):
        intent = "FALLBACK"
        clean_slots = {k: None for k in SLOTS}

    return {"intent": intent, "slots": clean_slots, "confidence": conf}

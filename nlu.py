import json
import re
from typing import Any, Dict, List, Optional

from schema import SLOTS, INTENTS, INTENT_SLOTS, RULES

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Return a JSON object extracted from text, or None if not found.
    
    :param text: Description
    :type text: str
    :return: Description
    :rtype: Dict[str, Any] | None
    """
    # Remove markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text)
    text = text.replace("```", "").strip()

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    end = None
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        return None

    candidate = text[start:end]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _get_last_assistant(dialogue_history: Optional[List[Dict[str, str]]]) -> str:
    """
    Return the last assistant message from dialogue history.
    
    :param dialogue_history: Description
    :type dialogue_history: Optional[List[Dict[str, str]]]
    :return: Description
    :rtype: str
    """
    if not dialogue_history:
        return ""
    for t in reversed(dialogue_history):
        if t.get("role") == "assistant":
            return t.get("content", "")
    return ""

def _assistant_asked_for_confirmation(last_assistant: str) -> bool:
    la = (last_assistant or "").strip().lower()
    if not la:
        return False

    # confirmation request patterns
    patterns = [
        r"\bdo you confirm\b",
        r"\bcan you confirm\b",
        r"\bis that correct\b",
        r"\bis this correct\b",
        r"\bdoes that look right\b",
        r"\bis that right\b",
        r"\bplease confirm\b",
        r"\bconfirm the details\b",
        r"\bwould you like to modify anything\b",
        r"\bor would you like to modify anything\b",
    ]

    if any(re.search(p, la) for p in patterns):
        return True

    # fallback: se è una domanda e contiene parole tipiche di conferma
    if "?" in la and re.search(r"\b(confirm|correct|right|modify)\b", la):
        return True

    return False


def _quick_intent_override(user_utterance: str, last_assistant: str) -> Optional[str]:
    """
    Quick heuristic rules to override intent based on simple patterns.
    :param user_utterance: Description
    :type user_utterance: str
    :param last_assistant: Description
    :type last_assistant: str
    """
    u = user_utterance.strip().lower()
    la = last_assistant.strip().lower()

    # Goodbye
    if re.search(r"\b(bye|goodbye|stop|exit|quit)\b", u):
        return "END_DIALOGUE"
    
    # Hi/Hello
    if re.search(r"\b(hi|hello|hey)\b", u):
        return "GREETING"
    
    # Yes/Confirm
    if re.search(r"\b(yes|yeah|yep|correct|right|ok|okay|sure|absolutely|of course|no|nope|nah)\b", u):
        if _assistant_asked_for_confirmation(la):
            return "CONFIRM_DETAILS"

    return None


def nlu_parse(
    pipe,
    user_utterance: str,
    dialogue_history: Optional[List[Dict[str, str]]] = None,
    current_intent: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return: {intent, slots{...}}
    """
    # Keep a short context
    history_text = ""
    if dialogue_history:
        last = dialogue_history[-2:]
        history_text = "\n".join([f"{t['role'].upper()}: {t['content']}" for t in last])

    last_assistant = _get_last_assistant(dialogue_history)
    forced_intent = _quick_intent_override(user_utterance, last_assistant, current_intent)

    if forced_intent:
        allowed_slots = INTENT_SLOTS.get(forced_intent, [])
        return {"intent": forced_intent, "slots": {k: None for k in allowed_slots}}

    system = (
        "You are an NLU module for a travel-planner dialogue system.\n"
        "Task: classify the user's intent and extract slot values.\n"
        f"Valid intents: {INTENTS}\n"
        f"Valid slots: {SLOTS}\n"
        f"Valid intent->slot constraints: {INTENT_SLOTS}\n\n"
        "Output MUST be a single JSON object and nothing else.\n"
        f"{RULES}\n"
        "- Put null for unknown slots.\n"
        "- Never invent details.\n"
    )

    user = (
        f"Dialogue context (may help):\n{history_text}\n\n"
        f"Last assistant message:\n{last_assistant}\n"
        f"User utterance:\n{user_utterance}\n\n"
        "Return JSON with keys: intent, slots.\n"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        out = pipe(messages, max_new_tokens=256)
    except Exception as e:
        print(f"Error calling pipe: {e}")
        return {"intent": "OOD", "slots": {k: None for k in SLOTS}}
    
    try:
        generated = out[0]["generated_text"]
        if isinstance(generated, list):
            text = generated[-1].get("content", "")
        else:
            text = str(generated)
    except (IndexError, KeyError, TypeError) as e:
        print(f"Error extracting generated text: {e}")
        return {"intent": "OOD", "slots": {k: None for k in SLOTS}}

    print(text)

    parsed = extract_json(text)
    if not parsed:
        return {"intent": "OOD", "slots": {k: None for k in SLOTS}}

    intent = parsed.get("intent", "OOD")
    if intent not in INTENTS:
        intent = "OOD"
    raw_slots = parsed.get("slots", {}) or {}

    # keep only allowed slots for the predicted intent
    allowed_slots = INTENT_SLOTS.get(intent, [])
    clean_slots = {k: raw_slots.get(k, None) for k in allowed_slots}

    return {"intent": intent, "slots": clean_slots}

import json
import re
from typing import Any, Dict, List, Optional

INTENTS = ["START_TRIP", "PROVIDE_INFO", "REQUEST_PLAN", "MODIFY_PLAN", "END_DIALOGUE"]
SLOTS = ["destination", "dates", "budget", "duration", "travel_style"]

JSON_SCHEMA_HINT = {
    "intent": "START_TRIP",
    "slots": {
        "destination": None,
        "dates": None,
        "budget": None,
        "duration": None,
        "travel_style": None,
    },
    "confidence": 0.0
}

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Estrae il primo oggetto JSON da una stringa.
    """
    # Prova a trovare un blocco {...}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return None

def nlu_parse(pipe, user_utterance: str, dialogue_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Ritorna: {intent, slots{...}, confidence}
    dialogue_history: lista di turni (role/content) utile per disambiguare (facoltativo).
    """
    history_text = ""
    if dialogue_history:
        # tienila breve: ultimi 4 turni
        last = dialogue_history[-4:]
        history_text = "\n".join([f"{t['role'].upper()}: {t['content']}" for t in last])

    system = (
        "You are an NLU module for a travel-planner dialogue system.\n"
        "Task: classify the user's intent and extract slot values.\n"
        f"Valid intents: {INTENTS}\n"
        f"Valid slots: {SLOTS}\n\n"
        "Output MUST be a single JSON object and nothing else.\n"
        "Rules:\n"
        "- If the user starts planning a trip (mentions destination/going somewhere) -> START_TRIP.\n"
        "- If the user provides missing info (dates, budget, duration, style) -> PROVIDE_INFO.\n"
        "- If the user explicitly asks for the itinerary/plan -> REQUEST_PLAN.\n"
        "- If the user asks to change the plan (add/remove day, change style/budget) -> MODIFY_PLAN.\n"
        "- If the user ends the conversation -> END_DIALOGUE.\n"
        "- Put null for unknown slots.\n"
        "- confidence is a float in [0,1].\n"
        "- Never invent details.\n"
    )

    user = (
        f"Dialogue context (may help):\n{history_text}\n\n"
        f"User utterance:\n{user_utterance}\n\n"
        "Return JSON with keys: intent, slots, confidence.\n"
        f"Example format:\n{json.dumps(JSON_SCHEMA_HINT)}"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    out = pipe(messages, max_new_tokens=256)
    # pipeline chat returns list; generated_text is conversation-like
    generated = out[0]["generated_text"]
    # Llama chat pipeline often returns list of messages; here we handle both cases:
    if isinstance(generated, list):
        text = generated[-1].get("content", "")
    else:
        text = str(generated)

    parsed = _extract_json(text)
    if not parsed:
        # fallback safe
        return {"intent": "PROVIDE_INFO", "slots": {k: None for k in SLOTS}, "confidence": 0.0}

    # sanitizzazione
    intent = parsed.get("intent", "PROVIDE_INFO")
    if intent not in INTENTS:
        intent = "PROVIDE_INFO"

    slots = parsed.get("slots", {})
    clean_slots = {k: slots.get(k, None) for k in SLOTS}

    conf = parsed.get("confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    return {"intent": intent, "slots": clean_slots, "confidence": conf}

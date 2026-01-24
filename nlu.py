import json
import re
from typing import Any, Dict, List, Optional

from schema import INTENTS, INTENT_SLOTS, RULES

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Return a JSON object extracted from text, or None if not found."""
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
    """Return the last assistant message from dialogue history."""
    if not dialogue_history:
        return ""
    for t in reversed(dialogue_history):
        if t.get("role") == "assistant":
            return t.get("content", "")
    return ""

def nlu_parse(
    pipe,
    user_utterance: str,
    system_prompt: str,
    dialogue_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    NLU module: classify intent and extract slots.
    Returns: {intent, slots{...}}
    """
    # Keep short context
    history_text = ""
    if dialogue_history:
        last = dialogue_history[-2:]
        history_text = "\n".join([f"{t['role'].upper()}: {t['content']}" for t in last])

    last_assistant = _get_last_assistant(dialogue_history)

    user = (
        f"Last assistant: {last_assistant}\n"
        f"Dialogue context:\n{history_text}\n\n"
        f"User utterance: {user_utterance}\n"
        "\nReturn JSON with keys: intent, slots."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]

    try:
        out = pipe(messages, max_new_tokens=256)
    except Exception as e:
        print(f"Error calling pipe: {e}")
        return {"intent": "OOD", "slots": {}}
    
    try:
        generated = out[0]["generated_text"]
        if isinstance(generated, list):
            text = generated[-1].get("content", "")
        else:
            text = str(generated)
    except (IndexError, KeyError, TypeError) as e:
        print(f"Error extracting generated text: {e}")
        return {"intent": "OOD", "slots": {}}

    parsed = extract_json(text)
    if not parsed:
        return {"intent": "OOD", "slots": {}}

    intent = parsed.get("intent", "OOD")
    if intent not in INTENTS:
        intent = "OOD"
    
    raw_slots = parsed.get("slots", {}) or {}

    # Keep only allowed slots for the predicted intent
    allowed_slots = INTENT_SLOTS.get(intent, [])
    clean_slots = {k: raw_slots.get(k, None) for k in allowed_slots}

    return {"intent": intent, "slots": clean_slots}

import json
import re
from typing import Any, Dict, List, Optional

from dm import DialogueState
from schema import INTENTS, INTENT_SLOTS, RULES

def state_context(state: DialogueState) -> str:
    """Generate a context-aware NLU system prompt based on dialogue state."""
    
    # If no prior action, return base prompt
    if not state.last_action:
        return sys_base_prompt
    
    # Context-specific prompts based on last action
    if state.last_action in ["ASK_CONFIRMATION", "OFFER_SLOT_CARRYOVER"]:
        return _sys_confirmation_prompt(state)
    
    elif state.last_action == "REQUEST_SLOT_CHANGE":
        return _sys_slot_change_prompt(state)
    
    # Fallback to base prompt
    return sys_base_prompt

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
        out = pipe(messages, max_new_tokens=100)
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
    
    # Special case: if confirmation slot is present, always keep it
    # This handles ASK_CONFIRMATION and OFFER_SLOT_CARRYOVER states
    if "confirmation" in raw_slots:
        clean_slots = {"confirmation": raw_slots["confirmation"]}
        # Also include other allowed slots if present
        for k in allowed_slots:
            if k in raw_slots:
                clean_slots[k] = raw_slots[k]
    elif "slot_name" in raw_slots:
        clean_slots = {"slot_name": raw_slots["slot_name"]}
        # Also include other allowed slots if present
        for k in allowed_slots:
            if k in raw_slots:
                clean_slots[k] = raw_slots[k]
    else:
        clean_slots = {k: raw_slots.get(k, None) for k in allowed_slots}

    return {"intent": intent, "slots": clean_slots}


sys_base_prompt = (
    "You are an NLU module for a travel booking dialogue system.\n"
    "Task: classify the user's intent and extract slot values.\n\n"
    f"Valid intents: {INTENTS}\n\n"
    f"Valid slots per intent: {INTENT_SLOTS}\n\n"
    f"{RULES}\n\n"
    "Output MUST be a single JSON object with keys: intent, slots\n"
    "- Put null for unknown slots.\n"
    "- Never invent details.\n"
    "- Only include slots relevant to the detected intent.\n"
)

def _sys_confirmation_prompt(state: DialogueState) -> str:
    """Generate confirmation prompt with state context."""
    return (
        "You are an NLU module for a travel booking dialogue system.\n"
        "The system just asked for confirmation.\n\n"
        f"Current intent: {state.current_intent}\n\n"
        "Task: Determine if the user's response is positive or negative.\n"
        "- Positive: yes, yeah, sure, okay, correct, right, confirm, proceed\n"
        "- Negative: no, nope, wrong, change, modify, different\n\n"
        f"Return the current intent with a 'confirmation' slot.\n"
        "Output MUST be a single JSON object with keys: intent, slots\n"
        f"Example positive: {{\"intent\": \"{state.current_intent}\", \"slots\": {{\"confirmation\": \"yes\"}}}}\n"
        f"Example negative: {{\"intent\": \"{state.current_intent}\", \"slots\": {{\"confirmation\": \"no\"}}}}\n"
    )

def _sys_slot_change_prompt(state: DialogueState) -> str:
    """Generate slot change prompt with state context."""
    return (
        "You are an NLU module for a travel booking dialogue system.\n"
        "The user is indicating which slot/field they want to change.\n\n"
        f"Current intent: {state.current_intent}\n"
        f"Valid slots: {INTENT_SLOTS.get(state.current_intent, [])}\n\n"
        "Task: Identify which slot name the user is referring to.\n"
        "Map user's words to the actual slot name.\n"
        "- 'destination' or 'city' or 'place' -> slot_name: 'destination'\n"
        "- 'activity' or 'category' or 'type' -> slot_name: 'activity_category'\n"
        "- 'budget' or 'price' -> slot_name: 'budget_level'\n"
        "- 'check-in' or 'arrival' -> slot_name: 'check_in_date'\n"
        "- 'check-out' or 'departure' -> slot_name: 'check_out_date'\n"
        "- 'guests' or 'people' -> slot_name: 'num_guests'\n"
        "- 'passengers' -> slot_name: 'num_passengers'\n"
        "- 'origin' or 'from' -> slot_name: 'origin'\n"
        "\nOutput MUST be a single JSON object with keys: intent, slots\n"
        f"Example if user says 'the destination': {{\"intent\": \"{state.current_intent}\", \"slots\": {{\"slot_name\": \"destination\"}}}}\n"
        f"Example if user says 'budget': {{\"intent\": \"{state.current_intent}\", \"slots\": {{\"slot_name\": \"budget_level\"}}}}\n"
    )
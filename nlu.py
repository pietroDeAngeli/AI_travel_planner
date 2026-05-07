import json
import re
from typing import Any, Dict, List, Optional
from datetime import date, datetime
from dateutil import parser as dateutil_parser

from schema import INTENTS, INTENT_SLOTS, ACTIVITY_CATEGORIES

import logging
logging.basicConfig(level=logging.DEBUG)

current_date = date.today().isoformat()

# ── Grounding helpers ──────────────────────────────────────────────

# Budget synonyms → canonical value
_BUDGET_SYNONYMS: Dict[str, str] = {
    # low
    "low": "low", "cheap": "low", "budget": "low", "economy": "low",
    "economic": "low", "inexpensive": "low", "affordable": "low",
    "backpacker": "low", "thrifty": "low", "basso": "low", "economico": "low",
    # medium
    "medium": "medium", "mid": "medium", "moderate": "medium",
    "standard": "medium", "average": "medium", "normal": "medium",
    "regular": "medium", "medio": "medium",
    # high
    "high": "high", "luxury": "high", "expensive": "high", "premium": "high",
    "first class": "high", "first-class": "high", "vip": "high",
    "deluxe": "high", "upscale": "high", "alto": "high", "lusso": "high",
}

# Number words → int  (covers 1-20 plus common words)
_NUM_WORDS: Dict[str, int] = {
    "zero": 0, "one": 1, "a": 1, "an": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "couple": 2, "pair": 2, "dozen": 12,
    "uno": 1, "due": 2, "tre": 3, "quattro": 4, "cinque": 5,
}

# Build reverse index: keyword → canonical activity category
_ACTIVITY_KEYWORD_MAP: Dict[str, str] = {}
for _cat, _keywords in ACTIVITY_CATEGORIES.items():
    _ACTIVITY_KEYWORD_MAP[_cat] = _cat          # category name maps to itself
    for _kw in _keywords:
        _ACTIVITY_KEYWORD_MAP[_kw] = _cat


def _ground_budget(value: Any) -> Optional[str]:
    """Normalise budget_level via synonym lookup."""
    s = str(value).lower().strip()
    if s in _BUDGET_SYNONYMS:
        return _BUDGET_SYNONYMS[s]
    # Regex fallback: see if any synonym appears as a substring
    for syn, canon in _BUDGET_SYNONYMS.items():
        if re.search(rf"\b{re.escape(syn)}\b", s):
            return canon
    return None


def _ground_activity_category(value: Any) -> Optional[str]:
    """Map raw activity text to a canonical category via keyword index."""
    s = str(value).lower().strip()
    # Direct hit
    if s in _ACTIVITY_KEYWORD_MAP:
        return _ACTIVITY_KEYWORD_MAP[s]
    # Check if any keyword appears in the string
    for kw, cat in sorted(_ACTIVITY_KEYWORD_MAP.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(kw)}\b", s):
            return cat
    return None


def _ground_number(value: Any) -> Optional[int]:
    """Extract a positive integer from a value (digit or word)."""
    if isinstance(value, (int, float)) and int(value) > 0:
        return int(value)
    s = str(value).lower().strip()
    # Try direct int parse
    m = re.search(r"\d+", s)
    if m:
        n = int(m.group())
        return n if n > 0 else None
    # Try number words
    for word, n in _NUM_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", s):
            return n if n > 0 else None
    return None


def _ground_date(value: Any) -> Optional[str]:
    """
    Parse a date from many formats and return YYYY-MM-DD.
    Accepts: 2026-03-15, 15/03/2026, March 15 2026, 15 Mar 2026, etc.
    """
    s = str(value).strip()
    if not s:
        return None
    # Already correct format
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    # Use dateutil for flexible parsing
    try:
        dt = dateutil_parser.parse(s, dayfirst=True, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        pass
    return None


def _ground_confirmation(value: Any) -> Any:
    """Map confirmation to True/False if possible."""
    if isinstance(value, bool):
        return value
    s = str(value).lower().strip()
    if re.match(r"^(yes|true|confirm|correct|ok|okay|sure|right|si|sì|yep|yeah|affirmative)$", s):
        return True
    if re.match(r"^(no|false|deny|incorrect|cancel|nope|nah|negative)$", s):
        return False
    return value   # ambiguous → let DM handle


def _ground_city_name(value: Any) -> Optional[str]:
    """Title-case normalisation for city / place names."""
    s = str(value).strip()
    if not s:
        return None
    # Remove stray punctuation at edges
    s = re.sub(r"^[\"']+|[\"']+$", "", s).strip()
    return s.title() if s else None


def _ground_preferred_time(value: Any) -> Optional[str]:
    """Normalise preferred_time: keep period words or HH:MM."""
    s = str(value).lower().strip()
    if not s:
        return None
    # Accept named periods as-is
    if re.match(r"^(morning|afternoon|evening|night)$", s):
        return s
    # Try to extract HH:MM from strings like "10am", "3:30 pm", "15:00"
    m = re.match(r"(\d{1,2}):?(\d{2})?\s*(am|pm)?", s)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0
        ampm = m.group(3)
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"
    return s  # keep raw if not parseable — still informative


def _ground_slots(slots: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalise every slot value against schema constraints.
    Uses regex + synonym maps to *fix* values where possible.
    Sets to None only when the value is truly unrecoverable.
    """
    grounded: Dict[str, Any] = {}
    for slot, value in slots.items():
        if value is None:
            grounded[slot] = None
            continue

        # ── budget_level ──
        if slot == "budget_level":
            result = _ground_budget(value)
            if result is None:
                logging.debug(f"[grounding] budget_level '{value}' → None (unrecognised)")
            else:
                logging.debug(f"[grounding] budget_level '{value}' → '{result}'")
            grounded[slot] = result

        # ── activity_category ──
        elif slot == "activity_category":
            result = _ground_activity_category(value)
            if result is None:
                logging.debug(f"[grounding] activity_category '{value}' → None (unrecognised)")
            else:
                logging.debug(f"[grounding] activity_category '{value}' → '{result}'")
            grounded[slot] = result

        # ── numeric slots ──
        elif slot in ("num_passengers", "num_guests"):
            result = _ground_number(value)
            if result is None:
                logging.debug(f"[grounding] {slot} '{value}' → None (not a valid number)")
            grounded[slot] = result

        # ── date slots ──
        elif slot in ("departure_date", "return_date", "check_in_date", "check_out_date"):
            result = _ground_date(value)
            if result is None:
                logging.debug(f"[grounding] {slot} '{value}' → None (cannot parse date)")
            grounded[slot] = result

        # ── confirmation ──
        elif slot == "confirmation":
            grounded[slot] = _ground_confirmation(value)

        # ── preferred_time ──
        elif slot == "preferred_time":
            grounded[slot] = _ground_preferred_time(value)

        # ── city / place names ──
        elif slot in ("destination", "origin", "city1", "city2"):
            grounded[slot] = _ground_city_name(value)

        # ── anything else (passthrough) ──
        else:
            grounded[slot] = value

    return grounded


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
        logging.error(f"Error calling pipe: {e}")
        return {"intent": "OOD", "slots": {}}
    
    try:
        generated = out[0]["generated_text"]
        if isinstance(generated, list):
            text = generated[-1].get("content", "")
        else:
            text = str(generated)
    except (IndexError, KeyError, TypeError) as e:
        logging.error(f"Error extracting generated text: {e}")
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

    grounded_slots = _ground_slots(clean_slots)
    return {"intent": intent, "slots": grounded_slots}


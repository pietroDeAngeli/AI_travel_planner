import json
import re
from typing import Any, Dict, List, Optional
from datetime import date, datetime, timedelta
from dateutil import parser as dateutil_parser

from schema import INTENTS, INTENT_SLOTS, ACTIVITY_CATEGORIES

import logging
logging.basicConfig(level=logging.DEBUG)

current_date = date.today().isoformat()
DEFAULT_YEAR = 2026

# ── Grounding helpers ──────────────────────────────────────────────

# Budget synonyms → canonical value
_BUDGET_SYNONYMS: Dict[str, str] = {
    # low
    "low": "low", "cheap": "low", "budget": "low", "economy": "low",
    "economic": "low", "inexpensive": "low", "affordable": "low",
    "backpacker": "low", "thrifty": "low",
    # medium
    "medium": "medium", "mid": "medium", "moderate": "medium",
    "standard": "medium", "average": "medium", "normal": "medium",
    # high
    "high": "high", "luxury": "high", "expensive": "high", "premium": "high",
    "first class": "high", "first-class": "high", "vip": "high",
    "deluxe": "high", "upscale": "high",
}

# Number words → int  (covers 1-20 plus common words)
_NUM_WORDS: Dict[str, int] = {
    "zero": 0, "one": 1, "a": 1, "an": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "couple": 2, "pair": 2, "dozen": 12,
    # solo-traveller expressions
    "me": 1, "myself": 1, "solo": 1, "alone": 1, "just me": 1, "only me": 1,
}

# Build reverse index: keyword → canonical activity category
_ACTIVITY_KEYWORD_MAP: Dict[str, str] = {}
for _cat, _keywords in ACTIVITY_CATEGORIES.items():
    _ACTIVITY_KEYWORD_MAP[_cat] = _cat          # category name maps to itself
    for _kw in _keywords:
        _ACTIVITY_KEYWORD_MAP[_kw] = _cat


# Hallucinated placeholder patterns produced by LLMs when they lack real data.
# Any slot value matching these is treated as null.
_PLACEHOLDER_RE = re.compile(
    r"\b(unknown|n/?a|not\s+(specified|provided|mentioned|given|stated)|"
    r"user.s?\s+location|current\s+location|tbd|tba|unspecified|none|null|"
    r"to\s+be\s+(confirmed|determined))\b",
    re.IGNORECASE,
)


def _is_placeholder(value: str) -> bool:
    return bool(_PLACEHOLDER_RE.search(value))


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


def _resolve_relative_date(s: str, reference_date: Optional[date] = None) -> Optional[str]:
    """
    Resolve relative date expressions to YYYY-MM-DD.
    reference_date: anchor for expressions like 'N days after' (e.g. check_in_date).
    Falls back to today when reference_date is None.
    """
    today = date.today()
    anchor = reference_date or today
    s_lower = s.lower().strip()

    if s_lower == "today":
        return today.strftime("%Y-%m-%d")
    if s_lower in ("tomorrow", "tmrw"):
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    if s_lower == "yesterday":
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")

    _WEEKDAYS = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
        # short forms
        "mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3,
        "thurs": 3, "fri": 4, "sat": 5, "sun": 6,
    }

    # "next <weekday>" — always means the NEXT occurrence (never today)
    m = re.match(r"(?:next|coming)\s+(\w+)", s_lower)
    if m:
        day_name = m.group(1)
        if day_name in _WEEKDAYS:
            target = _WEEKDAYS[day_name]
            days_ahead = (target - today.weekday()) % 7
            days_ahead = days_ahead if days_ahead > 0 else 7
            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # "this <weekday>" — nearest future occurrence, can be today
    m = re.match(r"this\s+(\w+)", s_lower)
    if m:
        day_name = m.group(1)
        if day_name in _WEEKDAYS:
            target = _WEEKDAYS[day_name]
            days_ahead = (target - today.weekday()) % 7
            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # Bare weekday name — nearest future occurrence (not today)
    if s_lower in _WEEKDAYS:
        target = _WEEKDAYS[s_lower]
        days_ahead = (target - today.weekday()) % 7
        days_ahead = days_ahead if days_ahead > 0 else 7
        return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # "in N days/weeks" — relative to today
    m = re.match(r"in\s+(\d+)\s+(day|days|week|weeks)", s_lower)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta = timedelta(days=n) if "day" in unit else timedelta(weeks=n)
        return (today + delta).strftime("%Y-%m-%d")

    # "N days/weeks after" — relative to anchor (check_in_date / departure_date)
    m = re.match(r"(\d+)\s+(day|days|week|weeks)\s+(?:after|later)", s_lower)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta = timedelta(days=n) if "day" in unit else timedelta(weeks=n)
        return (anchor + delta).strftime("%Y-%m-%d")

    # "after N days/weeks"
    m = re.match(r"after\s+(\d+)\s+(day|days|week|weeks)", s_lower)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta = timedelta(days=n) if "day" in unit else timedelta(weeks=n)
        return (anchor + delta).strftime("%Y-%m-%d")

    return None


def _ground_date(value: Any, reference_date: Optional[date] = None) -> Optional[str]:
    """
    Parse a date from many formats and return YYYY-MM-DD.
    reference_date: anchor for relative expressions like '7 days after' (e.g. check_in_date).
    """
    s = str(value).strip()
    if not s:
        return None
    # Already correct format
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    # Try relative expressions first (pass anchor for offset-from-start expressions)
    relative = _resolve_relative_date(s, reference_date=reference_date)
    if relative:
        return relative
    # Use dateutil for flexible absolute parsing; default year when omitted
    try:
        dt = dateutil_parser.parse(s, fuzzy=True, default=datetime(DEFAULT_YEAR, 1, 1))
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        pass
    return None


def _ground_confirmation(value: Any) -> Any:
    """Normalise confirmation to 'yes'/'no' strings (what the DM expects)."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    s = str(value).lower().strip()
    if re.match(r"^(yes|true|confirm|correct|ok|okay|sure|right|yep|yeah|affirmative)$", s):
        return "yes"
    if re.match(r"^(no|false|deny|incorrect|cancel|nope|nah|negative)$", s):
        return "no"
    return value   # ambiguous → let DM handle


_NON_CITY_RE = re.compile(
    r"^(\d+|[a-z]|unknown|n/?a|tbd|tba|here|there|home|anywhere|somewhere|"
    r"this|that|the|a|an|it|me|my|i)$",
    re.IGNORECASE,
)


def _ground_city_name(value: Any) -> Optional[str]:
    """Title-case normalisation for city / place names."""
    s = str(value).strip()
    if not s:
        return None
    # Remove stray punctuation at edges
    s = re.sub(r"^[\"']+|[\"']+$", "", s).strip()
    if not s:
        return None
    # Reject obvious non-city tokens
    if _NON_CITY_RE.match(s):
        logging.debug(f"[grounding] city '{s}' → None (not a valid city name)")
        return None
    return s.title()


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
    return s  # keep raw if not parseable


def _ground_slots(slots: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalise every slot value against schema constraints.
    Uses regex + synonym maps to fix values where possible.
    Sets to None only when the value is truly unrecoverable.
    """
    grounded: Dict[str, Any] = {}
    for slot, value in slots.items():
        if value is None:
            grounded[slot] = None
            continue

        # Reject hallucinated placeholder values before any normalisation
        if isinstance(value, str) and _is_placeholder(value):
            logging.debug(f"[grounding] {slot} '{value}' → None (placeholder/hallucination)")
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
            # Determine anchor: check_out_date uses check_in_date; return_date uses departure_date
            anchor_str = None
            if slot == "check_out_date":
                anchor_str = grounded.get("check_in_date") or slots.get("check_in_date")
            elif slot == "return_date":
                anchor_str = grounded.get("departure_date") or slots.get("departure_date")
            anchor = None
            if anchor_str and isinstance(anchor_str, str):
                try:
                    anchor = date.fromisoformat(anchor_str)
                except ValueError:
                    pass
            result = _ground_date(value, reference_date=anchor)
            if result is None:
                logging.debug(f"[grounding] {slot} '{value}' → None (cannot parse date)")
            grounded[slot] = result

        # ── confirmation ──
        elif slot == "confirmation":
            grounded[slot] = _ground_confirmation(value)

        # ── preferred_date ──
        elif slot == "preferred_date":
            grounded[slot] = _ground_date(value)

        # ── preferred_time ──
        elif slot == "preferred_time":
            grounded[slot] = _ground_preferred_time(value)

        # ── city / place names ──
        elif slot in ("destination", "origin", "city1", "city2"):
            grounded[slot] = _ground_city_name(value)

        # ── anything else (passthrough) ──
        else:
            grounded[slot] = value

    # Cross-slot check: origin and destination must differ
    origin = grounded.get("origin")
    destination = grounded.get("destination")
    if origin and destination and origin.lower() == destination.lower():
        logging.debug(f"[grounding] origin == destination ('{origin}') — clearing origin")
        grounded["origin"] = None

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
    # Keep short context: last 2 turns (user + assistant) so the NLU
    # understands what was being asked before interpreting the new utterance.
    history_text = ""
    if dialogue_history:
        for t in dialogue_history[-4:]:
            role = t.get("role", "").upper()
            history_text += f"{role}: {t['content']}\n"

    user = (
        f"Dialogue context:\n{history_text}\n"
        f"User utterance: {user_utterance}\n"
        "\nReturn JSON with keys: intent, slots."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]

    try:
        out = pipe(messages)
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


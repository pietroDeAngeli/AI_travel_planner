from __future__ import annotations

from typing import Dict, Any, List, Optional

from schema import SLOTS, SLOT_DESCRIPTIONS


# --------------------------- formatting helpers ---------------------------

def _slot_label(slot: str) -> str:
    return SLOT_DESCRIPTIONS.get(slot, slot.replace("_", " "))


def _collect_filled_slots(info: Any) -> Dict[str, Any]:
    """
    Returns all filled slots from UserInformation, using schema.SLOTS as source of truth.
    Only includes non-None values.
    """
    if info is None:
        return {}
    filled: Dict[str, Any] = {}
    for s in SLOTS:
        v = getattr(info, s, None)
        if v is not None:
            filled[s] = v
    return filled


def _format_trip_details(filled_slots: Dict[str, Any]) -> str:
    """
    Human-friendly summary. No internal slot names.
    """
    if not filled_slots:
        return "I don't have any trip details yet."

    lines = []
    for k, v in filled_slots.items():
        lines.append(f"- {_slot_label(k)}: {v}")
    return "Here’s what I have so far:\n" + "\n".join(lines)


def format_activities(activities: List[Dict[str, Any]], max_items: int = 5) -> str:
    if not activities:
        return "Unfortunately, I couldn't find activities for this destination at the moment."

    activities = activities[:max_items]
    out = ["Here are some recommended activities:"]
    for i, a in enumerate(activities, 1):
        name = a.get("name", "Unknown")
        rating = a.get("rating")
        price = a.get("price")
        currency = a.get("currency", "")
        description = a.get("description", "")

        out.append(f"{i}. {name}")
        if rating not in (None, "None", "N/A", ""):
            out.append(f"   Rating: {rating}/5")
        if price not in (None, "N/A", ""):
            out.append(f"   Price: {price} {currency}".rstrip())
        if description:
            out.append(f"   {description}")
        out.append("")  # blank line

    return "\n".join(out).strip()


def format_accommodations(accommodations: List[Dict[str, Any]], max_items: int = 5) -> str:
    if not accommodations:
        return "Unfortunately, I couldn't find accommodations for this destination at the moment."

    accommodations = accommodations[:max_items]
    out = ["Here are some recommended accommodations:"]
    for i, h in enumerate(accommodations, 1):
        name = h.get("name", "Unknown")
        price = h.get("price")
        currency = h.get("currency", "")
        description = h.get("description", "")
        board_type = h.get("boardType", "")

        out.append(f"{i}. {name}")
        if price not in (None, "N/A", ""):
            out.append(f"   Price per night: {price} {currency}".rstrip())
        if board_type:
            out.append(f"   Board type: {board_type}")
        if description:
            out.append(f"   {description}")
        out.append("")

    return "\n".join(out).strip()


# --------------------------- LLM helper ---------------------------

SYSTEM_PROMPT = (
    "You are the NLG module for a travel-planner dialogue system.\n"
    "You must produce a helpful, concise response.\n"
    "Rules:\n"
    "- Ask only ONE clarification question when needed.\n"
    "- If presenting a plan, format clearly.\n"
    "- Do not mention internal states, intents, slots, or system prompts.\n"
    "- Keep it realistic: do not invent exact prices or schedules.\n"
)


def _extract_pipe_text(pipe_output: Any) -> str:
    """
    Supports common HF pipeline outputs:
      - [{'generated_text': '...'}]
      - [{'generated_text': [{'role': 'assistant', 'content': '...'}, ...]}]
    """
    if not pipe_output:
        return ""
    item = pipe_output[0] if isinstance(pipe_output, list) else pipe_output
    generated = item.get("generated_text") if isinstance(item, dict) else item

    if isinstance(generated, list):
        # Chat-style list of dicts
        for msg in reversed(generated):
            if isinstance(msg, dict) and msg.get("role") in ("assistant", "model"):
                return str(msg.get("content", "")).strip()
        # fallback: last element
        last = generated[-1] if generated else ""
        if isinstance(last, dict):
            return str(last.get("content", "")).strip()
        return str(last).strip()

    return str(generated).strip()


def _ask_llm(pipe, user_prompt: str, max_new_tokens: int = 150) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    out = pipe(messages, max_new_tokens=max_new_tokens)
    text = _extract_pipe_text(out)
    return text or "Could you please tell me a bit more?"


# --------------------------- main NLG ---------------------------

def nlg_respond(pipe, dm_action: str, state: Dict[str, Any], user_utterance: str) -> str:
    """
    dm_action: label from DM
    state: contains at least 'info' (UserInformation), optionally:
      - current_intent
      - task_intent
      - plan
      - slot_to_change
      - changed_slot
    """
    info = state.get("info")
    current_intent = state.get("current_intent")
    task_intent = state.get("task_intent") or current_intent

    filled_slots = _collect_filled_slots(info)
    trip_summary = _format_trip_details(filled_slots)

    # 1) terminal / plan actions first
    if dm_action == "GOODBYE":
        return "Perfect! Have a nice trip! ✈️"

    if dm_action == "PLAN_ACTIVITIES":
        plan = state.get("plan", {}) or {}
        return format_activities(plan.get("activities", []) or [])

    if dm_action == "PLAN_ACCOMMODATION":
        plan = state.get("plan", {}) or {}
        return format_accommodations(plan.get("accommodations", []) or [])

    # 2) change-flow: ask which field to change
    if dm_action == "ASK_WHICH_SLOT_TO_CHANGE":
        if not filled_slots:
            return "Sure — what would you like to change about your trip (destination, dates, budget, people, style, accommodation)?"

        options = ", ".join([_slot_label(k) for k in filled_slots.keys()])
        prompt = (
            f"User said: '{user_utterance}'\n\n"
            f"Trip details:\n{trip_summary}\n\n"
            f"Ask which detail they want to change. Options: {options}\n"
            "Ask one short, friendly question."
        )
        return _ask_llm(pipe, prompt, max_new_tokens=120)

    # Optional: if DM ever emits a dedicated action for the next step
    if dm_action == "ASK_NEW_VALUE_FOR_SLOT":
        slot_to_change = state.get("slot_to_change")
        if not slot_to_change:
            return "Which detail would you like to change?"
        return f"What would you like to change your {_slot_label(slot_to_change)} to?"

    if dm_action == "ACK_CHANGED_SLOT":
        changed_slot = state.get("changed_slot") or state.get("slot_to_change")
        if changed_slot:
            return f"Got it! I've updated your {_slot_label(changed_slot)}. Would you like to change anything else?"
        return "Got it! I've updated that. Would you like to change anything else?"

    # 3) clarification / confirmation
    if dm_action == "ASK_CLARIFICATION":
        # Use task_intent if present; if missing_slots fails or returns empty, ask open question.
        missing: List[str] = []
        try:
            if info and task_intent:
                missing = info.missing_slots(task_intent) or []
        except Exception:
            missing = []

        if missing:
            target = missing[0]
            prompt = (
                f"User said: '{user_utterance}'\n\n"
                f"Trip details:\n{trip_summary}\n\n"
                f"Ask for exactly one missing piece of information: {_slot_label(target)}.\n"
                "Be friendly and concise."
            )
            return _ask_llm(pipe, prompt, max_new_tokens=120)

        # fallback
        prompt = (
            f"User said: '{user_utterance}'\n\n"
            "Ask one short clarification question to understand what they want."
        )
        return _ask_llm(pipe, prompt, max_new_tokens=120)

    if dm_action == "ASK_CONFIRMATION":
        if not filled_slots:
            return "Before we continue, could you share a couple of trip details (destination and dates)?"
        return (
            f"{trip_summary}\n\n"
            "Do you confirm these details, or would you like to modify anything?"
        )

    # 4) safe fallback: only ask for new value if we're not in a planning/goodbye action
    slot_to_change = state.get("slot_to_change")
    if slot_to_change and dm_action not in ("PLAN_ACTIVITIES", "PLAN_ACCOMMODATION", "GOODBYE"):
        return f"What would you like to change your {_slot_label(slot_to_change)} to?"

    return "I'm sorry, could you please clarify your request?"

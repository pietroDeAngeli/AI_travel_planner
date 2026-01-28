from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import copy
import json
import re

from data import TripContext
from schema import INTENTS, DM_ACTIONS, INTENT_SCHEMAS, build_dm_actions_prompt


# --- State ---

@dataclass
class DialogueState:
    """
    Dialogue State Tracker for multi-booking travel planner.
    Maintains context across multiple self-contained booking intents.
    """
    context: TripContext = field(default_factory=TripContext)

    # Current booking flow
    current_intent: Optional[str] = None  # The active booking intent

    # Carryover data when switching intents
    pending_carryover: Optional[Dict[str, Any]] = None

    # Track if carryover was offered (waiting for user response)
    awaiting_carryover_response: bool = False

    # Last action for tracking
    last_action: Optional[str] = None

    # Whether the current booking has been confirmed by the user
    confirmed: bool = False

    # Data for COMPARE_CITIES intent (not a booking, just informative)
    compare_cities_data: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            "DialogueState(\n"
            f"  current_intent = {self.current_intent}\n"
            f"  confirmed = {self.confirmed}\n"
            f"  context:\n{self.context}\n"
            f"  last_action = {self.last_action}\n"
            ")"
        )

    def get_current_booking(self):
        """Get the booking object for the current intent."""
        return self.context.get_booking(self.current_intent)

    def get_missing_slots(self) -> List[str]:
        """Get missing slots for the current booking."""
        booking = self.get_current_booking()
        if booking and hasattr(booking, "missing_slots"):
            return booking.missing_slots()
        return []

    def to_summary(self) -> Dict[str, Any]:
        """Create a summary of current state."""
        booking = self.get_current_booking()
        booking_data = booking.to_dict() if booking and hasattr(booking, "to_dict") else {}
        filled_slots = {k: v for k, v in booking_data.items() if v is not None}
        return {
            "current_intent": self.current_intent,
            "filled_slots": filled_slots,
            "missing_slots": self.get_missing_slots(),
            "confirmed": self.confirmed,
            "completed_bookings": list(getattr(self.context, "completed_intents", [])),
        }

    def reset_for_new_intent(self) -> None:
        """Reset confirmation state when switching to a new intent."""
        self.confirmed = False


# --- Shared helpers ---

def _get_complete_action(intent: str) -> str:
    """Get the completion action for an intent."""
    mapping = {
        "BOOK_FLIGHT": "COMPLETE_FLIGHT_BOOKING",
        "BOOK_ACCOMMODATION": "COMPLETE_ACCOMMODATION_BOOKING",
        "BOOK_ACTIVITY": "COMPLETE_ACTIVITY_BOOKING",
    }
    return mapping.get(intent, "ASK_CLARIFICATION")


def _normalize_yes_no(value: Any) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in {"yes", "y", "si", "sì", "true"}:
        return "yes"
    if v in {"no", "n", "false"}:
        return "no"
    return None


def _update_state_with_nlu(state: DialogueState, nlu_output: Dict[str, Any]) -> None:
    """
    Update the dialogue state with parsed intent and slots from NLU.
    """
    intent = nlu_output.get("intent")
    slots = nlu_output.get("slots", {}) or {}

    # Skip state updates for non-booking intents
    if intent not in INTENTS or intent in ("OOD", "COMPARE_CITIES"):
        return

    # Check if we're switching intents
    if state.current_intent and state.current_intent != intent:
        carryover = state.context.get_carryover_values(state.current_intent, intent)
        if carryover:
            state.pending_carryover = carryover
        state.reset_for_new_intent()

    state.current_intent = intent

    booking = state.get_current_booking()
    if booking and slots:
        for slot, value in slots.items():
            if value is not None and hasattr(booking, slot):
                setattr(booking, slot, value)


def _allowed_slots_for_intent(intent: Optional[str]) -> List[str]:
    """
    Optional guardrail: use INTENT_SCHEMAS if available.
    """
    if not intent:
        return []
    schema = INTENT_SCHEMAS.get(intent, {}) if isinstance(INTENT_SCHEMAS, dict) else {}
    required = schema.get("required_slots", [])
    optional = schema.get("optional_slots", [])
    out: List[str] = []
    if isinstance(required, list):
        out.extend(required)
    if isinstance(optional, list):
        out.extend(optional)
    # de-dup preserve order
    seen = set()
    dedup = []
    for s in out:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup


def _extract_assistant_response(outputs: Any) -> str:
    """
    HF chat pipelines can return:
      outputs[0]["generated_text"] as either:
        - list of messages (dicts) with "content"
        - a string
    """
    try:
        generated = outputs[0].get("generated_text")
        if isinstance(generated, list) and generated:
            return str(generated[-1].get("content", "")).strip()
        return str(generated or "").strip()
    except Exception:
        return str(outputs).strip()


# --- Rule-based fallback ---

def dm_decide_rule_based(
    state: DialogueState,
    nlu_output: Dict[str, Any],
    user_utterance: str = "",
) -> str:
    """
    Deterministic DM (reference behavior). Good for regression tests.
    """
    intent = nlu_output.get("intent")
    slots = nlu_output.get("slots", {}) or {}

    # 1) END_DIALOGUE
    if intent == "END_DIALOGUE":
        state.last_action = "GOODBYE"
        return "GOODBYE"

    # 2) OOD / unknown
    if intent == "OOD" or intent is None or intent not in INTENTS:
        state.last_action = "ASK_CLARIFICATION"
        return "ASK_CLARIFICATION"

    # 3) COMPARE_CITIES
    if intent == "COMPARE_CITIES":
        state.current_intent = "COMPARE_CITIES"
        for s in ["city1", "city2", "activity_category"]:
            if slots.get(s) is not None:
                state.compare_cities_data[s] = slots.get(s)

        city1 = state.compare_cities_data.get("city1")
        city2 = state.compare_cities_data.get("city2")
        activity_category = state.compare_cities_data.get("activity_category")

        if not city1:
            state.last_action = "REQUEST_MISSING_SLOT(city1)"
            return state.last_action
        if not city2:
            state.last_action = "REQUEST_MISSING_SLOT(city2)"
            return state.last_action
        if not activity_category:
            state.last_action = "REQUEST_MISSING_SLOT(activity_category)"
            return state.last_action

        state.last_action = "COMPARE_CITIES_RESULT"
        return "COMPARE_CITIES_RESULT"

    # 4) If we were waiting for confirmation
    if state.last_action == "ASK_CONFIRMATION":
        conf = _normalize_yes_no(slots.get("confirmation"))
        if conf == "no":
            state.last_action = "REQUEST_SLOT_CHANGE"
            return "REQUEST_SLOT_CHANGE"
        if conf == "yes":
            state.confirmed = True
            action = _get_complete_action(state.current_intent)
            state.context.mark_completed(state.current_intent)
            state.last_action = action
            return action
        return "ASK_CONFIRMATION"

    # 5) Carryover offer response
    if state.last_action == "OFFER_SLOT_CARRYOVER":
        conf = _normalize_yes_no(slots.get("confirmation"))
        if conf is None:
            state.last_action = "OFFER_SLOT_CARRYOVER"
            return "OFFER_SLOT_CARRYOVER"
        if conf == "yes":
            if state.pending_carryover:
                booking = state.get_current_booking()
                if booking:
                    for k, v in state.pending_carryover.items():
                        if hasattr(booking, k):
                            setattr(booking, k, v)
            state.pending_carryover = None
            state.awaiting_carryover_response = False
        elif conf == "no":
            state.pending_carryover = None
            state.awaiting_carryover_response = False
        # continue flow

    # 6) Slot change
    if state.last_action == "REQUEST_SLOT_CHANGE":
        slot_to_change = slots.get("slot_name")
        if slot_to_change:
            booking = state.get_current_booking()
            if booking and hasattr(booking, slot_to_change):
                setattr(booking, slot_to_change, None)
                action = f"REQUEST_MISSING_SLOT({slot_to_change})"
                state.last_action = action
                return action
        state.last_action = "REQUEST_SLOT_CHANGE"
        return "REQUEST_SLOT_CHANGE"

    # Update state now
    _update_state_with_nlu(state, nlu_output)

    # No task intent
    if not state.current_intent:
        state.last_action = "ASK_CLARIFICATION"
        return "ASK_CLARIFICATION"

    # Missing slots
    missing = state.get_missing_slots()

    # Offer carryover only if there are missing slots that could be filled
    if state.pending_carryover and not state.awaiting_carryover_response and missing:
        # Check if carryover has any values for the missing slots
        carryover_useful = any(k in missing for k in state.pending_carryover.keys())
        if carryover_useful:
            state.awaiting_carryover_response = True
            state.last_action = "OFFER_SLOT_CARRYOVER"
            return "OFFER_SLOT_CARRYOVER"
        else:
            # Carryover not useful, clear it
            state.pending_carryover = None
    if missing:
        action = f"REQUEST_MISSING_SLOT({missing[0]})"
        state.last_action = action
        return action

    # Ask confirmation
    if not state.confirmed:
        state.last_action = "ASK_CONFIRMATION"
        return "ASK_CONFIRMATION"

    # Complete
    action = _get_complete_action(state.current_intent)
    state.context.mark_completed(state.current_intent)
    state.last_action = action
    return action


# --- LLM prompts + parsing ---

def _build_dm_system_prompt() -> str:
    actions_prompt = build_dm_actions_prompt()

    return f"""You are a Dialogue Manager for a travel booking assistant.
Your job: decide the NEXT system action.

{actions_prompt}

You must follow this priority logic:
1) If intent is END_DIALOGUE -> GOODBYE.
2) If intent is OOD/unknown -> ASK_CLARIFICATION.
   Exception: if last_action expects confirmation/slot_name, infer it from user utterance anyway.
3) If intent is COMPARE_CITIES:
   - request missing: city1, then city2, then activity_category
   - if all filled -> COMPARE_CITIES_RESULT
4) If last_action was ASK_CONFIRMATION:
   - infer confirmation yes/no (prefer NLU slot, else user utterance)
   - yes -> COMPLETE_* for current intent
   - no -> REQUEST_SLOT_CHANGE
   - unclear -> ASK_CONFIRMATION
5) If last_action was OFFER_SLOT_CARRYOVER:
   - infer yes/no (prefer NLU slot, else user utterance)
   - unclear -> OFFER_SLOT_CARRYOVER
   - yes/no -> proceed to normal slot-checking after applying/ignoring carryover
6) If last_action was REQUEST_SLOT_CHANGE:
   - infer slot_name (prefer NLU slot_name, else user utterance + filled_slot_keys_now)
   - if known -> REQUEST_MISSING_SLOT(slot_name)
   - else -> REQUEST_SLOT_CHANGE
7) Otherwise:
   - if carryover_must_offer == true -> OFFER_SLOT_CARRYOVER   (MANDATORY)
   - else if missing_slots exist -> REQUEST_MISSING_SLOT(first_missing_slot)
   - else if not confirmed -> ASK_CONFIRMATION
   - else -> COMPLETE_* for current intent

CRITICAL GATING RULES (must be respected):
- If can_request_slot_change == false, you MUST NOT output REQUEST_SLOT_CHANGE.
- If carryover_must_offer == true, you MUST output OFFER_SLOT_CARRYOVER (unless END_DIALOGUE or OOD/unknown intent).

STRICT OUTPUT:
Return ONLY one JSON object:
{{
  "action": "<one of allowed actions OR REQUEST_MISSING_SLOT>",
  "slot": "<slot name if action is REQUEST_MISSING_SLOT>",
  "inferred_confirmation": "yes|no|unclear (optional)",
  "inferred_slot_name": "<slot name (optional)>"
}}

No extra text. No markdown. No code fences.
"""


def _build_dm_user_prompt(
    *,
    state_before: Dict[str, Any],
    state_after: Dict[str, Any],
    state: DialogueState,
    nlu_output: Dict[str, Any],
    user_utterance: str,
) -> str:
    intent = nlu_output.get("intent")
    slots = nlu_output.get("slots", {}) or {}

    missing_after = state_after.get("missing_slots", []) or []
    first_missing = missing_after[0] if missing_after else None

    # derived gating flags (computed, not guessed)
    confirmation_from_nlu = _normalize_yes_no(slots.get("confirmation"))
    # request_slot_change allowed only if we're in that loop OR we just got a negative confirmation
    can_request_slot_change = (
        state.last_action == "REQUEST_SLOT_CHANGE"
        or (state.last_action == "ASK_CONFIRMATION" and confirmation_from_nlu == "no")
    )

    carryover_must_offer = bool(state.pending_carryover) and (not state.awaiting_carryover_response)

    # helpful lists
    filled_after = state_after.get("filled_slots", {}) or {}
    filled_keys = list(filled_after.keys())

    schema_slots = _allowed_slots_for_intent(state_after.get("current_intent"))
    allowed_missing_slots_now = list(missing_after)
    compare_cities_slots = ["city1", "city2", "activity_category"]

    # make carryover salient
    carryover_keys = list((state.pending_carryover or {}).keys())

    return f"""STATE_BEFORE (previous turn, before applying NLU now):
{json.dumps(state_before, ensure_ascii=False)}

STATE_AFTER (after applying NLU now):
{json.dumps(state_after, ensure_ascii=False)}
- first_missing_slot: {first_missing}

GATING FLAGS (computed, must be respected):
- can_request_slot_change: {can_request_slot_change}
- carryover_must_offer: {carryover_must_offer}

SALIENT CARRYOVER INFO:
- pending_carryover_exists: {bool(state.pending_carryover)}
- awaiting_carryover_response: {state.awaiting_carryover_response}
- pending_carryover_keys: {json.dumps(carryover_keys, ensure_ascii=False)}

INTERNAL FLAGS:
- last_action: {state.last_action}
- compare_cities_data: {json.dumps(state.compare_cities_data, ensure_ascii=False)}

NLU OUTPUT (this turn):
- intent: {intent}
- slots: {json.dumps(slots, ensure_ascii=False)}

USER UTTERANCE:
{json.dumps(user_utterance, ensure_ascii=False)}

HELPFUL LISTS:
- allowed_missing_slots_now: {json.dumps(allowed_missing_slots_now, ensure_ascii=False)}
- compare_cities_slots: {json.dumps(compare_cities_slots, ensure_ascii=False)}
- filled_slot_keys_now: {json.dumps(filled_keys, ensure_ascii=False)}
- schema_slots_for_intent: {json.dumps(schema_slots, ensure_ascii=False)}

Return the action JSON now.
"""


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_llm_action(llm_response: str) -> Tuple[str, Dict[str, Any]]:
    """
    Returns: (action_string, meta)
    meta may contain inferred_confirmation / inferred_slot_name.
    """
    meta: Dict[str, Any] = {}
    raw = _strip_code_fences(llm_response)

    obj = None
    try:
        if "{" in raw and "}" in raw:
            start = raw.find("{")
            end = raw.rfind("}")
            obj = json.loads(raw[start : end + 1])
    except Exception:
        obj = None

    if isinstance(obj, dict):
        action = str(obj.get("action", "")).strip()
        slot = obj.get("slot", None)

        inf_conf = obj.get("inferred_confirmation", None)
        if isinstance(inf_conf, str):
            meta["inferred_confirmation"] = inf_conf.strip().lower()

        inf_slot = obj.get("inferred_slot_name", None)
        if isinstance(inf_slot, str):
            meta["inferred_slot_name"] = inf_slot.strip()

        if action == "REQUEST_MISSING_SLOT":
            if isinstance(slot, str) and re.match(r"^\w+$", slot.strip()):
                return f"REQUEST_MISSING_SLOT({slot.strip()})", meta
            return "ASK_CLARIFICATION", meta

        if re.match(r"^REQUEST_MISSING_SLOT\(\w+\)$", action):
            return action, meta

        if action in DM_ACTIONS:
            return action, meta

        return "ASK_CLARIFICATION", meta

    # Legacy fallback: plain action string
    action = raw.strip().strip("\"'")
    action = [l.strip() for l in action.split("\n") if l.strip()][0] if action else ""
    if re.match(r"^REQUEST_MISSING_SLOT\(\w+\)$", action):
        return action, meta
    if action in DM_ACTIONS:
        return action, meta
    return "ASK_CLARIFICATION", meta


# --- Side effects ---

def _apply_action_side_effects(
    state: DialogueState,
    action: str,
    nlu_output: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    meta = meta or {}
    intent = nlu_output.get("intent")
    slots = nlu_output.get("slots", {}) or {}

    # COMPARE_CITIES updates
    if intent == "COMPARE_CITIES" or state.current_intent == "COMPARE_CITIES":
        state.current_intent = "COMPARE_CITIES"
        for s in ["city1", "city2", "activity_category"]:
            if slots.get(s) is not None:
                state.compare_cities_data[s] = slots.get(s)

    # If LLM completes -> mark confirmed + completed
    if action.startswith("COMPLETE_"):
        state.confirmed = True
        if state.current_intent:
            state.context.mark_completed(state.current_intent)

    # Carryover accept/decline when last_action was OFFER_SLOT_CARRYOVER
    if state.last_action == "OFFER_SLOT_CARRYOVER":
        conf = _normalize_yes_no(slots.get("confirmation"))
        if conf is None:
            conf = meta.get("inferred_confirmation")
        conf = conf if conf in ("yes", "no") else None

        if conf == "yes":
            if state.pending_carryover:
                booking = state.get_current_booking()
                if booking:
                    for k, v in state.pending_carryover.items():
                        if hasattr(booking, k):
                            setattr(booking, k, v)
            state.pending_carryover = None
            state.awaiting_carryover_response = False
        elif conf == "no":
            state.pending_carryover = None
            state.awaiting_carryover_response = False
        # unclear -> leave as-is (will re-offer)

    # Slot change side-effect: reset chosen slot
    if state.last_action == "REQUEST_SLOT_CHANGE":
        slot_to_change = slots.get("slot_name") or meta.get("inferred_slot_name")
        if slot_to_change:
            booking = state.get_current_booking()
            if booking and hasattr(booking, slot_to_change):
                setattr(booking, slot_to_change, None)

    # Offer carryover flag
    if action == "OFFER_SLOT_CARRYOVER":
        state.awaiting_carryover_response = True

    # Update booking state with NLU (booking intents only)
    if intent not in ("OOD", "COMPARE_CITIES", "END_DIALOGUE", None):
        _update_state_with_nlu(state, nlu_output)

    # Update last action
    state.last_action = action


# --- Validation / guardrails ---

def _validate_and_correct_action(
    *,
    state: DialogueState,
    nlu_output: Dict[str, Any],
    state_after: Dict[str, Any],
    action: str,
) -> str:
    """
    Lightweight post-checks to enforce the suggested gating:
    - REQUEST_SLOT_CHANGE only when allowed
    - OFFER_SLOT_CARRYOVER must be emitted when pending_carryover and not awaiting response
    - REQUEST_MISSING_SLOT slot must be plausible
    """
    intent = nlu_output.get("intent")
    slots = nlu_output.get("slots", {}) or {}

    # If dialogue end / OOD, don't force carryover
    is_ood_or_unknown = (intent == "OOD" or intent is None or intent not in INTENTS)

    carryover_must_offer = bool(state.pending_carryover) and (not state.awaiting_carryover_response)
    if carryover_must_offer and not is_ood_or_unknown and intent != "END_DIALOGUE":
        # mandatory offer unless we're in a response loop that must be handled first
        if state.last_action not in ("ASK_CONFIRMATION", "REQUEST_SLOT_CHANGE", "OFFER_SLOT_CARRYOVER"):
            return "OFFER_SLOT_CARRYOVER"

    # Gate REQUEST_SLOT_CHANGE
    conf = _normalize_yes_no(slots.get("confirmation"))
    can_request_slot_change = (
        state.last_action == "REQUEST_SLOT_CHANGE"
        or (state.last_action == "ASK_CONFIRMATION" and conf == "no")
    )
    if action == "REQUEST_SLOT_CHANGE" and not can_request_slot_change:
        # Prefer continuing the normal flow deterministically
        return dm_decide_rule_based(state, nlu_output, "")

    # Validate REQUEST_MISSING_SLOT(slot)
    if action.startswith("REQUEST_MISSING_SLOT("):
        m = re.match(r"REQUEST_MISSING_SLOT\((\w+)\)", action)
        if not m:
            return dm_decide_rule_based(state, nlu_output, "")
        slot_name = m.group(1)

        allowed = set((state_after.get("missing_slots") or []))
        allowed |= {"city1", "city2", "activity_category"}
        allowed |= set(_allowed_slots_for_intent(state_after.get("current_intent")))

        # If allowed list is empty, don't over-restrict
        if allowed and slot_name not in allowed:
            return dm_decide_rule_based(state, nlu_output, "")

    return action


# --- LLM-based DM ---

def dm_decide(
    state: DialogueState,
    nlu_output: Dict[str, Any],
    user_utterance: str = "",
    llm_pipe=None,
) -> str:
    """
    LLM-based Dialogue Manager.
    - LLM decides the action following the priority logic.
    - LLM sees BOTH state_before and state_after (after applying NLU on a copy).
    - Added gating to reduce REQUEST_SLOT_CHANGE false positives and enforce carryover offering.
    """
    if llm_pipe is None:
        return dm_decide_rule_based(state, nlu_output, user_utterance)

    # BEFORE snapshot
    state_before = state.to_summary()

    # AFTER snapshot (apply NLU on a copy so missing_slots are current)
    state_for_prompt = copy.deepcopy(state)
    intent = nlu_output.get("intent")
    slots = nlu_output.get("slots", {}) or {}

    # Pre-update compare cities data for prompt copy
    if intent == "COMPARE_CITIES":
        state_for_prompt.current_intent = "COMPARE_CITIES"
        for s in ["city1", "city2", "activity_category"]:
            if slots.get(s) is not None:
                state_for_prompt.compare_cities_data[s] = slots.get(s)

    # Pre-update booking/carryover for prompt copy
    if intent not in ("OOD", "COMPARE_CITIES", "END_DIALOGUE", None):
        _update_state_with_nlu(state_for_prompt, nlu_output)

    state_after = state_for_prompt.to_summary()

    # Build prompts
    system_prompt = _build_dm_system_prompt()
    user_prompt = _build_dm_user_prompt(
        state_before=state_before,
        state_after=state_after,
        state=state,
        nlu_output=nlu_output,
        user_utterance=user_utterance,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Call LLM
    try:
        outputs = llm_pipe(
            messages,
            max_new_tokens=120,
            do_sample=False,
            temperature=0.0,
            pad_token_id=llm_pipe.tokenizer.pad_token_id,
        )

        assistant_response = _extract_assistant_response(outputs)
        action, meta = _parse_llm_action(assistant_response)

    except Exception as e:
        print(f"LLM error in DM: {e}")
        action, meta = "ASK_CLARIFICATION", {}

    # If LLM output is weak/invalid, fallback to rule-based
    if action == "ASK_CLARIFICATION":
        action = dm_decide_rule_based(state, nlu_output, user_utterance)
        meta = {}

    # Enforce suggested gating/corrections
    action = _validate_and_correct_action(
        state=state,
        nlu_output=nlu_output,
        state_after=state_after,
        action=action,
    )

    # Apply side effects on real state
    _apply_action_side_effects(state, action, nlu_output, meta=meta)
    return action

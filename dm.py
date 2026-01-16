from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from data import UserInformation
from schema import SLOTS  # per inferire slot da testo (opzionale)

@dataclass
class DialogueState:
    info: UserInformation = field(default_factory=UserInformation)
    current_intent: Optional[str] = None
    current_state: Optional[str] = None
    slot_to_change: Optional[str] = None
    changed_slot: Optional[str] = None  # <--- NEW: per NLG ACK_CHANGED_SLOT
    task_intent: Optional[str] = None
    plan: Optional[Dict[str, Any]] = None
    confirmed: bool = False

    def __str__(self) -> str:
        return (
            "DialogueState(\n"
            f"  current_state   = {self.current_state}\n"
            f"  current_intent  = {self.current_intent}\n"
            f"  task_intent     = {self.task_intent}\n"
            f"  confirmed       = {self.confirmed}\n"
            f"  slot_to_change  = {self.slot_to_change}\n"
            f"  changed_slot    = {self.changed_slot}\n"
            f"  user_info       = {self.info}\n"
            f"  plan            = {self.plan}\n"
            ")"
        )

def _infer_slot_to_change(user_utterance: str) -> Optional[str]:
    """
    Heuristica semplice per interpretare la risposta dell'utente dopo
    ASK_WHICH_SLOT_TO_CHANGE (es. 'destination', 'budget', 'dates', ecc.).
    """
    if not user_utterance:
        return None
    t = user_utterance.strip().lower()

    # match diretto su nomi slot o "pretty name"
    for s in SLOTS:
        if s in t or s.replace("_", " ") in t:
            return s

    # sinonimi comuni (minimi)
    if "budget" in t:
        return "budget_level"
    if "people" in t or "person" in t or "traveler" in t:
        return "num_people"
    if "start" in t and "date" in t or "departure" in t or "leave" in t:
        return "start_date"
    if "end" in t and "date" in t or "return" in t:
        return "end_date"
    if "hotel" in t or "hostel" in t or "airbnb" in t or "accommodation" in t:
        return "accommodation_type"
    if "style" in t:
        return "travel_style"
    if "destination" in t or "city" in t or "where" in t:
        return "destination"

    return None


def update_info(
    state: DialogueState,
    intent: str,
    new_slots: Dict[str, Any],
    user_utterance: Optional[str] = None,   # <--- NEW: serve per scegliere lo slot
) -> None:
    # set task intent (lasciamo come avevi)
    if intent in ("START_TRIP", "REQUEST_PLAN", "ACCOMMODATION_PREFERENCE"):
        state.task_intent = intent

    state.current_intent = intent
    state.info.update_info(new_slots)

    # se il sistema aveva appena chiesto "quale slot vuoi cambiare?"
    # e l'utente risponde con "destination/budget/..."
    if state.current_state == "ASK_WHICH_SLOT_TO_CHANGE" and state.slot_to_change is None:
        inferred = _infer_slot_to_change(user_utterance or "")
        if inferred:
            state.slot_to_change = inferred

    print(state)

    task = state.task_intent or intent
    missing_slots = state.info.missing_slots(task)
    if missing_slots:
        state.confirmed = False


def update_current_state(new_state: str, state: DialogueState) -> None:
    state.current_state = new_state


def dm_decide(intent: str, state: DialogueState) -> str:
    """
    Return one of DM_ACTIONS
    """
    task = state.task_intent or intent
    missing_slots = state.info.missing_slots(task)

    if intent == "END_DIALOGUE":
        return "GOODBYE"

    if intent == "CONFIRM_DETAILS":
        if state.task_intent is None:
            return "ASK_CLARIFICATION"
        state.confirmed = True
        if state.task_intent == "REQUEST_PLAN":
            return "PLAN_ACTIVITIES"
        if state.task_intent == "ACCOMMODATION_PREFERENCE":
            return "PLAN_ACCOMMODATION"
        return "ACK_UPDATE"

    # --- CHANGE FLOW ---
    if intent == "CHANGE_DETAILS":
        state.slot_to_change = None
        state.changed_slot = None
        return "ASK_WHICH_SLOT_TO_CHANGE"

    # NEW: se lo slot da cambiare è stato scelto, chiedi esplicitamente il nuovo valore
    if state.slot_to_change is not None and intent != "PROVIDE_CHANGE_VALUE":
        return "ASK_NEW_VALUE_FOR_SLOT"

    if intent == "PROVIDE_CHANGE_VALUE":
        if state.slot_to_change is None:
            return "ASK_WHICH_SLOT_TO_CHANGE"
        # qui assumiamo che l'update del valore sia già arrivato in new_slots
        # (o che venga gestito altrove). Noi almeno conserviamo quale slot è stato cambiato.
        state.changed_slot = state.slot_to_change
        state.slot_to_change = None
        state.confirmed = False
        return "ACK_CHANGED_SLOT"

    # --- PLANNING ---
    if intent == "REQUEST_PLAN" and state.confirmed:
        return "PLAN_ACTIVITIES"

    if intent == "ACCOMMODATION_PREFERENCE" and state.confirmed:
        return "PLAN_ACCOMMODATION"

    if intent == "START_TRIP":
        state.task_intent = intent
        if len(missing_slots) > 0:
            return "ASK_CLARIFICATION"
        else:
            return "ASK_CONFIRMATION"

    return "ASK_CLARIFICATION"

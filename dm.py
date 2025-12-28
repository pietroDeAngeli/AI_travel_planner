from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

REQUIRED_SLOTS = ["destination", "dates", "budget", "duration", "travel_style"]

@dataclass
class DialogueState:
    slots: Dict[str, Any] = field(default_factory=lambda: {k: None for k in REQUIRED_SLOTS})
    plan: Optional[Dict[str, Any]] = None
    missing_slots: List[str] = field(default_factory=list)

def update_state_with_slots(state: DialogueState, new_slots: Dict[str, Any]) -> None:
    for k, v in new_slots.items():
        if v is not None and k in state.slots:
            state.slots[k] = v

def compute_missing(state: DialogueState) -> List[str]:
    return [k for k in REQUIRED_SLOTS if not state.slots.get(k)]

def dm_decide(intent: str, state: DialogueState) -> str:
    """
    Ritorna una dm_action per NLG:
      - ASK_CLARIFICATION
      - ACK_UPDATE
      - SHOW_PLAN
      - GOODBYE
    """
    state.missing_slots = compute_missing(state)

    if intent == "END_DIALOGUE":
        return "GOODBYE"

    if intent in ("START_TRIP", "PROVIDE_INFO", "MODIFY_PLAN"):
        if state.missing_slots:
            return "ASK_CLARIFICATION"
        return "ACK_UPDATE"

    if intent == "REQUEST_PLAN":
        if state.missing_slots:
            return "ASK_CLARIFICATION"
        return "SHOW_PLAN"

    # fallback
    if state.missing_slots:
        return "ASK_CLARIFICATION"
    return "ACK_UPDATE"

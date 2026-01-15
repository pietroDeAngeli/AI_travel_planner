from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from data import UserInformation


@dataclass
class DialogueState:
    info: UserInformation = field(default_factory=UserInformation)
    current_intent: Optional[str] = None
    confirmed = False
    current_state = None
    slot_to_change: Optional[str] = None
    task_intent: Optional[str] = None
    plan: Optional[Dict[str, Any]] = None  # Stores activities, accommodations, etc.

def update_info(state: DialogueState,intent: str, new_slots: Dict[str, Any]) -> None:
    if intent in ("START_TRIP","REQUEST_PLAN","TRAVEL_METHOD","ACCOMMODATION_PREFERENCE","MODIFY_PLAN"):
        state.task_intent = intent
    state.current_intent = intent
    state.info.update_info(new_slots)

    missing_slots = state.info.missing_slots(intent)
    if len(missing_slots) > 0:
        state.confirmed = False

def update_current_state(new_state: str, state: DialogueState) -> None:
    state.current_state = new_state

def dm_decide(intent: str, state: DialogueState) -> str:
    """
    Return one of DM_ACTIONS
    """

    missing_slots = state.info.missing_slots(intent)

    if intent == "END_DIALOGUE":
        return "GOODBYE"
    
    if intent == "CONFIRM_DETAILS":
        state.confirmed = True
        if state.task_intent == "REQUEST_PLAN":
            return "PLAN_ACTIVITIES"
        if state.task_intent == "TRAVEL_METHOD":
            return "PLAN_TRAVEL_METHOD"
        if state.task_intent == "ACCOMMODATION_PREFERENCE":
            return "PLAN_ACCOMMODATION"
        return "ACK_UPDATE"
    
    if intent == "CHANGE_DETAILS":
        return "ASK_WHICH_SLOT_TO_CHANGE"
    
    if intent == "PROVIDE_CHANGE_VALUE":
        # User has provided the new value for the pending slot
        # The slot should already be updated via update_info()
        # Now acknowledge the change
        changed_slot = state.slot_to_change
        state.slot_to_change = None  # Clear the pending slot
        return "ACK_CHANGED_SLOT"
    
    if intent == "REQUEST_PLAN" and state.confirmed:
        return "PLAN_ACTIVITIES"
    
    if intent == "TRAVEL_METHOD" and state.confirmed:
        return "PLAN_TRAVEL_METHOD"
    
    if intent == "ACCOMMODATION_PREFERENCE" and state.confirmed:
        return "PLAN_ACCOMMODATION"

    if intent in ("START_TRIP", "MODIFY_PLAN", "REQUEST_PLAN", "TRAVEL_METHOD", "ACCOMMODATION_PREFERENCE"):
        if len(missing_slots) > 0:
            return "ASK_CLARIFICATION"
        else: # All slots present but there's no confirmation yet
            return "ASK_CONFIRMATION"
    
    return "ASK_CLARIFICATION"

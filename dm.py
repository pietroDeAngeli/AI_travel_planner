import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from data import TripContext
from schema import INTENTS


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

    def __str__(self) -> str:
        return (
            "DialogueState(\n"
            f"  current_intent = {self.current_intent}\n"
            f"  context:\n{self.context}\n"
            ")"
        )

    def get_current_booking(self):
        """Get the booking object for the current intent."""
        return self.context.get_booking(self.current_intent)

    def get_missing_slots(self) -> List[str]:
        """Get missing slots for the current booking."""
        booking = self.get_current_booking()
        if booking and hasattr(booking, 'missing_slots'):
            return booking.missing_slots()
        return []
    
    def to_summary(self) -> Dict[str, Any]:
        """Create a summary of current state."""
        booking = self.get_current_booking()
        booking_data = booking.to_dict() if booking else {}
        filled_slots = {k: v for k, v in booking_data.items() if v is not None}
        
        return {
            "current_intent": self.current_intent,
            "filled_slots": filled_slots,
            "missing_slots": self.get_missing_slots(),
            "completed_bookings": list(self.context.completed_intents),
        }


# =============================================================================
# RULE-BASED DIALOGUE MANAGER
# =============================================================================

# Confirmation keywords
CONFIRM_KEYWORDS = {"yes", "yeah", "yep", "sure", "ok", "okay", "correct", "right", "confirm", "si", "sì"}
DENY_KEYWORDS = {"no", "nope", "nah", "wrong", "change", "cancel", "modify", "different"}
GOODBYE_KEYWORDS = {"bye", "goodbye", "thanks", "thank", "done", "exit", "quit", "ciao", "grazie"}


def _is_confirmation(user_text: str) -> bool:
    """Check if user utterance is a confirmation."""
    words = set(user_text.lower().split())
    return bool(words & CONFIRM_KEYWORDS)


def _is_denial(user_text: str) -> bool:
    """Check if user utterance is a denial."""
    words = set(user_text.lower().split())
    return bool(words & DENY_KEYWORDS)


def _is_goodbye(user_text: str) -> bool:
    """Check if user wants to end the conversation."""
    words = set(user_text.lower().split())
    return bool(words & GOODBYE_KEYWORDS)


def _get_complete_action(intent: str) -> str:
    """Get the completion action for an intent."""
    mapping = {
        "BOOK_FLIGHT": "COMPLETE_FLIGHT_BOOKING",
        "BOOK_ACCOMMODATION": "COMPLETE_ACCOMMODATION_BOOKING",
        "BOOK_ACTIVITY": "COMPLETE_ACTIVITY_BOOKING",
    }
    return mapping.get(intent, "ASK_CLARIFICATION")


def dm_decide(
    state: DialogueState,
    nlu_output: Dict[str, Any],
    user_utterance: str = "",
) -> str:
    """
    Rule-based Dialogue Manager: decides the next system action.
    
    Decision priority:
    1. GOODBYE intent or keywords -> GOODBYE
    2. COMPARE_CITIES with both cities -> COMPARE_CITIES_RESULT
    3. Handle confirmation/denial after ASK_CONFIRMATION
    4. Handle carryover acceptance/rejection
    5. OOD intent -> ASK_CLARIFICATION
    6. Offer carryover if available
    7. Request missing slots
    8. All slots filled -> ASK_CONFIRMATION
    
    Args:
        state: Current dialogue state
        nlu_output: NLU result with intent and slots
        user_utterance: Raw user input (for confirmation detection)
    
    Returns:
        The next DM action string.
    """
    intent = nlu_output.get("intent", "OOD")
    slots = nlu_output.get("slots", {})
    user_text = user_utterance.lower().strip()
    
    # =========================================================================
    # 1. GOODBYE - End conversation
    # =========================================================================
    if intent == "GOODBYE" or _is_goodbye(user_text):
        state.last_action = "GOODBYE"
        return "GOODBYE"
    
    # =========================================================================
    # 2. COMPARE_CITIES - Informative intent (no booking)
    # =========================================================================
    if intent == "COMPARE_CITIES":
        city1 = slots.get("city1")
        city2 = slots.get("city2")
        if city1 and city2:
            state.last_action = "COMPARE_CITIES_RESULT"
            return "COMPARE_CITIES_RESULT"
        else:
            state.last_action = "ASK_CLARIFICATION"
            return "ASK_CLARIFICATION"
    
    # =========================================================================
    # 3. Handle user response after ASK_CONFIRMATION
    # =========================================================================
    if state.last_action == "ASK_CONFIRMATION":
        if _is_confirmation(user_text):
            # Complete the booking
            action = _get_complete_action(state.current_intent)
            state.context.mark_completed(state.current_intent)
            state.last_action = action
            return action
        elif _is_denial(user_text):
            # User wants to change something
            state.last_action = "HANDLE_DENIAL"
            return "HANDLE_DENIAL"
        # If neither, continue processing as new input
    
    # =========================================================================
    # 4. Handle carryover offer response
    # =========================================================================
    if state.awaiting_carryover_response:
        if _is_confirmation(user_text):
            # Apply carryover slots
            if state.pending_carryover:
                booking = state.get_current_booking()
                if booking:
                    for slot, value in state.pending_carryover.items():
                        if hasattr(booking, slot):
                            setattr(booking, slot, value)
            state.pending_carryover = None
            state.awaiting_carryover_response = False
            # Continue to check missing slots below
        elif _is_denial(user_text):
            # User declined carryover
            state.pending_carryover = None
            state.awaiting_carryover_response = False
            # Continue to check missing slots below
    
    # =========================================================================
    # 5. OOD - Out of domain
    # =========================================================================
    if intent == "OOD":
        state.last_action = "ASK_CLARIFICATION"
        return "ASK_CLARIFICATION"
    
    # =========================================================================
    # 6. Update state for booking intents
    # =========================================================================
    if intent in INTENTS:
        # Check for carryover opportunity when switching intents
        if state.current_intent and state.current_intent != intent:
            carryover = state.context.get_carryover_values(state.current_intent, intent)
            if carryover:
                state.pending_carryover = carryover
        
        state.current_intent = intent
    
    # Update slots in current booking
    booking = state.get_current_booking()
    if booking and slots:
        for slot, value in slots.items():
            if value is not None and hasattr(booking, slot):
                setattr(booking, slot, value)
    
    # =========================================================================
    # 7. Offer slot carryover if available
    # =========================================================================
    if state.pending_carryover and not state.awaiting_carryover_response:
        state.awaiting_carryover_response = True
        state.last_action = "OFFER_SLOT_CARRYOVER"
        return "OFFER_SLOT_CARRYOVER"
    
    # =========================================================================
    # 8. Request missing slots
    # =========================================================================
    missing = state.get_missing_slots()
    if missing:
        first_missing = missing[0]
        action = f"REQUEST_MISSING_SLOT({first_missing})"
        state.last_action = action
        return action
    
    # =========================================================================
    # 9. All slots filled -> Ask for confirmation
    # =========================================================================
    if booking:
        state.last_action = "ASK_CONFIRMATION"
        return "ASK_CONFIRMATION"
    
    # =========================================================================
    # Fallback
    # =========================================================================
    state.last_action = "ASK_CLARIFICATION"
    return "ASK_CLARIFICATION"

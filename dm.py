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
    
    # Whether the current booking has been confirmed by the user
    confirmed: bool = False

    def __str__(self) -> str:
        return (
            "DialogueState(\n"
            f"  current_intent = {self.current_intent}\n"
            f"  confirmed = {self.confirmed}\n"
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
            "confirmed": self.confirmed,
            "completed_bookings": list(self.context.completed_intents),
        }
    
    def reset_for_new_intent(self) -> None:
        """Reset confirmation state when switching to a new intent."""
        self.confirmed = False


# =============================================================================
# RULE-BASED DIALOGUE MANAGER
# =============================================================================

# Confirmation keywords
CONFIRM_KEYWORDS = {"yes", "yeah", "yep", "sure", "ok", "okay", "correct", "right", "confirm", "si", "sì", "affirmative", "absolutely", "definitely"}
DENY_KEYWORDS = {"no", "nope", "nah", "wrong", "change", "cancel", "modify", "different", "incorrect", "not"}


def _is_confirmation(user_text: str) -> bool:
    """Check if user utterance is a confirmation."""
    words = set(user_text.lower().split())
    return bool(words & CONFIRM_KEYWORDS)


def _is_denial(user_text: str) -> bool:
    """Check if user utterance is a denial."""
    words = set(user_text.lower().split())
    return bool(words & DENY_KEYWORDS)


def _get_complete_action(intent: str) -> str:
    """Get the completion action for an intent."""
    mapping = {
        "BOOK_FLIGHT": "COMPLETE_FLIGHT_BOOKING",
        "BOOK_ACCOMMODATION": "COMPLETE_ACCOMMODATION_BOOKING",
        "BOOK_ACTIVITY": "COMPLETE_ACTIVITY_BOOKING",
    }
    return mapping.get(intent, "ASK_CLARIFICATION")


def _update_state_with_nlu(state: DialogueState, nlu_output: Dict[str, Any]) -> None:
    """
    Update the dialogue state with parsed intent and slots from NLU.
    
    Args:
        state: Current dialogue state to update
        nlu_output: NLU result with intent and slots
    """
    intent = nlu_output.get("intent")
    slots = nlu_output.get("slots", {})
    
    # Skip state updates for non-booking intents
    if intent not in INTENTS or intent in ("OOD", "COMPARE_CITIES"):
        return
    
    # Check if we're switching intents
    if state.current_intent and state.current_intent != intent:
        # Calculate carryover values before switching
        carryover = state.context.get_carryover_values(state.current_intent, intent)
        if carryover:
            state.pending_carryover = carryover
        # Reset confirmation state for new intent
        state.reset_for_new_intent()
    
    # Set the current intent
    state.current_intent = intent
    
    # Update slots in current booking
    booking = state.get_current_booking()
    if booking and slots:
        for slot, value in slots.items():
            if value is not None and hasattr(booking, slot):
                setattr(booking, slot, value)


def dm_decide(
    pipe,  # Kept for compatibility, ignored in rule-based DM
    state: DialogueState,
    nlu_output: Dict[str, Any],
    user_utterance: str = "",
) -> str:
    """
    Strictly rule-based Dialogue Manager: decides the next system action.
    No LLM is used - all decisions are deterministic based on state and input.
    
    Decision priority:
    1. END_DIALOGUE intent -> GOODBYE
    2. OOD or unknown intent -> ASK_CLARIFICATION
    3. Determine task intent and compute missing slots
       - If missing slots exist -> REQUEST_MISSING_SLOT
    4. If no missing slots and not confirmed -> ASK_CONFIRMATION
    5. If confirmed -> return completion action for the intent
    
    Special handling:
    - Negative confirmation -> REQUEST_SLOT_CHANGE
    - Positive confirmation -> set confirmed=True and proceed
    
    Args:
        pipe: Pipeline object (ignored, kept for compatibility)
        state: Current dialogue state
        nlu_output: NLU result with intent and slots
        user_utterance: Raw user input (for confirmation detection)
    
    Returns:
        The next DM action string.
    """
    intent = nlu_output.get("intent")
    slots = nlu_output.get("slots", {})
    user_text = user_utterance.lower().strip()
    
    # =========================================================================
    # RULE 1: END_DIALOGUE intent -> GOODBYE
    # =========================================================================
    if intent == "END_DIALOGUE" or intent == "GOODBYE":
        state.last_action = "GOODBYE"
        return "GOODBYE"
    
    # =========================================================================
    # RULE 2: OOD or unknown/None intent -> ASK_CLARIFICATION
    # =========================================================================
    if intent == "OOD" or intent is None or intent not in INTENTS:
        state.last_action = "ASK_CLARIFICATION"
        return "ASK_CLARIFICATION"
    
    # =========================================================================
    # SPECIAL: Handle COMPARE_CITIES (informative, no booking flow)
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
    # HANDLE CONFIRMATION STATE (before updating state)
    # Check if we were waiting for confirmation and user responded
    # =========================================================================
    if state.last_action == "ASK_CONFIRMATION":
        if _is_denial(user_text):
            # Negative confirmation: user wants to change something
            state.last_action = "REQUEST_SLOT_CHANGE"
            return "REQUEST_SLOT_CHANGE"
        elif _is_confirmation(user_text):
            # Positive confirmation: mark as confirmed and proceed to completion
            state.confirmed = True
            action = _get_complete_action(state.current_intent)
            state.context.mark_completed(state.current_intent)
            state.last_action = action
            return action
        # If neither clear confirmation nor denial, continue processing as new input
    
    # =========================================================================
    # HANDLE CARRYOVER OFFER RESPONSE
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
        elif _is_denial(user_text):
            # User declined carryover
            state.pending_carryover = None
            state.awaiting_carryover_response = False
        # Continue to slot checking below
    
    # =========================================================================
    # UPDATE STATE with NLU output (intent + slots)
    # =========================================================================
    _update_state_with_nlu(state, nlu_output)
    
    # =========================================================================
    # RULE 3: Determine task intent and check for missing slots
    # =========================================================================
    task_intent = state.current_intent
    
    # If no task intent established yet, ask for clarification
    if not task_intent:
        state.last_action = "ASK_CLARIFICATION"
        return "ASK_CLARIFICATION"
    
    # Offer carryover if available and not yet offered
    if state.pending_carryover and not state.awaiting_carryover_response:
        state.awaiting_carryover_response = True
        state.last_action = "OFFER_SLOT_CARRYOVER"
        return "OFFER_SLOT_CARRYOVER"
    
    # Get missing slots for the current task intent
    missing_slots = state.get_missing_slots()
    
    if missing_slots:
        # There are missing required slots -> request the first one
        first_missing = missing_slots[0]
        action = f"REQUEST_MISSING_SLOT({first_missing})"
        state.last_action = action
        return action
    
    # =========================================================================
    # RULE 4: No missing slots and not confirmed -> ASK_CONFIRMATION
    # =========================================================================
    if not state.confirmed:
        state.last_action = "ASK_CONFIRMATION"
        return "ASK_CONFIRMATION"
    
    # =========================================================================
    # RULE 5: Confirmed -> return completion action for the intent
    # =========================================================================
    if state.confirmed:
        action = _get_complete_action(task_intent)
        state.context.mark_completed(task_intent)
        state.last_action = action
        return action
    
    # =========================================================================
    # FALLBACK: ASK_CONFIRMATION is the default
    # =========================================================================
    state.last_action = "ASK_CONFIRMATION"
    return "ASK_CONFIRMATION"

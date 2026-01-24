import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from data import TripContext
from schema import INTENTS, get_dm_actions_list, build_dm_actions_prompt


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
        """Create a summary of current state for LLM consumption."""
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
# LLM-BASED DIALOGUE MANAGER
# =============================================================================

def _build_dm_system_prompt() -> str:
    """Build dynamic DM system prompt with action rules."""
    return f"""You are a Dialogue Manager (DM) for a travel booking system.
Your task is to decide the next system action based on the dialogue state and NLU output.

{build_dm_actions_prompt()}

DECISION PRIORITY (follow in order):
1. If intent is OOD or unclear -> ASK_CLARIFICATION
2. If intent is COMPARE_CITIES and city1, city2 are filled -> COMPARE_CITIES_RESULT  
3. If user said goodbye/thanks/done -> GOODBYE
4. If last_action was ASK_CONFIRMATION:
   - User confirmed (yes/ok/sure) -> COMPLETE_<INTENT>_BOOKING
   - User denied (no/change) -> HANDLE_DENIAL
5. If last_action was OFFER_SLOT_CARRYOVER:
   - User accepted -> continue with booking flow
   - User denied -> HANDLE_DENIAL (reset carryover slots)
6. If pending_carryover exists and not yet offered -> OFFER_SLOT_CARRYOVER
7. If missing_slots is NOT empty -> REQUEST_MISSING_SLOT(first_missing_slot)
8. If missing_slots is empty (all slots filled) -> ASK_CONFIRMATION

OUTPUT FORMAT:
- For REQUEST_MISSING_SLOT, output: REQUEST_MISSING_SLOT(slot_name) where slot_name is the first missing slot
- For booking completion, use the specific action: COMPLETE_FLIGHT_BOOKING, COMPLETE_ACCOMMODATION_BOOKING, or COMPLETE_ACTIVITY_BOOKING
- For all other actions, output just the action name

Output ONLY a single action. No explanation.
"""

def _dm_llm_decide(
    pipe,
    state: DialogueState,
    intent: str,
    slots: Dict[str, Any],
) -> str:
    """
    Use LLM to decide the next dialogue action.
    Returns the action string.
    """
    from schema import is_valid_action
    
    state_summary = state.to_summary()
    state_summary["last_action"] = state.last_action
    state_summary["pending_carryover"] = state.pending_carryover
    
    system_prompt = _build_dm_system_prompt()
    
    user_prompt = f"""Current dialogue state:
{json.dumps(state_summary, indent=2)}

NLU output:
- Intent: {intent}
- Extracted slots: {json.dumps(slots)}

What action should the system take next?"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    text = "ASK_CLARIFICATION"
    try:
        out = pipe(messages, max_new_tokens=50)
        generated = out[0]["generated_text"]
        if isinstance(generated, list):
            text = generated[-1].get("content", "")
        else:
            text = str(generated)
    except Exception as e:
        print(f"[DM] LLM call failed: {e}")
        return "ASK_CLARIFICATION"
    
    # Clean and validate action
    action = text.strip().upper() if text else "ASK_CLARIFICATION"
    # Remove any trailing punctuation or extra text
    action = action.split()[0] if action else "ASK_CLARIFICATION"
    
    if not is_valid_action(action):
        print(f"[DM] Invalid action from LLM: {action}")
        return "ASK_CLARIFICATION"
    
    return action


def dm_decide(
    pipe,
    state: DialogueState,
    nlu_output: Dict[str, Any],
) -> str:
    """
    Dialogue Manager: decides the next system action.
    
    Args:
        pipe: LLM pipeline
        state: Current dialogue state
        nlu_output: NLU result with intent and slots
        user_utterance: Raw user input
    
    Returns:
        The next DM action string.
    """
    intent = nlu_output.get("intent", "OOD")
    slots = nlu_output.get("slots", {})

    # Update current intent for booking intents
    if intent in INTENTS and intent != "OOD":
        # Check for carryover opportunity when switching intents
        if state.current_intent and state.current_intent != intent:
            carryover = state.context.get_carryover_slots(state.current_intent, intent)
            if carryover:
                state.pending_carryover = carryover
        state.current_intent = intent
    
    # Update slots in current booking
    booking = state.get_current_booking()
    if booking and slots:
        for slot, value in slots.items():
            if value is not None and hasattr(booking, slot):
                setattr(booking, slot, value)

    # Get action from LLM
    action = _dm_llm_decide(pipe, state, intent, slots)
    state.last_action = action
    
    return action

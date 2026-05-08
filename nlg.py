from schema import DM_ACTIONS, parse_action, SLOT_DESCRIPTIONS, BUDGET_LEVELS, ACTIVITY_CATEGORIES
from dm import DialogueState
from typing import Dict, List, Optional

import logging
logging.basicConfig(level=logging.DEBUG)

# --- Mixed Initiative: proactive suggestions for each slot ---

POPULAR_DESTINATIONS = [
    "Rome", "Paris", "Barcelona", "London", "Amsterdam",
    "Prague", "Lisbon", "Vienna", "Berlin", "Athens"
]

MIXED_INITIATIVE_OPTIONS = {
    # Common
    "destination": f"Popular destinations: {', '.join(POPULAR_DESTINATIONS)}",
    "budget_level": f"Available budget levels: {', '.join(BUDGET_LEVELS)}",
    # Accommodation
    "check_in_date": "Tip: you can say a specific date like 'June 1st' or 'next Monday'",
    "check_out_date": "Tip: you can say a specific date or a duration like '3 nights'",
    "num_guests": "Usually 1-4 guests per room",
    # Activity
    "activity_category": f"Available categories: {', '.join(list(ACTIVITY_CATEGORIES.keys()))}",
    "preferred_time": "Available times: morning, afternoon, evening, or a specific time like 10:00",
    # Compare cities
    "city1": f"Popular cities to compare: {', '.join(POPULAR_DESTINATIONS[:6])}",
    "city2": f"Popular cities to compare: {', '.join(POPULAR_DESTINATIONS[:6])}",
}

GREETING_MESSAGE = """Hello! I'm your travel assistant. I can help you with:
- Booking flights
- Finding accommodation
- Discovering activities and tours
- Comparing cities for your travel plans

How can I help you today?
"""


def nlg_generate(
    pipe,
    action: str,
    state: DialogueState,
    dialogue_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    NLG module: generates the surface utterance
    based on the DM action and dialogue state.
    Optionally uses the last turn of dialogue_history for context-aware responses
    (e.g. to apologize for mistakes or acknowledge corrections).
    """
    base_action, slot_param = parse_action(action)
    
    if base_action not in DM_ACTIONS:
        base_action = "ASK_CLARIFICATION"
        slot_param = None

    prompt_builders = {
        "REQUEST_MISSING_SLOT": _prompt_request_missing_slot,
        "OFFER_SLOT_CARRYOVER": _prompt_offer_carryover,
        "ASK_CONFIRMATION": _prompt_ask_confirmation,
        "REQUEST_SLOT_CHANGE": _prompt_request_slot_change,
        "COMPLETE_FLIGHT_BOOKING": _prompt_complete_flight,
        "COMPLETE_ACCOMMODATION_BOOKING": _prompt_complete_accommodation,
        "COMPLETE_ACTIVITY_BOOKING": _prompt_complete_activity,
        "COMPARE_CITIES_RESULT": _prompt_compare_cities,
        "ASK_CLARIFICATION": _prompt_ask_clarification,
        "GOODBYE": _prompt_goodbye,
    }

    prompt_builder = prompt_builders.get(base_action, _prompt_ask_clarification)
    
    if "REQUEST_MISSING_SLOT" in base_action and slot_param:
        prompt = _prompt_request_missing_slot(state, slot_param)
    else:
        prompt = prompt_builder(state)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a polite and helpful travel assistant. Be CONCISE and friendly.\n"
                "Instead of simply requesting information, try to proactively suggest options"
            )
        },
    ]

    # Inject the last turn from dialogue history for context-aware generation.
    # This allows the NLG to produce more natural responses such as
    # "I apologize for the confusion, I've updated the destination to Rome"
    # instead of a generic "Got it, I've updated the destination to Rome".
    if dialogue_history:
        for turn in dialogue_history[-2:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({
        "role": "user",
        "content": prompt
    })

    try:
        out = pipe(messages)
    except Exception as e:
        logging.error(f"Error calling pipe: {e}")
        return "I'm sorry, I'm having trouble generating a response right now."

    return out[0]["generated_text"][-1]["content"].strip()


# --- Prompt builders ---

def _prompt_request_missing_slot(state: DialogueState, slot_name: str = None) -> str:
    if slot_name:
        slot = slot_name
    else:
        missing = state.get_missing_slots()
        slot = missing[0] if missing else "some information"
    
    slot_description = SLOT_DESCRIPTIONS.get(slot, slot)
    
    intent_context = {
        "BOOK_FLIGHT": "flight booking",
        "BOOK_ACCOMMODATION": "accommodation booking",
        "BOOK_ACTIVITY": "activity booking",
        "COMPARE_CITIES": "city comparison",
    }.get(state.current_intent, "request")

    # Mixed Initiative: get proactive suggestions for this slot
    suggestions = MIXED_INITIATIVE_OPTIONS.get(slot, "")
    suggestions_block = f"\nProactive suggestions to offer: {suggestions}" if suggestions else ""

    return f"""
You are helping a user with their {intent_context}.

Missing information needed: {slot}
Description: {slot_description}{suggestions_block}

Instead of simply asking for the missing information, SUGGEST one of the available options to help the user decide.
Start with a progress marker ("Great!", "Perfect!", "Almost there!") then ask the question.
Keep it to 1-2 short sentences.
"""
# For example, instead of "Where would you like to go?" say "Where would you like to go? Some popular choices are Rome, Paris, and Barcelona!"



def _prompt_offer_carryover(state: DialogueState) -> str:
    carryover = state.pending_carryover or {}
    values_str = ", ".join([f"{k}: {v}" for k, v in carryover.items()]) if carryover else "previous values"
    
    return f"""
The user is starting a new booking. You have information from their previous booking that could be reused.

Values available to reuse: {values_str}

Ask the user if they would like to use the same values for this booking.
Be concise and natural.
Example: "Great! Would you like to use the information from the previous booking?"
"""


def _prompt_request_slot_change(state: DialogueState) -> str:
    booking = state.get_current_booking()
    booking_data = booking.to_dict() if booking else {}
    filled_slots = {k: v for k, v in booking_data.items() if v is not None}
    
    return f"""
The user wants to change something in their booking.

Current booking details:
{filled_slots}

Ask which information they would like to change.
Be helpful and list the options briefly.
"""


def _prompt_compare_cities(state: DialogueState) -> str:
    data = state.compare_cities_data or {}
    city1 = data.get("city1", "City 1")
    city2 = data.get("city2", "City 2")
    category = data.get("activity_category", "general")

    return f"""
Compare {city1} and {city2} for {category} activities.
Be informative but concise (3-4 sentences).
Offer to help with booking accommodation in those cities.
"""


def _prompt_ask_confirmation(state: DialogueState) -> str:
    booking = state.get_current_booking()
    booking_data = booking.to_dict() if booking else {}
    
    # Filter out None values for cleaner display
    filled_slots = {k: v for k, v in booking_data.items() if v is not None}
    
    intent_name = {
        "BOOK_FLIGHT": "flight",
        "BOOK_ACCOMMODATION": "accommodation",
        "BOOK_ACTIVITY": "activity",
    }.get(state.current_intent, "booking")

    return f"""
Summarize the following {intent_name} details and ask for confirmation.

Details:
{filled_slots}

START with a positive marker like "Perfect!" or "Excellent!" to acknowledge completion.
GROUND the information by briefly repeating key details.
End with a clear confirmation question like "Should I proceed with this booking?"
Keep it concise but include all the details.
"""


def _prompt_complete_flight(state: DialogueState) -> str:
    flight = state.context.flight.to_dict()
    filled = {k: v for k, v in flight.items() if v is not None}
    
    return f"""
Confirm the flight booking with these details: {filled}

START with an enthusiastic marker like "Excellent!" or "All set!" 
Be brief. Mention the key details and ask if they need anything else.
"""


def _prompt_complete_accommodation(state: DialogueState) -> str:
    accommodation = state.context.accommodation.to_dict()
    filled = {k: v for k, v in accommodation.items() if v is not None}
    
    return f"""
Confirm the accommodation booking with these details: {filled}

START with a positive marker like "Perfect!" or "Great!"
Be brief. Mention the name of the accommodation. Ask if they need anything else.
"""

def _prompt_complete_activity(state: DialogueState) -> str:
    activity = state.context.activity.to_dict()
    filled = {k: v for k, v in activity.items() if v is not None}
    
    return f"""
Confirm the activity booking with these details: {filled}

START with an enthusiastic marker like "Wonderful!" or "All done!"
Be brief. Mention the key details and ask if they need anything else.
"""


def _prompt_ask_clarification(state: DialogueState) -> str:
    return """
Politely ask the user to clarify their request.
USE a friendly marker like "I'd be happy to help!" to show willingness.
Mention what you can help with:
- Flights
- Hotels/accommodation
- Activities
- Travel information

Keep it brief and helpful.
"""


def _prompt_goodbye(state: DialogueState) -> str:
    # Check if any bookings were completed
    completed = state.context.completed_intents
    
    if completed:
        return f"""
            Say goodbye to the user. They completed the following bookings: {completed}

            Include:
            - Brief farewell
            - Wish them a good trip

            Keep it warm and brief.
            """
    
    return """
        Say goodbye politely.
        Keep it brief and friendly.
    """

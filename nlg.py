from schema import DM_ACTIONS, parse_action, SLOT_DESCRIPTIONS
from dm import DialogueState

GREETING_MESSAGE = """Hello! I'm your travel assistant. I can help you with:
- Booking flights
- Finding accommodation
- Discovering activities and tours
- Comparing cities for your travel plans

How can I help you today?
"""


def nlg_generate(pipe, action: str, state: DialogueState) -> str:
    """
    NLG module: generates the surface utterance
    based on the DM action and dialogue state.
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
            "content": "You are a polite and helpful travel assistant. Be concise and friendly."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    out = pipe(
        messages,
        max_new_tokens=150,
        do_sample=False,
    )

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
    }.get(state.current_intent, "request")

    return f"""
You are helping a user with their {intent_context}.

Missing information needed: {slot}
Description: {slot_description}

Ask for this information directly and politely. Start with a progress marker ("Great!", "Perfect!", "Almost there!") then ask the question.
Do NOT use phrases like "just to confirm" or "to clarify".
Keep it to ONE short sentence.
"""


def _prompt_offer_carryover(state: DialogueState) -> str:
    carryover = state.pending_carryover or {}
    values_str = ", ".join([f"{k}: {v}" for k, v in carryover.items()]) if carryover else "previous values"
    
    return f"""
The user is starting a new booking. You have information from their previous booking that could be reused.

Values available to reuse: {values_str}

Ask the user if they would like to use the same values for this booking.
Be concise and natural. START with a positive marker like "Great!" or "Perfect!".
Example: "Great! Would you like to use the same dates and number of guests from your flight booking?"
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
    # Note: This would need actual API data in production
    return """
Provide a helpful comparison between the two cities mentioned by the user.
Focus on the activity category they're interested in.
Be informative but concise (3-4 sentences).
Offer to help with booking activities in either city.
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
Be brief (2-3 sentences max). Mention the key details and ask if they need anything else.
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

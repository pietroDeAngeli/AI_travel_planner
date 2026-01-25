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
    # Parse action to get base action name and parameter
    base_action, slot_param = parse_action(action)
    
    # Validate base action
    if base_action not in DM_ACTIONS:
        base_action = "ASK_CLARIFICATION"
        slot_param = None

    # Build the appropriate prompt based on action
    prompt_builders = {
        "REQUEST_MISSING_SLOT": _prompt_request_missing_slot,
        "OFFER_SLOT_CARRYOVER": _prompt_offer_carryover,
        "ASK_CONFIRMATION": _prompt_ask_confirmation,
        "HANDLE_DENIAL": _prompt_handle_denial,
        "COMPLETE_FLIGHT_BOOKING": _prompt_complete_flight,
        "COMPLETE_ACCOMMODATION_BOOKING": _prompt_complete_accommodation,
        "COMPLETE_ACTIVITY_BOOKING": _prompt_complete_activity,
        "COMPARE_CITIES_RESULT": _prompt_compare_cities,
        "ASK_CLARIFICATION": _prompt_ask_clarification,
        "GOODBYE": _prompt_goodbye,
    }

    prompt_builder = prompt_builders.get(base_action, _prompt_ask_clarification)
    
    # Pass slot_param if action is REQUEST_MISSING_SLOT
    if base_action == "REQUEST_MISSING_SLOT" and slot_param:
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
        temperature=0.7,
        do_sample=True,
    )

    return out[0]["generated_text"][-1]["content"].strip()


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

def _prompt_request_missing_slot(state: DialogueState, slot_name: str = None) -> str:
    # Use provided slot_name or get first missing
    if slot_name:
        slot = slot_name
    else:
        missing = state.get_missing_slots()
        slot = missing[0] if missing else "some information"
    
    # Get slot description if available
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

Ask ONLY about this information in a natural and polite way.
Keep it short (1 max).
"""


def _prompt_offer_carryover(state: DialogueState) -> str:
    carryover = state.pending_carryover or {}
    values_str = ", ".join([f"{k}: {v}" for k, v in carryover.items()]) if carryover else "previous values"
    
    return f"""
The user is starting a new booking. You have information from their previous booking that could be reused.

Values available to reuse: {values_str}

Ask the user if they would like to use the same values for this booking.
Be concise and natural.
Example: "Would you like to use the same dates and number of guests from your flight booking?"
"""


def _prompt_handle_denial(state: DialogueState) -> str:
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

End with a clear confirmation question.
Keep it concise but include all the details.
"""


def _prompt_complete_flight(state: DialogueState) -> str:
    flight = state.context.flight.to_dict()
    filled = {k: v for k, v in flight.items() if v is not None}
    
    return f"""
The user has confirmed their flight booking. Provide a summary and completion message.

Flight details:
{filled}

Include:
- Brief confirmation message
- Summary of the booking
- Ask if they need anything else (hotel, activities)

Keep it professional but friendly.
"""


def _prompt_complete_accommodation(state: DialogueState) -> str:
    accommodation = state.context.accommodation.to_dict()
    filled = {k: v for k, v in accommodation.items() if v is not None}
    
    return f"""
The user has confirmed their accommodation booking. Provide a summary and completion message.

Accommodation details:
{filled}

Include:
- Brief confirmation message
- Summary of the booking
- Ask if they need anything else (activities, more info)

Keep it professional but friendly.
"""


def _prompt_complete_activity(state: DialogueState) -> str:
    activity = state.context.activity.to_dict()
    filled = {k: v for k, v in activity.items() if v is not None}
    
    return f"""
The user has confirmed their activity booking. Provide a summary and completion message.

Activity details:
{filled}

Include:
- Brief confirmation message
- Summary of the booking
- Ask if they need anything else

Keep it professional but friendly.
"""


def _prompt_ask_clarification(state: DialogueState) -> str:
    return """
Politely ask the user to clarify their request.
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
- Thank them for using the service

Keep it warm and brief.
"""
    else:
        return """
Say goodbye politely.
Thank them for using the service.
Keep it brief and friendly.
"""

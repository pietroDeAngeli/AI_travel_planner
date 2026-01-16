INTENT_SCHEMAS = {
    "START_TRIP": {
        "slots": ["destination", "start_date", "end_date", "num_people"]
    },
    "REQUEST_PLAN": {
        "slots": ["destination", "start_date", "end_date", "num_people", "budget_level", "travel_style"]
    },
    "ACCOMMODATION_PREFERENCE": {
        "slots": ["destination", "start_date", "end_date", "num_people", "accommodation_type", "budget_level"]
    },
    "CHANGE_DETAILS": {
        "slots": ["destination", "start_date", "end_date", "num_people", "accommodation_type", "travel_style", "budget_level"]
    },
    "PROVIDE_CHANGE_VALUE": {
        "slots": ["destination", "start_date", "end_date", "num_people", "accommodation_type", "travel_style", "budget_level"]
    },
    "CONFIRM_DETAILS": {
        "slots": []
    },
    "END_DIALOGUE": {
        "slots": []
    },
    "FALLBACK": {
        "slots": []
    },
    "GREETING": {
        "slots": []
    },
}

INTENTS = list(INTENT_SCHEMAS.keys())
SLOTS = sorted(set(slot for schema in INTENT_SCHEMAS.values() for slot in schema["slots"]))
INTENT_SLOTS = {intent: schema["slots"] for intent, schema in INTENT_SCHEMAS.items()}

# Helper function to generate schema hint for a specific intent
def get_json_schema_hint(intent="START_TRIP"):
    slots = INTENT_SCHEMAS.get(intent, {}).get("slots", [])
    return {
        "intent": intent,
        "slots": {slot: None for slot in slots},
    }

JSON_SCHEMA_HINT = get_json_schema_hint()

RULES = (
    "Rules:\n"
    "- Use START_TRIP when the user expresses the goal of planning a trip OR provides core trip info from scratch "
    "(e.g., destination, dates, duration) without referring to changes.\n"
    "- Use REQUEST_PLAN when the user asks to see the itinerary/plan/summary of the current trip.\n"
    "- Use ACCOMMODATION_PREFERENCE when the user specifies lodging type or constraints about accommodation "
    "(hotel/hostel/airbnb, star rating, room type) as a preference/constraint.\n"
    "- Use CHANGE_DETAILS when the user explicitly wants to modify something already discussed "
    "(e.g., 'change', 'instead', 'not X but Y', 'make it cheaper', 'move the date'), or asks to change a detail "
    "but DOES NOT provide the new value.\n"
    "- Use PROVIDE_CHANGE_VALUE only when the system has just asked for the new value of a specific field "
    "and the user reply is mainly the value (e.g., 'Rome', 'next Friday', '3 days', 'budget: low').\n"
    "- Use CONFIRM_DETAILS only when the system has just asked a yes/no confirmation about the current details "
    "and the user confirms or denies (yes/ok/correct/no).\n"
    "- Use END_DIALOGUE when the user ends the conversation (bye/thanks, stop, quit).\n"
    "- Otherwise use FALLBACK.\n"
)

SLOT_DESCRIPTIONS = {
    "destination": "the destination city",
    "start_date": "the start date",
    "end_date": "the end date",
    "num_people": "the number of people traveling",
    "travel_style": "the travel style preference",
    "accommodation_type": "the accommodation type preference",
    "budget_level": "the budget level for the plan",
}

DM_ACTIONS = ["ASK_CLARIFICATION", "ASK_CONFIRMATION", "ACK_UPDATE", "PLAN_ACTIVITIES", "PLAN_ACCOMMODATION", "ASK_WHICH_SLOT_TO_CHANGE", "ACK_CHANGED_SLOT", "GOODBYE"]


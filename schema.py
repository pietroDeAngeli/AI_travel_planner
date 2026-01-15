INTENT_SCHEMAS = {
    "START_TRIP": {
        "slots": ["destination", "start_date", "end_date", "num_people"]
    },
    "REQUEST_PLAN": {
        "slots": ["destination", "start_date", "end_date", "num_people", "budget_level", "travel_style"]
    },
    "MODIFY_PLAN": {
        "slots": []
    },
    "ACCOMMODATION_PREFERENCE": {
        "slots": ["destination", "start_date", "end_date", "num_people", "accommodation_type", "budget_level"]
    },
    "CONFIRM_DETAILS": {
        "slots": []
    },
    "CHANGE_DETAILS": {
        "slots": []
    },
    "PROVIDE_CHANGE_VALUE": {
        "slots": []
    },
    "END_DIALOGUE": {
        "slots": []
    },
    "FALLBACK": {
        "slots": []
    }
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
        "confidence": 0.0
    }

JSON_SCHEMA_HINT = get_json_schema_hint()

SLOT_DESCRIPTIONS = {
    "destination": "the destination city",
    "start_date": "the start date",
    "end_date": "the end date",
    "num_people": "the number of people traveling",
    "overall_budget": "the overall budget",
    "travel_style": "the travel style preference",
    "accommodation_type": "the accommodation type preference",
    "budget_level": "the budget level for the plan",
}

DM_ACTIONS = ["ASK_CLARIFICATION", "ASK_CONFIRMATION", "ACK_UPDATE", "PLAN_ACTIVITIES", "PLAN_ACCOMMODATION", "ASK_WHICH_SLOT_TO_CHANGE", "ACK_CHANGED_SLOT", "GOODBYE"]


INTENT_SCHEMAS = {
    "GREETING": {
        "slots": []
    },
    "PLAN_TRIP": {
        "slots": ["destination", "start_date", "end_date", "num_people", "accommodation_type", "travel_style", "budget_level"]
    },
    "COMPARE_OPTIONS": {
        "slots": ["option1", "option2", "criteria", "compare_type"]
    },
    "REQUEST_INFORMATION": {
        "slots": ["destination", "entity_type", "budget_constraint"]
    },
    "CONFIRM_DETAILS": {
        "slots": []
    },
    "END_DIALOGUE": {
        "slots": []
    },
    "OOD": {
        "slots": []
    },
}

INTENTS = list(INTENT_SCHEMAS.keys())
SLOTS = sorted(set(slot for schema in INTENT_SCHEMAS.values() for slot in schema["slots"]))
INTENT_SLOTS = {intent: schema["slots"] for intent, schema in INTENT_SCHEMAS.items()}

# Helper function to generate schema hint for a specific intent
def get_json_schema_hint(intent="PLAN_TRIP"):
    slots = INTENT_SCHEMAS.get(intent, {}).get("slots", [])
    return {
        "intent": intent,
        "slots": {slot: None for slot in slots},
    }

JSON_SCHEMA_HINT = get_json_schema_hint()

RULES = (
    "Rules:\n"
    "- Use PLAN_TRIP when the user expresses the goal of planning a trip, or provides any core trip information "
    "(e.g., destination, dates, number of people, budget, travel style, accommodation), even if provided incrementally "
    "or as a modification of previous information.\n"
    
    "- Use REQUEST_INFORMATION when the user asks for general information about a destination or entity "
    "(e.g., attractions, events, museums, costs), without explicitly requesting a personalized trip plan.\n"
    
    "- Use COMPARE_OPTIONS when the user asks to compare two or more options "
    "(e.g., cities, accommodations, activities, or travel choices) based on explicit or implicit criteria.\n"
    
    "- Use CONFIRM_DETAILS only when the system has explicitly asked for a yes/no confirmation "
    "and the user responds with confirmation or rejection (e.g., yes, no, correct, that's right).\n"
    
    "- Use END_DIALOGUE when the user clearly intends to end the conversation "
    "(e.g., bye, thanks, stop, quit).\n"
    
    "- Use OOD when the user input does not match any of the above intents or is out of domain.\n"
)


SLOT_DESCRIPTIONS = {
    # -------- PLAN_TRIP --------
    "destination": "the city or destination the user is referring to",

    "start_date": "the starting date of the trip",
    "end_date": "the ending date of the trip",

    "num_people": "the number of travelers included in the trip",

    "travel_style": "the preferred type of travel experience "
                    "(e.g., culture, walking, relaxation, nightlife)",

    "accommodation_type": "the preferred type of accommodation "
                    "(e.g., hotel, hostel, apartment)",

    "budget_level": "the overall budget level for the trip "
                    "(e.g., low, medium, high)",

    # -------- REQUEST_INFORMATION --------
    "entity_type": "the type of entity the user is asking information about "
                   "(e.g., hotels, flights, activities, events)",

    "budget_constraint": "a budget constraint used to filter or contextualize "
                    "the requested information",

    # -------- COMPARE_OPTIONS --------
    "option1": "the first option to be compared "
               "(e.g., a city, hotel, flight, or activity)",

    "option2": "the second option to be compared "
               "(e.g., a city, hotel, flight, or activity)",

    "criteria": "the aspect used to compare the options "
                "(e.g., price, activities, comfort)",

    "compare_type": "the type of items being compared "
                "(e.g., destination, accommodation, flight, activity)",
}


DM_ACTIONS = [
    "GREETING",
    "ASK_CLARIFICATION",
    "ASK_CONFIRMATION",
    "ACK_UPDATE",
    "REQUEST_MISSING_SLOT",
    "PROPOSE_TRIP_PLAN",
    "ASK_WHICH_SLOT_TO_CHANGE",
    "ASK_NEW_VALUE_FOR_SLOT",
    "ACK_CHANGED_SLOT",
    "GOODBYE"
]

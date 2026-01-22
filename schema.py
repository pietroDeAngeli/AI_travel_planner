INTENT_SCHEMAS = {
    "PLAN_TRIP": {
        "slots": ["destination", "start_date", "end_date", "num_people", "accommodation_type", "travel_style", "budget_level"]
    },
    "REQUEST_INFORMATION": {
        "slots": ["destination", "entity_type", "budget_level"]
    },
    "COMPARE_OPTIONS": {
        "slots": ["option1", "option2", "criteria"]
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

ACTIVITY_CATEGORIES = {

    "adventure": [
        "adventure", "hiking", "trekking", "climbing",
        "kayak", "rafting", "bike", "biking", "cycling",
        "outdoor", "jeep", "atv", "quad", "safari"
    ],

    "cultural": [
        "museum", "gallery", "art", "exhibition",
        "cathedral", "church", "basilica",
        "palace", "royal", "historic", "history",
        "monument", "heritage", "archaeological",
    ],

    "food": [
        "food", "wine", "tapas", "gastronomy",
        "culinary", "tasting", "dinner", "lunch",
        "market", "cooking", "cooking class"
    ],

    "sport": [
        "stadium", "football", "soccer",
        "basketball", "tennis",
        "bernabeu", "arena", "olympic"
    ],

    "relax": [
        "spa", "wellness", "relax",
        "thermal", "bath", "cruise",
        "boat", "river", "panoramic", "sunset"
    ],

    "nature": [
        "nature", "park", "garden",
        "botanical", "natural",
        "scenic", "landscape",
        "mountain", "lake"
    ],

    "nightlife": [
        "night", "nightlife", "bar", "pub",
        "club", "show", "concert",
        "music", "live", "flamenco"
    ],

    "family": [
        "family", "kids", "children",
        "zoo", "aquarium",
        "theme park", "amusement", "park"
    ],

    "general": [],
}

BUDGET_LEVELS = ["low", "medium", "high"]

ENTITY_TYPES = ["hotels", "flights", "activities"]

ACCOMMODATION_TYPES = ["hotel", "hostel", "apartment", "bnb"]

RULES = (
    "Rules:\n"
    "- Use PLAN_TRIP when the user expresses the goal of planning a trip, or provides any core trip information "
    f"(e.g., {INTENT_SCHEMAS['PLAN_TRIP']['slots']}), even if provided incrementally "
    "or as a modification of previous information.\n"
    
    "- Use REQUEST_INFORMATION when the user asks for general information about a destination without explicitly requesting a " "personalized trip plan. The information that could be provided are entity types such as hotels, flights, or activities, "
    f"and a budget level\n"
    
    "- Use COMPARE_OPTIONS when the user asks to compare two cities based on a specific activity criterion. \n"

    "- Use END_DIALOGUE when the user clearly intends to end the conversation "
    "(e.g., bye, stop, quit).\n"
    
    "- Use OOD when the user input does not match any of the above intents or is out of domain.\n"
)


SLOT_DESCRIPTIONS = {
    # -------- PLAN_TRIP --------
    "destination": "the city or destination the user is referring to",

    "start_date": "the starting date of the trip",
    "end_date": "the ending date of the trip",

    "num_people": "the number of travelers included in the trip",

    "travel_style": "the preferred type of travel experience "
                    f"(e.g., {list(ACTIVITY_CATEGORIES.keys())})",

    "accommodation_type": "the preferred type of accommodation "
                    f"(e.g., {ACCOMMODATION_TYPES})",

    "budget_level": "the overall budget level for the trip "
                    f"(e.g., {BUDGET_LEVELS})",

    # -------- REQUEST_INFORMATION --------
    "entity_type": "the type of entity the user is asking information about "
                   f"(e.g., {ENTITY_TYPES})",

    # -------- COMPARE_OPTIONS --------
    "option1": "the first city to be compared",

    "option2": "the second city to be compared",


    "compare_type": "the type of activity being compared "
                f"(e.g., {list(ACTIVITY_CATEGORIES.keys())})",
}


DM_ACTIONS = [
    "REQUEST_MISSING_SLOT",   # ask for one missing slot
    "ASK_CONFIRMATION",       # confirm collected information
    "PROPOSE_TRIP_PLAN",      # final goal for PLAN_TRIP
    "PROVIDE_INFORMATION",    # REQUEST_INFORMATION intent
    "PROVIDE_COMPARISON",     # COMPARE_OPTIONS intent
    "HANDLE_SLOT_CHANGE",     # generic modification flow
    "ASK_CLARIFICATION",      # fallback / OOD / ambiguity
    "GOODBYE"                 # end dialogue
]

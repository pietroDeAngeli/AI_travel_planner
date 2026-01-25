# ===========================
# Intent / Slot Schemas and functions
# ===========================

INTENT_SCHEMAS = {
    "BOOK_FLIGHT": {
        "slots": ["origin", "destination", "departure_date", "return_date", 
                  "num_passengers", "budget_level"],
        "description": "User wants to book or search for flights",
        "rule": "User mentions flights, flying, air travel, or provides origin city",
        "examples": ["I need a flight to Rome", "Fly from Milan to Paris", "Book tickets to Barcelona"]
    },
    "BOOK_ACCOMMODATION": {
        "slots": ["destination", "check_in_date", "check_out_date", 
                  "num_guests", "budget_level"],
        "description": "User wants to book or search for accommodation",
        "rule": "User mentions hotel, hostel, apartment, stay, accommodation, or lodging",
        "examples": ["Find me a hotel in Rome", "I need a place to stay", "Book a hostel"]
    },
    "BOOK_ACTIVITY": {
        "slots": ["destination", "activity_category", "budget_level"],
        "description": "User wants to book tours, experiences, or activities",
        "rule": "User mentions tours, activities, things to do, experiences",
        "examples": ["I want to go hiking", "Book a cooking class", "Find a museum tour"]
    },
    "COMPARE_CITIES": {
        "slots": ["city1", "city2", "activity_category"],
        "description": "User wants to compare two cities' activities for travel",
        "rule": "User wants to compare features of two cities",
        "examples": ["Compare Paris and London for sightseeing", 
                     "Which is better for food, Rome or Barcelona?"]
    },    
    "OOD": {
        "slots": [],
        "description": "Out of domain - request not supported",
        "rule": "User request is not related to travel booking, or is unclear",
        "examples": ["What's the weather?", "Tell me a joke", "How are you?"]
    },
}

INTENTS = list(INTENT_SCHEMAS.keys())
SLOTS = sorted(set(slot for schema in INTENT_SCHEMAS.values() for slot in schema["slots"]))
INTENT_SLOTS = {intent: schema["slots"] for intent, schema in INTENT_SCHEMAS.items()}

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
        "arena", "olympic"
    ],
    "relax": [
        "spa", "wellness", "relax",
        "thermal", "bath", "cruise",
        "boat", "river", "panoramic", "sunset"
    ],
    "nature": [
        "nature", "park", "garden",
        "botanical",
        "scenic", "landscape",
        "mountain", "lake"
    ],
    "nightlife": [
        "night", "nightlife", "bar", "pub",
        "club", "show", "concert",
        "music", "live",
    ],
    "family": [
        "family", "kids", "children",
        "zoo", "aquarium",
        "theme park", "park"
    ],
    "general": [],
}

BUDGET_LEVELS = ["low", "medium", "high"]

SLOT_DESCRIPTIONS = {
    # -------- Common --------
    "destination": "The city or place the user wants to visit\n",
    "budget_level": f"Budget preference: {BUDGET_LEVELS}\n",
    
    # -------- BOOK_FLIGHT --------
    "origin": "Departure city/airport\n",
    "departure_date": "Flight departure date\n",
    "return_date": "Flight return date (for round trips)\n",
    "num_passengers": "Number of travelers for the flight\n",
    
    # -------- BOOK_ACCOMMODATION --------
    "check_in_date": "Hotel check-in date\n",
    "check_out_date": "Hotel check-out date\n",
    "num_guests": "Number of guests staying\n",
    
    # -------- BOOK_ACTIVITY --------
    "activity_category": f"Type of activity: {list(ACTIVITY_CATEGORIES.keys())}\n",
    
    # -------- COMPARE_CITIES --------
    "city1": "First city for comparison\n",
    "city2": "Second city for comparison\n",
    "activity_category": f"Type of activity: {list(ACTIVITY_CATEGORIES.keys())}\n",
}

def _build_rules() -> str:
    rules = ["Rules for intent classification:\n"]

    for intent, schema in INTENT_SCHEMAS.items():
        rules.append(f"{intent}:")
        rules.append(f"- Rule: {schema['rule']}")
        
        examples = schema.get("examples", [])
        if examples:
            rules.append("- Examples:")
            for ex in examples:
                rules.append(f"  - {ex}")
        
        rules.append("")

    return "\n".join(rules)


RULES = _build_rules()

def get_json_schema_hint(intent: str) -> dict:
    """Generate a JSON schema hint for NLU output."""
    slots = INTENT_SCHEMAS.get(intent, {}).get("slots", [])
    return {
        "intent": intent,
        "slots": {slot: None for slot in slots},
    }

# ===========================
# Actions for Dialogue Manager
# ===========================

DM_ACTIONS = {
    # Slot collection
    "REQUEST_MISSING_SLOT": {
        "description": "Ask for a specific missing slot. Format: REQUEST_MISSING_SLOT(slot_name)",
        "rule": "When missing_slots is not empty for the current booking intent",
    },
    "OFFER_SLOT_CARRYOVER": {
        "description": "Offer to reuse values from a previous booking",
        "rule": "When switching intents and there are shared slot values from completed bookings",
    },
    
    # Confirmation flow
    "ASK_CONFIRMATION": {
        "description": "Confirm all collected information before completing a booking",
        "rule": "When all required slots are filled (missing_slots is empty) and the intent is not confirmed yet",
    },
    "HANDLE_DENIAL": {
        "description": "Handle when user denies confirmation and wants to modify something",
        "rule": "When user responds negatively to ASK_CONFIRMATION or OFFER_SLOT_CARRYOVER",
    },
    
    # Booking completion
    "COMPLETE_FLIGHT_BOOKING": {
        "description": "Finalize and confirm a flight booking",
        "rule": "When current_intent is BOOK_FLIGHT and user confirms positively the booking",
    },
    "COMPLETE_ACCOMMODATION_BOOKING": {
        "description": "Finalize and confirm an accommodation booking",
        "rule": "When current_intent is BOOK_ACCOMMODATION and user confirms positively the booking",
    },
    "COMPLETE_ACTIVITY_BOOKING": {
        "description": "Finalize and confirm an activity booking",
        "rule": "When current_intent is BOOK_ACTIVITY and user confirms positively the booking",
    },
    
    # Compare cities
    "COMPARE_CITIES_RESULT": {
        "description": "Provide comparison between two cities for activities",
        "rule": "When intent is COMPARE_CITIES AND required slots are filled AND confirmed",
    },
    
    # Dialogue management
    "ASK_CLARIFICATION": {
        "description": "Request clarification for unclear or out-of-domain input",
        "rule": "When intent is OOD or the user's request is ambiguous",
    },
    "GOODBYE": {
        "description": "End the dialogue",
        "rule": "When user says goodbye, thanks, or has no more requests",
    },
}

# Booking intents that require slot filling
BOOKING_INTENTS = ["BOOK_FLIGHT", "BOOK_ACCOMMODATION", "BOOK_ACTIVITY"]


def get_dm_actions_list() -> list:
    """Return list of action names for validation."""
    return list(DM_ACTIONS.keys())


def is_valid_action(action: str) -> bool:
    """
    Check if action is valid, including parameterized actions.
    Handles REQUEST_MISSING_SLOT(slot_name) format.
    """
    if action in DM_ACTIONS:
        return True
    # Check for parameterized actions like REQUEST_MISSING_SLOT(destination)
    if action.startswith("REQUEST_MISSING_SLOT(") and action.endswith(")"):
        return True
    return False


def parse_action(action: str) -> tuple:
    """
    Parse an action string into (action_name, parameter).
    Returns (action, None) for simple actions.
    Returns (action_name, slot_name) for REQUEST_MISSING_SLOT(slot_name).
    """
    if action.startswith("REQUEST_MISSING_SLOT(") and action.endswith(")"):
        slot_name = action[len("REQUEST_MISSING_SLOT("):-1]
        return ("REQUEST_MISSING_SLOT", slot_name)
    return (action, None)


def build_dm_actions_prompt() -> str:
    """
    Build the actions section of the DM system prompt from DM_ACTIONS.
    Returns a formatted string with action names, descriptions, and rules.
    """
    lines = ["Available actions:\n"]
    
    for action, info in DM_ACTIONS.items():
        lines.append(f"- {action}: {info['rule']}")
    
    return "\n".join(lines)
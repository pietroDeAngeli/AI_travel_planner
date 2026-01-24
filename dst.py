from dm import DialogueState
from schema import INTENTS, INTENT_SLOTS, RULES

def state_context(state: DialogueState) -> str:
    """Generate a context-aware NLU system prompt based on dialogue state."""
    
    # Base system prompt for normal dialogue flow
    base_prompt = (
        "You are an NLU module for a travel booking dialogue system.\n"
        "Task: classify the user's intent and extract slot values.\n\n"
        f"Valid intents: {INTENTS}\n\n"
        f"Valid slots per intent: {INTENT_SLOTS}\n\n"
        f"{RULES}\n\n"
        "Output MUST be a single JSON object with keys: intent, slots\n"
        "- Put null for unknown slots.\n"
        "- Never invent details.\n"
        "- Only include slots relevant to the detected intent.\n"
    )
    
    # If no prior action, return base prompt
    if not state.last_action:
        return base_prompt
    
    # Context-specific prompts based on last action
    if state.last_action in ["ASK_CONFIRMATION", "OFFER_SLOT_CARRYOVER"]:
        return (
            "You are an NLU module for a travel booking dialogue system.\n"
            "The system just asked for confirmation.\n\n"
            f"Current intent: {state.current_intent}\n\n"
            "Task: Determine if the user's response is positive or negative.\n"
            "- Positive: yes, yeah, sure, okay, correct, right, confirm, proceed\n"
            "- Negative: no, nope, wrong, change, modify, different\n\n"
            f"Return the current intent with a 'confirmation' slot.\n"
            "Output MUST be a single JSON object with keys: intent, slots\n"
            f"Example positive: {{\"intent\": \"{state.current_intent}\", \"slots\": {{\"confirmation\": \"yes\"}}}}\n"
            f"Example negative: {{\"intent\": \"{state.current_intent}\", \"slots\": {{\"confirmation\": \"no\"}}}}\n"
        )
    
    elif state.last_action == "HANDLE_DENIAL":
        missing = state.get_missing_slots()
        return (
            "You are an NLU module for a travel booking dialogue system.\n"
            "The user wants to modify their booking.\n\n"
            f"Current intent: {state.current_intent}\n"
            f"Valid slots: {INTENT_SLOTS.get(state.current_intent, [])}\n\n"
            "Task: Extract the new slot value(s) the user provides.\n"
            "Output MUST be a single JSON object with keys: intent, slots\n"
            f"Example: {{\"intent\": \"{state.current_intent}\", \"slots\": {{\"destination\": \"Rome\"}}}}\n"
        )
    
    elif state.last_action == "REQUEST_MISSING_SLOT":
        missing = state.get_missing_slots()
        next_slot = missing[0] if missing else "unknown"
        return (
            "You are an NLU module for a travel booking dialogue system.\n"
            f"The system just asked for: {next_slot}\n\n"
            f"Current intent: {state.current_intent}\n"
            f"Missing slots: {missing}\n\n"
            "Task: Extract the slot value from the user's response.\n"
            f"Focus on extracting: {next_slot}\n"
            f"Output format: {{\"intent\": \"{state.current_intent}\", \"slots\": {{\"{next_slot}\": \"value\"}}}}\n"
            "If the user provides other information, extract all relevant slots.\n"
        )
    
    # Fallback to base prompt
    return base_prompt
       

    

    
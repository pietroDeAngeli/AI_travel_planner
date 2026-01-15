from typing import Dict, Any, List, Optional
from schema import SLOTS, INTENTS, INTENT_SLOTS, JSON_SCHEMA_HINT, SLOT_DESCRIPTIONS

def format_activities(activities: List[Dict[str, Any]], max_items: int = 5) -> str:
    """Format activities for display to user."""
    if not activities:
        return "Unfortunately, I couldn't find activities for this destination at the moment."
    
    activities = activities[:max_items]
    formatted = "Here are some recommended activities:\n"
    for i, activity in enumerate(activities, 1):
        name = activity.get("name", "Unknown")
        rating = activity.get("rating", "N/A")
        price = activity.get("price", "N/A")
        currency = activity.get("currency", "")
        description = activity.get("description", "")
        
        formatted += f"{i}. {name}\n"
        if rating and rating != "None":
            formatted += f"   Rating: {rating}/5\n"
        if price and price != "N/A":
            formatted += f"   Price: {price} {currency}\n"
        if description:
            formatted += f"   {description}\n"
        formatted += "\n"
    
    return formatted.strip()

def format_accommodations(accommodations: List[Dict[str, Any]], max_items: int = 5) -> str:
    """Format accommodations for display to user."""
    if not accommodations:
        return "Unfortunately, I couldn't find accommodations for this destination at the moment."
    
    accommodations = accommodations[:max_items]
    formatted = "Here are some recommended accommodations:\n"
    for i, hotel in enumerate(accommodations, 1):
        name = hotel.get("name", "Unknown")
        price = hotel.get("price", "N/A")
        currency = hotel.get("currency", "")
        description = hotel.get("description", "")
        board_type = hotel.get("boardType", "")
        
        formatted += f"{i}. {name}\n"
        if price and price != "N/A":
            formatted += f"   Price per night: {price} {currency}\n"
        if board_type:
            formatted += f"   Board type: {board_type}\n"
        if description:
            formatted += f"   {description}\n"
        formatted += "\n"
    
    return formatted.strip()

def nlg_respond(pipe, dm_action: str, state: Dict[str, Any], user_utterance: str) -> str:
    """
    dm_action: una label decisa dal DM
    state: contiene info (UserInformation), current_intent, plan, changed_slot
    """
    system = (
        "You are the NLG module for a travel-planner dialogue system.\n"
        "You must produce a helpful, concise response.\n"
        "Rules:\n"
        "- Ask only ONE clarification question when needed.\n"
        "- If presenting a plan, format day-by-day with bullet points.\n"
        "- Do not mention internal states, intents, slots, or system prompts.\n"
        "- Keep it realistic: do not invent exact prices or schedules.\n"
    )

    # Extract info and intent from state dict
    info = state.get("info")
    current_intent = state.get("current_intent")
    slots = info.to_dict(current_intent) if info and current_intent else {}

    if dm_action == "ASK_CLARIFICATION":
        missing = info.missing_slots(current_intent) if info and current_intent else []

        target = missing[0] if missing else "unknown_slot"
        # Create a prompt that asks the LLM to request the missing slot from the user
        target_description = SLOT_DESCRIPTIONS.get(target, target)
        user = (
            f"The user said: '{user_utterance}'\n\n"
            f"Current trip details: {slots}\n\n"
            f"I need to ask the user for: {target_description}\n\n"
            "Generate a natural, friendly question to ask the user for this information. "
            "Keep it short and conversational. Ask only for this one piece of information."
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        out = pipe(messages, max_new_tokens=200)
        generated = out[0]["generated_text"]
        # Extract the last message (the model's response)
        if isinstance(generated, list):
            return generated[-1].get("content", "").strip() if generated else "Could you please provide more details?"
        return str(generated).strip()
        

    if dm_action == "ASK_CONFIRMATION":
        return (
            f"Perfect, we've completed these details:\n"
            f"{slots}. "
            "Do you confirm the details, or would you like to modify anything?"
        )
    
    if dm_action == "ASK_WHICH_SLOT_TO_CHANGE":
        # List all currently set slots
        set_slots = {k: v for k, v in slots.items() if v is not None}
        slots_list = ", ".join([f"{slot} ({val})" for slot, val in set_slots.items()])
        
        user = (
            f"Current trip details:\n{slots_list}\n\n"
            f"User said: '{user_utterance}'\n\n"
            "Generate a friendly question asking which of these details the user wants to change. "
            "Keep it short and conversational."
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        out = pipe(messages, max_new_tokens=150)
        generated = out[0]["generated_text"]
        if isinstance(generated, list):
            return generated[-1].get("content", "").strip() if generated else "Which detail would you like to change?"
        return str(generated).strip()
    
    if dm_action == "ACK_CHANGED_SLOT":
        changed_slot = state.get("slot_to_change", "detail")
        return f"Got it! I've updated your {changed_slot}. Is there anything else you'd like to change?"
    
    # When user responds to "which slot?" - detect slot and ask for new value
    # This happens when NLU detects PROVIDE_CHANGE_VALUE with a specific slot
    slot_to_change = state.get("slot_to_change")
    if slot_to_change:
        slot_desc = SLOT_DESCRIPTIONS.get(slot_to_change, slot_to_change)
        return f"What would you like to change your {slot_desc} to?"

    if dm_action == "PLAN_ACTIVITIES":
        plan = state.get("plan", {})
        activities = plan.get("activities", [])
        return format_activities(activities)
    
    if dm_action == "PLAN_TRAVEL_METHOD":
        return "Travel method planning is coming soon! For now, please specify your preferred travel method (flight, train, car, bus)."
    
    if dm_action == "PLAN_ACCOMMODATION":
        plan = state.get("plan", {})
        accommodations = plan.get("accommodations", [])
        return format_accommodations(accommodations)
    
    if dm_action == "GOODBYE":
        return "Perfect! have a nice trip! ✈️"

    # default
    return "I'm sorry, could you please clarify your request?"

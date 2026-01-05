from typing import Dict, Any, List, Optional
from schema import SLOTS, INTENTS, INTENT_SLOTS, JSON_SCHEMA_HINT

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
        slot_descriptions = {
            "destination": "the destination city",
            "start_date": "the start date",
            "end_date": "the end date",
            "num_people": "the number of people traveling",
            "overall_budget": "the overall budget",
            "travel_style": "the travel style preference",
            "travel_method": "the preferred travel method",
            "accommodation_type": "the accommodation type preference",
            "budget_level": "the budget level for the plan",
            "leaving_time_preference": "the preferred leaving time"
        }
        
        target_description = slot_descriptions.get(target, target)
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
        slot_descriptions = {
            "destination": "the destination city",
            "start_date": "the start date",
            "end_date": "the end date",
            "num_people": "the number of people traveling",
            "overall_budget": "the overall budget",
            "travel_style": "the travel style preference",
            "travel_method": "the preferred travel method",
            "accommodation_type": "the accommodation type preference",
            "budget_level": "the budget level for the plan",
            "leaving_time_preference": "the preferred leaving time"
        }
        
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
        slot_descriptions = {
            "destination": "the destination city",
            "start_date": "the start date",
            "end_date": "the end date",
            "num_people": "the number of people traveling",
            "overall_budget": "the overall budget",
            "travel_style": "the travel style preference",
            "travel_method": "the preferred travel method",
            "accommodation_type": "the accommodation type preference",
            "budget_level": "the budget level for the plan",
            "leaving_time_preference": "the preferred leaving time"
        }
        slot_desc = slot_descriptions.get(slot_to_change, slot_to_change)
        return f"What would you like to change your {slot_desc} to?"

    if dm_action == "GOODBYE":
        return "Perfect! have a nice trip! ✈️"

    # default
    return "I'm sorry, could you please clarify your request?"

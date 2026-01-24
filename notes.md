# schema.py
I decided to design the schema this way because keeping all togher confuses the NLU making the intent understanding not trivial. Here I have more separated intents. It's true the intents are not self contained but they actually are because the values are independent among each other even if some slots have the same name. This could impact on the UX so only for this case if we have already filled a "shared" attribute the system becomes system initiative and the user asks the user if he wants to use the same values, if it answers yes they are kept otherwise the system asks for new ones. 

The schema file contains:
 - Intent-slots schemas, with the rule and some examples for each intent.
 - Slots description
 - DM actions with descriptions and rules
 - Predefined values for *budget_levels* and *activity_types*
 - Helper functions: `parse_action()`, `is_valid_action()`, `build_dm_actions_prompt()`

# dst.py
The Dialogue State Tracker module checks if the state is in the normal booking phase or if it is in a control state which are:
 - CARRYOVER (OFFER_SLOT_CARRYOVER)
 - ASK CONFIRMATION (ASK_CONFIRMATION)
 - HANDLE DENIAL (HANDLE_DENIAL)
 - REQUEST MISSING SLOT (REQUEST_MISSING_SLOT)

If the last action were one of the previous mentioned this module returns a different system prompt designed so that the NLU can generate the correct output. Using this trick we maintain the NLU stateless, handling only user utterances and the DM only takes as input the intent and slot filled by the NLU and it doesn't need the user utterance to understand if we're dealing with a control interaction of the dialogue. 

# nlu.py
The NLU is stateless. It just fill a json containing the intent and the slots it finds. 
It doesn't handle dialogue states because it is handled by the DST.
It is composed by an LLM prompted with the system message which suggests the model what should it do and with an user message the user utterance. 

# dm.py
The DM is LLM-based since it's more flexible and it's easier to change the behaviour without manually re-design the logic of this block. Given the NLU output (intent + slots), the current dialogue state, and a set of action rules defined in schema.py, it generates an output action.

Key features:
 - Uses `build_dm_actions_prompt()` to generate action rules dynamically from schema
 - Follows a clear decision priority: OOD -> confirmation responses -> carryover -> missing slots -> confirmation
 - Outputs parameterized actions like `REQUEST_MISSING_SLOT(destination)`
 - Updates slots in the booking objects when NLU extracts them
 - Detects carryover opportunities when user switches between booking intents

The DM also updates the state internally:
 - Sets `current_intent` based on NLU output
 - Updates booking slots from NLU extracted values
 - Detects `pending_carryover` when switching intents
 - Tracks `last_action` for context-aware decisions

# data.py
Contains the dataclasses for storing booking information:
 - `FlightBooking`, `AccommodationBooking`, `ActivityBooking`
 - `TripContext` which holds all bookings and tracks completed intents
 - `CARRYOVER_SLOTS` mapping for shared slots between booking types
 - `get_carryover_slots()` to detect reusable values when switching intents

# main.py
This works as orchestrator: calls the components in order:
1. DST - generates context-aware system prompt for NLU
2. NLU - parses user utterance into intent + slots  
3. DM - decides next action and updates state
4. API - calls Amadeus APIs for booking completions
5. NLG - generates natural language response
6. State update - marks bookings as completed after COMPLETE_* actions

# nlg.py
Generates natural language responses based on DM actions. Uses:
 - `parse_action()` to handle parameterized actions like `REQUEST_MISSING_SLOT(slot_name)`
 - Different prompt builders for each action type
 - Access to dialogue state for personalized responses

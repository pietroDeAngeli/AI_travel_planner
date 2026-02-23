from llm import make_llm
from nlu import nlu_parse
from dm import DialogueState, dm_decide, state_context
from nlg import nlg_generate, GREETING_MESSAGE
from schema import parse_action
from amadeus import search_activities, search_accommodation
from intent_splitter import split_intents, IntentQueue
import argparse

import logging
logging.basicConfig(level=logging.DEBUG)

class Color:
    RESET = "\033[0m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"

def _is_flow_idle(state: DialogueState) -> bool:
    """
    Check if the current dialogue flow is idle (no active slot-filling in progress).
    Returns True when it's safe to inject a queued intent.
    """
    # No active intent — idle
    if state.current_intent is None:
        return True
    # Last action was a completion or result — flow just finished
    if state.last_action in (
        "COMPLETE_FLIGHT_BOOKING", "COMPLETE_ACCOMMODATION_BOOKING",
        "COMPLETE_ACTIVITY_BOOKING", "COMPARE_CITIES_RESULT",
        "GOODBYE", None,
    ):
        return True
    return False


def run(debug: bool = False, use_splitter: bool = True, use_llm_dm: bool = True):
    if debug:
        print(f"{Color.GREEN}------------------> DEBUG MODE ENABLED <------------------{Color.RESET}")
        print(f"{Color.GREEN}USE_SPLITTER={use_splitter}  USE_LLM_DM={use_llm_dm}{Color.RESET}")

    pipe = make_llm()
    if debug:
        if pipe is None:
            print(f"{Color.RED}Error: LLM pipeline could not be created.{Color.RESET}")
            print(f"{Color.RED}----------------------------------------------------------{Color.RESET}")

    state = DialogueState()
    intent_queue = IntentQueue()  # Queue for multi-intent handling
    if debug:
        print(f"{Color.GREEN}Initial Dialogue State:{Color.RESET}")
        print(state)
        print(f"{Color.GREEN}----------------------------------------------------------{Color.RESET}")
    history = []

    print(f"{Color.BLUE}{GREETING_MESSAGE}{Color.RESET}")

    while True:
        # --- Determine input for this turn ---
        # If there's a queued intent AND current booking flow is done, auto-process it.
        # Otherwise, always wait for user input.
        processing_queued = False

        if (use_splitter
            and intent_queue.has_pending()
            and _is_flow_idle(state)):
            # Auto-process next queued intent
            current_input = intent_queue.pop()
            processing_queued = True
            if debug:
                print(f"{Color.GREEN}[SPLIT] Auto-processing queued intent: {current_input}{Color.RESET}")
            # Show user what we're processing
            print(f"{Color.BLUE}BOT: Now let me help you with: \"{current_input}\"{Color.RESET}\n")
        else:
            user = input(f"{Color.YELLOW}YOU: {Color.RESET}").strip()
            if not user:
                continue

            # Intent splitting: only on fresh user input, not during active booking
            if use_splitter and _is_flow_idle(state):
                current_input, pending = split_intents(pipe, user)
                if pending:
                    intent_queue.add(pending)
                    if debug:
                        print(f"{Color.GREEN}[SPLIT] Detected {len(pending) + 1} intents{Color.RESET}")
                        print(f"{Color.GREEN}[SPLIT] Processing: {current_input}{Color.RESET}")
                        print(f"{Color.GREEN}[SPLIT] Queued: {pending}{Color.RESET}")
            else:
                # During active booking flow, user input goes directly to NLU
                # If user says something that cancels/overrides, clear the queue
                current_input = user

        # DST
        system_prompt = state_context(state)

        # NLU
        nlu_output = nlu_parse(pipe=pipe, user_utterance=current_input, system_prompt=system_prompt, dialogue_history=history)
        if debug:
            print(f"{Color.GREEN}NLU Output:{Color.RESET}")
            print(nlu_output)
            print(f"{Color.GREEN}----------------------------------------------------------{Color.RESET}")

        # DM
        action = dm_decide(state, nlu_output, current_input, llm_pipe=pipe if use_llm_dm else None)
        base_action, slot_param = parse_action(action)
        
        if debug:
            print(f"{Color.GREEN}DM Action:{Color.RESET}")
            print(f"  Action: {action}")
            print(f"{Color.GREEN}Dialogue State:{Color.RESET}")
            print(state)
            print(f"{Color.GREEN}----------------------------------------------------------{Color.RESET}")
        # API calls (only for completion actions)
        api_results = None
        
        if base_action == "COMPLETE_FLIGHT_BOOKING":
            flight = state.context.flight
            if debug:
                print(f"{Color.GREEN}[API] Would search flights: {flight.to_dict()}{Color.RESET}")
            pass
        
        elif base_action == "COMPLETE_ACCOMMODATION_BOOKING":
            accommodation = state.context.accommodation
            if accommodation.destination and accommodation.check_in_date and accommodation.check_out_date:
                adults = accommodation.num_guests or 1
                budget = accommodation.budget_level or "medium"
                
                ratings = {
                    "low": "1,2",
                    "medium": "3,4",
                    "high": "5",
                }.get(budget, "3,4")
                
                try:
                    api_results = search_accommodation(
                        city=accommodation.destination,
                        ratings=ratings,
                        num_adults=adults,
                        start_date=accommodation.check_in_date,
                        end_date=accommodation.check_out_date,
                    )
                except Exception as e:
                    if debug:
                        print(f"{Color.RED}[API] Accommodation search failed: {e}{Color.RESET}")
        
        elif base_action == "COMPLETE_ACTIVITY_BOOKING":
            activity = state.context.activity
            if activity.destination:
                try:
                    api_results = search_activities(
                        city=activity.destination,
                        activity_type=activity.activity_category or "cultural",
                    )
                except Exception as e:
                    if debug:
                        print(f"{Color.RED}[API] Activity search failed: {e}{Color.RESET}")
        
        elif base_action == "COMPARE_CITIES_RESULT":
            pass

        # Only pass the top accommodation option forward (NLG + post-processing).
        if base_action == "COMPLETE_ACCOMMODATION_BOOKING" and isinstance(api_results, list):
            api_results = api_results[:1]

        # NLG
        response = nlg_generate(pipe, action, state, dialogue_history=history)
        
        if api_results and base_action in ["COMPLETE_ACCOMMODATION_BOOKING", "COMPLETE_ACTIVITY_BOOKING"]:
            if isinstance(api_results, list) and len(api_results) > 0:
                max_items = 1 if base_action == "COMPLETE_ACCOMMODATION_BOOKING" else 3
                header = "Here is the top option I found:" if max_items == 1 else "Here are some options I found:"

                results_summary = f"\n\n{header}\n"
                for i, result in enumerate(api_results[:max_items], 1):
                    if isinstance(result, dict):
                        name = result.get("name", "Option")
                        price = result.get("price", "N/A")
                        results_summary += f"{i}. {name} - {price}\n"
                response += results_summary
        
        # Notify user about pending intents after completing a booking
        if intent_queue.has_pending() and base_action in (
            "COMPLETE_FLIGHT_BOOKING", "COMPLETE_ACCOMMODATION_BOOKING",
            "COMPLETE_ACTIVITY_BOOKING", "COMPARE_CITIES_RESULT",
        ):
            next_intent = intent_queue.peek()
            response += f"\n\nI also noted your earlier request: \"{next_intent}\". I'll handle that next!"

        # If user ends dialogue, clear the queue
        if base_action == "GOODBYE":
            intent_queue.clear()

        if debug:
            if api_results:
                print("API Results:", api_results[:2] if isinstance(api_results, list) else api_results)
            print("----------------------------------------------------------")

        # History
        if not processing_queued:
            history.append({"role": "user", "content": current_input})
        else:
            history.append({"role": "user", "content": f"[auto-queued] {current_input}"})
        history.append({"role": "assistant", "content": response})

        print(f"{Color.BLUE}BOT: {response}{Color.RESET}\n")
        
        if base_action == "GOODBYE":
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Travel Planner")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--no-splitter",
        action="store_true",
        help="Disable intent splitter (single intent per message)"
    )
    parser.add_argument(
        "--rule-based-dm",
        action="store_true",
        help="Use rule-based DM instead of LLM-assisted DM"
    )
    args = parser.parse_args()
    
    run(
        debug=args.debug,
        use_splitter=not args.no_splitter,
        use_llm_dm=not args.rule_based_dm,
    )

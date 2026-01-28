from llm import make_llm
from nlu import nlu_parse, state_context
from dm import DialogueState, dm_decide
from nlg import nlg_generate, GREETING_MESSAGE
from schema import parse_action
from amadeus import search_activities, search_accomodation
from intent_splitter import split_intents, has_multiple_intents, IntentQueue
import argparse

class Color:
    RESET = "\033[0m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"

def update_state_after_action(state: DialogueState, action: str, api_results=None):
    """Update state after an action (e.g., mark booking completed)."""
    base_action, _ = parse_action(action)
    
    if base_action in ["COMPLETE_FLIGHT_BOOKING", "COMPLETE_ACCOMMODATION_BOOKING", "COMPLETE_ACTIVITY_BOOKING"]:
        if state.current_intent:
            state.context.mark_completed(state.current_intent)
            if api_results:
                pass
    
    if base_action == "GOODBYE":
        pass


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
        user = input(f"{Color.YELLOW}YOU: {Color.RESET}").strip()
        if not user:
            continue

        # Intent splitting
        if use_splitter:
            if intent_queue.has_pending():
                current_input = intent_queue.pop()
                if debug:
                    print(f"{Color.GREEN}[SPLIT] Processing queued intent: {current_input}{Color.RESET}")
            else:
                if has_multiple_intents(pipe, user):
                    current_input, pending = split_intents(pipe, user)
                    if pending:
                        intent_queue.add(pending)
                        if debug:
                            print(f"{Color.GREEN}[SPLIT] Detected {len(pending) + 1} intents{Color.RESET}")
                            print(f"{Color.GREEN}[SPLIT] Processing: {current_input}{Color.RESET}")
                            print(f"{Color.GREEN}[SPLIT] Queued: {pending}{Color.RESET}")
                else:
                    current_input = user
        else:
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
                    api_results = search_accomodation(
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

        # State update
        update_state_after_action(state, action, api_results)

        # NLG
        response = nlg_generate(pipe, action, state)
        
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
        
        if intent_queue.has_pending():
            next_intent = intent_queue.peek()
            response += f"\n\n(I also noticed you mentioned: \"{next_intent}\" - I'll help with that next!)"

        if debug:
            if api_results:
                print("API Results:", api_results[:2] if isinstance(api_results, list) else api_results)
            print("----------------------------------------------------------")

        # History
        history.append({"role": "user", "content": user})
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

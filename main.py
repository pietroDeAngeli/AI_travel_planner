from llm import make_llm
from dst import state_context
from nlu import nlu_parse
from dm import DialogueState, dm_decide
from nlg import nlg_generate, GREETING_MESSAGE
from schema import parse_action
from amadeus import search_activities, search_accomodation
import argparse

class Color:
    RESET = "\033[0m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"

def update_state_after_action(state: DialogueState, action: str, api_results=None):
    """
    Update dialogue state after DM decision and API calls.
    Handles completion actions and marks bookings as done.
    """
    base_action, _ = parse_action(action)
    
    # Handle booking completions
    if base_action in ["COMPLETE_FLIGHT_BOOKING", "COMPLETE_ACCOMMODATION_BOOKING", "COMPLETE_ACTIVITY_BOOKING"]:
        # Mark current booking as completed
        if state.current_intent:
            state.context.mark_completed(state.current_intent)
            if api_results:
                # Could store API results in state if needed
                pass
    
    # Handle GOODBYE - could clean up state
    if base_action == "GOODBYE":
        pass  # State will be discarded anyway


def run(debug: bool = False):
    if debug:
        print("------------------> DEBUG MODE ENABLED <------------------")

    pipe = make_llm()
    if debug:
        if pipe is None:
            print("Error: LLM pipeline could not be created.")
            print("----------------------------------------------------------")

    state = DialogueState()
    if debug:
        print("Initial Dialogue State:")
        print(state)
        print("----------------------------------------------------------")
    history = []

    print(f"{Color.GREEN}{GREETING_MESSAGE}{Color.RESET}")

    while True:
        user = input("YOU: ").strip()
        if not user:
            continue

        # =========================
        # 1) DST - Generate context-aware prompt
        # =========================
        system_prompt = state_context(state)

        # =========================
        # 2) NLU - Parse intent and slots
        # =========================
        nlu_output = nlu_parse(pipe, user, system_prompt, dialogue_history=history)
        if debug:
            print("NLU Output:")
            print(nlu_output)
            print("----------------------------------------------------------")

        # =========================
        # 3) DM - Decide action (also updates state internally)
        # =========================
        action = dm_decide(state, nlu_output, user)
        base_action, slot_param = parse_action(action)
        
        if debug:
            print("DM Action:")
            print(f"  Action: {action}")
            print("Dialogue State:")
            print(state)
            print("----------------------------------------------------------")

        # =========================
        # 4) API CALLS (if needed)
        # =========================
        api_results = None
        
        if base_action == "COMPLETE_FLIGHT_BOOKING":
            # Flight API calls would go here
            flight = state.context.flight
            if debug:
                print(f"[API] Would search flights: {flight.to_dict()}")
            # api_results = search_flights(...)
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
                        print(f"[API] Accommodation search failed: {e}")
        
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
                        print(f"[API] Activity search failed: {e}")
        
        elif base_action == "COMPARE_CITIES_RESULT":
            # Would call compare_options from amadeus.py
            pass

        # =========================
        # 5) Update state after action
        # =========================
        update_state_after_action(state, action, api_results)

        # =========================
        # 6) NLG - Generate response
        # =========================
        response = nlg_generate(pipe, action, state)
        
        # Append API results summary if available
        if api_results and base_action in ["COMPLETE_ACCOMMODATION_BOOKING", "COMPLETE_ACTIVITY_BOOKING"]:
            if isinstance(api_results, list) and len(api_results) > 0:
                results_summary = f"\n\nHere are some options I found:\n"
                for i, result in enumerate(api_results[:3], 1):
                    if isinstance(result, dict):
                        name = result.get("name", "Option")
                        price = result.get("price", "N/A")
                        results_summary += f"{i}. {name} - {price}\n"
                response += results_summary
        
        if debug:
            if api_results:
                print("API Results:", api_results[:2] if isinstance(api_results, list) else api_results)
            print("----------------------------------------------------------")

        # =========================
        # 7) Update history
        # =========================
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": response})

        print(f"{Color.GREEN}BOT: {response}{Color.RESET}\n")
        
        # Exit on goodbye
        if base_action == "GOODBYE":
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Travel Planner")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    args = parser.parse_args()
    
    run(debug=args.debug)

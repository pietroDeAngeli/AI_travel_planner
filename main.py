from llm import make_llm
from nlu import nlu_parse
from dm import DialogueState, update_info, dm_decide
from nlg import nlg_respond
from amadeus import search_activities, search_accomodation

def run_demo():
    pipe = make_llm()
    state = DialogueState()
    history = []

    print("Travel Planner (type 'stop' to exit)\n")

    while True:
        user = input("YOU: ").strip()
        if not user:
            continue

        # hard exit
        if user.lower() in ("stop", "exit", "quit"):
            break

        # NLU
        nlu = nlu_parse(pipe, user, dialogue_history=history)
        intent, slots = nlu["intent"], nlu["slots"]

        # Detect which slot user wants to change (from user utterance)
        # This is a simple heuristic - in a real system, NLU would extract this
        if intent == "CHANGE_DETAILS" or (intent == "PROVIDE_CHANGE_VALUE" and state.slot_to_change is None):
            # Try to detect slot name from user utterance
            slot_keywords = {
                "destination": ["rome", "paris", "destination", "city", "where", "go to"],
                "start_date": ["start", "departure", "begin", "when"],
                "end_date": ["end", "return", "back"],
                "num_people": ["people", "persons", "travelers", "guests", "how many"],
                "overall_budget": ["budget", "money", "cost", "spend"],
                "travel_style": ["style", "adventure", "relax", "culture", "beach"],
                "travel_method": ["flight", "train", "car", "bus", "drive", "fly", "transport"],
                "accommodation_type": ["hotel", "hostel", "airbnb", "accommodation", "stay"],
                "budget_level": ["budget", "level", "low", "medium", "high", "expensive"],
                "leaving_time_preference": ["time", "morning", "afternoon", "evening", "leaving"]
            }
            
            user_lower = user.lower()
            for slot_name, keywords in slot_keywords.items():
                if any(kw in user_lower for kw in keywords):
                    state.slot_to_change = slot_name
                    break

        # Update state
        update_info(state, intent, slots)

        # DM decision
        action = dm_decide(intent, state)

        if action == "GOODBYE":
            state = DialogueState()
        elif action == "PLAN_ACTIVITIES":
            if state.info.destination:
                activities = search_activities(destination=state.info.destination)
                state.plan = {"activities": activities if activities else []}
            else:
                state.plan = {"activities": []}

        elif action == "PLAN_TRAVEL_METHOD":
            # Plan travel method - TODO: integrate flight/train search
            state.plan = {"travel_method": "pending"}
            
        elif action == "PLAN_ACCOMMODATION":
            if state.info.destination and state.info.start_date and state.info.end_date:
                checkIn = state.info.start_date
                checkout = state.info.end_date
                adults = state.info.num_people or 1
                budget = state.info.budget_level
                ratings = "1,2,3,4,5"

                if budget == "low":
                    ratings = "1,2"
                elif budget == "medium":
                    ratings = "3,4"
                else:
                    ratings = "5"

                accomodations = search_accomodation(city=state.info.destination, ratings=ratings, num_adults=adults, start_date=checkIn, end_date=checkout)
                state.plan = {"accommodations": accomodations if accomodations else []}
            else:
                state.plan = {"accommodations": []}

        # NLG
        response = nlg_respond(pipe, action, {
            "info": state.info,
            "current_intent": state.current_intent,
            "plan": state.plan,
            "slot_to_change": state.slot_to_change
        }, user)

        # update history
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": response})

        print(f"BOT: {response}\n")

if __name__ == "__main__":
    run_demo()

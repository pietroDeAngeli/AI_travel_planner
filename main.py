from llm import make_llm
from nlu import nlu_parse
from dm import DialogueState, dm_decide
from nlg import nlg_respond
from amadeus import search_activities, search_accomodation


def run_demo():
    pipe = make_llm()
    state = DialogueState()
    history = []

    print("🌍Welcome to your Travel Planning Assistant!")
    print("=" * 70)
    print("\nI can help you with:")
    print("  🗺️  Plan Your Trip")
    print("      Tell me where you want to go, when, and with how many people.")
    print("      I'll organize your trip based on your budget (low/medium/high),")
    print("      accommodation preferences (hotel/hostel/apartment/bnb), and")
    print("      travel style (adventure, cultural, food, sport, relax, nature,")
    print("      nightlife, or family-friendly).")
    print("\n  ℹ️  Get Travel Information")
    print("      Ask about hotels, flights, or activities in any destination.")
    print("\n  ⚖️  Compare Destinations")
    print("      Compare two cities based on travel styles.")
    print("\nJust tell me where you'd like to go or what you're interested in!")
    print("=" * 70)
    print()

    while True:
        user = input("YOU: ").strip()
        if not user:
            continue

        # =========================
        # 1) NLU
        # =========================
        nlu_output = nlu_parse(pipe, user, dialogue_history=history)

        # =========================
        # 2) DM (LLM-based)
        # =========================
        action = dm_decide(pipe, state, nlu_output, user)

        # =========================
        # 3) ORCHESTRATOR LOGIC
        # =========================
        payload = {
            "info": state.info,
            "current_intent": state.current_intent,
            "task_intent": state.task_intent,
            "plan": None,
        }

        # ---- Tool calling (Amadeus) ----
        if action in {"PROPOSE_TRIP_PLAN", "PROVIDE_INFORMATION"}:
            activities = []
            accommodations = []

            # Activities
            if state.info.destination:
                activities = search_activities(
                    destination=state.info.destination
                )

            # Accommodations
            if (
                state.info.destination
                and state.info.start_date
                and state.info.end_date
            ):
                adults = state.info.num_people or 1
                budget = state.info.budget_level or "medium"

                ratings = {
                    "low": "1,2",
                    "medium": "3,4",
                    "high": "5",
                }.get(budget, "3,4")

                accommodations = search_accomodation(
                    city=state.info.destination,
                    ratings=ratings,
                    num_adults=adults,
                    start_date=state.info.start_date,
                    end_date=state.info.end_date,
                )

            state.plan = {
                "activities": activities,
                "accommodations": accommodations,
            }

            payload["plan"] = state.plan

        # ---- Missing slot handling ----
        if action == "REQUEST_MISSING_SLOT":
            missing = state.info.missing_slots(state.task_intent)
            payload["missing_slot"] = missing[0] if missing else None

        # ---- Confirmation ----
        if action == "ASK_CONFIRMATION":
            payload["summary"] = state.info.to_dict(state.task_intent)

        # ---- End dialogue ----
        if action == "GOODBYE":
            state = DialogueState()

        # =========================
        # 4) NLG
        # =========================
        response = nlg_respond(pipe, action, payload, user)

        # =========================
        # 5) Update history
        # =========================
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": response})

        print(f"BOT: {response}\n")


if __name__ == "__main__":
    run_demo()

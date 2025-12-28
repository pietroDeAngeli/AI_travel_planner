from llm import make_llm
from nlu import nlu_parse
from dm import DialogueState, update_state_with_slots, dm_decide
from planner import simple_plan_from_slots
from nlg import nlg_respond

def run_demo():
    pipe = make_llm()
    state = DialogueState()
    history = []

    print("Travel Planner (scrivi 'stop' per uscire)\n")

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

        # Update state
        update_state_with_slots(state, slots)

        # DM decision
        action = dm_decide(intent, state)

        # If we need to show plan and plan missing, generate it
        if action == "SHOW_PLAN" and state.plan is None:
            state.plan = simple_plan_from_slots(state.slots)

        # NLG
        response = nlg_respond(pipe, action, {
            "slots": state.slots,
            "plan": state.plan,
            "missing_slots": state.missing_slots
        }, user)

        # update history
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": response})

        print(f"BOT: {response}\n")

if __name__ == "__main__":
    run_demo()

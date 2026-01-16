from llm import make_llm
from nlu import nlu_parse


ABLATION_TESTS = [
    # --- A) RULE OVERRIDES / HEURISTICS ---
    {
        "name": "greeting-only",
        "history": [],
        "user": "hey",
        "expect_intent": "GREETING",
        "expect_slots": {},
        "purpose": "Check rule override for greetings"
    },
    {
        "name": "end-dialogue",
        "history": [],
        "user": "thanks, bye!",
        "expect_intent": "END_DIALOGUE",
        "expect_slots": {},
        "purpose": "Check rule override for ending"
    },

    # --- B) START_TRIP vs PROVIDE_CONSTRAINTS (same info, different framing) ---
    {
        "name": "start-trip-goal",
        "history": [],
        "user": "I want to plan a trip to Rome",
        "expect_intent": "START_TRIP",
        "expect_slots": {"destination": "Rome"},
        "purpose": "Goal statement should trigger START_TRIP"
    },
    {
        "name": "constraints-without-goal",
        "history": [],
        "user": "Rome, 3 days, medium budget",
        "expect_intent": "PROVIDE_CONSTRAINTS",
        "expect_slots": {"destination": "Rome", "budget_level": "medium"},
        "purpose": "Bare constraints without explicit planning phrase"
    },

    # --- C) ACCOMMODATION_PREFERENCE ---
    {
        "name": "accommodation-type",
        "history": [],
        "user": "I prefer a hostel",
        "expect_intent": "ACCOMMODATION_PREFERENCE",
        "expect_slots": {"accommodation_type": "hostel"},
        "purpose": "Accommodation preference extraction"
    },

    # --- D) CHANGE_DETAILS vs PROVIDE_CHANGE_VALUE (context dependent) ---
    {
        "name": "change-details-no-value",
        "history": [{"role": "assistant", "content": "What would you like to change? (destination, dates, budget)"}],
        "user": "Change the destination",
        "expect_intent": "CHANGE_DETAILS",
        "expect_slots": {},
        "purpose": "Change request without new value"
    },
    {
        "name": "provide-change-value-after-ask",
        "history": [{"role": "assistant", "content": "Ok, what should the new destination be?"}],
        "user": "Florence",
        "expect_intent": "PROVIDE_CHANGE_VALUE",
        "expect_slots": {},  # value may be handled by DM; if you expect destination here, set it.
        "purpose": "Value-only answer should become PROVIDE_CHANGE_VALUE"
    },
    {
        "name": "change-details-with-value",
        "history": [],
        "user": "Not Rome, make it Florence instead",
        "expect_intent": "CHANGE_DETAILS",
        "expect_slots": {"destination": "Florence"},
        "purpose": "Change with explicit new value (no system prompt needed)"
    },

    # --- E) CONFIRM_DETAILS (context dependent) ---
    {
        "name": "confirm-after-confirm-question",
        "history": [{"role": "assistant", "content": "So: Rome, 3 days, medium budget. Is that correct?"}],
        "user": "yes, correct",
        "expect_intent": "CONFIRM_DETAILS",
        "expect_slots": {},
        "purpose": "Confirmation should be CONFIRM_DETAILS only after confirm-question"
    },
    {
        "name": "yes-without-context",
        "history": [],
        "user": "yes",
        "expect_intent": "FALLBACK",
        "expect_slots": {},
        "purpose": "Without context, bare yes should NOT auto-confirm"
    },

    # --- F) REQUEST_PLAN ---
    {
        "name": "request-plan",
        "history": [{"role": "assistant", "content": "Great. I have your preferences."}],
        "user": "Can you show me the itinerary?",
        "expect_intent": "REQUEST_PLAN",
        "expect_slots": {},
        "purpose": "Explicit plan request"
    },

    # --- G) Schema adherence (ablation target: slots should not leak across intents) ---
    {
        "name": "no-slots-on-greeting",
        "history": [],
        "user": "hello, I like hotels",
        "expect_intent": "GREETING",
        "expect_slots": {},  # if rule override forces greeting, slots must stay empty
        "purpose": "If greeting override triggers, ensure slots cleaned to schema"
    },
]

def run_ablation_tests():
    pipe = make_llm()
    passed = 0

    for t in ABLATION_TESTS:
        nlu = nlu_parse(pipe, t["user"], dialogue_history=t["history"])
        intent_ok = (nlu.get("intent") == t["expect_intent"])

        # slots: controlla che almeno le chiavi attese matchino
        got_slots = nlu.get("slots") or {}
        exp_slots = t["expect_slots"] or {}
        slots_ok = all(got_slots.get(k) == v for k, v in exp_slots.items())

        ok = intent_ok and slots_ok
        passed += int(ok)

        print(f"[{'PASS' if ok else 'FAIL'}] {t['name']} — {t['purpose']}")
        print("  user:", t["user"])
        print("  expected:", t["expect_intent"], exp_slots)
        print("  got     :", nlu.get("intent"), got_slots)
        print()

    print(f"Summary: {passed}/{len(ABLATION_TESTS)} passed")


if __name__ == "__main__":
    run_ablation_tests()

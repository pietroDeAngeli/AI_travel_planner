import re
import json
from typing import Any, Dict, List, Tuple
from collections import defaultdict

from llm import make_llm
from nlu import nlu_parse
from schema import INTENT_SLOTS, INTENTS


# -------------------------
# Helpers for robust checks
# -------------------------

NOT_NONE = "__NOT_NONE__"
ANY = "__ANY__"

def _match_value(expected: Any, got: Any) -> bool:
    """
    expected can be:
      - exact value (str/int)
      - list/tuple/set of acceptable values
      - {"re": "<regex>"} to match strings
      - "__NOT_NONE__" (value must be not None)
      - "__ANY__" (always ok)
    """
    if expected == ANY:
        return True

    if expected == NOT_NONE:
        return got is not None

    if isinstance(expected, (list, tuple, set)):
        return got in expected

    if isinstance(expected, dict) and "re" in expected:
        if got is None:
            return False
        return re.search(expected["re"], str(got)) is not None

    # numeric tolerance: allow "4" vs 4
    if isinstance(expected, int):
        if isinstance(got, int):
            return got == expected
        if isinstance(got, str) and got.strip().isdigit():
            return int(got.strip()) == expected
        return False

    return got == expected


def _check_schema_keys(expected_intent: str, got_slots: Dict[str, Any]) -> Tuple[bool, str]:
    expected_keys = list(INTENT_SLOTS.get(expected_intent, []))
    got_keys = list(got_slots.keys())

    if set(got_keys) != set(expected_keys):
        return False, f"Slot keys mismatch. expected={expected_keys}, got={got_keys}"

    return True, ""


def _check_expected_slots(exp_slots: Dict[str, Any], got_slots: Dict[str, Any]) -> Tuple[bool, str]:
    for k, exp in exp_slots.items():
        if k not in got_slots:
            return False, f"Missing expected slot '{k}' in output"
        if not _match_value(exp, got_slots.get(k)):
            return False, f"Slot '{k}' mismatch. expected={exp}, got={got_slots.get(k)}"
    return True, ""


def _call_nlu(pipe, user: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Compatible with both nlu_parse signatures (with/without current_intent kw).
    """
    try:
        return nlu_parse(pipe, user, dialogue_history=history, current_intent=None)
    except TypeError:
        return nlu_parse(pipe, user, dialogue_history=history)


# -------------------------
# Test cases aligned with schema.py
# -------------------------

TEST_DIALOGUES = [
    # ========== GREETING ==========
    {
        "name": "01_greeting_simple",
        "history": [],
        "user": "hello",
        "expect_intent": "GREETING",
        "expect_slots": {},
        "purpose": "Simple greeting"
    },
    {
        "name": "02_greeting_hey",
        "history": [{"role": "assistant", "content": "How can I help?"}],
        "user": "hey there",
        "expect_intent": "GREETING",
        "expect_slots": {},
        "purpose": "Informal greeting"
    },

    # ========== PLAN_TRIP ==========
    {
        "name": "03_plan_trip_with_destination",
        "history": [],
        "user": "I want to plan a trip to Rome",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {"destination": ["Rome", "Roma"]},
        "purpose": "PLAN_TRIP with destination"
    },
    {
        "name": "04_plan_trip_full_details",
        "history": [{"role": "assistant", "content": "Tell me your trip details"}],
        "user": "Trip to Paris from 2026-03-10 to 2026-03-15 for 2 people",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": ["Paris"],
            "start_date": {"re": r"2026.*03.*10|03.*10.*2026"},
            "end_date": {"re": r"2026.*03.*15|03.*15.*2026"},
            "num_people": 2
        },
        "purpose": "PLAN_TRIP with multiple slots"
    },
    {
        "name": "05_plan_trip_destination_only",
        "history": [{"role": "assistant", "content": "Where would you like to go?"}],
        "user": "Barcelona",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {"destination": ["Barcelona"]},
        "purpose": "Destination-only response with context"
    },
    {
        "name": "06_plan_trip_with_budget_style",
        "history": [],
        "user": "I want to go to Milan, medium budget, culture style",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": ["Milan", "Milano"],
            "budget_level": ["medium"],
            "travel_style": NOT_NONE
        },
        "purpose": "PLAN_TRIP with budget and style"
    },
    {
        "name": "07_plan_trip_accommodation",
        "history": [{"role": "assistant", "content": "What are your preferences?"}],
        "user": "I'd like a hotel in Rome",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": ["Rome", "Roma"],
            "accommodation_type": ["hotel"]
        },
        "purpose": "PLAN_TRIP with accommodation preference"
    },
    {
        "name": "08_plan_trip_dates_people",
        "history": [{"role": "assistant", "content": "When and how many people?"}],
        "user": "From May 1st to May 5th, 2026, we are 3 people",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "start_date": NOT_NONE,
            "end_date": NOT_NONE,
            "num_people": 3
        },
        "purpose": "PLAN_TRIP with dates and num_people"
    },

    # ========== COMPARE_OPTIONS ==========
    {
        "name": "09_compare_two_cities",
        "history": [],
        "user": "Can you compare Rome and Milan?",
        "expect_intent": "COMPARE_OPTIONS",
        "expect_slots": {
            "city1": ["Rome", "Roma"],
            "city2": ["Milan", "Milano"]
        },
        "purpose": "Compare two cities"
    },
    {
        "name": "10_compare_with_criteria",
        "history": [],
        "user": "Compare Paris and London by price",
        "expect_intent": "COMPARE_OPTIONS",
        "expect_slots": {
            "city1": ["Paris"],
            "city2": ["London"],
            "criteria": ["price"]
        },
        "purpose": "Compare with explicit criteria"
    },
    {
        "name": "11_compare_activities",
        "history": [{"role": "assistant", "content": "What would you like to know?"}],
        "user": "Which has more activities, Barcelona or Madrid?",
        "expect_intent": "COMPARE_OPTIONS",
        "expect_slots": {
            "city1": ["Barcelona"],
            "city2": ["Madrid"],
            "criteria": NOT_NONE,
            "compare_type": NOT_NONE
        },
        "purpose": "Compare by activities criterion"
    },

    # ========== REQUEST_INFORMATION ==========
    {
        "name": "12_request_info_about_destination",
        "history": [],
        "user": "Tell me about Rome",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {
            "destination": ["Rome", "Roma"]
        },
        "purpose": "Request general information"
    },
    {
        "name": "13_request_info_attractions",
        "history": [],
        "user": "What attractions are there in Paris?",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {
            "destination": ["Paris"],
            "info_type": NOT_NONE
        },
        "purpose": "Request specific info type"
    },
    {
        "name": "14_request_info_museums",
        "history": [{"role": "assistant", "content": "What information do you need?"}],
        "user": "Show me museums in Florence",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {
            "destination": ["Florence", "Firenze"],
            "entity_type": NOT_NONE,
            "info_type": NOT_NONE
        },
        "purpose": "Request museums information"
    },
    {
        "name": "15_request_info_events",
        "history": [],
        "user": "Are there any events in Berlin?",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {
            "destination": ["Berlin"],
            "info_type": ["events", "event"]
        },
        "purpose": "Request events information"
    },

    # ========== CONFIRM_DETAILS ==========
    {
        "name": "16_confirm_yes",
        "history": [
            {"role": "assistant", "content": "Rome, May 1-5, 2 people. Is that correct?"}
        ],
        "user": "yes",
        "expect_intent": "CONFIRM_DETAILS",
        "expect_slots": {},
        "purpose": "Confirmation with 'yes'"
    },
    {
        "name": "17_confirm_correct",
        "history": [
            {"role": "assistant", "content": "Do you confirm these details?"}
        ],
        "user": "that's correct",
        "expect_intent": "CONFIRM_DETAILS",
        "expect_slots": {},
        "purpose": "Confirmation with 'correct'"
    },
    {
        "name": "18_confirm_no",
        "history": [
            {"role": "assistant", "content": "Is this okay?"}
        ],
        "user": "no",
        "expect_intent": "CONFIRM_DETAILS",
        "expect_slots": {},
        "purpose": "Rejection in confirmation context"
    },

    # ========== END_DIALOGUE ==========
    {
        "name": "19_end_goodbye",
        "history": [
            {"role": "assistant", "content": "Anything else?"}
        ],
        "user": "bye",
        "expect_intent": "END_DIALOGUE",
        "expect_slots": {},
        "purpose": "End with 'bye'"
    },
    {
        "name": "20_end_quit",
        "history": [],
        "user": "quit",
        "expect_intent": "END_DIALOGUE",
        "expect_slots": {},
        "purpose": "End with 'quit'"
    },
    {
        "name": "21_end_thanks_bye",
        "history": [{"role": "assistant", "content": "Is there anything else?"}],
        "user": "thanks, goodbye",
        "expect_intent": "END_DIALOGUE",
        "expect_slots": {},
        "purpose": "End with thanks and goodbye"
    },

    # ========== OOD ==========
    {
        "name": "22_fallback_off_topic",
        "history": [],
        "user": "What's the weather like?",
        "expect_intent": "OOD",
        "expect_slots": {},
        "purpose": "Off-topic question"
    },
    {
        "name": "23_fallback_unclear",
        "history": [{"role": "assistant", "content": "How can I help?"}],
        "user": "maybe",
        "expect_intent": "OOD",
        "expect_slots": {},
        "purpose": "Unclear single word"
    },
    {
        "name": "24_fallback_random",
        "history": [],
        "user": "I like pizza",
        "expect_intent": "OOD",
        "expect_slots": {},
        "purpose": "Random non-task statement"
    },

    # ========== Multi-turn scenarios ==========
    {
        "name": "25_plan_trip_incremental_destination",
        "history": [
            {"role": "user", "content": "I want to plan a trip"},
            {"role": "assistant", "content": "Great! Where would you like to go?"}
        ],
        "user": "Rome",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {"destination": ["Rome", "Roma"]},
        "purpose": "Incremental PLAN_TRIP: destination"
    },
    {
        "name": "26_plan_trip_incremental_dates",
        "history": [
            {"role": "user", "content": "Rome"},
            {"role": "assistant", "content": "When would you like to go?"}
        ],
        "user": "From March 10 to March 15, 2026",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "start_date": NOT_NONE,
            "end_date": NOT_NONE
        },
        "purpose": "Incremental PLAN_TRIP: dates"
    },
    {
        "name": "27_compare_incremental_criteria",
        "history": [
            {"role": "user", "content": "Compare Rome and Milan"},
            {"role": "assistant", "content": "What criteria would you like to use?"}
        ],
        "user": "price",
        "expect_intent": "COMPARE_OPTIONS",
        "expect_slots": {"criteria": ["price"]},
        "purpose": "Incremental COMPARE_OPTIONS: criteria"
    },
    {
        "name": "28_request_info_incremental_type",
        "history": [
            {"role": "user", "content": "Tell me about Paris"},
            {"role": "assistant", "content": "What type of information do you need?"}
        ],
        "user": "museums",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {
            "info_type": ["museums", "museum"]
        },
        "purpose": "Incremental REQUEST_INFORMATION: info_type"
    },  

    # ========== Edge cases ==========
    {
        "name": "29_yes_without_confirm_context",
        "history": [
            {"role": "assistant", "content": "What would you like to do?"}
        ],
        "user": "yes",
        "expect_intent": "OOD",
        "expect_slots": {},
        "purpose": "Yes without confirmation question"
    },
    {
        "name": "30_plan_trip_italian",
        "history": [],
        "user": "Vorrei organizzare un viaggio a Roma",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {"destination": ["Rome", "Roma"]},
        "purpose": "Italian language PLAN_TRIP"
    },

    # ========== Overinformative users ==========
    {
        "name": "31_overinformative_plan_trip_all",
        "history": [],
        "user": "I want to plan a trip to Paris from March 1st to March 10th 2026 for 2 people with a budget of 3000 euros, staying in a hotel with breakfast included",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": ["Paris"],
            "start_date": NOT_NONE,
            "end_date": NOT_NONE,
            "num_people": 2,
            "budget": NOT_NONE,
            "accommodation_type": ["hotel"]
        },
        "purpose": "Overinformative: all PLAN_TRIP details in one utterance"
    },
    {
        "name": "32_overinformative_compare_detailed",
        "history": [],
        "user": "I need to compare Rome and Florence based on price, activities, and cultural attractions because I'm planning a trip in April and I have a limited budget of 2000 euros",
        "expect_intent": "COMPARE_OPTIONS",
        "expect_slots": {
            "city1": ["Rome", "Roma"],
            "city2": ["Florence", "Firenze"],
            "criteria": NOT_NONE
        },
        "purpose": "Overinformative: compare with extra context and multiple criteria"
    },
    {
        "name": "33_overinformative_request_info_verbose",
        "history": [],
        "user": "Could you please tell me everything about Barcelona including museums, restaurants, historical sites, nightlife, transportation, and local events happening in summer?",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {
            "destination": ["Barcelona"],
            "info_type": NOT_NONE
        },
        "purpose": "Overinformative: request with many details"
    },
    {
        "name": "34_overinformative_book_activity_specific",
        "history": [],
        "user": "I want to book a guided museum tour in Rome on March 15th at 10 AM for 4 adults and 2 children under 12 years old with English speaking guide and audio equipment",
        "expect_intent": "BOOK_ACTIVITY",
        "expect_slots": {
            "activity_type": NOT_NONE,
            "destination": ["Rome", "Roma"],
            "date": NOT_NONE,
            "num_people": NOT_NONE
        },
        "purpose": "Overinformative: book with many specific details"
    },

    # ========== Underinformative users ==========
    {
        "name": "35_underinformative_plan_trip_minimal",
        "history": [],
        "user": "plan a trip",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {},
        "purpose": "Underinformative: PLAN_TRIP without any details"
    },
    {
        "name": "36_underinformative_compare_vague",
        "history": [],
        "user": "compare them",
        "expect_intent": "COMPARE_OPTIONS",
        "expect_slots": {},
        "purpose": "Underinformative: compare without specifying what"
    },
    {
        "name": "37_underinformative_request_info_minimal",
        "history": [{"role": "assistant", "content": "What would you like to know?"}],
        "user": "tell me about it",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {},
        "purpose": "Underinformative: request without destination"
    },
    {
        "name": "38_underinformative_book_activity_vague",
        "history": [],
        "user": "book something",
        "expect_intent": "BOOK_ACTIVITY",
        "expect_slots": {},
        "purpose": "Underinformative: book without activity type or destination"
    },
    {
        "name": "39_underinformative_single_word",
        "history": [
            {"role": "user", "content": "I want to plan a trip"},
            {"role": "assistant", "content": "Where would you like to go?"}
        ],
        "user": "Paris",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {"destination": ["Paris"]},
        "purpose": "Underinformative: single word response to fill slot"
    },
    {
        "name": "40_underinformative_implicit_confirm",
        "history": [
            {"role": "assistant", "content": "Would you like to book a hotel in Rome?"}
        ],
        "user": "sure",
        "expect_intent": "CONFIRM",
        "expect_slots": {},
        "purpose": "Underinformative: minimal confirmation"
    },
]


# -------------------------
# Statistics tracking
# -------------------------

class TestStatistics:
    def __init__(self):
        self.results = []
        self.intent_confusion = defaultdict(lambda: defaultdict(int))
        
    def add_result(self, test_name: str, expected_intent: str, got_intent: str, 
                   intent_ok: bool, schema_ok: bool, slots_ok: bool, mode: str):
        self.results.append({
            "test": test_name,
            "expected_intent": expected_intent,
            "got_intent": got_intent,
            "intent_ok": intent_ok,
            "schema_ok": schema_ok,
            "slots_ok": slots_ok,
            "mode": mode,
            "passed": intent_ok and schema_ok and slots_ok
        })
        self.intent_confusion[expected_intent][got_intent] += 1
    
    def get_per_intent_stats(self, mode: str) -> Dict:
        """Calculate precision, recall, F1 per intent for a given mode"""
        mode_results = [r for r in self.results if r["mode"] == mode]
        
        stats = {}
        for intent in INTENTS:
            tp = sum(1 for r in mode_results if r["expected_intent"] == intent and r["got_intent"] == intent)
            fp = sum(1 for r in mode_results if r["expected_intent"] != intent and r["got_intent"] == intent)
            fn = sum(1 for r in mode_results if r["expected_intent"] == intent and r["got_intent"] != intent)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            stats[intent] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": tp + fn
            }
        
        return stats
    
    def get_overall_accuracy(self, mode: str) -> float:
        mode_results = [r for r in self.results if r["mode"] == mode]
        if not mode_results:
            return 0.0
        passed = sum(1 for r in mode_results if r["passed"])
        return passed / len(mode_results)
    
    def print_report(self, mode: str):
        print(f"\n{'='*90}")
        print(f"DETAILED REPORT FOR MODE: {mode}")
        print(f"{'='*90}")
        
        # Overall accuracy
        accuracy = self.get_overall_accuracy(mode)
        print(f"\nOverall Accuracy: {accuracy:.2%}")
        
        # Per-intent statistics
        stats = self.get_per_intent_stats(mode)
        print(f"\nPer-Intent Statistics:")
        print(f"{'Intent':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 70)
        
        for intent in sorted(stats.keys()):
            s = stats[intent]
            if s["support"] > 0:
                print(f"{intent:<25} {s['precision']:>10.2%} {s['recall']:>10.2%} {s['f1']:>10.2%} {s['support']:>10}")
        
        # Confusion matrix
        print(f"\nConfusion Matrix (Expected → Got):")
        print(f"{'Expected':<20}", end="")
        for intent in sorted(INTENTS):
            print(f"{intent[:8]:>10}", end="")
        print()
        print("-" * 100)
        
        for expected in sorted(INTENTS):
            if expected in self.intent_confusion:
                print(f"{expected:<20}", end="")
                for got in sorted(INTENTS):
                    count = self.intent_confusion[expected].get(got, 0)
                    print(f"{count:>10}", end="")
                print()


# -------------------------
# Runner: history ablation
# -------------------------

ABLATION_MODES = [
    ("full", None),     # use full history
    ("last2", 2),       # keep only last 2 turns
    ("last6", 6),       # keep last 6 turns
    ("none", 0),        # no history
]


def _slice_history(history: List[Dict[str, str]], k: int | None) -> List[Dict[str, str]]:
    if k is None:
        return history
    if k <= 0:
        return []
    return history[-k:]


def run_ablation_tests():
    pipe = make_llm()
    
    stats_per_mode = {mode: TestStatistics() for mode, _ in ABLATION_MODES}
    summary = {mode: {"pass": 0, "total": 0} for mode, _ in ABLATION_MODES}

    for t in TEST_DIALOGUES:
        print("=" * 90)
        print(f"TEST: {t['name']} — {t['purpose']}")
        print(f"USER: {t['user']}")
        print(f"EXPECT: intent={t['expect_intent']} slots={t['expect_slots']}")
        print("-" * 90)

        for mode, k in ABLATION_MODES:
            hist = _slice_history(t["history"], k)
            nlu = _call_nlu(pipe, t["user"], hist)

            got_intent = nlu.get("intent")
            got_slots = nlu.get("slots") or {}

            intent_ok = (got_intent == t["expect_intent"])
            schema_ok, schema_msg = _check_schema_keys(t["expect_intent"], got_slots) if intent_ok else (False, "Skipped (intent mismatch)")
            slots_ok, slots_msg = _check_expected_slots(t["expect_slots"], got_slots) if intent_ok else (False, "Skipped (intent mismatch)")

            ok = intent_ok and schema_ok and slots_ok

            summary[mode]["total"] += 1
            summary[mode]["pass"] += int(ok)
            
            stats_per_mode[mode].add_result(
                t["name"], t["expect_intent"], got_intent,
                intent_ok, schema_ok, slots_ok, mode
            )

            print(f"[{mode:5}] {'✓ PASS' if ok else '✗ FAIL'}")
            print(f"  got intent: {got_intent}")
            print(f"  got slots : {got_slots}")
            if not intent_ok:
                print(f"  reason   : intent mismatch (expected {t['expect_intent']})")
            elif not schema_ok:
                print(f"  reason   : {schema_msg}")
            elif not slots_ok:
                print(f"  reason   : {slots_msg}")
            print()

    # Print summary
    print("\n" + "=" * 90)
    print("SUMMARY (by history ablation mode)")
    print("=" * 90)
    for mode in summary:
        p = summary[mode]["pass"]
        tot = summary[mode]["total"]
        pct = (p / tot * 100) if tot > 0 else 0
        print(f"- {mode:5}: {p:3d}/{tot:3d} passed ({pct:5.1f}%)")
    
    # Print detailed reports for each mode
    for mode, _ in ABLATION_MODES:
        stats_per_mode[mode].print_report(mode)
    
    # Save results to JSON
    results_file = "nlu_test_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "summary": summary,
            "detailed_results": [r for stats in stats_per_mode.values() for r in stats.results]
        }, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    run_ablation_tests()

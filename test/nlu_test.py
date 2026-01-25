import re
import json
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    return nlu_parse(pipe, user, dialogue_history=history)


# -------------------------
# Test cases aligned with schema.py
# -------------------------

TEST_DIALOGUES = [
    # ==========================================================================
    # BOOK_FLIGHT TESTS
    # ==========================================================================
    
    # ---- Under-informative ----
    {
        "name": "01_flight_minimal",
        "history": [],
        "user": "I need a flight",
        "expect_intent": "BOOK_FLIGHT",
        "expect_slots": {},
        "purpose": "Under-informative: flight with no details"
    },
    {
        "name": "02_flight_destination_only",
        "history": [],
        "user": "I want to fly to Rome",
        "expect_intent": "BOOK_FLIGHT",
        "expect_slots": {"destination": "Rome"},
        "purpose": "Under-informative: only destination"
    },
    
    # ---- Normal ----
    {
        "name": "03_flight_origin_dest",
        "history": [],
        "user": "I need a flight from Milan to Paris",
        "expect_intent": "BOOK_FLIGHT",
        "expect_slots": {
            "origin": "Milan",
            "destination": "Paris"
        },
        "purpose": "Normal: origin and destination"
    },
    {
        "name": "04_flight_with_dates",
        "history": [],
        "user": "Book a flight to Barcelona on March 15th returning March 20th",
        "expect_intent": "BOOK_FLIGHT",
        "expect_slots": {
            "destination": "Barcelona",
            "departure_date": NOT_NONE,
            "return_date": NOT_NONE
        },
        "purpose": "Normal: destination with dates"
    },
    
    # ---- Over-informative ----
    {
        "name": "05_flight_overinformative",
        "history": [],
        "user": "I need to book a flight from London Heathrow to Madrid Barajas for 3 passengers on April 10th 2026, returning April 17th, we have a medium budget and prefer morning flights with no layovers",
        "expect_intent": "BOOK_FLIGHT",
        "expect_slots": {
            "origin": NOT_NONE,
            "destination": NOT_NONE,
            "departure_date": NOT_NONE,
            "return_date": NOT_NONE,
            "num_passengers": 3,
            "budget_level": "medium"
        },
        "purpose": "Over-informative: all slots plus extra details"
    },
    
    # ---- Multi-turn / Mixed Initiative ----
    {
        "name": "06_flight_multiturn_origin",
        "history": [
            {"role": "user", "content": "I want to book a flight to Vienna"},
            {"role": "assistant", "content": "Where will you be departing from?"}
        ],
        "user": "from Berlin",
        "expect_intent": "BOOK_FLIGHT",
        "expect_slots": {"origin": "Berlin"},
        "purpose": "Multi-turn: providing origin after prompt"
    },
    {
        "name": "07_flight_multiturn_passengers",
        "history": [
            {"role": "user", "content": "Flight from Rome to Amsterdam on May 5th"},
            {"role": "assistant", "content": "How many passengers?"}
        ],
        "user": "4 people",
        "expect_intent": "BOOK_FLIGHT",
        "expect_slots": {"num_passengers": 4},
        "purpose": "Multi-turn: providing passengers count"
    },

    # ==========================================================================
    # BOOK_ACCOMMODATION TESTS
    # ==========================================================================
    
    # ---- Under-informative ----
    {
        "name": "08_accommodation_minimal",
        "history": [],
        "user": "I need a hotel",
        "expect_intent": "BOOK_ACCOMMODATION",
        "expect_slots": {},
        "purpose": "Under-informative: hotel with no details"
    },
    
    # ---- Normal ----
    {
        "name": "09_accommodation_dest_dates",
        "history": [],
        "user": "Find me a hotel in Prague from June 10 to June 15",
        "expect_intent": "BOOK_ACCOMMODATION",
        "expect_slots": {
            "destination": "Prague",
            "check_in_date": NOT_NONE,
            "check_out_date": NOT_NONE
        },
        "purpose": "Normal: destination with dates"
    },
    {
        "name": "10_accommodation_hostel",
        "history": [],
        "user": "I need a place to stay in Amsterdam for 2 guests",
        "expect_intent": "BOOK_ACCOMMODATION",
        "expect_slots": {
            "destination": "Amsterdam",
            "num_guests": 2
        },
        "purpose": "Normal: destination with guests"
    },
    
    # ---- Over-informative ----
    {
        "name": "11_accommodation_overinformative",
        "history": [],
        "user": "I'm looking for a luxury hotel in Paris near the Eiffel Tower, checking in on July 1st and checking out on July 7th 2026, for 2 guests, high budget, preferably with a pool and free breakfast",
        "expect_intent": "BOOK_ACCOMMODATION",
        "expect_slots": {
            "destination": "Paris",
            "check_in_date": NOT_NONE,
            "check_out_date": NOT_NONE,
            "num_guests": 2,
            "budget_level": "high"
        },
        "purpose": "Over-informative: all slots plus amenities"
    },

    # ==========================================================================
    # BOOK_ACTIVITY TESTS
    # ==========================================================================
    
    # ---- Under-informative ----
    {
        "name": "12_activity_minimal",
        "history": [],
        "user": "I want to do something fun",
        "expect_intent": "BOOK_ACTIVITY",
        "expect_slots": {},
        "purpose": "Under-informative: activity with no details"
    },
    
    # ---- Normal ----
    {
        "name": "13_activity_destination_category",
        "history": [],
        "user": "I want to go hiking in the Swiss Alps",
        "expect_intent": "BOOK_ACTIVITY",
        "expect_slots": {
            "destination": NOT_NONE,
            "activity_category": "adventure"
        },
        "purpose": "Normal: destination with activity category"
    },
    {
        "name": "14_activity_museum",
        "history": [],
        "user": "Book a museum tour in Florence",
        "expect_intent": "BOOK_ACTIVITY",
        "expect_slots": {
            "destination": "Florence",
            "activity_category": "cultural"
        },
        "purpose": "Normal: cultural activity"
    },
    
    # ---- Multi-turn ----
    {
        "name": "15_activity_multiturn",
        "history": [
            {"role": "user", "content": "I want to book an activity in Rome"},
            {"role": "assistant", "content": "What type of activity are you interested in?"}
        ],
        "user": "food and wine tasting",
        "expect_intent": "BOOK_ACTIVITY",
        "expect_slots": {"activity_category": "food"},
        "purpose": "Multi-turn: providing activity type"
    },

    # ==========================================================================
    # COMPARE_CITIES TESTS
    # ==========================================================================
    
    {
        "name": "16_compare_minimal",
        "history": [],
        "user": "Compare cities",
        "expect_intent": "COMPARE_CITIES",
        "expect_slots": {},
        "purpose": "Under-informative: compare with no cities"
    },
    {
        "name": "17_compare_two_cities",
        "history": [],
        "user": "Compare Paris and London for sightseeing",
        "expect_intent": "COMPARE_CITIES",
        "expect_slots": {
            "city1": "Paris",
            "city2": "London",
            "activity_category": NOT_NONE
        },
        "purpose": "Normal: two cities with category"
    },
    {
        "name": "18_compare_question_form",
        "history": [],
        "user": "Which is better for food, Rome or Barcelona?",
        "expect_intent": "COMPARE_CITIES",
        "expect_slots": {
            "city1": ("Rome", "Barcelona"),
            "city2": ("Rome", "Barcelona"),
            "activity_category": "food"
        },
        "purpose": "Normal: comparison as question"
    },

    # ==========================================================================
    # GOODBYE TESTS
    # ==========================================================================
    
    {
        "name": "19_goodbye_simple",
        "history": [],
        "user": "goodbye",
        "expect_intent": "GOODBYE",
        "expect_slots": {},
        "purpose": "End: simple goodbye"
    },
    {
        "name": "20_goodbye_thanks",
        "history": [
            {"role": "assistant", "content": "Your flight is booked!"}
        ],
        "user": "Thanks, that's all I needed",
        "expect_intent": "GOODBYE",
        "expect_slots": {},
        "purpose": "End: thanks and closure"
    },

    # ==========================================================================
    # OOD (Out of Domain) - FALLBACK POLICY TESTS
    # ==========================================================================
    
    {
        "name": "21_ood_weather",
        "history": [],
        "user": "What's the weather like in Paris?",
        "expect_intent": "OOD",
        "expect_slots": {},
        "purpose": "OOD/Fallback: weather question"
    },
    {
        "name": "22_ood_random",
        "history": [],
        "user": "Tell me a joke",
        "expect_intent": "OOD",
        "expect_slots": {},
        "purpose": "OOD/Fallback: unrelated request"
    },
    {
        "name": "23_ood_unclear",
        "history": [],
        "user": "maybe something",
        "expect_intent": "OOD",
        "expect_slots": {},
        "purpose": "OOD/Fallback: vague unclear input"
    },

    # ==========================================================================
    # MIXED INITIATIVE - User provides info unprompted
    # ==========================================================================
    
    {
        "name": "24_mixed_initiative_all_at_once",
        "history": [],
        "user": "I want to fly from New York to Tokyo on December 1st for 2 passengers with a high budget",
        "expect_intent": "BOOK_FLIGHT",
        "expect_slots": {
            "origin": NOT_NONE,
            "destination": "Tokyo",
            "departure_date": NOT_NONE,
            "num_passengers": 2,
            "budget_level": "high"
        },
        "purpose": "Mixed initiative: user provides all info unprompted"
    },
    {
        "name": "25_mixed_initiative_switch_intent",
        "history": [
            {"role": "user", "content": "I want to book a flight to Madrid"},
            {"role": "assistant", "content": "Where are you departing from?"}
        ],
        "user": "Actually, I also need a hotel there from March 5 to March 10",
        "expect_intent": "BOOK_ACCOMMODATION",
        "expect_slots": {
            "destination": "Madrid",
            "check_in_date": NOT_NONE,
            "check_out_date": NOT_NONE
        },
        "purpose": "Mixed initiative: user switches to new intent"
    },

    # ==========================================================================
    # NOISE & ROBUSTNESS
    # ==========================================================================
    
    {
        "name": "26_noise_typos",
        "history": [],
        "user": "I wnat to book a flihgt to Barselona",
        "expect_intent": "BOOK_FLIGHT",
        "expect_slots": {
            "destination": NOT_NONE
        },
        "purpose": "Robustness: spelling errors"
    },
    {
        "name": "27_noise_filler_words",
        "history": [],
        "user": "um so like I kind of want to maybe find a hotel in uh Vienna you know",
        "expect_intent": "BOOK_ACCOMMODATION",
        "expect_slots": {
            "destination": "Vienna"
        },
        "purpose": "Robustness: filler words"
    },
    {
        "name": "28_noise_informal",
        "history": [],
        "user": "yo I need to bounce to Berlin next week, hook me up with some flights",
        "expect_intent": "BOOK_FLIGHT",
        "expect_slots": {
            "destination": "Berlin"
        },
        "purpose": "Robustness: informal/slang language"
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

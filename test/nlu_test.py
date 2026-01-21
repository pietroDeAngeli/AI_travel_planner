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
    return nlu_parse(pipe, user, dialogue_history=history)


# -------------------------
# Test cases aligned with schema.py
# -------------------------

TEST_DIALOGUES = [
    # ========== GREETING ==========
    {
        "name": "01_greeting_hello",
        "history": [],
        "user": "hello",
        "expect_intent": "GREETING",
        "expect_slots": {},
        "purpose": "Simple greeting"
    },
    {
        "name": "02_greeting_hi",
        "history": [],
        "user": "hi there",
        "expect_intent": "GREETING",
        "expect_slots": {},
        "purpose": "Casual greeting"
    },
    {
        "name": "03_greeting_good_morning",
        "history": [],
        "user": "good morning",
        "expect_intent": "GREETING",
        "expect_slots": {},
        "purpose": "Polite greeting"
    },

    # ========== PLAN_TRIP - Underinformative ==========
    {
        "name": "04_plan_trip_minimal",
        "history": [],
        "user": "I want to plan a trip",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {},
        "purpose": "Underinformative: no details"
    },
    {
        "name": "05_plan_trip_destination_only",
        "history": [],
        "user": "I want to go to Paris",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {"destination": "Paris"},
        "purpose": "Underinformative: only destination"
    },

    # ========== PLAN_TRIP - Normal ==========
    {
        "name": "06_plan_trip_dest_dates",
        "history": [],
        "user": "I want to visit Rome from March 10 to March 15",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Rome",
            "start_date": NOT_NONE,
            "end_date": NOT_NONE
        },
        "purpose": "Normal: destination and dates"
    },
    {
        "name": "07_plan_trip_dest_people",
        "history": [],
        "user": "Trip to Barcelona for 3 people",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Barcelona",
            "num_people": 3
        },
        "purpose": "Normal: destination and people"
    },

    # ========== PLAN_TRIP - Overinformative ==========
    {
        "name": "08_plan_trip_overinformative_all",
        "history": [],
        "user": "I want to plan a trip to London from May 1st to May 10th 2026 for 4 people, we prefer hotels, our budget is medium, and we like cultural activities",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "London",
            "start_date": NOT_NONE,
            "end_date": NOT_NONE,
            "num_people": 4,
            "accommodation_type": "hotel",
            "budget_level": "medium",
            "travel_style": NOT_NONE
        },
        "purpose": "Overinformative: all slots at once"
    },
    {
        "name": "09_plan_trip_overinformative_verbose",
        "history": [],
        "user": "I'm planning a vacation to Berlin next month from June 5 to June 12 with my family of 2 adults and want to stay in an apartment because we need a kitchen",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Berlin",
            "start_date": NOT_NONE,
            "end_date": NOT_NONE,
            "num_people": 2,
            "accommodation_type": "apartment"
        },
        "purpose": "Overinformative: extra context"
    },

    # ========== PLAN_TRIP - Multi-turn ==========
    {
        "name": "10_plan_trip_followup_dest",
        "history": [
            {"role": "user", "content": "I want to plan a trip"},
            {"role": "assistant", "content": "Great! Where would you like to go?"}
        ],
        "user": "Milan",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {"destination": "Milan"},
        "purpose": "Multi-turn: destination after prompt"
    },
    {
        "name": "11_plan_trip_followup_dates",
        "history": [
            {"role": "user", "content": "I want to go to Paris"},
            {"role": "assistant", "content": "When would you like to travel?"}
        ],
        "user": "from April 20 to April 25",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "start_date": NOT_NONE,
            "end_date": NOT_NONE
        },
        "purpose": "Multi-turn: dates after prompt"
    },
    {
        "name": "12_plan_trip_followup_people",
        "history": [
            {"role": "user", "content": "Trip to Rome in May"},
            {"role": "assistant", "content": "How many people will be traveling?"}
        ],
        "user": "just 2 of us",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {"num_people": 2},
        "purpose": "Multi-turn: people after prompt"
    },

    # ========== COMPARE_OPTIONS - Underinformative ==========
    {
        "name": "13_compare_minimal",
        "history": [],
        "user": "compare them",
        "expect_intent": "COMPARE_OPTIONS",
        "expect_slots": {},
        "purpose": "Underinformative: no options"
    },

    # ========== COMPARE_OPTIONS - Normal ==========
    {
        "name": "14_compare_two_cities",
        "history": [],
        "user": "Compare Paris and London",
        "expect_intent": "COMPARE_OPTIONS",
        "expect_slots": {
            "option1": "Paris",
            "option2": "London"
        },
        "purpose": "Normal: two cities"
    },
    {
        "name": "15_compare_with_criteria",
        "history": [],
        "user": "Compare Rome and Florence by price",
        "expect_intent": "COMPARE_OPTIONS",
        "expect_slots": {
            "option1": "Rome",
            "option2": "Florence",
            "criteria": "price"
        },
        "purpose": "Normal: with criteria"
    },

    # ========== COMPARE_OPTIONS - Overinformative ==========
    {
        "name": "16_compare_overinformative",
        "history": [],
        "user": "I need to compare Madrid and Barcelona based on price, activities, and cultural attractions because I'm planning a trip in summer and want to decide which destination offers better value for money",
        "expect_intent": "COMPARE_OPTIONS",
        "expect_slots": {
            "option1": "Madrid",
            "option2": "Barcelona",
            "criteria": NOT_NONE
        },
        "purpose": "Overinformative: detailed comparison"
    },

    # ========== COMPARE_OPTIONS - Multi-turn ==========
    {
        "name": "17_compare_followup_criteria",
        "history": [
            {"role": "user", "content": "Compare Berlin and Munich"},
            {"role": "assistant", "content": "What criteria would you like to use?"}
        ],
        "user": "activities",
        "expect_intent": "COMPARE_OPTIONS",
        "expect_slots": {"criteria": "activities"},
        "purpose": "Multi-turn: criteria after prompt"
    },

    # ========== REQUEST_INFORMATION - Underinformative ==========
    {
        "name": "18_request_info_minimal",
        "history": [],
        "user": "tell me about it",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {},
        "purpose": "Underinformative: no destination"
    },

    # ========== REQUEST_INFORMATION - Normal ==========
    {
        "name": "19_request_info_destination",
        "history": [],
        "user": "Tell me about Vienna",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {"destination": "Vienna"},
        "purpose": "Normal: general information"
    },
    {
        "name": "20_request_info_with_entity",
        "history": [],
        "user": "What hotels are available in Prague?",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {
            "destination": "Prague",
            "entity_type": "hotels"
        },
        "purpose": "Normal: specific entity type"
    },
    {
        "name": "21_request_info_activities",
        "history": [],
        "user": "Show me activities in Amsterdam",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {
            "destination": "Amsterdam",
            "entity_type": "activities"
        },
        "purpose": "Normal: activities"
    },

    # ========== REQUEST_INFORMATION - Overinformative ==========
    {
        "name": "22_request_info_overinformative",
        "history": [],
        "user": "Could you please tell me everything about Copenhagen including hotels, flights, activities, museums, restaurants, and local events because I'm planning a comprehensive 10-day trip there in summer",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {
            "destination": "Copenhagen",
            "entity_type": NOT_NONE
        },
        "purpose": "Overinformative: detailed request"
    },

    # ========== REQUEST_INFORMATION - Multi-turn ==========
    {
        "name": "23_request_info_followup_entity",
        "history": [
            {"role": "user", "content": "Tell me about Athens"},
            {"role": "assistant", "content": "What type of information do you need?"}
        ],
        "user": "hotels",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {"entity_type": "hotels"},
        "purpose": "Multi-turn: entity after prompt"
    },

    # ========== CONFIRM_DETAILS ==========
    {
        "name": "24_confirm_yes",
        "history": [
            {"role": "assistant", "content": "Paris, May 1-5, 2 people. Is that correct?"}
        ],
        "user": "yes",
        "expect_intent": "CONFIRM_DETAILS",
        "expect_slots": {},
        "purpose": "Confirmation: yes"
    },
    {
        "name": "25_confirm_correct",
        "history": [
            {"role": "assistant", "content": "Do you confirm these details?"}
        ],
        "user": "that's correct",
        "expect_intent": "CONFIRM_DETAILS",
        "expect_slots": {},
        "purpose": "Confirmation: correct"
    },
    {
        "name": "26_confirm_no",
        "history": [
            {"role": "assistant", "content": "Is this okay?"}
        ],
        "user": "no",
        "expect_intent": "CONFIRM_DETAILS",
        "expect_slots": {},
        "purpose": "Confirmation: rejection"
    },
    {
        "name": "27_confirm_needs_change",
        "history": [
            {"role": "assistant", "content": "Are these details correct?"}
        ],
        "user": "not quite, I need to change something",
        "expect_intent": "CONFIRM_DETAILS",
        "expect_slots": {},
        "purpose": "Confirmation: rejection with explanation"
    },

    # ========== END_DIALOGUE ==========
    {
        "name": "28_end_goodbye",
        "history": [],
        "user": "goodbye",
        "expect_intent": "END_DIALOGUE",
        "expect_slots": {},
        "purpose": "End: goodbye"
    },
    {
        "name": "29_end_bye",
        "history": [],
        "user": "bye",
        "expect_intent": "END_DIALOGUE",
        "expect_slots": {},
        "purpose": "End: bye"
    },
    {
        "name": "30_end_thanks",
        "history": [{"role": "assistant", "content": "Is there anything else?"}],
        "user": "no thanks, that's all",
        "expect_intent": "END_DIALOGUE",
        "expect_slots": {},
        "purpose": "End: thanks and closure"
    },

    # ========== OOD (Out of Domain) ==========
    {
        "name": "31_ood_weather",
        "history": [],
        "user": "What's the weather like today?",
        "expect_intent": "OOD",
        "expect_slots": {},
        "purpose": "OOD: weather question"
    },
    {
        "name": "32_ood_unclear",
        "history": [],
        "user": "maybe",
        "expect_intent": "OOD",
        "expect_slots": {},
        "purpose": "OOD: unclear word"
    },
    {
        "name": "33_ood_random",
        "history": [],
        "user": "I like pizza",
        "expect_intent": "OOD",
        "expect_slots": {},
        "purpose": "OOD: random statement"
    },
    {
        "name": "34_ood_yes_no_context",
        "history": [
            {"role": "assistant", "content": "What would you like to do?"}
        ],
        "user": "yes",
        "expect_intent": "OOD",
        "expect_slots": {},
        "purpose": "OOD: yes without confirmation"
    },

    # ========== Additional Edge Cases ==========
    {
        "name": "35_plan_trip_change_destination",
        "history": [
            {"role": "user", "content": "I want to go to Paris"},
            {"role": "assistant", "content": "Great! When would you like to go?"}
        ],
        "user": "actually, let's go to Rome instead",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {"destination": "Rome"},
        "purpose": "Edge: changing slot value"
    },
    {
        "name": "36_compare_question_form",
        "history": [],
        "user": "Which is better, Venice or Florence?",
        "expect_intent": "COMPARE_OPTIONS",
        "expect_slots": {
            "option1": "Venice",
            "option2": "Florence"
        },
        "purpose": "Edge: comparison as question"
    },
    {
        "name": "37_request_info_events",
        "history": [],
        "user": "Are there any events in Dublin?",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {
            "destination": "Dublin",
            "entity_type": "events"
        },
        "purpose": "Edge: question about events"
    },
    {
        "name": "38_plan_trip_budget_accommodation",
        "history": [],
        "user": "I need a low budget hostel trip to Lisbon",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Lisbon",
            "budget_level": "low",
            "accommodation_type": "hostel"
        },
        "purpose": "Edge: budget and accommodation"
    },
    {
        "name": "39_plan_trip_travel_style",
        "history": [],
        "user": "I want a relaxing beach vacation in Nice",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Nice",
            "travel_style": NOT_NONE
        },
        "purpose": "Edge: travel style"
    },
    {
        "name": "40_compare_with_type",
        "history": [],
        "user": "Compare hotels in Brussels and Bruges",
        "expect_intent": "COMPARE_OPTIONS",
        "expect_slots": {
            "option1": "Brussels",
            "option2": "Bruges",
            "compare_type": NOT_NONE
        },
        "purpose": "Edge: comparing specific type"
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

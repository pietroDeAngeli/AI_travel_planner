"""
Dialogue Manager Test Suite

Tests the LLM-based DM against the rule-based DM (ground truth).
Calculates action classification F1 score.
"""

import json
import copy
from typing import Any, Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm import make_llm
from dm import DialogueState, dm_decide, dm_decide_rule_based
from schema import DM_ACTIONS, is_valid_action, parse_action


# --- Test case definition ---

@dataclass
class DMTestCase:
    """A single test case for the Dialogue Manager."""
    name: str
    purpose: str
    # Initial state configuration
    current_intent: str = None
    last_action: str = None
    confirmed: bool = False
    # Filled slots for the current booking
    filled_slots: Dict[str, Any] = field(default_factory=dict)
    # Compare cities data (if applicable)
    compare_cities_data: Dict[str, Any] = field(default_factory=dict)
    # Pending carryover
    pending_carryover: Dict[str, Any] = None
    awaiting_carryover_response: bool = False
    # NLU output for this turn
    nlu_output: Dict[str, Any] = field(default_factory=dict)
    # User utterance
    user_utterance: str = ""


def setup_state(tc: DMTestCase) -> DialogueState:
    """Create a DialogueState from a test case configuration."""
    state = DialogueState()
    state.current_intent = tc.current_intent
    state.last_action = tc.last_action
    state.confirmed = tc.confirmed
    state.compare_cities_data = tc.compare_cities_data.copy()
    state.pending_carryover = tc.pending_carryover
    state.awaiting_carryover_response = tc.awaiting_carryover_response
    
    # Set up booking slots based on current intent
    if tc.current_intent and tc.filled_slots:
        booking = state.get_current_booking()
        if booking:
            for slot, value in tc.filled_slots.items():
                if hasattr(booking, slot):
                    setattr(booking, slot, value)
    
    return state


# --- Test cases ---

TEST_CASES: List[DMTestCase] = [
    # ==========================================================================
    # RULE 1: END_DIALOGUE -> GOODBYE
    # ==========================================================================
    DMTestCase(
        name="01_end_dialogue_simple",
        purpose="END_DIALOGUE intent should return GOODBYE",
        nlu_output={"intent": "END_DIALOGUE", "slots": {}},
        user_utterance="goodbye"
    ),
    DMTestCase(
        name="02_end_dialogue_mid_booking",
        purpose="END_DIALOGUE during booking should still return GOODBYE",
        current_intent="BOOK_FLIGHT",
        filled_slots={"origin": "Rome", "destination": "Paris"},
        nlu_output={"intent": "END_DIALOGUE", "slots": {}},
        user_utterance="thanks, bye"
    ),

    # ==========================================================================
    # RULE 2: OOD -> ASK_CLARIFICATION
    # ==========================================================================
    DMTestCase(
        name="03_ood_simple",
        purpose="OOD intent should return ASK_CLARIFICATION",
        nlu_output={"intent": "OOD", "slots": {}},
        user_utterance="what's the weather?"
    ),
    DMTestCase(
        name="04_ood_none_intent",
        purpose="None intent should return ASK_CLARIFICATION",
        nlu_output={"intent": None, "slots": {}},
        user_utterance="hmm maybe"
    ),
    DMTestCase(
        name="05_ood_unknown_intent",
        purpose="Unknown intent should return ASK_CLARIFICATION",
        nlu_output={"intent": "UNKNOWN_INTENT", "slots": {}},
        user_utterance="do something weird"
    ),

    # ==========================================================================
    # RULE 3: COMPARE_CITIES slot collection
    # ==========================================================================
    DMTestCase(
        name="06_compare_cities_no_slots",
        purpose="COMPARE_CITIES with no slots should request city1",
        nlu_output={"intent": "COMPARE_CITIES", "slots": {}},
        user_utterance="compare cities"
    ),
    DMTestCase(
        name="07_compare_cities_city1_only",
        purpose="COMPARE_CITIES with city1 should request city2",
        compare_cities_data={"city1": "Paris"},
        nlu_output={"intent": "COMPARE_CITIES", "slots": {}},
        user_utterance="Paris"
    ),
    DMTestCase(
        name="08_compare_cities_both_cities",
        purpose="COMPARE_CITIES with both cities should request activity_category",
        compare_cities_data={"city1": "Paris", "city2": "London"},
        nlu_output={"intent": "COMPARE_CITIES", "slots": {}},
        user_utterance="London"
    ),
    DMTestCase(
        name="09_compare_cities_all_slots",
        purpose="COMPARE_CITIES with all slots should return COMPARE_CITIES_RESULT",
        compare_cities_data={"city1": "Paris", "city2": "London", "activity_category": "cultural"},
        nlu_output={"intent": "COMPARE_CITIES", "slots": {}},
        user_utterance="cultural"
    ),
    DMTestCase(
        name="10_compare_cities_all_at_once",
        purpose="COMPARE_CITIES with all slots in NLU should return COMPARE_CITIES_RESULT",
        nlu_output={
            "intent": "COMPARE_CITIES",
            "slots": {"city1": "Rome", "city2": "Barcelona", "activity_category": "food"}
        },
        user_utterance="compare Rome and Barcelona for food"
    ),

    # ==========================================================================
    # RULE 4: Confirmation handling
    # ==========================================================================
    DMTestCase(
        name="11_confirmation_positive",
        purpose="Positive confirmation should complete booking",
        current_intent="BOOK_FLIGHT",
        last_action="ASK_CONFIRMATION",
        filled_slots={
            "origin": "Rome",
            "destination": "Paris",
            "departure_date": "2026-03-15",
            "num_passengers": 2,
            "budget_level": "medium"
        },
        nlu_output={"intent": "BOOK_FLIGHT", "slots": {"confirmation": "yes"}},
        user_utterance="yes, confirm"
    ),
    DMTestCase(
        name="12_confirmation_negative",
        purpose="Negative confirmation should request slot change",
        current_intent="BOOK_FLIGHT",
        last_action="ASK_CONFIRMATION",
        filled_slots={
            "origin": "Rome",
            "destination": "Paris",
            "departure_date": "2026-03-15",
            "num_passengers": 2,
            "budget_level": "medium"
        },
        nlu_output={"intent": "BOOK_FLIGHT", "slots": {"confirmation": "no"}},
        user_utterance="no, I want to change something"
    ),
    DMTestCase(
        name="13_confirmation_accommodation",
        purpose="Positive confirmation for accommodation should complete",
        current_intent="BOOK_ACCOMMODATION",
        last_action="ASK_CONFIRMATION",
        filled_slots={
            "destination": "Paris",
            "check_in_date": "2026-03-15",
            "check_out_date": "2026-03-20",
            "num_guests": 2,
            "budget_level": "high"
        },
        nlu_output={"intent": "BOOK_ACCOMMODATION", "slots": {"confirmation": "yes"}},
        user_utterance="yes please"
    ),
    DMTestCase(
        name="14_confirmation_activity",
        purpose="Positive confirmation for activity should complete",
        current_intent="BOOK_ACTIVITY",
        last_action="ASK_CONFIRMATION",
        filled_slots={
            "destination": "Florence",
            "activity_category": "cultural",
            "budget_level": "medium"
        },
        nlu_output={"intent": "BOOK_ACTIVITY", "slots": {"confirmation": "yes"}},
        user_utterance="confirm it"
    ),

    # ==========================================================================
    # RULE 5: Slot change handling
    # ==========================================================================
    DMTestCase(
        name="15_slot_change_with_name",
        purpose="Slot change with slot_name should request that slot",
        current_intent="BOOK_FLIGHT",
        last_action="REQUEST_SLOT_CHANGE",
        filled_slots={
            "origin": "Rome",
            "destination": "Paris",
            "departure_date": "2026-03-15",
            "num_passengers": 2,
            "budget_level": "medium"
        },
        nlu_output={"intent": "BOOK_FLIGHT", "slots": {"slot_name": "destination"}},
        user_utterance="I want to change the destination"
    ),
    DMTestCase(
        name="16_slot_change_no_name",
        purpose="Slot change without slot_name should ask again",
        current_intent="BOOK_FLIGHT",
        last_action="REQUEST_SLOT_CHANGE",
        filled_slots={
            "origin": "Rome",
            "destination": "Paris",
            "departure_date": "2026-03-15",
            "num_passengers": 2,
            "budget_level": "medium"
        },
        nlu_output={"intent": "BOOK_FLIGHT", "slots": {}},
        user_utterance="something"
    ),

    # ==========================================================================
    # RULE 6: Missing slots - BOOK_FLIGHT
    # ==========================================================================
    DMTestCase(
        name="17_flight_missing_all",
        purpose="Flight with no slots should request origin",
        nlu_output={"intent": "BOOK_FLIGHT", "slots": {}},
        user_utterance="I want to book a flight"
    ),
    DMTestCase(
        name="18_flight_missing_destination",
        purpose="Flight with origin should request destination",
        current_intent="BOOK_FLIGHT",
        filled_slots={"origin": "Rome"},
        nlu_output={"intent": "BOOK_FLIGHT", "slots": {}},
        user_utterance="from Rome"
    ),
    DMTestCase(
        name="19_flight_missing_date",
        purpose="Flight with origin+dest should request departure_date",
        current_intent="BOOK_FLIGHT",
        filled_slots={"origin": "Rome", "destination": "Paris"},
        nlu_output={"intent": "BOOK_FLIGHT", "slots": {}},
        user_utterance="to Paris"
    ),
    DMTestCase(
        name="20_flight_missing_passengers",
        purpose="Flight missing passengers should request num_passengers",
        current_intent="BOOK_FLIGHT",
        filled_slots={
            "origin": "Rome",
            "destination": "Paris",
            "departure_date": "2026-03-15"
        },
        nlu_output={"intent": "BOOK_FLIGHT", "slots": {}},
        user_utterance="March 15"
    ),
    DMTestCase(
        name="21_flight_missing_budget",
        purpose="Flight missing budget should request budget_level",
        current_intent="BOOK_FLIGHT",
        filled_slots={
            "origin": "Rome",
            "destination": "Paris",
            "departure_date": "2026-03-15",
            "num_passengers": 2
        },
        nlu_output={"intent": "BOOK_FLIGHT", "slots": {}},
        user_utterance="2 passengers"
    ),
    DMTestCase(
        name="22_flight_all_slots_filled",
        purpose="Flight with all slots should ask confirmation",
        current_intent="BOOK_FLIGHT",
        filled_slots={
            "origin": "Rome",
            "destination": "Paris",
            "departure_date": "2026-03-15",
            "num_passengers": 2,
            "budget_level": "medium"
        },
        nlu_output={"intent": "BOOK_FLIGHT", "slots": {}},
        user_utterance="medium budget"
    ),

    # ==========================================================================
    # RULE 7: Missing slots - BOOK_ACCOMMODATION
    # ==========================================================================
    DMTestCase(
        name="23_accommodation_missing_all",
        purpose="Accommodation with no slots should request destination",
        nlu_output={"intent": "BOOK_ACCOMMODATION", "slots": {}},
        user_utterance="I need a hotel"
    ),
    DMTestCase(
        name="24_accommodation_missing_dates",
        purpose="Accommodation with destination should request check_in_date",
        current_intent="BOOK_ACCOMMODATION",
        filled_slots={"destination": "Paris"},
        nlu_output={"intent": "BOOK_ACCOMMODATION", "slots": {}},
        user_utterance="in Paris"
    ),
    DMTestCase(
        name="25_accommodation_all_filled",
        purpose="Accommodation with all slots should ask confirmation",
        current_intent="BOOK_ACCOMMODATION",
        filled_slots={
            "destination": "Paris",
            "check_in_date": "2026-03-15",
            "check_out_date": "2026-03-20",
            "num_guests": 2,
            "budget_level": "high"
        },
        nlu_output={"intent": "BOOK_ACCOMMODATION", "slots": {}},
        user_utterance="high budget"
    ),

    # ==========================================================================
    # RULE 8: Missing slots - BOOK_ACTIVITY
    # ==========================================================================
    DMTestCase(
        name="26_activity_missing_all",
        purpose="Activity with no slots should request destination",
        nlu_output={"intent": "BOOK_ACTIVITY", "slots": {}},
        user_utterance="I want to do something fun"
    ),
    DMTestCase(
        name="27_activity_missing_category",
        purpose="Activity with destination should request activity_category",
        current_intent="BOOK_ACTIVITY",
        filled_slots={"destination": "Florence"},
        nlu_output={"intent": "BOOK_ACTIVITY", "slots": {}},
        user_utterance="in Florence"
    ),
    DMTestCase(
        name="28_activity_all_filled",
        purpose="Activity with all slots should ask confirmation",
        current_intent="BOOK_ACTIVITY",
        filled_slots={
            "destination": "Florence",
            "activity_category": "cultural",
            "budget_level": "medium"
        },
        nlu_output={"intent": "BOOK_ACTIVITY", "slots": {}},
        user_utterance="medium"
    ),

    # ==========================================================================
    # RULE 9: Carryover handling
    # ==========================================================================
    DMTestCase(
        name="29_carryover_offer",
        purpose="Pending carryover should offer carryover",
        current_intent="BOOK_ACCOMMODATION",
        pending_carryover={"destination": "Paris"},
        awaiting_carryover_response=False,
        nlu_output={"intent": "BOOK_ACCOMMODATION", "slots": {}},
        user_utterance="now I need a hotel"
    ),
    DMTestCase(
        name="30_carryover_accept",
        purpose="Accepting carryover should continue slot collection",
        current_intent="BOOK_ACCOMMODATION",
        last_action="OFFER_SLOT_CARRYOVER",
        pending_carryover={"destination": "Paris"},
        awaiting_carryover_response=True,
        nlu_output={"intent": "BOOK_ACCOMMODATION", "slots": {"confirmation": "yes"}},
        user_utterance="yes, use the same destination"
    ),
    DMTestCase(
        name="31_carryover_decline",
        purpose="Declining carryover should continue slot collection",
        current_intent="BOOK_ACCOMMODATION",
        last_action="OFFER_SLOT_CARRYOVER",
        pending_carryover={"destination": "Paris"},
        awaiting_carryover_response=True,
        nlu_output={"intent": "BOOK_ACCOMMODATION", "slots": {"confirmation": "no"}},
        user_utterance="no, different city"
    ),

    # ==========================================================================
    # EDGE CASES
    # ==========================================================================
    DMTestCase(
        name="32_new_booking_all_slots_at_once",
        purpose="New flight with all slots at once should ask confirmation",
        nlu_output={
            "intent": "BOOK_FLIGHT",
            "slots": {
                "origin": "Rome",
                "destination": "Paris",
                "departure_date": "2026-03-15",
                "num_passengers": 2,
                "budget_level": "medium"
            }
        },
        user_utterance="book a flight from Rome to Paris on March 15 for 2 people, medium budget"
    ),
    DMTestCase(
        name="33_activity_all_slots_at_once",
        purpose="New activity with all slots at once should ask confirmation",
        nlu_output={
            "intent": "BOOK_ACTIVITY",
            "slots": {
                "destination": "Florence",
                "activity_category": "cultural",
                "budget_level": "low"
            }
        },
        user_utterance="book a cultural activity in Florence, low budget"
    ),
]


# -------------------------
# Statistics and Metrics
# -------------------------

class DMTestStatistics:
    """Track test results and compute metrics."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        # For F1 calculation: track per-action TP, FP, FN
        self.action_tp: Dict[str, int] = defaultdict(int)
        self.action_fp: Dict[str, int] = defaultdict(int)
        self.action_fn: Dict[str, int] = defaultdict(int)
    
    def add_result(
        self,
        test_name: str,
        expected_action: str,
        got_action: str,
        passed: bool
    ):
        self.results.append({
            "test_name": test_name,
            "expected_action": expected_action,
            "got_action": got_action,
            "passed": passed,
        })
        
        # Parse actions to base action for metrics
        expected_base, _ = parse_action(expected_action)
        got_base, _ = parse_action(got_action)
        
        if expected_base == got_base:
            self.action_tp[expected_base] += 1
        else:
            self.action_fn[expected_base] += 1
            self.action_fp[got_base] += 1
    
    def get_accuracy(self) -> float:
        if not self.results:
            return 0.0
        passed = sum(1 for r in self.results if r["passed"])
        return passed / len(self.results)
    
    def get_per_action_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute precision, recall, F1 for each action."""
        all_actions = set(self.action_tp.keys()) | set(self.action_fp.keys()) | set(self.action_fn.keys())
        metrics = {}
        
        for action in all_actions:
            tp = self.action_tp[action]
            fp = self.action_fp[action]
            fn = self.action_fn[action]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            support = tp + fn
            
            metrics[action] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        
        return metrics
    
    def get_macro_f1(self) -> float:
        """Compute macro-averaged F1 score."""
        metrics = self.get_per_action_metrics()
        if not metrics:
            return 0.0
        f1_scores = [m["f1"] for m in metrics.values() if m["support"] > 0]
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    def get_weighted_f1(self) -> float:
        """Compute weighted F1 score (weighted by support)."""
        metrics = self.get_per_action_metrics()
        if not metrics:
            return 0.0
        
        total_support = sum(m["support"] for m in metrics.values())
        if total_support == 0:
            return 0.0
        
        weighted_f1 = sum(m["f1"] * m["support"] for m in metrics.values()) / total_support
        return weighted_f1
    
    def print_report(self):
        print(f"\n{'='*90}")
        print("DIALOGUE MANAGER TEST REPORT")
        print(f"{'='*90}")
        
        # Overall accuracy
        accuracy = self.get_accuracy()
        macro_f1 = self.get_macro_f1()
        weighted_f1 = self.get_weighted_f1()
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:    {accuracy:.2%}")
        print(f"  Macro F1:    {macro_f1:.4f}")
        print(f"  Weighted F1: {weighted_f1:.4f}")
        
        # Per-action statistics
        metrics = self.get_per_action_metrics()
        print(f"\nPer-Action Statistics:")
        print(f"{'Action':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 75)
        
        for action in sorted(metrics.keys()):
            m = metrics[action]
            if m["support"] > 0 or m["fp"] > 0:
                print(f"{action:<30} {m['precision']:>10.2%} {m['recall']:>10.2%} {m['f1']:>10.4f} {m['support']:>10}")
        
        # Failed tests
        failed = [r for r in self.results if not r["passed"]]
        if failed:
            print(f"\nFailed Tests ({len(failed)}):")
            print("-" * 75)
            for r in failed:
                print(f"  {r['test_name']}")
                print(f"    Expected: {r['expected_action']}")
                print(f"    Got:      {r['got_action']}")


def normalize_action(action: str) -> str:
    """Normalize action for comparison (handle REQUEST_MISSING_SLOT variations)."""
    return action


def run_dm_tests(use_llm: bool = True):
    """
    Run DM tests comparing LLM-based DM against rule-based (ground truth).
    
    Args:
        use_llm: If True, test LLM-based DM. If False, test rule-based against itself.
    """
    pipe = None
    if use_llm:
        print("Loading LLM for DM testing...")
        pipe = make_llm()
        print("LLM loaded.\n")
    
    stats = DMTestStatistics()
    
    print("=" * 90)
    print("DIALOGUE MANAGER TESTS")
    print("Testing: LLM-based DM vs Rule-based DM (ground truth)" if use_llm else "Testing: Rule-based DM self-test")
    print("=" * 90)
    
    for tc in TEST_CASES:
        print(f"\nTEST: {tc.name}")
        print(f"  Purpose: {tc.purpose}")
        print(f"  User: \"{tc.user_utterance}\"")
        print(f"  NLU: intent={tc.nlu_output.get('intent')}, slots={tc.nlu_output.get('slots', {})}")
        
        # Create fresh states for both DMs
        state_rule = setup_state(tc)
        state_llm = setup_state(tc)
        
        # Get ground truth from rule-based DM
        expected_action = dm_decide_rule_based(
            state_rule,
            tc.nlu_output,
            tc.user_utterance
        )
        
        # Get LLM-based DM action
        if use_llm:
            got_action = dm_decide(
                state_llm,
                tc.nlu_output,
                tc.user_utterance,
                llm_pipe=pipe
            )
        else:
            # Self-test: both should be identical
            got_action = dm_decide_rule_based(
                state_llm,
                tc.nlu_output,
                tc.user_utterance
            )
        
        # Compare actions
        passed = (normalize_action(expected_action) == normalize_action(got_action))
        
        stats.add_result(tc.name, expected_action, got_action, passed)
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Expected: {expected_action}")
        print(f"  Got:      {got_action}")
        print(f"  Result:   {status}")
    
    # Print report
    stats.print_report()
    
    # Save results to JSON
    results_file = os.path.join(os.path.dirname(__file__), "dm_test_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "accuracy": stats.get_accuracy(),
            "macro_f1": stats.get_macro_f1(),
            "weighted_f1": stats.get_weighted_f1(),
            "per_action_metrics": stats.get_per_action_metrics(),
            "detailed_results": stats.results
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Dialogue Manager")
    parser.add_argument("--no-llm", action="store_true", help="Run without LLM (rule-based self-test)")
    args = parser.parse_args()
    
    run_dm_tests(use_llm=not args.no_llm)

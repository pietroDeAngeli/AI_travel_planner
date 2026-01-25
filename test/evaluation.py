"""
Evaluation Module for Task-Oriented Dialogue System (AI Travel Planner)

System-level evaluation covering:
1. Task Success Rate
2. Slot Filling Accuracy
3. Dialogue Manager Accuracy
4. Dialogue Efficiency

Uses manually defined gold standard test dialogues.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dm import DialogueState, dm_decide
from schema import INTENT_SLOTS


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DialogueTurn:
    """A single turn in a dialogue."""
    user_utterance: str
    nlu_output: Dict[str, Any]  # Simulated NLU output
    expected_action: str        # Gold standard DM action
    expected_slots: Dict[str, Any] = field(default_factory=dict)  # Expected slot values after this turn


@dataclass
class GoldDialogue:
    """A complete dialogue with gold standard annotations."""
    name: str
    description: str
    intent: str
    turns: List[DialogueTurn]
    expected_final_slots: Dict[str, Any]  # Final slot values when task completes
    expected_final_action: str            # Final action (e.g., COMPLETE_FLIGHT_BOOKING)
    is_successful: bool = True            # Whether this dialogue should succeed


@dataclass
class TurnResult:
    """Result of evaluating a single turn."""
    turn_idx: int
    user_utterance: str
    expected_action: str
    actual_action: str
    action_correct: bool
    expected_slots: Dict[str, Any]
    actual_slots: Dict[str, Any]
    slot_matches: Dict[str, bool]


@dataclass 
class DialogueResult:
    """Result of evaluating a complete dialogue."""
    dialogue_name: str
    intent: str
    turns: List[TurnResult]
    task_success: bool
    final_action_correct: bool
    total_turns: int
    slot_precision: float
    slot_recall: float
    slot_f1: float
    dm_accuracy: float


# =============================================================================
# GOLD STANDARD TEST DIALOGUES
# =============================================================================

GOLD_DIALOGUES: List[GoldDialogue] = [
    # =========================================================================
    # BOOK_FLIGHT - Successful dialogues
    # =========================================================================
    GoldDialogue(
        name="flight_simple_success",
        description="Simple flight booking with all slots provided incrementally",
        intent="BOOK_FLIGHT",
        turns=[
            DialogueTurn(
                user_utterance="I want to book a flight to Rome",
                nlu_output={"intent": "BOOK_FLIGHT", "slots": {"destination": "Rome"}},
                expected_action="REQUEST_MISSING_SLOT(origin)",
                expected_slots={"destination": "Rome"}
            ),
            DialogueTurn(
                user_utterance="From Milan",
                nlu_output={"intent": "BOOK_FLIGHT", "slots": {"origin": "Milan"}},
                expected_action="REQUEST_MISSING_SLOT(departure_date)",
                expected_slots={"origin": "Milan", "destination": "Rome"}
            ),
            DialogueTurn(
                user_utterance="March 15th",
                nlu_output={"intent": "BOOK_FLIGHT", "slots": {"departure_date": "2026-03-15"}},
                expected_action="REQUEST_MISSING_SLOT(num_passengers)",
                expected_slots={"origin": "Milan", "destination": "Rome", "departure_date": "2026-03-15"}
            ),
            DialogueTurn(
                user_utterance="2 passengers",
                nlu_output={"intent": "BOOK_FLIGHT", "slots": {"num_passengers": 2}},
                expected_action="REQUEST_MISSING_SLOT(budget_level)",
                expected_slots={"origin": "Milan", "destination": "Rome", "departure_date": "2026-03-15", "num_passengers": 2}
            ),
            DialogueTurn(
                user_utterance="Medium budget",
                nlu_output={"intent": "BOOK_FLIGHT", "slots": {"budget_level": "medium"}},
                expected_action="ASK_CONFIRMATION",
                expected_slots={"origin": "Milan", "destination": "Rome", "departure_date": "2026-03-15", "num_passengers": 2, "budget_level": "medium"}
            ),
            DialogueTurn(
                user_utterance="Yes, confirm",
                nlu_output={"intent": "BOOK_FLIGHT", "slots": {}},
                expected_action="COMPLETE_FLIGHT_BOOKING",
                expected_slots={"origin": "Milan", "destination": "Rome", "departure_date": "2026-03-15", "num_passengers": 2, "budget_level": "medium"}
            ),
        ],
        expected_final_slots={"origin": "Milan", "destination": "Rome", "departure_date": "2026-03-15", "num_passengers": 2, "budget_level": "medium"},
        expected_final_action="COMPLETE_FLIGHT_BOOKING",
        is_successful=True
    ),
    
    GoldDialogue(
        name="flight_all_at_once",
        description="Flight booking with all slots provided in first turn",
        intent="BOOK_FLIGHT",
        turns=[
            DialogueTurn(
                user_utterance="Book a flight from Paris to London on April 10th for 3 passengers, low budget",
                nlu_output={"intent": "BOOK_FLIGHT", "slots": {
                    "origin": "Paris", "destination": "London", 
                    "departure_date": "2026-04-10", "num_passengers": 3, "budget_level": "low"
                }},
                expected_action="ASK_CONFIRMATION",
                expected_slots={"origin": "Paris", "destination": "London", "departure_date": "2026-04-10", "num_passengers": 3, "budget_level": "low"}
            ),
            DialogueTurn(
                user_utterance="Yes",
                nlu_output={"intent": "BOOK_FLIGHT", "slots": {}},
                expected_action="COMPLETE_FLIGHT_BOOKING",
                expected_slots={"origin": "Paris", "destination": "London", "departure_date": "2026-04-10", "num_passengers": 3, "budget_level": "low"}
            ),
        ],
        expected_final_slots={"origin": "Paris", "destination": "London", "departure_date": "2026-04-10", "num_passengers": 3, "budget_level": "low"},
        expected_final_action="COMPLETE_FLIGHT_BOOKING",
        is_successful=True
    ),

    GoldDialogue(
        name="flight_with_denial",
        description="Flight booking where user denies confirmation and changes a slot",
        intent="BOOK_FLIGHT",
        turns=[
            DialogueTurn(
                user_utterance="Flight from Berlin to Madrid on May 5th, 1 passenger, high budget",
                nlu_output={"intent": "BOOK_FLIGHT", "slots": {
                    "origin": "Berlin", "destination": "Madrid",
                    "departure_date": "2026-05-05", "num_passengers": 1, "budget_level": "high"
                }},
                expected_action="ASK_CONFIRMATION",
                expected_slots={"origin": "Berlin", "destination": "Madrid", "departure_date": "2026-05-05", "num_passengers": 1, "budget_level": "high"}
            ),
            DialogueTurn(
                user_utterance="No, change the destination",
                nlu_output={"intent": "BOOK_FLIGHT", "slots": {}},
                expected_action="REQUEST_SLOT_CHANGE",
                expected_slots={"origin": "Berlin", "destination": "Madrid", "departure_date": "2026-05-05", "num_passengers": 1, "budget_level": "high"}
            ),
            DialogueTurn(
                user_utterance="Barcelona instead",
                nlu_output={"intent": "BOOK_FLIGHT", "slots": {"destination": "Barcelona"}},
                expected_action="ASK_CONFIRMATION",
                expected_slots={"origin": "Berlin", "destination": "Barcelona", "departure_date": "2026-05-05", "num_passengers": 1, "budget_level": "high"}
            ),
            DialogueTurn(
                user_utterance="Yes, that's correct",
                nlu_output={"intent": "BOOK_FLIGHT", "slots": {}},
                expected_action="COMPLETE_FLIGHT_BOOKING",
                expected_slots={"origin": "Berlin", "destination": "Barcelona", "departure_date": "2026-05-05", "num_passengers": 1, "budget_level": "high"}
            ),
        ],
        expected_final_slots={"origin": "Berlin", "destination": "Barcelona", "departure_date": "2026-05-05", "num_passengers": 1, "budget_level": "high"},
        expected_final_action="COMPLETE_FLIGHT_BOOKING",
        is_successful=True
    ),

    # =========================================================================
    # BOOK_ACCOMMODATION - Successful dialogues
    # =========================================================================
    GoldDialogue(
        name="accommodation_success",
        description="Hotel booking with incremental slot filling",
        intent="BOOK_ACCOMMODATION",
        turns=[
            DialogueTurn(
                user_utterance="I need a hotel in Prague",
                nlu_output={"intent": "BOOK_ACCOMMODATION", "slots": {"destination": "Prague"}},
                expected_action="REQUEST_MISSING_SLOT(check_in_date)",
                expected_slots={"destination": "Prague"}
            ),
            DialogueTurn(
                user_utterance="Check in June 1st, check out June 5th",
                nlu_output={"intent": "BOOK_ACCOMMODATION", "slots": {"check_in_date": "2026-06-01", "check_out_date": "2026-06-05"}},
                expected_action="REQUEST_MISSING_SLOT(num_guests)",
                expected_slots={"destination": "Prague", "check_in_date": "2026-06-01", "check_out_date": "2026-06-05"}
            ),
            DialogueTurn(
                user_utterance="2 guests, medium budget",
                nlu_output={"intent": "BOOK_ACCOMMODATION", "slots": {"num_guests": 2, "budget_level": "medium"}},
                expected_action="ASK_CONFIRMATION",
                expected_slots={"destination": "Prague", "check_in_date": "2026-06-01", "check_out_date": "2026-06-05", "num_guests": 2, "budget_level": "medium"}
            ),
            DialogueTurn(
                user_utterance="Confirm",
                nlu_output={"intent": "BOOK_ACCOMMODATION", "slots": {}},
                expected_action="COMPLETE_ACCOMMODATION_BOOKING",
                expected_slots={"destination": "Prague", "check_in_date": "2026-06-01", "check_out_date": "2026-06-05", "num_guests": 2, "budget_level": "medium"}
            ),
        ],
        expected_final_slots={"destination": "Prague", "check_in_date": "2026-06-01", "check_out_date": "2026-06-05", "num_guests": 2, "budget_level": "medium"},
        expected_final_action="COMPLETE_ACCOMMODATION_BOOKING",
        is_successful=True
    ),

    # =========================================================================
    # BOOK_ACTIVITY - Successful dialogues
    # =========================================================================
    GoldDialogue(
        name="activity_success",
        description="Activity booking - museum tour",
        intent="BOOK_ACTIVITY",
        turns=[
            DialogueTurn(
                user_utterance="I want to book a museum tour in Florence",
                nlu_output={"intent": "BOOK_ACTIVITY", "slots": {"destination": "Florence", "activity_category": "cultural"}},
                expected_action="REQUEST_MISSING_SLOT(budget_level)",
                expected_slots={"destination": "Florence", "activity_category": "cultural"}
            ),
            DialogueTurn(
                user_utterance="Low budget",
                nlu_output={"intent": "BOOK_ACTIVITY", "slots": {"budget_level": "low"}},
                expected_action="ASK_CONFIRMATION",
                expected_slots={"destination": "Florence", "activity_category": "cultural", "budget_level": "low"}
            ),
            DialogueTurn(
                user_utterance="Yes please",
                nlu_output={"intent": "BOOK_ACTIVITY", "slots": {}},
                expected_action="COMPLETE_ACTIVITY_BOOKING",
                expected_slots={"destination": "Florence", "activity_category": "cultural", "budget_level": "low"}
            ),
        ],
        expected_final_slots={"destination": "Florence", "activity_category": "cultural", "budget_level": "low"},
        expected_final_action="COMPLETE_ACTIVITY_BOOKING",
        is_successful=True
    ),

    # =========================================================================
    # COMPARE_CITIES - Successful dialogues
    # =========================================================================
    GoldDialogue(
        name="compare_cities_success",
        description="Compare two cities for activities",
        intent="COMPARE_CITIES",
        turns=[
            DialogueTurn(
                user_utterance="Compare Paris and London",
                nlu_output={"intent": "COMPARE_CITIES", "slots": {"city1": "Paris", "city2": "London"}},
                expected_action="COMPARE_CITIES_RESULT",
                expected_slots={"city1": "Paris", "city2": "London"}
            ),
        ],
        expected_final_slots={"city1": "Paris", "city2": "London"},
        expected_final_action="COMPARE_CITIES_RESULT",
        is_successful=True
    ),

    GoldDialogue(
        name="compare_cities_incremental",
        description="Compare cities with incremental slot filling",
        intent="COMPARE_CITIES",
        turns=[
            DialogueTurn(
                user_utterance="Compare cities",
                nlu_output={"intent": "COMPARE_CITIES", "slots": {}},
                expected_action="REQUEST_MISSING_SLOT(city1)",
                expected_slots={}
            ),
            DialogueTurn(
                user_utterance="Rome",
                nlu_output={"intent": "COMPARE_CITIES", "slots": {"city1": "Rome"}},
                expected_action="REQUEST_MISSING_SLOT(city2)",
                expected_slots={"city1": "Rome"}
            ),
            DialogueTurn(
                user_utterance="Barcelona",
                nlu_output={"intent": "COMPARE_CITIES", "slots": {"city2": "Barcelona"}},
                expected_action="COMPARE_CITIES_RESULT",
                expected_slots={"city1": "Rome", "city2": "Barcelona"}
            ),
        ],
        expected_final_slots={"city1": "Rome", "city2": "Barcelona"},
        expected_final_action="COMPARE_CITIES_RESULT",
        is_successful=True
    ),

    # =========================================================================
    # GOODBYE - Successful dialogues
    # =========================================================================
    GoldDialogue(
        name="goodbye_simple",
        description="User says goodbye",
        intent="GOODBYE",
        turns=[
            DialogueTurn(
                user_utterance="Goodbye",
                nlu_output={"intent": "GOODBYE", "slots": {}},
                expected_action="GOODBYE",
                expected_slots={}
            ),
        ],
        expected_final_slots={},
        expected_final_action="GOODBYE",
        is_successful=True
    ),

    # =========================================================================
    # OOD - Out of domain handling
    # =========================================================================
    GoldDialogue(
        name="ood_weather",
        description="Out of domain - weather question",
        intent="OOD",
        turns=[
            DialogueTurn(
                user_utterance="What's the weather like?",
                nlu_output={"intent": "OOD", "slots": {}},
                expected_action="ASK_CLARIFICATION",
                expected_slots={}
            ),
        ],
        expected_final_slots={},
        expected_final_action="ASK_CLARIFICATION",
        is_successful=True  # OOD handling is considered success if clarification is asked
    ),

    GoldDialogue(
        name="ood_unclear",
        description="Out of domain - unclear input",
        intent="OOD",
        turns=[
            DialogueTurn(
                user_utterance="maybe something",
                nlu_output={"intent": "OOD", "slots": {}},
                expected_action="ASK_CLARIFICATION",
                expected_slots={}
            ),
        ],
        expected_final_slots={},
        expected_final_action="ASK_CLARIFICATION",
        is_successful=True
    ),

    # =========================================================================
    # EDGE CASES
    # =========================================================================
    GoldDialogue(
        name="flight_minimal_then_complete",
        description="User provides minimal info, then completes",
        intent="BOOK_FLIGHT",
        turns=[
            DialogueTurn(
                user_utterance="I need a flight",
                nlu_output={"intent": "BOOK_FLIGHT", "slots": {}},
                expected_action="REQUEST_MISSING_SLOT(origin)",
                expected_slots={}
            ),
            DialogueTurn(
                user_utterance="From New York to Tokyo, March 20, 2 people, high budget",
                nlu_output={"intent": "BOOK_FLIGHT", "slots": {
                    "origin": "New York", "destination": "Tokyo",
                    "departure_date": "2026-03-20", "num_passengers": 2, "budget_level": "high"
                }},
                expected_action="ASK_CONFIRMATION",
                expected_slots={"origin": "New York", "destination": "Tokyo", "departure_date": "2026-03-20", "num_passengers": 2, "budget_level": "high"}
            ),
            DialogueTurn(
                user_utterance="Yes",
                nlu_output={"intent": "BOOK_FLIGHT", "slots": {}},
                expected_action="COMPLETE_FLIGHT_BOOKING",
                expected_slots={"origin": "New York", "destination": "Tokyo", "departure_date": "2026-03-20", "num_passengers": 2, "budget_level": "high"}
            ),
        ],
        expected_final_slots={"origin": "New York", "destination": "Tokyo", "departure_date": "2026-03-20", "num_passengers": 2, "budget_level": "high"},
        expected_final_action="COMPLETE_FLIGHT_BOOKING",
        is_successful=True
    ),
]


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def normalize_action(action: str) -> str:
    """Normalize action for comparison (handle parameterized actions)."""
    # Extract base action for comparison
    if action.startswith("REQUEST_MISSING_SLOT("):
        return action  # Keep full action for exact match
    return action


def compare_actions(expected: str, actual: str) -> bool:
    """Compare two actions, handling parameterized actions."""
    # Exact match
    if expected == actual:
        return True
    
    # For REQUEST_MISSING_SLOT, check if both request a slot (flexible matching)
    if expected.startswith("REQUEST_MISSING_SLOT(") and actual.startswith("REQUEST_MISSING_SLOT("):
        # Extract slot names
        exp_slot = expected[len("REQUEST_MISSING_SLOT("):-1]
        act_slot = actual[len("REQUEST_MISSING_SLOT("):-1]
        # For evaluation, we can be strict or flexible
        return exp_slot == act_slot
    
    return False


def compute_slot_metrics(expected: Dict[str, Any], actual: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 for slot filling.
    
    Returns: (precision, recall, f1)
    """
    if not expected and not actual:
        return 1.0, 1.0, 1.0
    
    if not expected:
        return 0.0, 1.0, 0.0
    
    if not actual:
        return 0.0, 0.0, 0.0
    
    # Count matches
    true_positives = 0
    for slot, exp_value in expected.items():
        if slot in actual and actual[slot] == exp_value:
            true_positives += 1
    
    precision = true_positives / len(actual) if actual else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def get_actual_slots(state: DialogueState, intent: str) -> Dict[str, Any]:
    """Extract actual slot values from dialogue state."""
    if intent == "COMPARE_CITIES":
        return {k: v for k, v in state.compare_cities_data.items() if v is not None}
    
    booking = state.get_current_booking()
    if booking:
        data = booking.to_dict()
        # Remove 'completed' field
        return {k: v for k, v in data.items() if v is not None and k != "completed"}
    
    return {}


def evaluate_dialogue(dialogue: GoldDialogue, verbose: bool = False) -> DialogueResult:
    """
    Evaluate a single dialogue against gold standard.
    
    Args:
        dialogue: Gold standard dialogue to evaluate
        verbose: Whether to print detailed output
    
    Returns:
        DialogueResult with evaluation metrics
    """
    state = DialogueState()
    turn_results: List[TurnResult] = []
    correct_actions = 0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating: {dialogue.name}")
        print(f"Description: {dialogue.description}")
        print(f"{'='*60}")
    
    for idx, turn in enumerate(dialogue.turns):
        # Run DM decision
        actual_action = dm_decide(state, turn.nlu_output, turn.user_utterance)
        
        # Get actual slots from state
        actual_slots = get_actual_slots(state, dialogue.intent)
        
        # Compare actions
        action_correct = compare_actions(turn.expected_action, actual_action)
        if action_correct:
            correct_actions += 1
        
        # Compare slots
        slot_matches = {}
        for slot, exp_value in turn.expected_slots.items():
            slot_matches[slot] = actual_slots.get(slot) == exp_value
        
        turn_result = TurnResult(
            turn_idx=idx,
            user_utterance=turn.user_utterance,
            expected_action=turn.expected_action,
            actual_action=actual_action,
            action_correct=action_correct,
            expected_slots=turn.expected_slots,
            actual_slots=actual_slots,
            slot_matches=slot_matches
        )
        turn_results.append(turn_result)
        
        if verbose:
            status = "PASS" if action_correct else "FAIL"
            print(f"\nTurn {idx + 1}: {turn.user_utterance}")
            print(f"  Expected action: {turn.expected_action}")
            print(f"  Actual action:   {actual_action} [{status}]")
            print(f"  Slots: {actual_slots}")
    
    # Compute final metrics
    final_actual_slots = get_actual_slots(state, dialogue.intent)
    slot_precision, slot_recall, slot_f1 = compute_slot_metrics(
        dialogue.expected_final_slots, final_actual_slots
    )
    
    dm_accuracy = correct_actions / len(dialogue.turns) if dialogue.turns else 0.0
    
    # Determine task success
    final_action_correct = compare_actions(
        dialogue.expected_final_action, 
        turn_results[-1].actual_action if turn_results else ""
    )
    
    # Task is successful if:
    # 1. Final action is correct
    # 2. All expected slots are filled correctly
    task_success = final_action_correct and slot_f1 == 1.0
    
    result = DialogueResult(
        dialogue_name=dialogue.name,
        intent=dialogue.intent,
        turns=turn_results,
        task_success=task_success,
        final_action_correct=final_action_correct,
        total_turns=len(dialogue.turns),
        slot_precision=slot_precision,
        slot_recall=slot_recall,
        slot_f1=slot_f1,
        dm_accuracy=dm_accuracy
    )
    
    if verbose:
        print(f"\n--- Results ---")
        print(f"Task Success: {'PASS' if task_success else 'FAIL'}")
        print(f"DM Accuracy: {dm_accuracy:.2%}")
        print(f"Slot F1: {slot_f1:.2%}")
    
    return result


def evaluate_all(dialogues: List[GoldDialogue] = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Evaluate all gold standard dialogues.
    
    Args:
        dialogues: List of dialogues to evaluate (defaults to GOLD_DIALOGUES)
        verbose: Whether to print detailed output
    
    Returns:
        Dictionary with aggregate metrics
    """
    if dialogues is None:
        dialogues = GOLD_DIALOGUES
    
    results: List[DialogueResult] = []
    
    for dialogue in dialogues:
        result = evaluate_dialogue(dialogue, verbose=verbose)
        results.append(result)
    
    # Aggregate metrics
    total_dialogues = len(results)
    successful_dialogues = sum(1 for r in results if r.task_success)
    
    total_turns = sum(r.total_turns for r in results)
    total_correct_actions = sum(
        sum(1 for t in r.turns if t.action_correct) 
        for r in results
    )
    
    avg_slot_precision = sum(r.slot_precision for r in results) / total_dialogues if total_dialogues else 0
    avg_slot_recall = sum(r.slot_recall for r in results) / total_dialogues if total_dialogues else 0
    avg_slot_f1 = sum(r.slot_f1 for r in results) / total_dialogues if total_dialogues else 0
    
    avg_turns = sum(r.total_turns for r in results) / total_dialogues if total_dialogues else 0
    
    # Per-intent breakdown
    intent_stats = defaultdict(lambda: {"total": 0, "success": 0, "dm_acc": [], "turns": []})
    for r in results:
        intent_stats[r.intent]["total"] += 1
        intent_stats[r.intent]["success"] += int(r.task_success)
        intent_stats[r.intent]["dm_acc"].append(r.dm_accuracy)
        intent_stats[r.intent]["turns"].append(r.total_turns)
    
    summary = {
        "total_dialogues": total_dialogues,
        "successful_dialogues": successful_dialogues,
        "task_success_rate": successful_dialogues / total_dialogues if total_dialogues else 0,
        "dm_accuracy": total_correct_actions / total_turns if total_turns else 0,
        "avg_slot_precision": avg_slot_precision,
        "avg_slot_recall": avg_slot_recall,
        "avg_slot_f1": avg_slot_f1,
        "avg_turns_per_dialogue": avg_turns,
        "total_turns": total_turns,
        "per_intent": {
            intent: {
                "total": stats["total"],
                "success": stats["success"],
                "success_rate": stats["success"] / stats["total"] if stats["total"] else 0,
                "avg_dm_accuracy": sum(stats["dm_acc"]) / len(stats["dm_acc"]) if stats["dm_acc"] else 0,
                "avg_turns": sum(stats["turns"]) / len(stats["turns"]) if stats["turns"] else 0
            }
            for intent, stats in intent_stats.items()
        },
        "detailed_results": results
    }
    
    return summary


def print_summary_table(summary: Dict[str, Any]) -> None:
    """Print a formatted summary table of evaluation results."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY - AI TRAVEL PLANNER DIALOGUE SYSTEM")
    print("=" * 80)
    
    print("\n1. OVERALL METRICS")
    print("-" * 40)
    print(f"  Total Dialogues:        {summary['total_dialogues']}")
    print(f"  Successful Dialogues:   {summary['successful_dialogues']}")
    print(f"  Task Success Rate:      {summary['task_success_rate']:.2%}")
    print(f"  DM Accuracy:            {summary['dm_accuracy']:.2%}")
    
    print("\n2. SLOT FILLING METRICS")
    print("-" * 40)
    print(f"  Avg Precision:          {summary['avg_slot_precision']:.2%}")
    print(f"  Avg Recall:             {summary['avg_slot_recall']:.2%}")
    print(f"  Avg F1 Score:           {summary['avg_slot_f1']:.2%}")
    
    print("\n3. EFFICIENCY METRICS")
    print("-" * 40)
    print(f"  Total Turns:            {summary['total_turns']}")
    print(f"  Avg Turns/Dialogue:     {summary['avg_turns_per_dialogue']:.2f}")
    
    print("\n4. PER-INTENT BREAKDOWN")
    print("-" * 80)
    print(f"{'Intent':<25} {'Total':>8} {'Success':>8} {'Rate':>10} {'DM Acc':>10} {'Avg Turns':>10}")
    print("-" * 80)
    
    for intent, stats in sorted(summary['per_intent'].items()):
        print(f"{intent:<25} {stats['total']:>8} {stats['success']:>8} "
              f"{stats['success_rate']:>10.2%} {stats['avg_dm_accuracy']:>10.2%} "
              f"{stats['avg_turns']:>10.2f}")
    
    print("\n5. INDIVIDUAL DIALOGUE RESULTS")
    print("-" * 80)
    print(f"{'Dialogue':<35} {'Intent':<20} {'Success':>8} {'DM Acc':>10} {'Slot F1':>10}")
    print("-" * 80)
    
    for r in summary['detailed_results']:
        status = "PASS" if r.task_success else "FAIL"
        print(f"{r.dialogue_name:<35} {r.intent:<20} {status:>8} "
              f"{r.dm_accuracy:>10.2%} {r.slot_f1:>10.2%}")
    
    print("\n" + "=" * 80)


def run_evaluation(verbose: bool = False) -> Dict[str, Any]:
    """
    Run the complete evaluation pipeline.
    
    Args:
        verbose: Whether to print detailed per-turn output
    
    Returns:
        Summary dictionary with all metrics
    """
    print("Starting Dialogue System Evaluation...")
    print(f"Evaluating {len(GOLD_DIALOGUES)} gold standard dialogues\n")
    
    summary = evaluate_all(verbose=verbose)
    print_summary_table(summary)
    
    return summary


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate AI Travel Planner Dialogue System")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed per-turn output")
    args = parser.parse_args()
    
    run_evaluation(verbose=args.verbose)

"""
Pipeline Evaluation Test Suite

Evaluates the complete dialogue pipeline (NLU → DM → NLG) using:
1. Template-based synthetic dialogue generation
2. LLM-based dialogue generation (few-shot)

Metrics:
- Task Success Rate
- Slot Filling Accuracy (Precision, Recall, F1)
- Dialogue Manager Action Accuracy
- Dialogue Efficiency (turns to completion)
- End-to-End Response Appropriateness
"""

import copy
import json
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from tqdm import tqdm

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm import make_llm
from nlu import nlu_parse
from dm import DialogueState, dm_decide, dm_decide_rule_based, state_context
from nlg import nlg_generate
from intent_splitter import split_intents, IntentQueue
from schema import (parse_action)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GeneratedDialogue:
    """A dialogue instance for evaluation."""
    name: str
    intent: str
    generation_method: str  # "static", "template", "llm", etc.
    turns: List[Dict[str, Any]]  # Each turn: {user_utterance, provided_slots, expected_action}
    expected_final_slots: Dict[str, Any]
    expected_task_success: bool
    is_multi_intent: bool = False
    expected_intents: List[str] = field(default_factory=list)


# =============================================================================
# STATIC TEST DIALOGUES (NO RANDOM, NO LLM GENERATION)
# =============================================================================

def get_static_dialogues() -> List[GeneratedDialogue]:
    """Return a fixed, deterministic set of evaluation dialogues."""

    dialogues = []

    # --------------------------------------------------
    # 1. Flight – Full booking
    # --------------------------------------------------
    dialogues.append(GeneratedDialogue(
        name="flight_full_static",
        intent="BOOK_FLIGHT",
        generation_method="static",
        turns=[
            {
                "user_utterance": "Book a flight from Rome to Paris on 2026-06-10 for 2 passengers, medium budget",
                "provided_slots": {
                    "origin": "Rome",
                    "destination": "Paris",
                    "departure_date": "2026-06-10",
                    "num_passengers": 2,
                    "budget_level": "medium"
                },
                "expected_action": "ASK_CONFIRMATION"
            },
            {
                "user_utterance": "Yes",
                "provided_slots": {"confirmation": "yes"},
                "expected_action": "COMPLETE_FLIGHT_BOOKING"
            }
        ],
        expected_final_slots={
            "origin": "Rome",
            "destination": "Paris",
            "departure_date": "2026-06-10",
            "num_passengers": 2,
            "budget_level": "medium"
        },
        expected_task_success=True,
    ))

    # --------------------------------------------------
    # 2. Flight – Incremental
    # --------------------------------------------------
    dialogues.append(GeneratedDialogue(
        name="flight_incremental_static",
        intent="BOOK_FLIGHT",
        generation_method="static",
        turns=[
            {
                "user_utterance": "I want to fly to London",
                "provided_slots": {"destination": "London"},
                "expected_action": "REQUEST_MISSING_SLOT(origin)"
            },
            {
                "user_utterance": "From Berlin",
                "provided_slots": {"origin": "Berlin"},
                "expected_action": "REQUEST_MISSING_SLOT(departure_date)"
            },
            {
                "user_utterance": "On 2026-07-01",
                "provided_slots": {"departure_date": "2026-07-01"},
                "expected_action": "REQUEST_MISSING_SLOT(num_passengers)"
            },
            {
                "user_utterance": "2 people",
                "provided_slots": {"num_passengers": 2},
                "expected_action": "REQUEST_MISSING_SLOT(budget_level)"
            },
            {
                "user_utterance": "Low budget",
                "provided_slots": {"budget_level": "low"},
                "expected_action": "ASK_CONFIRMATION"
            },
            {
                "user_utterance": "Confirm",
                "provided_slots": {"confirmation": "yes"},
                "expected_action": "COMPLETE_FLIGHT_BOOKING"
            }
        ],
        expected_final_slots={
            "origin": "Berlin",
            "destination": "London",
            "departure_date": "2026-07-01",
            "num_passengers": 2,
            "budget_level": "low"
        },
        expected_task_success=True,
    ))

    # --------------------------------------------------
    # 3. Accommodation – Full
    # --------------------------------------------------
    dialogues.append(GeneratedDialogue(
        name="hotel_full_static",
        intent="BOOK_ACCOMMODATION",
        generation_method="static",
        turns=[
            {
                "user_utterance": "Book a hotel in Madrid from 2026-08-01 to 2026-08-05 for 2 guests, high budget",
                "provided_slots": {
                    "destination": "Madrid",
                    "check_in_date": "2026-08-01",
                    "check_out_date": "2026-08-05",
                    "num_guests": 2,
                    "budget_level": "high"
                },
                "expected_action": "ASK_CONFIRMATION"
            },
            {
                "user_utterance": "Yes please",
                "provided_slots": {"confirmation": "yes"},
                "expected_action": "COMPLETE_ACCOMMODATION_BOOKING"
            }
        ],
        expected_final_slots={
            "destination": "Madrid",
            "check_in_date": "2026-08-01",
            "check_out_date": "2026-08-05",
            "num_guests": 2,
            "budget_level": "high"
        },
        expected_task_success=True,
    ))

    # --------------------------------------------------
    # 4. Compare Cities
    # --------------------------------------------------
    dialogues.append(GeneratedDialogue(
        name="compare_static",
        intent="COMPARE_CITIES",
        generation_method="static",
        turns=[
            {
                "user_utterance": "Compare Rome and Paris for culture",
                "provided_slots": {
                    "city1": "Rome",
                    "city2": "Paris",
                    "activity_category": "culture"
                },
                "expected_action": "COMPARE_CITIES_RESULT"
            }
        ],
        expected_final_slots={
            "city1": "Rome",
            "city2": "Paris",
            "activity_category": "culture"
        },
        expected_task_success=True,
    ))

    # --------------------------------------------------
    # 5. End Dialogue
    # --------------------------------------------------
    dialogues.append(GeneratedDialogue(
        name="end_dialogue_static",
        intent="END_DIALOGUE",
        generation_method="static",
        turns=[
            {
                "user_utterance": "Goodbye",
                "provided_slots": {},
                "expected_action": "GOODBYE"
            }
        ],
        expected_final_slots={},
        expected_task_success=True,
    ))

    # ==========================================================
    # UNDER-INFORMATIVE USER DIALOGUES
    # Users provide very little information per turn, forcing
    # the system to request missing slots incrementally.
    # ==========================================================

    # --------------------------------------------------
    # 6. Flight – Under-informative (minimal start)
    # User says only "I want to fly" — no slots at all.
    # --------------------------------------------------
    dialogues.append(GeneratedDialogue(
        name="flight_under_minimal",
        intent="BOOK_FLIGHT",
        generation_method="static",
        turns=[
            {
                "user_utterance": "I want to fly",
                "provided_slots": {},
                "expected_action": "REQUEST_MISSING_SLOT(destination)"
            },
            {
                "user_utterance": "Tokyo",
                "provided_slots": {"destination": "Tokyo"},
                "expected_action": "REQUEST_MISSING_SLOT(origin)"
            },
            {
                "user_utterance": "Milan",
                "provided_slots": {"origin": "Milan"},
                "expected_action": "REQUEST_MISSING_SLOT(departure_date)"
            },
            {
                "user_utterance": "2026-09-15",
                "provided_slots": {"departure_date": "2026-09-15"},
                "expected_action": "REQUEST_MISSING_SLOT(num_passengers)"
            },
            {
                "user_utterance": "Just me",
                "provided_slots": {"num_passengers": 1},
                "expected_action": "REQUEST_MISSING_SLOT(budget_level)"
            },
            {
                "user_utterance": "High",
                "provided_slots": {"budget_level": "high"},
                "expected_action": "ASK_CONFIRMATION"
            },
            {
                "user_utterance": "Yes",
                "provided_slots": {"confirmation": "yes"},
                "expected_action": "COMPLETE_FLIGHT_BOOKING"
            }
        ],
        expected_final_slots={
            "origin": "Milan",
            "destination": "Tokyo",
            "departure_date": "2026-09-15",
            "num_passengers": 1,
            "budget_level": "high"
        },
        expected_task_success=True,
    ))

    # --------------------------------------------------
    # 7. Accommodation – Under-informative (vague start)
    # User says "I need a place to stay" — zero details.
    # --------------------------------------------------
    dialogues.append(GeneratedDialogue(
        name="hotel_under_vague",
        intent="BOOK_ACCOMMODATION",
        generation_method="static",
        turns=[
            {
                "user_utterance": "I need a place to stay",
                "provided_slots": {},
                "expected_action": "REQUEST_MISSING_SLOT(destination)"
            },
            {
                "user_utterance": "Barcelona",
                "provided_slots": {"destination": "Barcelona"},
                "expected_action": "REQUEST_MISSING_SLOT(check_in_date)"
            },
            {
                "user_utterance": "2026-10-01",
                "provided_slots": {"check_in_date": "2026-10-01"},
                "expected_action": "REQUEST_MISSING_SLOT(check_out_date)"
            },
            {
                "user_utterance": "2026-10-05",
                "provided_slots": {"check_out_date": "2026-10-05"},
                "expected_action": "REQUEST_MISSING_SLOT(num_guests)"
            },
            {
                "user_utterance": "3",
                "provided_slots": {"num_guests": 3},
                "expected_action": "REQUEST_MISSING_SLOT(budget_level)"
            },
            {
                "user_utterance": "Medium",
                "provided_slots": {"budget_level": "medium"},
                "expected_action": "ASK_CONFIRMATION"
            },
            {
                "user_utterance": "Yes, confirm",
                "provided_slots": {"confirmation": "yes"},
                "expected_action": "COMPLETE_ACCOMMODATION_BOOKING"
            }
        ],
        expected_final_slots={
            "destination": "Barcelona",
            "check_in_date": "2026-10-01",
            "check_out_date": "2026-10-05",
            "num_guests": 3,
            "budget_level": "medium"
        },
        expected_task_success=True,
    ))

    # --------------------------------------------------
    # 8. Activity – Under-informative (only destination)
    # User mentions only the city, nothing else.
    # --------------------------------------------------
    dialogues.append(GeneratedDialogue(
        name="activity_under_sparse",
        intent="BOOK_ACTIVITY",
        generation_method="static",
        turns=[
            {
                "user_utterance": "I want to do something in Rome",
                "provided_slots": {"destination": "Rome"},
                "expected_action": "REQUEST_MISSING_SLOT(activity_category)"
            },
            {
                "user_utterance": "Cultural",
                "provided_slots": {"activity_category": "cultural"},
                "expected_action": "REQUEST_MISSING_SLOT(budget_level)"
            },
            {
                "user_utterance": "Low",
                "provided_slots": {"budget_level": "low"},
                "expected_action": "REQUEST_MISSING_SLOT(preferred_time)"
            },
            {
                "user_utterance": "Morning",
                "provided_slots": {"preferred_time": "morning"},
                "expected_action": "ASK_CONFIRMATION"
            },
            {
                "user_utterance": "Yes",
                "provided_slots": {"confirmation": "yes"},
                "expected_action": "COMPLETE_ACTIVITY_BOOKING"
            }
        ],
        expected_final_slots={
            "destination": "Rome",
            "activity_category": "cultural",
            "budget_level": "low",
            "preferred_time": "morning"
        },
        expected_task_success=True,
    ))

    # --------------------------------------------------
    # 9. Compare – Under-informative (no details)
    # User says "Compare cities" without specifying which.
    # --------------------------------------------------
    dialogues.append(GeneratedDialogue(
        name="compare_under_incomplete",
        intent="COMPARE_CITIES",
        generation_method="static",
        turns=[
            {
                "user_utterance": "I'd like to compare two cities",
                "provided_slots": {},
                "expected_action": "REQUEST_MISSING_SLOT(city1)"
            },
            {
                "user_utterance": "London",
                "provided_slots": {"city1": "London"},
                "expected_action": "REQUEST_MISSING_SLOT(city2)"
            },
            {
                "user_utterance": "Amsterdam",
                "provided_slots": {"city2": "Amsterdam"},
                "expected_action": "REQUEST_MISSING_SLOT(activity_category)"
            },
            {
                "user_utterance": "Nightlife",
                "provided_slots": {"activity_category": "nightlife"},
                "expected_action": "COMPARE_CITIES_RESULT"
            }
        ],
        expected_final_slots={
            "city1": "London",
            "city2": "Amsterdam",
            "activity_category": "nightlife"
        },
        expected_task_success=True,
    ))

    # --------------------------------------------------
    # 10. Flight – Under-informative (terse one-word answers)
    # User gives bare minimum single-word responses.
    # --------------------------------------------------
    dialogues.append(GeneratedDialogue(
        name="flight_under_terse",
        intent="BOOK_FLIGHT",
        generation_method="static",
        turns=[
            {
                "user_utterance": "Flight",
                "provided_slots": {},
                "expected_action": "REQUEST_MISSING_SLOT(destination)"
            },
            {
                "user_utterance": "Lisbon",
                "provided_slots": {"destination": "Lisbon"},
                "expected_action": "REQUEST_MISSING_SLOT(origin)"
            },
            {
                "user_utterance": "Vienna",
                "provided_slots": {"origin": "Vienna"},
                "expected_action": "REQUEST_MISSING_SLOT(departure_date)"
            },
            {
                "user_utterance": "2026-12-20",
                "provided_slots": {"departure_date": "2026-12-20"},
                "expected_action": "REQUEST_MISSING_SLOT(num_passengers)"
            },
            {
                "user_utterance": "4",
                "provided_slots": {"num_passengers": 4},
                "expected_action": "REQUEST_MISSING_SLOT(budget_level)"
            },
            {
                "user_utterance": "Low",
                "provided_slots": {"budget_level": "low"},
                "expected_action": "ASK_CONFIRMATION"
            },
            {
                "user_utterance": "Yes",
                "provided_slots": {"confirmation": "yes"},
                "expected_action": "COMPLETE_FLIGHT_BOOKING"
            }
        ],
        expected_final_slots={
            "origin": "Vienna",
            "destination": "Lisbon",
            "departure_date": "2026-12-20",
            "num_passengers": 4,
            "budget_level": "low"
        },
        expected_task_success=True,
    ))

    # --------------------------------------------------
    # 11. Accommodation – Under-informative (one slot at a time)
    # User provides exactly one slot per turn.
    # --------------------------------------------------
    dialogues.append(GeneratedDialogue(
        name="hotel_under_one_by_one",
        intent="BOOK_ACCOMMODATION",
        generation_method="static",
        turns=[
            {
                "user_utterance": "Hotel",
                "provided_slots": {},
                "expected_action": "REQUEST_MISSING_SLOT(destination)"
            },
            {
                "user_utterance": "Prague",
                "provided_slots": {"destination": "Prague"},
                "expected_action": "REQUEST_MISSING_SLOT(check_in_date)"
            },
            {
                "user_utterance": "2026-11-10",
                "provided_slots": {"check_in_date": "2026-11-10"},
                "expected_action": "REQUEST_MISSING_SLOT(check_out_date)"
            },
            {
                "user_utterance": "2026-11-14",
                "provided_slots": {"check_out_date": "2026-11-14"},
                "expected_action": "REQUEST_MISSING_SLOT(num_guests)"
            },
            {
                "user_utterance": "1",
                "provided_slots": {"num_guests": 1},
                "expected_action": "REQUEST_MISSING_SLOT(budget_level)"
            },
            {
                "user_utterance": "High",
                "provided_slots": {"budget_level": "high"},
                "expected_action": "ASK_CONFIRMATION"
            },
            {
                "user_utterance": "Confirm",
                "provided_slots": {"confirmation": "yes"},
                "expected_action": "COMPLETE_ACCOMMODATION_BOOKING"
            }
        ],
        expected_final_slots={
            "destination": "Prague",
            "check_in_date": "2026-11-10",
            "check_out_date": "2026-11-14",
            "num_guests": 1,
            "budget_level": "high"
        },
        expected_task_success=True,
    ))

    # ==========================================================
    # OVER-INFORMATIVE USER DIALOGUES
    # Users provide more information than needed: extra narrative,
    # irrelevant details, optional slots, or info for other intents.
    # ==========================================================

    # --------------------------------------------------
    # 12. Flight – Over-informative (verbose narrative)
    # User provides all slots + return_date + irrelevant story.
    # System should extract relevant slots and ignore the rest.
    # --------------------------------------------------
    dialogues.append(GeneratedDialogue(
        name="flight_over_verbose",
        intent="BOOK_FLIGHT",
        generation_method="static",
        turns=[
            {
                "user_utterance": "I really need to visit my grandmother in Athens, she's been quite ill lately and I haven't seen her since Christmas. I'm flying from Munich on 2026-07-20, it'll be just me traveling alone, and I'd like to keep it cheap so low budget please. Oh and I'm coming back on 2026-07-27.",
                "provided_slots": {
                    "origin": "Munich",
                    "destination": "Athens",
                    "departure_date": "2026-07-20",
                    "num_passengers": 1,
                    "budget_level": "low"
                },
                "expected_action": "ASK_CONFIRMATION"
            },
            {
                "user_utterance": "Yes that looks right, thanks!",
                "provided_slots": {"confirmation": "yes"},
                "expected_action": "COMPLETE_FLIGHT_BOOKING"
            }
        ],
        expected_final_slots={
            "origin": "Munich",
            "destination": "Athens",
            "departure_date": "2026-07-20",
            "num_passengers": 1,
            "budget_level": "low"
        },
        expected_task_success=True,
    ))

    # --------------------------------------------------
    # 13. Accommodation – Over-informative (extra slot domains)
    # User provides hotel info AND mentions flight details.
    # System should focus on accommodation slots only.
    # --------------------------------------------------
    dialogues.append(GeneratedDialogue(
        name="hotel_over_extra_domains",
        intent="BOOK_ACCOMMODATION",
        generation_method="static",
        turns=[
            {
                "user_utterance": "I'm arriving in Dublin by flight from Rome on 2026-09-01 and I need a hotel. Check in September 1st, check out September 6th 2026, 2 guests, medium budget. I also want to rent a car but let's do the hotel first.",
                "provided_slots": {
                    "destination": "Dublin",
                    "check_in_date": "2026-09-01",
                    "check_out_date": "2026-09-06",
                    "num_guests": 2,
                    "budget_level": "medium"
                },
                "expected_action": "ASK_CONFIRMATION"
            },
            {
                "user_utterance": "Perfect, confirm the hotel",
                "provided_slots": {"confirmation": "yes"},
                "expected_action": "COMPLETE_ACCOMMODATION_BOOKING"
            }
        ],
        expected_final_slots={
            "destination": "Dublin",
            "check_in_date": "2026-09-01",
            "check_out_date": "2026-09-06",
            "num_guests": 2,
            "budget_level": "medium"
        },
        expected_task_success=True,
    ))

    # --------------------------------------------------
    # 14. Activity – Over-informative (extra preferences)
    # User provides all activity slots + irrelevant details like
    # weather preference, clothing, dietary restrictions.
    # --------------------------------------------------
    dialogues.append(GeneratedDialogue(
        name="activity_over_detailed",
        intent="BOOK_ACTIVITY",
        generation_method="static",
        turns=[
            {
                "user_utterance": "I'd love a food tour in Barcelona, preferably in the evening around 7pm. Medium budget. I'm vegetarian and I hope it doesn't rain. Should I wear formal clothes?",
                "provided_slots": {
                    "destination": "Barcelona",
                    "activity_category": "food",
                    "budget_level": "medium",
                    "preferred_time": "evening"
                },
                "expected_action": "ASK_CONFIRMATION"
            },
            {
                "user_utterance": "Sounds great, go ahead",
                "provided_slots": {"confirmation": "yes"},
                "expected_action": "COMPLETE_ACTIVITY_BOOKING"
            }
        ],
        expected_final_slots={
            "destination": "Barcelona",
            "activity_category": "food",
            "budget_level": "medium",
            "preferred_time": "evening"
        },
        expected_task_success=True,
    ))

    # --------------------------------------------------
    # 15. Compare – Over-informative (extra narrative + opinions)
    # User provides comparison request with lots of opinions
    # and tangential context.
    # --------------------------------------------------
    dialogues.append(GeneratedDialogue(
        name="compare_over_narrative",
        intent="COMPARE_CITIES",
        generation_method="static",
        turns=[
            {
                "user_utterance": "My wife and I are debating where to go for our anniversary. I think Berlin is amazing for nightlife but she prefers Prague. Can you compare Berlin and Prague for nightlife? We've been to both before but years ago.",
                "provided_slots": {
                    "city1": "Berlin",
                    "city2": "Prague",
                    "activity_category": "nightlife"
                },
                "expected_action": "COMPARE_CITIES_RESULT"
            }
        ],
        expected_final_slots={
            "city1": "Berlin",
            "city2": "Prague",
            "activity_category": "nightlife"
        },
        expected_task_success=True,
    ))

    # --------------------------------------------------
    # 16. Flight – Over-informative (all slots + return date)
    # User provides every possible slot including the optional
    # return_date in a single dense utterance.
    # --------------------------------------------------
    dialogues.append(GeneratedDialogue(
        name="flight_over_complete",
        intent="BOOK_FLIGHT",
        generation_method="static",
        turns=[
            {
                "user_utterance": "Book a round trip flight from Stockholm to Istanbul departing 2026-08-10 returning 2026-08-20 for 3 passengers, high budget. I prefer morning flights and aisle seats.",
                "provided_slots": {
                    "origin": "Stockholm",
                    "destination": "Istanbul",
                    "departure_date": "2026-08-10",
                    "num_passengers": 3,
                    "budget_level": "high"
                },
                "expected_action": "ASK_CONFIRMATION"
            },
            {
                "user_utterance": "Confirmed",
                "provided_slots": {"confirmation": "yes"},
                "expected_action": "COMPLETE_FLIGHT_BOOKING"
            }
        ],
        expected_final_slots={
            "origin": "Stockholm",
            "destination": "Istanbul",
            "departure_date": "2026-08-10",
            "num_passengers": 3,
            "budget_level": "high"
        },
        expected_task_success=True,
    ))

    return dialogues

# =============================================================================
# PIPELINE EVALUATION
# =============================================================================

@dataclass
class PipelineMetrics:
    """Aggregated metrics for pipeline evaluation."""
    # Task-level
    task_success_rate: float = 0.0
    
    # Slot-level
    slot_precision: float = 0.0
    slot_recall: float = 0.0
    slot_f1: float = 0.0
    
    # DM-level
    dm_accuracy: float = 0.0
    dm_action_f1: float = 0.0
    
    # Efficiency
    avg_turns: float = 0.0
    
    # Per-intent breakdown
    per_intent_success: Dict[str, float] = field(default_factory=dict)
    per_intent_dm_accuracy: Dict[str, float] = field(default_factory=dict)
    
    # Intent Splitter metrics
    splitter_detection_accuracy: float = 0.0  # How often it correctly detects multi-intent
    splitter_split_accuracy: float = 0.0  # How often it splits correctly
    multi_intent_success_rate: float = 0.0  # Task success for multi-intent dialogues


@dataclass
class DialogueEvalResult:
    """Result of evaluating a single dialogue."""
    dialogue_name: str
    intent: str
    generation_method: str
    dm_mode: str  # "llm" or "rule"
    task_success: bool
    num_turns: int
    dm_correct_actions: int
    dm_total_actions: int
    slot_tp: int
    slot_fp: int
    slot_fn: int
    errors: List[str]
    # Splitter-specific
    is_multi_intent: bool = False
    splitter_detected_correctly: bool = True
    splitter_split_correctly: bool = True
    expected_intents_count: int = 1
    detected_intents_count: int = 1


class DualPipelineEvaluator:
    """
    Evaluates the pipeline with BOTH LLM-based and rule-based DM in a single run.
    NLU is called once per turn; the same output is fed to both DMs independently.
    Each DM maintains its own DialogueState so their flows evolve separately.
    """

    def __init__(self, pipe, use_splitter: bool = False):
        self.pipe = pipe
        self.use_splitter = use_splitter

        # Separate result lists per DM mode
        self.results_llm: List[DialogueEvalResult] = []
        self.results_rule: List[DialogueEvalResult] = []

        # Separate action-level counters per DM mode
        self.action_tp_llm: Dict[str, int] = defaultdict(int)
        self.action_fp_llm: Dict[str, int] = defaultdict(int)
        self.action_fn_llm: Dict[str, int] = defaultdict(int)

        self.action_tp_rule: Dict[str, int] = defaultdict(int)
        self.action_fp_rule: Dict[str, int] = defaultdict(int)
        self.action_fn_rule: Dict[str, int] = defaultdict(int)

        # Splitter metrics (shared – the splitter runs once)
        self.splitter_stats = {
            "total_multi_intent": 0,
            "detected_correctly": 0,
            "split_correctly": 0,
        }

    # -----------------------------------------------------------------
    # Single-dialogue evaluation (dual)
    # -----------------------------------------------------------------
    def run_dialogue_dual(
        self, dialogue: GeneratedDialogue
    ) -> Tuple[DialogueEvalResult, DialogueEvalResult]:
        """
        Run one dialogue through the pipeline.
        NLU is called ONCE per turn; the same nlu_output is
        deep-copied and given to both DMs independently.
        Returns (result_llm, result_rule).
        """
        # Independent states for the two DMs
        state_rule = DialogueState()
        state_llm = DialogueState()

        intent_queue = IntentQueue() if self.use_splitter else None

        # Shared history used as NLU context (uses rule-based NLG
        # responses so the context is deterministic and reproducible)
        shared_history: List[Dict[str, str]] = []

        # Per-DM tracking
        errors_llm: List[str] = []
        errors_rule: List[str] = []
        dm_correct_llm = 0
        dm_correct_rule = 0
        dm_total = 0

        # Slot metrics are shared (NLU output is the same)
        slot_tp, slot_fp, slot_fn = 0, 0, 0

        task_success_llm = True
        task_success_rule = True

        # Splitter variables (shared)
        is_multi_intent = dialogue.is_multi_intent
        splitter_detected_correctly = True
        splitter_split_correctly = True
        expected_intents_count = (
            len(dialogue.expected_intents) if dialogue.expected_intents else 1
        )
        detected_intents_count = 1

        for turn_idx, turn in enumerate(dialogue.turns):
            user_utterance = turn["user_utterance"]
            expected_action = turn.get("expected_action", "")
            provided_slots = turn.get("provided_slots", {})
            is_multi_intent_input = turn.get("is_multi_intent_input", False)

            # 0. Intent Splitter (shared, called once)
            current_input = user_utterance
            if self.use_splitter and is_multi_intent_input:
                self.splitter_stats["total_multi_intent"] += 1
                current_input, pending = split_intents(self.pipe, user_utterance)
                detected_multi = len(pending) > 0
                detected_intents_count = 1 + len(pending)

                if detected_multi == is_multi_intent:
                    splitter_detected_correctly = True
                    self.splitter_stats["detected_correctly"] += 1
                else:
                    splitter_detected_correctly = False
                    err = (f"Turn {turn_idx}: Splitter detection failed - "
                           f"expected multi-intent={is_multi_intent}, got {detected_multi}")
                    errors_llm.append(err)
                    errors_rule.append(err)

                if intent_queue and pending:
                    intent_queue.add(pending)

                if abs(detected_intents_count - expected_intents_count) <= 1:
                    splitter_split_correctly = True
                    self.splitter_stats["split_correctly"] += 1
                else:
                    splitter_split_correctly = False
                    err = (f"Turn {turn_idx}: Splitter split count wrong - "
                           f"expected {expected_intents_count}, got {detected_intents_count}")
                    errors_llm.append(err)
                    errors_rule.append(err)

            # 1. DST context — use rule-based state (deterministic)
            system_prompt = state_context(state_rule)

            # 2. NLU — called ONCE
            nlu_output = nlu_parse(
                self.pipe,
                current_input,
                system_prompt,
                dialogue_history=shared_history,
            )

            # Deep-copy so the two DMs don't cross-contaminate
            nlu_for_llm = copy.deepcopy(nlu_output)
            nlu_for_rule = copy.deepcopy(nlu_output)

            # 3a. DM — Rule-based
            action_rule = dm_decide_rule_based(state_rule, nlu_for_rule)

            # 3b. DM — LLM-based
            action_llm = dm_decide(
                state_llm, nlu_for_llm, current_input, llm_pipe=self.pipe
            )

            # --- Evaluate DM actions ---
            dm_total += 1
            expected_base, _ = parse_action(expected_action)

            # Rule-based
            actual_base_rule, _ = parse_action(action_rule)
            if expected_base == actual_base_rule:
                dm_correct_rule += 1
                self.action_tp_rule[expected_base] += 1
            else:
                self.action_fn_rule[expected_base] += 1
                self.action_fp_rule[actual_base_rule] += 1
                errors_rule.append(
                    f"Turn {turn_idx}: expected {expected_action}, got {action_rule}"
                )

            # LLM-based
            actual_base_llm, _ = parse_action(action_llm)
            if expected_base == actual_base_llm:
                dm_correct_llm += 1
                self.action_tp_llm[expected_base] += 1
            else:
                self.action_fn_llm[expected_base] += 1
                self.action_fp_llm[actual_base_llm] += 1
                errors_llm.append(
                    f"Turn {turn_idx}: expected {expected_action}, got {action_llm}"
                )

            # --- Evaluate slot extraction (shared NLU output) ---
            nlu_slots = nlu_output.get("slots", {}) or {}
            for slot, expected_val in provided_slots.items():
                if slot == "confirmation":
                    continue
                got_val = nlu_slots.get(slot)
                if got_val is not None and str(got_val).lower() == str(expected_val).lower():
                    slot_tp += 1
                elif got_val is not None:
                    slot_fp += 1
                    slot_fn += 1
                else:
                    slot_fn += 1

            # --- Shared history (rule-based NLG for deterministic context) ---
            shared_history.append({"role": "user", "content": user_utterance})
            try:
                response = nlg_generate(self.pipe, action_rule, state_rule)
                shared_history.append({"role": "assistant", "content": response})
            except Exception as e:
                shared_history.append(
                    {"role": "assistant", "content": f"[NLG Error: {e}]"}
                )

        # --- Task success (check final action per DM) ---
        def _check_task_success(
            state: DialogueState, dialogue: GeneratedDialogue
        ) -> bool:
            if not dialogue.turns:
                return True
            final_expected = dialogue.turns[-1].get("expected_action", "")
            final_expected_base, _ = parse_action(final_expected)
            completion_actions = [
                "COMPLETE_FLIGHT_BOOKING",
                "COMPLETE_ACCOMMODATION_BOOKING",
                "COMPLETE_ACTIVITY_BOOKING",
                "COMPARE_CITIES_RESULT",
                "GOODBYE",
            ]
            if final_expected_base in completion_actions:
                actual_final_base, _ = parse_action(state.last_action or "")
                if actual_final_base != final_expected_base:
                    return False
                booking = state.get_current_booking()
                if booking:
                    for slot, expected_value in dialogue.expected_final_slots.items():
                        actual_value = getattr(booking, slot, None)
                        if actual_value is None:
                            return False
                        if str(actual_value).lower() != str(expected_value).lower():
                            return False
            return True

        task_success_rule = _check_task_success(state_rule, dialogue)
        task_success_llm = _check_task_success(state_llm, dialogue)

        # Build result objects
        common = dict(
            dialogue_name=dialogue.name,
            intent=dialogue.intent,
            generation_method=dialogue.generation_method,
            num_turns=len(dialogue.turns),
            dm_total_actions=dm_total,
            slot_tp=slot_tp,
            slot_fp=slot_fp,
            slot_fn=slot_fn,
            is_multi_intent=is_multi_intent,
            splitter_detected_correctly=splitter_detected_correctly,
            splitter_split_correctly=splitter_split_correctly,
            expected_intents_count=expected_intents_count,
            detected_intents_count=detected_intents_count,
        )

        result_llm = DialogueEvalResult(
            **common,
            dm_mode="llm",
            task_success=task_success_llm,
            dm_correct_actions=dm_correct_llm,
            errors=errors_llm,
        )
        result_rule = DialogueEvalResult(
            **common,
            dm_mode="rule",
            task_success=task_success_rule,
            dm_correct_actions=dm_correct_rule,
            errors=errors_rule,
        )
        return result_llm, result_rule

    # -----------------------------------------------------------------
    # Evaluate all dialogues
    # -----------------------------------------------------------------
    def evaluate_all(
        self, dialogues: List[GeneratedDialogue]
    ) -> Tuple[PipelineMetrics, PipelineMetrics]:
        """Evaluate all dialogues. Returns (metrics_llm, metrics_rule)."""
        self.results_llm = []
        self.results_rule = []
        self.action_tp_llm = defaultdict(int)
        self.action_fp_llm = defaultdict(int)
        self.action_fn_llm = defaultdict(int)
        self.action_tp_rule = defaultdict(int)
        self.action_fp_rule = defaultdict(int)
        self.action_fn_rule = defaultdict(int)

        for dialogue in tqdm(dialogues, desc="Evaluating dialogues (dual DM)", unit="dlg"):
            r_llm, r_rule = self.run_dialogue_dual(dialogue)
            self.results_llm.append(r_llm)
            self.results_rule.append(r_rule)

        metrics_llm = self._compute_metrics(
            self.results_llm, self.action_tp_llm, self.action_fp_llm, self.action_fn_llm
        )
        metrics_rule = self._compute_metrics(
            self.results_rule, self.action_tp_rule, self.action_fp_rule, self.action_fn_rule
        )
        return metrics_llm, metrics_rule

    # -----------------------------------------------------------------
    # Metric computation (reused for both DMs)
    # -----------------------------------------------------------------
    @staticmethod
    def _compute_metrics(
        results: List[DialogueEvalResult],
        action_tp: Dict[str, int],
        action_fp: Dict[str, int],
        action_fn: Dict[str, int],
    ) -> PipelineMetrics:
        metrics = PipelineMetrics()
        if not results:
            return metrics

        # Task success
        metrics.task_success_rate = sum(1 for r in results if r.task_success) / len(results)

        # Slot metrics (shared NLU, identical for both)
        total_tp = sum(r.slot_tp for r in results)
        total_fp = sum(r.slot_fp for r in results)
        total_fn = sum(r.slot_fn for r in results)
        metrics.slot_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        metrics.slot_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        metrics.slot_f1 = (
            2 * metrics.slot_precision * metrics.slot_recall
            / (metrics.slot_precision + metrics.slot_recall)
            if (metrics.slot_precision + metrics.slot_recall) > 0 else 0
        )

        # DM accuracy
        total_dm_correct = sum(r.dm_correct_actions for r in results)
        total_dm_actions = sum(r.dm_total_actions for r in results)
        metrics.dm_accuracy = total_dm_correct / total_dm_actions if total_dm_actions > 0 else 0

        # DM action F1 (macro)
        action_f1s = []
        for action in set(action_tp.keys()) | set(action_fn.keys()):
            tp = action_tp[action]
            fp = action_fp[action]
            fn = action_fn[action]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            if tp + fn > 0:
                action_f1s.append(f1)
        metrics.dm_action_f1 = sum(action_f1s) / len(action_f1s) if action_f1s else 0

        # Efficiency
        metrics.avg_turns = sum(r.num_turns for r in results) / len(results)

        # Per-intent breakdown
        intent_results = defaultdict(list)
        for r in results:
            intent_results[r.intent].append(r)
        for intent, ires in intent_results.items():
            success_count = sum(1 for r in ires if r.task_success)
            metrics.per_intent_success[intent] = success_count / len(ires) if ires else 0
            dm_c = sum(r.dm_correct_actions for r in ires)
            dm_t = sum(r.dm_total_actions for r in ires)
            metrics.per_intent_dm_accuracy[intent] = dm_c / dm_t if dm_t > 0 else 0

        # Splitter metrics
        multi = [r for r in results if r.is_multi_intent]
        if multi:
            metrics.splitter_detection_accuracy = sum(1 for r in multi if r.splitter_detected_correctly) / len(multi)
            metrics.splitter_split_accuracy = sum(1 for r in multi if r.splitter_split_correctly) / len(multi)
            metrics.multi_intent_success_rate = sum(1 for r in multi if r.task_success) / len(multi)

        return metrics

    # -----------------------------------------------------------------
    # Printing
    # -----------------------------------------------------------------
    def _print_single_report(
        self,
        label: str,
        metrics: PipelineMetrics,
        results: List[DialogueEvalResult],
        action_tp: Dict[str, int],
        action_fp: Dict[str, int],
        action_fn: Dict[str, int],
    ):
        print(f"\n{'=' * 90}")
        print(f"PIPELINE EVALUATION — {label}")
        print("=" * 90)

        print(f"\n{'Overall Metrics':^40}")
        print("-" * 50)
        print(f"Task Success Rate:     {metrics.task_success_rate:.2%}")
        print(f"DM Action Accuracy:    {metrics.dm_accuracy:.2%}")
        print(f"DM Action Macro F1:    {metrics.dm_action_f1:.4f}")
        print(f"Slot Precision:        {metrics.slot_precision:.2%}")
        print(f"Slot Recall:           {metrics.slot_recall:.2%}")
        print(f"Slot F1:               {metrics.slot_f1:.4f}")
        print(f"Avg Turns/Dialogue:    {metrics.avg_turns:.1f}")

        # Splitter
        multi = [r for r in results if r.is_multi_intent]
        if multi and self.use_splitter:
            print(f"\n{'Intent Splitter Metrics':^40}")
            print("-" * 50)
            print(f"Detection Accuracy:    {metrics.splitter_detection_accuracy:.2%}")
            print(f"Split Accuracy:        {metrics.splitter_split_accuracy:.2%}")
            print(f"Multi-Intent Success:  {metrics.multi_intent_success_rate:.2%}")
            print(f"Multi-Intent Dialogues: {len(multi)}")

        print(f"\n{'Per-Intent Task Success':^40}")
        print("-" * 50)
        for intent in sorted(metrics.per_intent_success.keys()):
            print(f"{intent:<25} {metrics.per_intent_success[intent]:.2%}")

        print(f"\n{'Per-Intent DM Accuracy':^40}")
        print("-" * 50)
        for intent in sorted(metrics.per_intent_dm_accuracy.keys()):
            print(f"{intent:<25} {metrics.per_intent_dm_accuracy[intent]:.2%}")

        print(f"\n{'Per-Action F1 Scores':^40}")
        print("-" * 50)
        for action in sorted(action_tp.keys()):
            tp = action_tp[action]
            fp = action_fp[action]
            fn = action_fn[action]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            support = tp + fn
            print(f"{action:<30} P={precision:.2f} R={recall:.2f} F1={f1:.4f} (n={support})")

        # Failures
        failed = [r for r in results if not r.task_success]
        if failed:
            print(f"\n{'Failed Dialogues':^40}")
            print("-" * 50)
            for r in failed[:10]:
                print(f"\n{r.dialogue_name} ({r.intent})")
                for err in r.errors[:3]:
                    print(f"  - {err}")

        # By generation method
        print(f"\n{'By Generation Method':^40}")
        print("-" * 50)
        for method in sorted({r.generation_method for r in results}):
            mr = [r for r in results if r.generation_method == method]
            if mr:
                sr = sum(1 for r in mr if r.task_success) / len(mr)
                print(f"{method:<15} Success Rate: {sr:.2%} (n={len(mr)})")

    def print_comparative_report(
        self,
        metrics_llm: PipelineMetrics,
        metrics_rule: PipelineMetrics,
    ):
        """Print both individual reports + a side-by-side comparison."""
        # Individual reports
        self._print_single_report(
            "RULE-BASED DM",
            metrics_rule,
            self.results_rule,
            self.action_tp_rule,
            self.action_fp_rule,
            self.action_fn_rule,
        )
        self._print_single_report(
            "LLM-BASED DM",
            metrics_llm,
            self.results_llm,
            self.action_tp_llm,
            self.action_fp_llm,
            self.action_fn_llm,
        )

        # Comparative summary
        print(f"\n{'=' * 90}")
        print("COMPARATIVE SUMMARY  (same NLU input)")
        print("=" * 90)
        header = f"{'Metric':<30} {'Rule-Based':>12} {'LLM-Based':>12} {'Delta':>10}"
        print(header)
        print("-" * len(header))

        rows = [
            ("Task Success Rate", metrics_rule.task_success_rate, metrics_llm.task_success_rate),
            ("DM Action Accuracy", metrics_rule.dm_accuracy, metrics_llm.dm_accuracy),
            ("DM Action Macro F1", metrics_rule.dm_action_f1, metrics_llm.dm_action_f1),
            ("Slot Precision", metrics_rule.slot_precision, metrics_llm.slot_precision),
            ("Slot Recall", metrics_rule.slot_recall, metrics_llm.slot_recall),
            ("Slot F1", metrics_rule.slot_f1, metrics_llm.slot_f1),
            ("Avg Turns/Dialogue", metrics_rule.avg_turns, metrics_llm.avg_turns),
        ]
        for name, val_rule, val_llm in rows:
            delta = val_llm - val_rule
            sign = "+" if delta >= 0 else ""
            if name == "Avg Turns/Dialogue":
                print(f"{name:<30} {val_rule:>12.1f} {val_llm:>12.1f} {sign}{delta:>9.1f}")
            else:
                print(f"{name:<30} {val_rule:>11.2%} {val_llm:>11.2%} {sign}{delta:>9.2%}")

        # Per-intent comparison
        all_intents = sorted(
            set(metrics_rule.per_intent_success.keys())
            | set(metrics_llm.per_intent_success.keys())
        )
        if all_intents:
            print(f"\n{'Per-Intent Task Success Comparison':^60}")
            print("-" * 60)
            for intent in all_intents:
                vr = metrics_rule.per_intent_success.get(intent, 0)
                vl = metrics_llm.per_intent_success.get(intent, 0)
                d = vl - vr
                sign = "+" if d >= 0 else ""
                print(f"{intent:<25} {vr:>11.2%} {vl:>11.2%} {sign}{d:>9.2%}")


# =============================================================================
# HELPERS — serialise one set of results to a JSON file
# =============================================================================

def _save_results_json(
    filepath: str,
    metrics: PipelineMetrics,
    dm_mode: str,
    use_splitter: bool,
    results: List[DialogueEvalResult],
    num_dialogues: int,
) -> None:
    payload = {
        "metrics": {
            "task_success_rate": metrics.task_success_rate,
            "dm_accuracy": metrics.dm_accuracy,
            "dm_action_f1": metrics.dm_action_f1,
            "slot_precision": metrics.slot_precision,
            "slot_recall": metrics.slot_recall,
            "slot_f1": metrics.slot_f1,
            "avg_turns": metrics.avg_turns,
            "per_intent_success": metrics.per_intent_success,
            "per_intent_dm_accuracy": metrics.per_intent_dm_accuracy,
            "splitter_detection_accuracy": metrics.splitter_detection_accuracy,
            "splitter_split_accuracy": metrics.splitter_split_accuracy,
            "multi_intent_success_rate": metrics.multi_intent_success_rate,
        },
        "config": {
            "dm_mode": dm_mode,
            "use_splitter": use_splitter,
        },
        "dialogues_evaluated": num_dialogues,
        "detailed_results": [
            {
                "name": r.dialogue_name,
                "intent": r.intent,
                "method": r.generation_method,
                "dm_mode": r.dm_mode,
                "task_success": r.task_success,
                "num_turns": r.num_turns,
                "dm_accuracy": (
                    r.dm_correct_actions / r.dm_total_actions
                    if r.dm_total_actions > 0 else 0
                ),
                "is_multi_intent": r.is_multi_intent,
                "splitter_detected_correctly": r.splitter_detected_correctly,
                "splitter_split_correctly": r.splitter_split_correctly,
                "expected_intents_count": r.expected_intents_count,
                "detected_intents_count": r.detected_intents_count,
                "errors": r.errors,
            }
            for r in results
        ],
    }
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_pipeline_evaluation(use_splitter: bool = False):
    """
    Run the complete pipeline evaluation with BOTH DM modes
    in a single run (shared NLU).  Produces two output files.
    """
    print("Loading LLM...")
    pipe = make_llm()
    print("LLM loaded.\n")

    all_dialogues = get_static_dialogues()
    print(f"Loaded {len(all_dialogues)} static dialogues.")
    print(f"Intent Splitter: {'Enabled' if use_splitter else 'Disabled'}")
    print("DM modes: Rule-based + LLM-based (dual evaluation)\n")

    evaluator = DualPipelineEvaluator(pipe, use_splitter=use_splitter)
    metrics_llm, metrics_rule = evaluator.evaluate_all(all_dialogues)

    # Print comparative report to stdout
    evaluator.print_comparative_report(metrics_llm, metrics_rule)

    # Save results to two separate JSON files
    base_dir = os.path.dirname(__file__)
    splitter_tag = "splitter" if use_splitter else "no_splitter"

    rule_file = os.path.join(base_dir, f"pipeline_eval_results_rule_dm_{splitter_tag}.json")
    llm_file = os.path.join(base_dir, f"pipeline_eval_results_llm_dm_{splitter_tag}.json")

    _save_results_json(
        rule_file, metrics_rule, "rule", use_splitter,
        evaluator.results_rule, len(all_dialogues),
    )
    _save_results_json(
        llm_file, metrics_llm, "llm", use_splitter,
        evaluator.results_llm, len(all_dialogues),
    )

    print(f"\nResults saved to:")
    print(f"  Rule-based DM → {rule_file}")
    print(f"  LLM-based DM  → {llm_file}")

    return metrics_llm, metrics_rule, evaluator


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline Evaluation (Dual DM)")
    parser.add_argument(
        "--use-splitter", action="store_true",
        help="Enable intent splitter for multi-intent handling",
    )
    args = parser.parse_args()

    run_pipeline_evaluation(use_splitter=args.use_splitter)

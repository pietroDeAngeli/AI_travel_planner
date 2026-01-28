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

import json
import random
import copy
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from tqdm import tqdm

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm import make_llm
from nlu import nlu_parse, state_context
from dm import DialogueState, dm_decide, dm_decide_rule_based
from nlg import nlg_generate
from intent_splitter import split_intents, has_multiple_intents, IntentQueue
from schema import (
    INTENT_SCHEMAS, INTENT_SLOTS, INTENTS, DM_ACTIONS,
    ACTIVITY_CATEGORIES, BUDGET_LEVELS, parse_action
)


# --- Template-based dialogue generation ---

# Slot value pools for template filling
CITIES = [
    "Rome", "Paris", "London", "Barcelona", "Berlin", "Amsterdam",
    "Vienna", "Prague", "Madrid", "Milan", "Florence", "Venice",
    "Athens", "Lisbon", "Dublin", "Brussels", "Munich", "Zurich"
]

DATES_FUTURE = [
    (datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d")
    for d in [30, 45, 60, 75, 90, 120, 150, 180]
]

NUM_PASSENGERS = [1, 2, 3, 4, 5]
NUM_GUESTS = [1, 2, 3, 4]

ACTIVITY_TYPES = list(ACTIVITY_CATEGORIES.keys())


# --- User utterance templates ---

FLIGHT_TEMPLATES = {
    "full": [
        "Book a flight from {origin} to {destination} on {departure_date} for {num_passengers} passengers, {budget_level} budget",
        "I need to fly from {origin} to {destination} on {departure_date}, {num_passengers} people, {budget_level} budget please",
        "Flight from {origin} to {destination}, departing {departure_date}, {num_passengers} travelers, {budget_level} budget",
    ],
    "partial_dest_only": [
        "I want to fly to {destination}",
        "Book a flight to {destination}",
        "I need a flight to {destination}",
    ],
    "partial_origin_dest": [
        "Flight from {origin} to {destination}",
        "I want to fly from {origin} to {destination}",
        "Book a flight from {origin} to {destination} please",
    ],
    "slot_origin": [
        "From {origin}",
        "{origin}",
        "Departing from {origin}",
    ],
    "slot_destination": [
        "To {destination}",
        "{destination}",
        "Going to {destination}",
    ],
    "slot_date": [
        "On {departure_date}",
        "{departure_date}",
        "Departing {departure_date}",
    ],
    "slot_passengers": [
        "{num_passengers} passengers",
        "{num_passengers} people",
        "For {num_passengers}",
    ],
    "slot_budget": [
        "{budget_level} budget",
        "{budget_level}",
        "Budget is {budget_level}",
    ],
}

ACCOMMODATION_TEMPLATES = {
    "full": [
        "Book a hotel in {destination} from {check_in_date} to {check_out_date} for {num_guests} guests, {budget_level} budget",
        "I need accommodation in {destination}, check in {check_in_date}, check out {check_out_date}, {num_guests} guests, {budget_level}",
    ],
    "partial_dest_only": [
        "I need a hotel in {destination}",
        "Find me accommodation in {destination}",
        "Book a hotel in {destination}",
    ],
    "slot_destination": [
        "In {destination}",
        "{destination}",
    ],
    "slot_dates": [
        "From {check_in_date} to {check_out_date}",
        "Check in {check_in_date}, check out {check_out_date}",
    ],
    "slot_guests": [
        "{num_guests} guests",
        "For {num_guests} people",
    ],
    "slot_budget": [
        "{budget_level} budget",
        "{budget_level}",
    ],
}

ACTIVITY_TEMPLATES = {
    "full": [
        "Book a {activity_category} activity in {destination}, {budget_level} budget",
        "I want to do something {activity_category} in {destination}, budget {budget_level}",
    ],
    "partial_dest_only": [
        "I want to do something in {destination}",
        "Find activities in {destination}",
    ],
    "partial_dest_category": [
        "I want a {activity_category} activity in {destination}",
        "{activity_category} things to do in {destination}",
    ],
    "slot_destination": [
        "In {destination}",
        "{destination}",
    ],
    "slot_category": [
        "{activity_category}",
        "Something {activity_category}",
    ],
    "slot_budget": [
        "{budget_level}",
        "{budget_level} budget",
    ],
}

COMPARE_CITIES_TEMPLATES = {
    "full": [
        "Compare {city1} and {city2} for {activity_category}",
        "Which is better for {activity_category}, {city1} or {city2}?",
    ],
    "partial_cities_only": [
        "Compare {city1} and {city2}",
    ],
    "slot_city1": [
        "{city1}",
    ],
    "slot_city2": [
        "{city2}",
    ],
    "slot_category": [
        "{activity_category}",
        "For {activity_category}",
    ],
}

CONFIRMATION_TEMPLATES = {
    "yes": ["Yes", "Yes please", "Confirm", "That's correct", "Looks good", "Perfect"],
    "no": ["No", "No, I want to change something", "Not quite", "Change it"],
}

# Multi-intent templates for testing the intent splitter
MULTI_INTENT_TEMPLATES = [
    {
        "template": "Book a flight to {destination} and find me a hotel there",
        "intents": ["BOOK_FLIGHT", "BOOK_ACCOMMODATION"],
        "slots_per_intent": [
            {"destination": "{destination}"},
            {"destination": "{destination}"},
        ],
    },
    {
        "template": "I need to fly to {destination}. Also, can you show me some {activity_category} activities?",
        "intents": ["BOOK_FLIGHT", "BOOK_ACTIVITY"],
        "slots_per_intent": [
            {"destination": "{destination}"},
            {"destination": "{destination}", "activity_category": "{activity_category}"},
        ],
    },
    {
        "template": "Book a hotel in {destination} and I'd like to do something {activity_category} as well",
        "intents": ["BOOK_ACCOMMODATION", "BOOK_ACTIVITY"],
        "slots_per_intent": [
            {"destination": "{destination}"},
            {"destination": "{destination}", "activity_category": "{activity_category}"},
        ],
    },
    {
        "template": "I want to go to {destination}. Book flights, find a hotel, and show me {activity_category} things to do",
        "intents": ["BOOK_FLIGHT", "BOOK_ACCOMMODATION", "BOOK_ACTIVITY"],
        "slots_per_intent": [
            {"destination": "{destination}"},
            {"destination": "{destination}"},
            {"destination": "{destination}", "activity_category": "{activity_category}"},
        ],
    },
]

END_DIALOGUE_TEMPLATES = [
    "Goodbye", "Thanks, bye", "That's all", "I'm done", "Thank you, goodbye"
]


@dataclass
class GeneratedDialogue:
    """A synthetically generated dialogue for testing."""
    name: str
    intent: str
    generation_method: str  # "template", "llm", or "multi_intent"
    turns: List[Dict[str, Any]]  # Each turn: {user_utterance, expected_slots, expected_action}
    expected_final_slots: Dict[str, Any]
    expected_task_success: bool
    is_multi_intent: bool = False  # True if this is a multi-intent dialogue
    expected_intents: List[str] = field(default_factory=list)  # For multi-intent dialogues


def generate_slot_values(intent: str) -> Dict[str, Any]:
    """Generate random slot values for an intent."""
    if intent == "BOOK_FLIGHT":
        origin, dest = random.sample(CITIES, 2)
        return {
            "origin": origin,
            "destination": dest,
            "departure_date": random.choice(DATES_FUTURE),
            "return_date": None,  # Optional
            "num_passengers": random.choice(NUM_PASSENGERS),
            "budget_level": random.choice(BUDGET_LEVELS),
        }
    elif intent == "BOOK_ACCOMMODATION":
        check_in = random.choice(DATES_FUTURE)
        check_in_dt = datetime.strptime(check_in, "%Y-%m-%d")
        check_out = (check_in_dt + timedelta(days=random.randint(2, 7))).strftime("%Y-%m-%d")
        return {
            "destination": random.choice(CITIES),
            "check_in_date": check_in,
            "check_out_date": check_out,
            "num_guests": random.choice(NUM_GUESTS),
            "budget_level": random.choice(BUDGET_LEVELS),
        }
    elif intent == "BOOK_ACTIVITY":
        return {
            "destination": random.choice(CITIES),
            "activity_category": random.choice(ACTIVITY_TYPES),
            "budget_level": random.choice(BUDGET_LEVELS),
        }
    elif intent == "COMPARE_CITIES":
        city1, city2 = random.sample(CITIES, 2)
        return {
            "city1": city1,
            "city2": city2,
            "activity_category": random.choice(ACTIVITY_TYPES),
        }
    return {}


def generate_template_dialogue_full(intent: str, idx: int) -> GeneratedDialogue:
    """Generate a dialogue where user provides all info at once, then confirms."""
    slots = generate_slot_values(intent)
    
    # Select template and fill
    if intent == "BOOK_FLIGHT":
        template = random.choice(FLIGHT_TEMPLATES["full"])
        utterance = template.format(**slots)
        required_slots = {k: v for k, v in slots.items() if v is not None and k != "return_date"}
    elif intent == "BOOK_ACCOMMODATION":
        template = random.choice(ACCOMMODATION_TEMPLATES["full"])
        utterance = template.format(**slots)
        required_slots = slots
    elif intent == "BOOK_ACTIVITY":
        template = random.choice(ACTIVITY_TEMPLATES["full"])
        utterance = template.format(**slots)
        required_slots = slots
    elif intent == "COMPARE_CITIES":
        template = random.choice(COMPARE_CITIES_TEMPLATES["full"])
        utterance = template.format(**slots)
        required_slots = slots
    else:
        return None
    
    # Build turns
    turns = [
        {
            "user_utterance": utterance,
            "provided_slots": required_slots,
            "expected_action": "ASK_CONFIRMATION" if intent != "COMPARE_CITIES" else "COMPARE_CITIES_RESULT",
        }
    ]
    
    # Add confirmation turn for booking intents
    if intent in ["BOOK_FLIGHT", "BOOK_ACCOMMODATION", "BOOK_ACTIVITY"]:
        confirm_utterance = random.choice(CONFIRMATION_TEMPLATES["yes"])
        completion_action = {
            "BOOK_FLIGHT": "COMPLETE_FLIGHT_BOOKING",
            "BOOK_ACCOMMODATION": "COMPLETE_ACCOMMODATION_BOOKING",
            "BOOK_ACTIVITY": "COMPLETE_ACTIVITY_BOOKING",
        }[intent]
        turns.append({
            "user_utterance": confirm_utterance,
            "provided_slots": {"confirmation": "yes"},
            "expected_action": completion_action,
        })
    
    return GeneratedDialogue(
        name=f"template_{intent.lower()}_{idx}_full",
        intent=intent,
        generation_method="template",
        turns=turns,
        expected_final_slots=required_slots,
        expected_task_success=True,
    )


def generate_template_dialogue_incremental(intent: str, idx: int) -> GeneratedDialogue:
    """Generate a dialogue with incremental slot filling."""
    slots = generate_slot_values(intent)
    turns = []
    accumulated_slots = {}
    
    if intent == "BOOK_FLIGHT":
        # Turn 1: destination only
        utterance = random.choice(FLIGHT_TEMPLATES["partial_dest_only"]).format(destination=slots["destination"])
        accumulated_slots["destination"] = slots["destination"]
        turns.append({
            "user_utterance": utterance,
            "provided_slots": {"destination": slots["destination"]},
            "expected_action": "REQUEST_MISSING_SLOT(origin)",
        })
        
        # Turn 2: origin
        utterance = random.choice(FLIGHT_TEMPLATES["slot_origin"]).format(origin=slots["origin"])
        accumulated_slots["origin"] = slots["origin"]
        turns.append({
            "user_utterance": utterance,
            "provided_slots": {"origin": slots["origin"]},
            "expected_action": "REQUEST_MISSING_SLOT(departure_date)",
        })
        
        # Turn 3: date
        utterance = random.choice(FLIGHT_TEMPLATES["slot_date"]).format(departure_date=slots["departure_date"])
        accumulated_slots["departure_date"] = slots["departure_date"]
        turns.append({
            "user_utterance": utterance,
            "provided_slots": {"departure_date": slots["departure_date"]},
            "expected_action": "REQUEST_MISSING_SLOT(num_passengers)",
        })
        
        # Turn 4: passengers
        utterance = random.choice(FLIGHT_TEMPLATES["slot_passengers"]).format(num_passengers=slots["num_passengers"])
        accumulated_slots["num_passengers"] = slots["num_passengers"]
        turns.append({
            "user_utterance": utterance,
            "provided_slots": {"num_passengers": slots["num_passengers"]},
            "expected_action": "REQUEST_MISSING_SLOT(budget_level)",
        })
        
        # Turn 5: budget
        utterance = random.choice(FLIGHT_TEMPLATES["slot_budget"]).format(budget_level=slots["budget_level"])
        accumulated_slots["budget_level"] = slots["budget_level"]
        turns.append({
            "user_utterance": utterance,
            "provided_slots": {"budget_level": slots["budget_level"]},
            "expected_action": "ASK_CONFIRMATION",
        })
        
        # Turn 6: confirm
        turns.append({
            "user_utterance": random.choice(CONFIRMATION_TEMPLATES["yes"]),
            "provided_slots": {"confirmation": "yes"},
            "expected_action": "COMPLETE_FLIGHT_BOOKING",
        })
        
    elif intent == "BOOK_ACCOMMODATION":
        # Turn 1: destination
        utterance = random.choice(ACCOMMODATION_TEMPLATES["partial_dest_only"]).format(destination=slots["destination"])
        accumulated_slots["destination"] = slots["destination"]
        turns.append({
            "user_utterance": utterance,
            "provided_slots": {"destination": slots["destination"]},
            "expected_action": "REQUEST_MISSING_SLOT(check_in_date)",
        })
        
        # Turn 2: dates
        utterance = random.choice(ACCOMMODATION_TEMPLATES["slot_dates"]).format(
            check_in_date=slots["check_in_date"],
            check_out_date=slots["check_out_date"]
        )
        accumulated_slots["check_in_date"] = slots["check_in_date"]
        accumulated_slots["check_out_date"] = slots["check_out_date"]
        turns.append({
            "user_utterance": utterance,
            "provided_slots": {"check_in_date": slots["check_in_date"], "check_out_date": slots["check_out_date"]},
            "expected_action": "REQUEST_MISSING_SLOT(num_guests)",
        })
        
        # Turn 3: guests
        utterance = random.choice(ACCOMMODATION_TEMPLATES["slot_guests"]).format(num_guests=slots["num_guests"])
        accumulated_slots["num_guests"] = slots["num_guests"]
        turns.append({
            "user_utterance": utterance,
            "provided_slots": {"num_guests": slots["num_guests"]},
            "expected_action": "REQUEST_MISSING_SLOT(budget_level)",
        })
        
        # Turn 4: budget
        utterance = random.choice(ACCOMMODATION_TEMPLATES["slot_budget"]).format(budget_level=slots["budget_level"])
        accumulated_slots["budget_level"] = slots["budget_level"]
        turns.append({
            "user_utterance": utterance,
            "provided_slots": {"budget_level": slots["budget_level"]},
            "expected_action": "ASK_CONFIRMATION",
        })
        
        # Turn 5: confirm
        turns.append({
            "user_utterance": random.choice(CONFIRMATION_TEMPLATES["yes"]),
            "provided_slots": {"confirmation": "yes"},
            "expected_action": "COMPLETE_ACCOMMODATION_BOOKING",
        })
        
    elif intent == "BOOK_ACTIVITY":
        # Turn 1: destination
        utterance = random.choice(ACTIVITY_TEMPLATES["partial_dest_only"]).format(destination=slots["destination"])
        accumulated_slots["destination"] = slots["destination"]
        turns.append({
            "user_utterance": utterance,
            "provided_slots": {"destination": slots["destination"]},
            "expected_action": "REQUEST_MISSING_SLOT(activity_category)",
        })
        
        # Turn 2: category
        utterance = random.choice(ACTIVITY_TEMPLATES["slot_category"]).format(activity_category=slots["activity_category"])
        accumulated_slots["activity_category"] = slots["activity_category"]
        turns.append({
            "user_utterance": utterance,
            "provided_slots": {"activity_category": slots["activity_category"]},
            "expected_action": "REQUEST_MISSING_SLOT(budget_level)",
        })
        
        # Turn 3: budget
        utterance = random.choice(ACTIVITY_TEMPLATES["slot_budget"]).format(budget_level=slots["budget_level"])
        accumulated_slots["budget_level"] = slots["budget_level"]
        turns.append({
            "user_utterance": utterance,
            "provided_slots": {"budget_level": slots["budget_level"]},
            "expected_action": "ASK_CONFIRMATION",
        })
        
        # Turn 4: confirm
        turns.append({
            "user_utterance": random.choice(CONFIRMATION_TEMPLATES["yes"]),
            "provided_slots": {"confirmation": "yes"},
            "expected_action": "COMPLETE_ACTIVITY_BOOKING",
        })
    
    elif intent == "COMPARE_CITIES":
        # Turn 1: partial
        utterance = random.choice(COMPARE_CITIES_TEMPLATES["partial_cities_only"]).format(
            city1=slots["city1"], city2=slots["city2"]
        )
        accumulated_slots["city1"] = slots["city1"]
        accumulated_slots["city2"] = slots["city2"]
        turns.append({
            "user_utterance": utterance,
            "provided_slots": {"city1": slots["city1"], "city2": slots["city2"]},
            "expected_action": "REQUEST_MISSING_SLOT(activity_category)",
        })
        
        # Turn 2: category
        utterance = random.choice(COMPARE_CITIES_TEMPLATES["slot_category"]).format(
            activity_category=slots["activity_category"]
        )
        accumulated_slots["activity_category"] = slots["activity_category"]
        turns.append({
            "user_utterance": utterance,
            "provided_slots": {"activity_category": slots["activity_category"]},
            "expected_action": "COMPARE_CITIES_RESULT",
        })
    
    final_slots = {k: v for k, v in slots.items() if v is not None and k != "return_date"}
    
    return GeneratedDialogue(
        name=f"template_{intent.lower()}_{idx}_incremental",
        intent=intent,
        generation_method="template",
        turns=turns,
        expected_final_slots=final_slots,
        expected_task_success=True,
    )


def generate_template_dialogues(num_per_intent: int = 5) -> List[GeneratedDialogue]:
    """Generate template-based dialogues for all intents."""
    dialogues = []
    
    for intent in ["BOOK_FLIGHT", "BOOK_ACCOMMODATION", "BOOK_ACTIVITY", "COMPARE_CITIES"]:
        for i in range(num_per_intent):
            # Half full, half incremental
            if i % 2 == 0:
                d = generate_template_dialogue_full(intent, i)
            else:
                d = generate_template_dialogue_incremental(intent, i)
            if d:
                dialogues.append(d)
    
    # Add some END_DIALOGUE and OOD cases
    dialogues.append(GeneratedDialogue(
        name="template_end_dialogue_0",
        intent="END_DIALOGUE",
        generation_method="template",
        turns=[{"user_utterance": random.choice(END_DIALOGUE_TEMPLATES), "provided_slots": {}, "expected_action": "GOODBYE"}],
        expected_final_slots={},
        expected_task_success=True,
    ))
    
    dialogues.append(GeneratedDialogue(
        name="template_ood_0",
        intent="OOD",
        generation_method="template",
        turns=[{"user_utterance": "What's the weather like today?", "provided_slots": {}, "expected_action": "ASK_CLARIFICATION"}],
        expected_final_slots={},
        expected_task_success=True,
    ))
    
    return dialogues


def generate_multi_intent_dialogues(num_dialogues: int = 4) -> List[GeneratedDialogue]:
    """Generate dialogues with multiple intents for testing the intent splitter."""
    dialogues = []
    
    for idx, template_info in enumerate(MULTI_INTENT_TEMPLATES[:num_dialogues]):
        # Generate slot values
        destination = random.choice(CITIES)
        activity_category = random.choice(ACTIVITY_TYPES)
        
        # Format the multi-intent utterance
        utterance = template_info["template"].format(
            destination=destination,
            activity_category=activity_category,
        )
        
        intents = template_info["intents"]
        
        # Create turns - just the initial multi-intent utterance
        # The splitter should handle breaking it apart
        turns = [
            {
                "user_utterance": utterance,
                "provided_slots": {"destination": destination},
                "expected_action": "REQUEST_MISSING_SLOT",  # Will request more info for first intent
                "is_multi_intent_input": True,
            }
        ]
        
        dialogues.append(GeneratedDialogue(
            name=f"multi_intent_{idx}_{len(intents)}_intents",
            intent=intents[0],  # Primary intent
            generation_method="multi_intent",
            turns=turns,
            expected_final_slots={"destination": destination},
            expected_task_success=True,
            is_multi_intent=True,
            expected_intents=intents,
        ))
    
    return dialogues


# --- LLM-based dialogue generation ---

LLM_GENERATION_PROMPT = """You are a data generator for a travel booking dialogue system.
Generate a realistic user dialogue for the following scenario.

Task: {intent_description}
Required slots: {slots}
Slot values to use: {slot_values}

Generate a natural multi-turn dialogue where the user gradually provides information.
Output format - return a JSON array of turns, each with:
- "user_utterance": what the user says
- "provided_slots": dict of slots provided in this turn

Example for BOOK_FLIGHT:

# --- Evaluation functions ---
  {{"user_utterance": "2 people", "provided_slots": {{"num_passengers": 2}}}},
  {{"user_utterance": "Medium budget", "provided_slots": {{"budget_level": "medium"}}}},
  {{"user_utterance": "Yes, confirm", "provided_slots": {{"confirmation": "yes"}}}}
]

Now generate for:
Intent: {intent}
Slots: {slot_values}

Return ONLY the JSON array, no other text."""


def generate_llm_dialogue(pipe, intent: str, idx: int) -> Optional[GeneratedDialogue]:
    """Generate a dialogue using the LLM with few-shot prompting."""
    slots = generate_slot_values(intent)
    
    intent_descriptions = {
        "BOOK_FLIGHT": "User wants to book a flight",
        "BOOK_ACCOMMODATION": "User wants to book a hotel/accommodation",
        "BOOK_ACTIVITY": "User wants to book an activity or tour",
        "COMPARE_CITIES": "User wants to compare two cities for travel",
    }
    
    if intent not in intent_descriptions:
        return None
    
    slot_names = INTENT_SLOTS.get(intent, [])
    
    prompt = LLM_GENERATION_PROMPT.format(
        intent_description=intent_descriptions[intent],
        slots=slot_names,
        slot_values=json.dumps(slots),
        intent=intent,
    )
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates dialogue data in JSON format."},
        {"role": "user", "content": prompt},
    ]
    
    try:
        outputs = pipe(
            messages,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            pad_token_id=pipe.tokenizer.pad_token_id,
        )
        
        response = outputs[0]["generated_text"][-1]["content"]
        
        # Parse JSON from response
        # Try to find JSON array in the response
        import re
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            turns_data = json.loads(json_match.group())
        else:
            print(f"Warning: Could not parse LLM response for {intent}")
            return None
        
        # Convert to our format and add expected actions
        turns = []
        accumulated = {}
        required_slots = set(INTENT_SLOTS.get(intent, []))
        
        for turn_data in turns_data:
            user_utterance = turn_data.get("user_utterance", "")
            provided_slots = turn_data.get("provided_slots", {})
            
            # Update accumulated slots
            accumulated.update({k: v for k, v in provided_slots.items() if v is not None})
            
            # Determine expected action based on accumulated slots
            if provided_slots.get("confirmation") == "yes":
                expected_action = {
                    "BOOK_FLIGHT": "COMPLETE_FLIGHT_BOOKING",
                    "BOOK_ACCOMMODATION": "COMPLETE_ACCOMMODATION_BOOKING",
                    "BOOK_ACTIVITY": "COMPLETE_ACTIVITY_BOOKING",
                }.get(intent, "ASK_CONFIRMATION")
            elif provided_slots.get("confirmation") == "no":
                expected_action = "REQUEST_SLOT_CHANGE"
            else:
                # Check what's still missing
                filled = set(k for k, v in accumulated.items() if v is not None and k != "confirmation")
                missing = required_slots - filled
                
                if not missing:
                    if intent == "COMPARE_CITIES":
                        expected_action = "COMPARE_CITIES_RESULT"
                    else:
                        expected_action = "ASK_CONFIRMATION"
                else:
                    first_missing = sorted(missing)[0]  # Deterministic order
                    expected_action = f"REQUEST_MISSING_SLOT({first_missing})"
            
            turns.append({
                "user_utterance": user_utterance,
                "provided_slots": provided_slots,
                "expected_action": expected_action,
            })
        
        final_slots = {k: v for k, v in slots.items() if v is not None and k != "return_date"}
        
        return GeneratedDialogue(
            name=f"llm_{intent.lower()}_{idx}",
            intent=intent,
            generation_method="llm",
            turns=turns,
            expected_final_slots=final_slots,
            expected_task_success=True,
        )
        
    except Exception as e:
        print(f"Error generating LLM dialogue for {intent}: {e}")
        return None


def generate_llm_dialogues(pipe, num_per_intent: int = 3) -> List[GeneratedDialogue]:
    """Generate LLM-based dialogues for all booking intents."""
    dialogues = []
    
    for intent in ["BOOK_FLIGHT", "BOOK_ACCOMMODATION", "BOOK_ACTIVITY", "COMPARE_CITIES"]:
        for i in range(num_per_intent):
            d = generate_llm_dialogue(pipe, intent, i)
            if d:
                dialogues.append(d)
    
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


class PipelineEvaluator:
    """Evaluates the complete dialogue pipeline."""
    
    def __init__(self, pipe, use_llm_dm: bool = True, use_splitter: bool = False):
        self.pipe = pipe
        self.use_llm_dm = use_llm_dm
        self.use_splitter = use_splitter
        self.results: List[DialogueEvalResult] = []
        
        # Action-level metrics
        self.action_tp: Dict[str, int] = defaultdict(int)
        self.action_fp: Dict[str, int] = defaultdict(int)
        self.action_fn: Dict[str, int] = defaultdict(int)
        
        # Splitter metrics tracking
        self.splitter_stats = {
            "total_multi_intent": 0,
            "detected_correctly": 0,
            "split_correctly": 0,
        }
    
    def run_dialogue(self, dialogue: GeneratedDialogue) -> DialogueEvalResult:
        """Run a single dialogue through the pipeline and evaluate."""
        state = DialogueState()
        intent_queue = IntentQueue() if self.use_splitter else None
        history = []
        errors = []
        dm_correct = 0
        dm_total = 0
        slot_tp, slot_fp, slot_fn = 0, 0, 0
        task_success = True
        
        # Splitter evaluation variables
        is_multi_intent = dialogue.is_multi_intent
        splitter_detected_correctly = True
        splitter_split_correctly = True
        expected_intents_count = len(dialogue.expected_intents) if dialogue.expected_intents else 1
        detected_intents_count = 1
        
        for turn_idx, turn in enumerate(dialogue.turns):
            user_utterance = turn["user_utterance"]
            expected_action = turn.get("expected_action", "")
            provided_slots = turn.get("provided_slots", {})
            is_multi_intent_input = turn.get("is_multi_intent_input", False)
            
            # 0. Intent Splitter (if enabled)
            current_input = user_utterance
            if self.use_splitter and is_multi_intent_input:
                # Track multi-intent detection
                self.splitter_stats["total_multi_intent"] += 1
                
                # Check if splitter detects it as multi-intent
                detected_multi = has_multiple_intents(self.pipe, user_utterance)
                if detected_multi == is_multi_intent:
                    splitter_detected_correctly = True
                    self.splitter_stats["detected_correctly"] += 1
                else:
                    splitter_detected_correctly = False
                    errors.append(f"Turn {turn_idx}: Splitter detection failed - expected multi-intent={is_multi_intent}, got {detected_multi}")
                
                # Attempt to split
                if detected_multi:
                    current_input, pending = split_intents(self.pipe, user_utterance)
                    detected_intents_count = 1 + len(pending)
                    
                    # Add pending to queue
                    if intent_queue and pending:
                        intent_queue.add(pending)
                    
                    # Check if split count is correct (allow ±1 tolerance)
                    if abs(detected_intents_count - expected_intents_count) <= 1:
                        splitter_split_correctly = True
                        self.splitter_stats["split_correctly"] += 1
                    else:
                        splitter_split_correctly = False
                        errors.append(f"Turn {turn_idx}: Splitter split count wrong - expected {expected_intents_count}, got {detected_intents_count}")
            
            # 1. DST - Generate context
            system_prompt = state_context(state)
            
            # 2. NLU - Parse user input (use current_input if splitter is active)
            nlu_output = nlu_parse(
                self.pipe,
                current_input,
                system_prompt,
                dialogue_history=history
            )
            
            # 3. DM - Decide action
            if self.use_llm_dm:
                action = dm_decide(state, nlu_output, current_input, llm_pipe=self.pipe)
            else:
                action = dm_decide_rule_based(state, nlu_output, current_input)
            
            # Evaluate DM action
            dm_total += 1
            expected_base, _ = parse_action(expected_action)
            actual_base, _ = parse_action(action)
            
            if expected_base == actual_base:
                dm_correct += 1
                self.action_tp[expected_base] += 1
            else:
                self.action_fn[expected_base] += 1
                self.action_fp[actual_base] += 1
                errors.append(f"Turn {turn_idx}: expected {expected_action}, got {action}")
            
            # Evaluate slot extraction
            nlu_slots = nlu_output.get("slots", {})
            for slot, expected_val in provided_slots.items():
                if slot == "confirmation":
                    continue
                got_val = nlu_slots.get(slot)
                if got_val is not None and str(got_val).lower() == str(expected_val).lower():
                    slot_tp += 1
                elif got_val is not None:
                    slot_fp += 1
                    slot_fn += 1  # Wrong value
                else:
                    slot_fn += 1  # Missing
            
            # Check for extra slots in NLU output
            for slot, val in nlu_slots.items():
                if slot not in provided_slots and val is not None and slot != "confirmation":
                    # Could be carryover or hallucination - count as FP for strict eval
                    pass
            
            # Update history
            history.append({"role": "user", "content": user_utterance})
            
            # Generate response (optional - for completeness)
            try:
                response = nlg_generate(self.pipe, action, state)
                history.append({"role": "assistant", "content": response})
            except Exception as e:
                history.append({"role": "assistant", "content": f"[NLG Error: {e}]"})
            
            # Check if task failed
            if actual_base == "ASK_CLARIFICATION" and expected_base != "ASK_CLARIFICATION":
                # Pipeline got confused
                pass
        
        # Check final action for task success
        if dialogue.turns:
            final_expected = dialogue.turns[-1].get("expected_action", "")
            final_expected_base, _ = parse_action(final_expected)
            
            # Task is successful if we reached the expected final action
            completion_actions = ["COMPLETE_FLIGHT_BOOKING", "COMPLETE_ACCOMMODATION_BOOKING", 
                                   "COMPLETE_ACTIVITY_BOOKING", "COMPARE_CITIES_RESULT", "GOODBYE"]
            
            if final_expected_base in completion_actions:
                # Check if state.last_action matches
                actual_final_base, _ = parse_action(state.last_action or "")
                task_success = (actual_final_base == final_expected_base)
        
        return DialogueEvalResult(
            dialogue_name=dialogue.name,
            intent=dialogue.intent,
            generation_method=dialogue.generation_method,
            task_success=task_success,
            num_turns=len(dialogue.turns),
            dm_correct_actions=dm_correct,
            dm_total_actions=dm_total,
            slot_tp=slot_tp,
            slot_fp=slot_fp,
            slot_fn=slot_fn,
            errors=errors,
            is_multi_intent=is_multi_intent,
            splitter_detected_correctly=splitter_detected_correctly,
            splitter_split_correctly=splitter_split_correctly,
            expected_intents_count=expected_intents_count,
            detected_intents_count=detected_intents_count,
        )
    
    def evaluate_all(self, dialogues: List[GeneratedDialogue]) -> PipelineMetrics:
        """Evaluate all dialogues and compute aggregate metrics."""
        self.results = []
        self.action_tp = defaultdict(int)
        self.action_fp = defaultdict(int)
        self.action_fn = defaultdict(int)
        
        for dialogue in tqdm(dialogues, desc="Evaluating dialogues", unit="dialogue"):
            result = self.run_dialogue(dialogue)
            self.results.append(result)
        
        # Compute aggregate metrics
        metrics = PipelineMetrics()
        
        # Task success rate
        successful = sum(1 for r in self.results if r.task_success)
        metrics.task_success_rate = successful / len(self.results) if self.results else 0
        
        # Slot metrics
        total_tp = sum(r.slot_tp for r in self.results)
        total_fp = sum(r.slot_fp for r in self.results)
        total_fn = sum(r.slot_fn for r in self.results)
        
        metrics.slot_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        metrics.slot_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        metrics.slot_f1 = 2 * metrics.slot_precision * metrics.slot_recall / (metrics.slot_precision + metrics.slot_recall) if (metrics.slot_precision + metrics.slot_recall) > 0 else 0
        
        # DM accuracy
        total_dm_correct = sum(r.dm_correct_actions for r in self.results)
        total_dm_actions = sum(r.dm_total_actions for r in self.results)
        metrics.dm_accuracy = total_dm_correct / total_dm_actions if total_dm_actions > 0 else 0
        
        # DM action F1 (macro)
        action_f1s = []
        for action in set(self.action_tp.keys()) | set(self.action_fn.keys()):
            tp = self.action_tp[action]
            fp = self.action_fp[action]
            fn = self.action_fn[action]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            if tp + fn > 0:  # Only count actions that appear in ground truth
                action_f1s.append(f1)
        metrics.dm_action_f1 = sum(action_f1s) / len(action_f1s) if action_f1s else 0
        
        # Efficiency
        metrics.avg_turns = sum(r.num_turns for r in self.results) / len(self.results) if self.results else 0
        
        # Per-intent breakdown
        intent_results = defaultdict(list)
        for r in self.results:
            intent_results[r.intent].append(r)
        
        for intent, results in intent_results.items():
            success_count = sum(1 for r in results if r.task_success)
            metrics.per_intent_success[intent] = success_count / len(results) if results else 0
            
            dm_correct = sum(r.dm_correct_actions for r in results)
            dm_total = sum(r.dm_total_actions for r in results)
            metrics.per_intent_dm_accuracy[intent] = dm_correct / dm_total if dm_total > 0 else 0
        
        # Intent Splitter metrics
        multi_intent_results = [r for r in self.results if r.is_multi_intent]
        if multi_intent_results:
            metrics.splitter_detection_accuracy = sum(1 for r in multi_intent_results if r.splitter_detected_correctly) / len(multi_intent_results)
            metrics.splitter_split_accuracy = sum(1 for r in multi_intent_results if r.splitter_split_correctly) / len(multi_intent_results)
            metrics.multi_intent_success_rate = sum(1 for r in multi_intent_results if r.task_success) / len(multi_intent_results)
        
        return metrics
    
    def print_report(self, metrics: PipelineMetrics):
        """Print a detailed evaluation report."""
        print("\n" + "=" * 90)
        print("PIPELINE EVALUATION REPORT")
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
        
        # Intent Splitter metrics (if applicable)
        multi_intent_results = [r for r in self.results if r.is_multi_intent]
        if multi_intent_results and self.use_splitter:
            print(f"\n{'Intent Splitter Metrics':^40}")
            print("-" * 50)
            print(f"Detection Accuracy:    {metrics.splitter_detection_accuracy:.2%}")
            print(f"Split Accuracy:        {metrics.splitter_split_accuracy:.2%}")
            print(f"Multi-Intent Success:  {metrics.multi_intent_success_rate:.2%}")
            print(f"Multi-Intent Dialogues: {len(multi_intent_results)}")
        
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
        for action in sorted(self.action_tp.keys()):
            tp = self.action_tp[action]
            fp = self.action_fp[action]
            fn = self.action_fn[action]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            support = tp + fn
            print(f"{action:<30} P={precision:.2f} R={recall:.2f} F1={f1:.4f} (n={support})")
        
        # Print failures
        failed = [r for r in self.results if not r.task_success]
        if failed:
            print(f"\n{'Failed Dialogues':^40}")
            print("-" * 50)
            for r in failed[:10]:  # Show first 10
                print(f"\n{r.dialogue_name} ({r.intent})")
                for err in r.errors[:3]:  # Show first 3 errors
                    print(f"  - {err}")
        
        # By generation method
        print(f"\n{'By Generation Method':^40}")
        print("-" * 50)
        for method in ["template", "llm"]:
            method_results = [r for r in self.results if r.generation_method == method]
            if method_results:
                success_rate = sum(1 for r in method_results if r.task_success) / len(method_results)
                print(f"{method:<15} Success Rate: {success_rate:.2%} (n={len(method_results)})")


def run_pipeline_evaluation(
    num_template_dialogues: int = 5,
    use_llm_dm: bool = False,
    use_splitter: bool = False,
    include_multi_intent: bool = True,
    seed: int = 42
):
    """Run the complete pipeline evaluation."""
    random.seed(seed)
    
    print("Loading LLM...")
    pipe = make_llm()
    print("LLM loaded.\n")
    
    # Generate dialogues (template-based only)
    print("Generating template-based dialogues...")
    template_dialogues = generate_template_dialogues(num_template_dialogues)
    print(f"Generated {len(template_dialogues)} template dialogues.\n")
    
    all_dialogues = template_dialogues
    
    # Generate multi-intent dialogues if enabled
    if include_multi_intent:
        print("Generating multi-intent dialogues...")
        multi_intent_dialogues = generate_multi_intent_dialogues(num_dialogues=4)
        print(f"Generated {len(multi_intent_dialogues)} multi-intent dialogues.\n")
        all_dialogues = template_dialogues + multi_intent_dialogues
    
    # Evaluate
    print(f"Evaluating {len(all_dialogues)} dialogues...")
    print(f"Using {'LLM-based' if use_llm_dm else 'Rule-based'} DM")
    print(f"Intent Splitter: {'Enabled' if use_splitter else 'Disabled'}\n")
    
    evaluator = PipelineEvaluator(pipe, use_llm_dm=use_llm_dm, use_splitter=use_splitter)
    metrics = evaluator.evaluate_all(all_dialogues)
    
    # Print report
    evaluator.print_report(metrics)
    
    # Save results
    results_file = os.path.join(os.path.dirname(__file__), "pipeline_eval_results.json")
    with open(results_file, "w") as f:
        json.dump({
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
                # Splitter metrics
                "splitter_detection_accuracy": metrics.splitter_detection_accuracy,
                "splitter_split_accuracy": metrics.splitter_split_accuracy,
                "multi_intent_success_rate": metrics.multi_intent_success_rate,
            },
            "config": {
                "use_llm_dm": use_llm_dm,
                "use_splitter": use_splitter,
                "include_multi_intent": include_multi_intent,
            },
            "dialogues_evaluated": len(all_dialogues),
            "detailed_results": [
                {
                    "name": r.dialogue_name,
                    "intent": r.intent,
                    "method": r.generation_method,
                    "task_success": r.task_success,
                    "num_turns": r.num_turns,
                    "dm_accuracy": r.dm_correct_actions / r.dm_total_actions if r.dm_total_actions > 0 else 0,
                    "is_multi_intent": r.is_multi_intent,
                    "splitter_detected_correctly": r.splitter_detected_correctly,
                    "splitter_split_correctly": r.splitter_split_correctly,
                    "expected_intents_count": r.expected_intents_count,
                    "detected_intents_count": r.detected_intents_count,
                    "errors": r.errors,
                }
                for r in evaluator.results
            ]
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return metrics, evaluator


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline Evaluation")
    parser.add_argument("--template-dialogues", type=int, default=5, help="Number of template dialogues per intent")
    parser.add_argument("--llm-dm", action="store_true", help="Use LLM-based DM instead of rule-based")
    parser.add_argument("--use-splitter", action="store_true", help="Enable intent splitter for multi-intent handling")
    parser.add_argument("--no-multi-intent", action="store_true", help="Disable multi-intent test dialogues")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    run_pipeline_evaluation(
        num_template_dialogues=args.template_dialogues,
        use_llm_dm=args.llm_dm,
        use_splitter=args.use_splitter,
        include_multi_intent=not args.no_multi_intent,
        seed=args.seed
    )

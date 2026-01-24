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
    # ========== PLAN_TRIP - Underinformative ==========
    {
        "name": "01_plan_trip_minimal",
        "history": [],
        "user": "I want to plan a trip",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {},
        "purpose": "Underinformative: no details"
    },
    {
        "name": "02_plan_trip_destination_only",
        "history": [],
        "user": "I want to go to Paris",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {"destination": "Paris"},
        "purpose": "Underinformative: only destination"
    },

    # ========== PLAN_TRIP - Normal ==========
    {
        "name": "03_plan_trip_dest_dates",
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
        "name": "04_plan_trip_dest_people",
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
        "name": "05_plan_trip_overinformative_all",
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
        "name": "06_plan_trip_overinformative_verbose",
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
        "name": "07_plan_trip_followup_dest",
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
        "name": "08_plan_trip_followup_dates",
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
        "name": "09_plan_trip_followup_people",
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
        "name": "10_compare_minimal",
        "history": [],
        "user": "compare them",
        "expect_intent": "COMPARE_OPTIONS",
        "expect_slots": {},
        "purpose": "Underinformative: no options"
    },

    # ========== COMPARE_OPTIONS - Normal ==========
    {
        "name": "11_compare_two_cities",
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
        "name": "12_compare_with_criteria",
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
        "name": "13_compare_overinformative",
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
        "name": "14_compare_followup_criteria",
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
        "name": "15_request_info_minimal",
        "history": [],
        "user": "tell me about it",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {},
        "purpose": "Underinformative: no destination"
    },

    # ========== REQUEST_INFORMATION - Normal ==========
    {
        "name": "16_request_info_destination",
        "history": [],
        "user": "Tell me about Vienna",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {"destination": "Vienna"},
        "purpose": "Normal: general information"
    },
    {
        "name": "17_request_info_with_entity",
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
        "name": "18_request_info_activities",
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
        "name": "19_request_info_overinformative",
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
        "name": "20_request_info_followup_entity",
        "history": [
            {"role": "user", "content": "Tell me about Athens"},
            {"role": "assistant", "content": "What type of information do you need?"}
        ],
        "user": "hotels",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {"entity_type": "hotels"},
        "purpose": "Multi-turn: entity after prompt"
    },

    # ========== END_DIALOGUE ==========
    {
        "name": "21_end_goodbye",
        "history": [],
        "user": "goodbye",
        "expect_intent": "END_DIALOGUE",
        "expect_slots": {},
        "purpose": "End: goodbye"
    },
    {
        "name": "22_end_bye",
        "history": [],
        "user": "bye",
        "expect_intent": "END_DIALOGUE",
        "expect_slots": {},
        "purpose": "End: bye"
    },
    {
        "name": "23_end_thanks",
        "history": [{"role": "assistant", "content": "Is there anything else?"}],
        "user": "no thanks, that's all",
        "expect_intent": "END_DIALOGUE",
        "expect_slots": {},
        "purpose": "End: thanks and closure"
    },

    # ========== OOD (Out of Domain) ==========
    {
        "name": "24_ood_weather",
        "history": [],
        "user": "What's the weather like today?",
        "expect_intent": "OOD",
        "expect_slots": {},
        "purpose": "OOD: weather question"
    },
    {
        "name": "25_ood_unclear",
        "history": [],
        "user": "maybe",
        "expect_intent": "OOD",
        "expect_slots": {},
        "purpose": "OOD: unclear word"
    },
    {
        "name": "26_ood_random",
        "history": [],
        "user": "I like pizza",
        "expect_intent": "OOD",
        "expect_slots": {},
        "purpose": "OOD: random statement"
    },
    {
        "name": "27_ood_yes_no_context",
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
        "name": "28_plan_trip_change_destination",
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
        "name": "29_compare_question_form",
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
        "name": "30_request_info_events",
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
        "name": "31_plan_trip_budget_accommodation",
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
        "name": "32_plan_trip_travel_style",
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
        "name": "33_compare_with_type",
        "history": [],
        "user": "Compare hotels in Brussels and Bruges",
        "expect_intent": "COMPARE_OPTIONS",
        "expect_slots": {
            "option1": "Brussels",
            "option2": "Bruges",
            "criteria": NOT_NONE
        },
        "purpose": "Edge: comparing specific type"
    },

    # ========== UNDER-INFORMATIVE: Very minimal/vague requests ==========
    {
        "name": "34_underinform_just_go",
        "history": [],
        "user": "I want to go somewhere",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {},
        "purpose": "Under-informative: vague destination"
    },
    {
        "name": "35_underinform_just_trip",
        "history": [],
        "user": "trip",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {},
        "purpose": "Under-informative: single word"
    },
    {
        "name": "36_underinform_vague_time",
        "history": [],
        "user": "I want to travel soon",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {},
        "purpose": "Under-informative: vague timeframe"
    },
    {
        "name": "37_underinform_incomplete_compare",
        "history": [],
        "user": "which one is better?",
        "expect_intent": "COMPARE_OPTIONS",
        "expect_slots": {},
        "purpose": "Under-informative: no options specified"
    },
    {
        "name": "38_underinform_just_info",
        "history": [],
        "user": "information please",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {},
        "purpose": "Under-informative: no topic specified"
    },
    {
        "name": "39_underinform_where",
        "history": [],
        "user": "where can I go?",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {},
        "purpose": "Under-informative: open-ended question"
    },
    {
        "name": "40_underinform_just_destination",
        "history": [],
        "user": "Tokyo",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {"destination": "Tokyo"},
        "purpose": "Under-informative: just city name"
    },
    {
        "name": "41_underinform_elliptical",
        "history": [
            {"role": "assistant", "content": "Where would you like to go?"}
        ],
        "user": "there",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {},
        "purpose": "Under-informative: elliptical reference"
    },
    {
        "name": "42_underinform_partial_date",
        "history": [],
        "user": "I want to go to Berlin in May",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Berlin",
            "start_date": NOT_NONE
        },
        "purpose": "Under-informative: incomplete date info"
    },
    {
        "name": "43_underinform_pronoun_heavy",
        "history": [
            {"role": "user", "content": "Tell me about Paris"},
            {"role": "assistant", "content": "What would you like to know?"}
        ],
        "user": "that one",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {},
        "purpose": "Under-informative: pronoun reference"
    },

    # ========== OVER-INFORMATIVE: Excessive details, noise, irrelevant info ==========
    {
        "name": "44_overinform_life_story",
        "history": [],
        "user": "Hi, my name is John and I'm a software engineer from California and I've always wanted to visit Spain, specifically Madrid, because my grandmother was from there and she used to tell me stories about it, so I'm thinking about going there sometime between July 15 and July 22, 2026, with my wife who is 35 years old",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Madrid",
            "start_date": NOT_NONE,
            "end_date": NOT_NONE,
            "num_people": 2
        },
        "purpose": "Over-informative: excessive backstory"
    },
    {
        "name": "45_overinform_redundant",
        "history": [],
        "user": "I want to plan a trip, a vacation, a holiday to Paris, France, the capital of France, from June 1st to June 5th, that's 5 days, for 2 people, me and my partner",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Paris",
            "start_date": NOT_NONE,
            "end_date": NOT_NONE,
            "num_people": 2
        },
        "purpose": "Over-informative: redundant phrasing"
    },
    {
        "name": "46_overinform_mixed_irrelevant",
        "history": [],
        "user": "Compare Rome and Athens for cultural activities, also I heard the weather is nice in both places and I really like Italian food but Greek food is great too, and my friend visited Rome last year",
        "expect_intent": "COMPARE_OPTIONS",
        "expect_slots": {
            "option1": "Rome",
            "option2": "Athens",
            "criteria": NOT_NONE
        },
        "purpose": "Over-informative: irrelevant details"
    },
    {
        "name": "47_overinform_specifications",
        "history": [],
        "user": "I need information about hotels in Vienna, specifically 4-star or 5-star hotels with breakfast included, near the city center, with WiFi, parking, and a gym, that accept credit cards and have English-speaking staff",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {
            "destination": "Vienna",
            "entity_type": "hotels"
        },
        "purpose": "Over-informative: excessive specifications"
    },
    {
        "name": "48_overinform_multiple_questions",
        "history": [],
        "user": "I want to go to Barcelona but I'm not sure when, maybe in spring or summer, possibly April or May or June, for about a week or maybe 10 days, with either 2 or 3 people depending on if my brother can come, and we need to decide on budget",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Barcelona"
        },
        "purpose": "Over-informative: multiple alternatives"
    },
    {
        "name": "49_overinform_justification",
        "history": [],
        "user": "I want to compare Amsterdam and Copenhagen because I'm trying to decide where to go for my birthday and I have limited vacation days and my budget is around $2000 and I prefer cities with good public transportation",
        "expect_intent": "COMPARE_OPTIONS",
        "expect_slots": {
            "option1": "Amsterdam",
            "option2": "Copenhagen"
        },
        "purpose": "Over-informative: excessive justification"
    },
    {
        "name": "50_overinform_all_slots_verbose",
        "history": [],
        "user": "I'm planning to visit Dublin, Ireland from August 10, 2026 until August 20, 2026 (that's 10 days) for a total of 4 travelers (2 adults and 2 children ages 12 and 14), we want to stay in a family-friendly hotel or maybe an apartment with kitchen facilities, our budget is medium-range around $150-200 per night, and we're interested in cultural activities like museums and historical sites plus some outdoor activities if the weather is good",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Dublin",
            "start_date": NOT_NONE,
            "end_date": NOT_NONE,
            "num_people": 4,
            "accommodation_type": ANY,
            "budget_level": "medium",
            "travel_style": NOT_NONE
        },
        "purpose": "Over-informative: all slots with extra context"
    },
    {
        "name": "51_overinform_parentheticals",
        "history": [],
        "user": "Tell me about Brussels (Belgium) hotels (3-star or higher) for my trip (business trip actually) next month (probably)",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {
            "destination": "Brussels",
            "entity_type": "hotels"
        },
        "purpose": "Over-informative: excessive parentheticals"
    },
    {
        "name": "52_overinform_options_list",
        "history": [],
        "user": "I'm considering several destinations: Paris, London, Rome, Barcelona, and Amsterdam, but I really want to compare Paris and London first",
        "expect_intent": "COMPARE_OPTIONS",
        "expect_slots": {
            "option1": "Paris",
            "option2": "London"
        },
        "purpose": "Over-informative: multiple options listed"
    },
    {
        "name": "53_overinform_conditional",
        "history": [],
        "user": "If the flights are cheap enough, I'd like to go to Stockholm from December 20 to December 27, but if not, maybe just a weekend trip, unless the hotels are too expensive, then I'll reconsider",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Stockholm",
            "start_date": NOT_NONE,
            "end_date": NOT_NONE
        },
        "purpose": "Over-informative: conditional statements"
    },

    # ========== NOISE & AMBIGUITY: Mixed signals, typos, unclear intent ==========
    {
        "name": "54_noise_mixed_intents",
        "history": [],
        "user": "I want to go to Prague and also can you tell me about hotels there or maybe compare it with Vienna",
        "expect_intent": ANY,  # Could be PLAN_TRIP, REQUEST_INFORMATION, or COMPARE_OPTIONS
        "expect_slots": {},
        "purpose": "Noise: multiple possible intents"
    },
    {
        "name": "55_noise_typos",
        "history": [],
        "user": "I wnat to travle to Barselona in Apirl",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": NOT_NONE
        },
        "purpose": "Noise: spelling errors"
    },
    {
        "name": "56_noise_informal_slang",
        "history": [],
        "user": "yo wanna check out some dope activities in Berlin ya know",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {
            "destination": "Berlin",
            "entity_type": NOT_NONE
        },
        "purpose": "Noise: informal language"
    },
    {
        "name": "57_noise_filler_words",
        "history": [],
        "user": "um well like I think maybe I kind of want to sort of go to uh Paris you know",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Paris"
        },
        "purpose": "Noise: filler words"
    },
    {
        "name": "58_ambiguous_this_that",
        "history": [
            {"role": "assistant", "content": "Would you like to know about hotels or activities?"}
        ],
        "user": "yes both of those",
        "expect_intent": ANY,
        "expect_slots": {},
        "purpose": "Ambiguous: unclear reference"
    },
    {
        "name": "59_ambiguous_numbers",
        "history": [],
        "user": "Trip to Milan for three to five people in 2 or 3 weeks",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Milan",
            "num_people": NOT_NONE
        },
        "purpose": "Ambiguous: range of values"
    },
    {
        "name": "60_noise_code_switching",
        "history": [],
        "user": "I want to visit la Ciudad de Barcelona pour voir las Ramblas",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": NOT_NONE
        },
        "purpose": "Noise: code-switching languages"
    },
    {
        "name": "61_ambiguous_time_reference",
        "history": [],
        "user": "I want to go to Rome next month",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Rome",
            "start_date": ANY  # Depends on current date context
        },
        "purpose": "Ambiguous: relative time reference"
    },
    {
        "name": "62_noise_emojis_special_chars",
        "history": [],
        "user": "Paris!!! :) from 5/10 to 5/15 :D",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Paris",
            "start_date": NOT_NONE,
            "end_date": NOT_NONE
        },
        "purpose": "Noise: emojis and punctuation"
    },
    {
        "name": "63_ambiguous_entity_vs_destination",
        "history": [],
        "user": "hotels in Paris",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {
            "destination": "Paris",
            "entity_type": "hotels"
        },
        "purpose": "Ambiguous: could be info request or trip planning"
    },

    # ========== INCREMENTAL INFORMATION: Building up context ==========
    {
        "name": "64_incremental_destination_first",
        "history": [],
        "user": "London",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "London"
        },
        "purpose": "Incremental: start with destination"
    },
    {
        "name": "65_incremental_add_dates",
        "history": [
            {"role": "user", "content": "London"},
            {"role": "assistant", "content": "When would you like to go?"}
        ],
        "user": "March 5 to 10",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "start_date": NOT_NONE,
            "end_date": NOT_NONE
        },
        "purpose": "Incremental: add dates"
    },
    {
        "name": "66_incremental_add_people",
        "history": [
            {"role": "user", "content": "London"},
            {"role": "assistant", "content": "When?"},
            {"role": "user", "content": "March 5 to 10"},
            {"role": "assistant", "content": "How many people?"}
        ],
        "user": "3 people",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "num_people": 3
        },
        "purpose": "Incremental: add people count"
    },
    {
        "name": "67_incremental_add_budget",
        "history": [
            {"role": "user", "content": "Trip to Venice for 2"},
            {"role": "assistant", "content": "What's your budget?"}
        ],
        "user": "medium budget",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "budget_level": "medium"
        },
        "purpose": "Incremental: add budget"
    },
    {
        "name": "68_incremental_add_accommodation",
        "history": [
            {"role": "user", "content": "Prague in June"},
            {"role": "assistant", "content": "What type of accommodation?"}
        ],
        "user": "hotel",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "accommodation_type": "hotel"
        },
        "purpose": "Incremental: add accommodation type"
    },
    {
        "name": "69_incremental_add_style",
        "history": [
            {"role": "user", "content": "Tokyo for 4 people"},
            {"role": "assistant", "content": "What's your travel style?"}
        ],
        "user": "cultural activities and museums",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "travel_style": NOT_NONE
        },
        "purpose": "Incremental: add travel style"
    },
    {
        "name": "70_incremental_modify_slot",
        "history": [
            {"role": "user", "content": "Oslo from June 1 to June 10"},
            {"role": "assistant", "content": "Got it, Oslo June 1-10"}
        ],
        "user": "actually make it June 15 to June 20",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "start_date": NOT_NONE,
            "end_date": NOT_NONE
        },
        "purpose": "Incremental: modify existing slot"
    },
    {
        "name": "71_incremental_change_destination",
        "history": [
            {"role": "user", "content": "I want to go to Paris"},
            {"role": "assistant", "content": "Great! When?"}
        ],
        "user": "wait, I changed my mind, make it Rome",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Rome"
        },
        "purpose": "Incremental: change destination mid-conversation"
    },

    # ========== SHORT vs LONG utterances ==========
    {
        "name": "72_ultra_short_ok",
        "history": [
            {"role": "assistant", "content": "Is this plan okay?"}
        ],
        "user": "ok",
        "expect_intent": ANY,  # Could be confirmation or unclear
        "expect_slots": {},
        "purpose": "Ultra-short: single word response"
    },
    {
        "name": "73_ultra_short_no",
        "history": [
            {"role": "assistant", "content": "Shall we proceed?"}
        ],
        "user": "no",
        "expect_intent": ANY,
        "expect_slots": {},
        "purpose": "Ultra-short: single word negative"
    },
    {
        "name": "74_ultra_long_stream",
        "history": [],
        "user": "So I've been thinking about taking a vacation for a while now and I was talking to my colleague yesterday and she mentioned that she went to Iceland last summer and had an amazing time and showed me pictures and everything looked so beautiful especially the waterfalls and the northern lights and she said the food was interesting too although a bit expensive but that got me thinking that maybe I should go there too but then my partner suggested maybe we should go somewhere warmer like Greece or Spain instead because it's been so cold here lately and we could use some sun and beach time but then again Iceland sounds so unique and different from anywhere I've been before so I'm kind of torn between the two options and I was wondering if you could help me compare them",
        "expect_intent": ANY,  # Complex, could be PLAN_TRIP or COMPARE_OPTIONS
        "expect_slots": {},
        "purpose": "Ultra-long: stream of consciousness"
    },
    {
        "name": "75_long_but_focused",
        "history": [],
        "user": "I would like to plan a comprehensive trip to Copenhagen, Denmark from September 15th to September 25th, 2026 for myself and my spouse (2 people total), with a medium budget range, preferring to stay in a boutique hotel or bed and breakfast, and we are particularly interested in cultural activities such as visiting museums, art galleries, and historical landmarks, as well as trying local cuisine",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Copenhagen",
            "start_date": NOT_NONE,
            "end_date": NOT_NONE,
            "num_people": 2,
            "accommodation_type": ANY,
            "budget_level": "medium",
            "travel_style": NOT_NONE
        },
        "purpose": "Long: detailed but on-topic"
    },

    # ========== NATURAL CONVERSATIONAL language ==========
    {
        "name": "76_conversational_casual",
        "history": [],
        "user": "hey so I'm thinking maybe I'll head to Berlin sometime soon",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Berlin"
        },
        "purpose": "Natural: casual tone"
    },
    {
        "name": "77_conversational_question",
        "history": [],
        "user": "can you help me find some cool things to do in Amsterdam?",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {
            "destination": "Amsterdam",
            "entity_type": NOT_NONE
        },
        "purpose": "Natural: question form"
    },
    {
        "name": "78_conversational_hesitation",
        "history": [],
        "user": "I'm not sure but I think I want to compare Paris and London maybe",
        "expect_intent": "COMPARE_OPTIONS",
        "expect_slots": {
            "option1": "Paris",
            "option2": "London"
        },
        "purpose": "Natural: hesitation markers"
    },
    {
        "name": "79_conversational_thinking_aloud",
        "history": [],
        "user": "let me see... what about Vienna? yeah, Vienna sounds good",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Vienna"
        },
        "purpose": "Natural: thinking aloud"
    },
    {
        "name": "80_conversational_correction",
        "history": [],
        "user": "I want to go to... wait, sorry, I meant Brussels not Amsterdam",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Brussels"
        },
        "purpose": "Natural: self-correction"
    },
    {
        "name": "81_conversational_elaboration",
        "history": [
            {"role": "assistant", "content": "Where to?"}
        ],
        "user": "well, I was thinking somewhere in Italy, you know, like Rome or maybe Florence, but let's go with Rome",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Rome"
        },
        "purpose": "Natural: elaboration and narrowing"
    },
    {
        "name": "82_conversational_implicit",
        "history": [
            {"role": "user", "content": "Tell me about Madrid"},
            {"role": "assistant", "content": "What would you like to know?"}
        ],
        "user": "what's there to see",
        "expect_intent": "REQUEST_INFORMATION",
        "expect_slots": {
            "entity_type": NOT_NONE
        },
        "purpose": "Natural: implicit reference"
    },
    {
        "name": "83_conversational_colloquial",
        "history": [],
        "user": "gonna hit up Stockholm next weekend",
        "expect_intent": "PLAN_TRIP",
        "expect_slots": {
            "destination": "Stockholm"
        },
        "purpose": "Natural: colloquial expression"
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

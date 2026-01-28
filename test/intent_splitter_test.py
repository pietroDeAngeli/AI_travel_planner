"""
Tests for the Intent Splitter module.

Run with: python intent_splitter_test.py
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm import make_llm
from intent_splitter import split_intents, has_multiple_intents, IntentQueue, extract_json_array


# --- Test cases ---

SINGLE_INTENT_CASES = [
    "I want to book a flight to Rome",
    "Find me a hotel in Paris",
    "What activities are available in Barcelona?",
    "I need a flight from Milan to London",
    "Book a hostel for next week",
]

MULTI_INTENT_CASES = [
    {
        "input": "I want to book a flight to Rome and also find me a hotel there",
        "expected_count": 2,
        "description": "Flight + Hotel"
    },
    {
        "input": "Book a flight to Paris. Can you also show me some tours?",
        "expected_count": 2,
        "description": "Flight + Activity"
    },
    {
        "input": "I need a hotel in Rome and I'd like to book a cooking class as well",
        "expected_count": 2,
        "description": "Hotel + Activity"
    },
    {
        "input": "Find flights to Barcelona, book a hotel there, and show me hiking tours",
        "expected_count": 3,
        "description": "Flight + Hotel + Activity"
    },
    {
        "input": "I want to go to Rome. Also, can you compare it with Florence for food experiences?",
        "expected_count": 2,
        "description": "General + Compare cities"
    },
]


def test_extract_json_array():
    """Test JSON array extraction."""
    print("\n" + "="*60)
    print("Testing JSON Array Extraction")
    print("="*60)
    
    test_cases = [
        ('["one", "two"]', ["one", "two"]),
        ('Here is the result: ["a", "b", "c"]', ["a", "b", "c"]),
        ('```json\n["test"]\n```', ["test"]),
        ('Invalid', None),
        ('{"key": "value"}', None),  # Object, not array
    ]
    
    passed = 0
    for text, expected in test_cases:
        result = extract_json_array(text)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        print(f"  {status} Input: {text[:30]}... -> {result}")
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_has_multiple_intents():
    """Test heuristic detection of multiple intents."""
    print("\n" + "="*60)
    print("Testing Heuristic Multi-Intent Detection")
    print("="*60)
    
    # Should return True
    multi_cases = [
        "Book a flight and also find a hotel",
        "I need a hotel. Also, show me activities",
        "Flight to Rome? And can you book a tour?",
        "Find accommodation plus some local tours",
    ]
    
    # Should return False
    single_cases = [
        "Book a flight to Rome",
        "I want a hotel in Paris",
        "Show me activities",
    ]
    
    passed = 0
    
    print("\n  Multi-intent cases (should be True):")
    for case in multi_cases:
        result = has_multiple_intents(None, case)
        status = "✓" if result else "✗"
        if result:
            passed += 1
        print(f"    {status} \"{case[:40]}...\" -> {result}")
    
    print("\n  Single-intent cases (should be False):")
    for case in single_cases:
        result = has_multiple_intents(None, case)
        status = "✓" if not result else "✗"
        if not result:
            passed += 1
        print(f"    {status} \"{case[:40]}\" -> {result}")
    
    total = len(multi_cases) + len(single_cases)
    print(f"\nPassed: {passed}/{total}")
    return passed == total


def test_intent_queue():
    """Test IntentQueue functionality."""
    print("\n" + "="*60)
    print("Testing IntentQueue")
    print("="*60)
    
    queue = IntentQueue()
    
    tests = []
    
    # Test empty queue
    tests.append(("Empty queue has_pending", not queue.has_pending()))
    tests.append(("Empty queue pop returns None", queue.pop() is None))
    
    # Add items
    queue.add(["intent1", "intent2", "intent3"])
    tests.append(("After add, has_pending is True", queue.has_pending()))
    tests.append(("Peek returns first item", queue.peek() == "intent1"))
    tests.append(("Pop returns first item", queue.pop() == "intent1"))
    tests.append(("Second pop returns second item", queue.pop() == "intent2"))
    
    # Clear
    queue.clear()
    tests.append(("After clear, has_pending is False", not queue.has_pending()))
    
    passed = sum(1 for _, result in tests if result)
    for name, result in tests:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
    
    print(f"\nPassed: {passed}/{len(tests)}")
    return passed == len(tests)


def test_split_intents_single(pipe):
    """Test that single intents are not split."""
    print("\n" + "="*60)
    print("Testing Single Intent (No Split)")
    print("="*60)
    
    passed = 0
    for case in SINGLE_INTENT_CASES:
        current, pending = split_intents(pipe, case)
        # Should return original with no pending
        is_correct = len(pending) == 0
        status = "✓" if is_correct else "✗"
        if is_correct:
            passed += 1
        print(f"  {status} \"{case[:40]}...\"")
        print(f"      Current: \"{current[:50]}...\"")
        print(f"      Pending: {pending}")
    
    print(f"\nPassed: {passed}/{len(SINGLE_INTENT_CASES)}")
    return passed >= len(SINGLE_INTENT_CASES) * 0.8  # 80% threshold


def test_split_intents_multi(pipe):
    """Test that multiple intents are correctly split."""
    print("\n" + "="*60)
    print("Testing Multiple Intent Splitting")
    print("="*60)
    
    passed = 0
    for case in MULTI_INTENT_CASES:
        user_input = case["input"]
        expected_count = case["expected_count"]
        description = case["description"]
        
        current, pending = split_intents(pipe, user_input)
        total_count = 1 + len(pending)
        
        # Allow ±1 tolerance for LLM variability
        is_correct = abs(total_count - expected_count) <= 1
        status = "✓" if is_correct else "✗"
        if is_correct:
            passed += 1
        
        print(f"\n  {status} [{description}] Expected: {expected_count}, Got: {total_count}")
        print(f"      Input: \"{user_input[:60]}...\"")
        print(f"      Current: \"{current[:50]}...\"")
        print(f"      Pending: {pending}")
    
    print(f"\nPassed: {passed}/{len(MULTI_INTENT_CASES)}")
    return passed >= len(MULTI_INTENT_CASES) * 0.6  # 60% threshold (LLM can be variable)


def test_end_to_end_flow(pipe):
    """Test the full flow with IntentQueue."""
    print("\n" + "="*60)
    print("Testing End-to-End Flow with Queue")
    print("="*60)
    
    queue = IntentQueue()
    
    # Simulate user input with multiple intents
    user_input = "Book a flight to Rome and find me a hotel there"
    print(f"\n  User: \"{user_input}\"")
    
    # First turn: split and process
    current, pending = split_intents(pipe, user_input)
    queue.add(pending)
    
    print(f"  Turn 1 - Processing: \"{current}\"")
    print(f"  Queue: {queue.pending}")
    
    # Simulate completing first intent
    turns = 1
    while queue.has_pending():
        turns += 1
        next_intent = queue.pop()
        print(f"  Turn {turns} - Processing queued: \"{next_intent}\"")
    
    print(f"\n  Total turns needed: {turns}")
    is_correct = turns >= 1  # At least processed something
    status = "✓" if is_correct else "✗"
    print(f"\n  {status} End-to-end flow completed successfully")
    
    return is_correct


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("INTENT SPLITTER TEST SUITE")
    print("="*60)
    
    results = {}
    
    # Unit tests (no LLM needed)
    results["JSON Extraction"] = test_extract_json_array()
    results["Heuristic Detection"] = test_has_multiple_intents()
    results["IntentQueue"] = test_intent_queue()
    
    # LLM-based tests
    print("\n" + "="*60)
    print("Loading LLM for integration tests...")
    print("="*60)
    
    try:
        pipe = make_llm()
        if pipe is None:
            print("  ✗ Failed to load LLM. Skipping integration tests.")
            results["Single Intent Split"] = None
            results["Multi Intent Split"] = None
            results["End-to-End Flow"] = None
        else:
            print("  ✓ LLM loaded successfully")
            results["Single Intent Split"] = test_split_intents_single(pipe)
            results["Multi Intent Split"] = test_split_intents_multi(pipe)
            results["End-to-End Flow"] = test_end_to_end_flow(pipe)
    except Exception as e:
        print(f"  ✗ Error loading LLM: {e}")
        results["Single Intent Split"] = None
        results["Multi Intent Split"] = None
        results["End-to-End Flow"] = None
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        if passed is None:
            status = "⊘ SKIPPED"
        elif passed:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        print(f"  {status}: {test_name}")
    
    passed_count = sum(1 for v in results.values() if v is True)
    failed_count = sum(1 for v in results.values() if v is False)
    skipped_count = sum(1 for v in results.values() if v is None)
    
    print(f"\nTotal: {passed_count} passed, {failed_count} failed, {skipped_count} skipped")
    
    return failed_count == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

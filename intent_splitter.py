"""
Intent Splitter Module

Splits user input containing multiple intents into separate sentences.
This handles over-informative users who provide multiple requests at once.

Strategy: Process only one intent at a time (the first/most important one)
and queue the rest for subsequent turns. Queued intents are only processed
after the current booking flow is completed.
"""

import json
import re
import logging
from typing import List, Tuple, Optional

from schema import INTENT_SCHEMAS

logger = logging.getLogger(__name__)

# Build intent descriptions for the prompt
INTENT_DESCRIPTIONS = "\n".join([
    f"- {intent}: {schema['description']}"
    for intent, schema in INTENT_SCHEMAS.items()
    if intent not in ["OOD", "END_DIALOGUE"]
])

SPLIT_PROMPT = f"""You are an intent splitter for a travel assistant.

Given a user input, determine if it contains multiple DISTINCT booking requests/intents.
If it does, split it into separate sentences, each representing a single intent.

Possible travel intents:
{INTENT_DESCRIPTIONS}

RULES:
1. If the input has only ONE intent, return it as-is in a single-element list.
2. If the input has MULTIPLE intents, split into separate sentences.
3. Each sentence MUST be self-contained: repeat shared details (city, dates, budget, number of people) in every sentence.
4. Preserve ALL important details (dates, locations, numbers) in each relevant sentence.
5. Ignore greetings or filler words when splitting.
6. Two sentences about the SAME topic (e.g. "I want a hotel. Budget should be medium.") are ONE intent, not two.
7. Only split when the sentences refer to DIFFERENT booking types (flight vs hotel vs activity vs compare).

OUTPUT FORMAT: Return ONLY a JSON array of strings.
Example single intent: ["I want to book a flight to Rome on December 1st for 2 passengers"]
Example multiple intents: ["I want to book a flight to Rome on December 1st for 2 passengers", "Find me a hotel in Rome for 2 guests checking in December 1st"]

Only output the JSON array, no other text.
"""

def extract_json_array(text: str) -> Optional[List[str]]:
    """Extract a JSON array from text."""
    text = re.sub(r"```(?:json)?", "", text)
    text = text.replace("```", "").strip()

    start = text.find("[")
    if start == -1:
        return None

    depth = 0
    end = None
    for i in range(start, len(text)):
        c = text[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        return None

    candidate = text[start:end]
    try:
        result = json.loads(candidate)
        if isinstance(result, list) and all(isinstance(item, str) for item in result):
            return result
        return None
    except json.JSONDecodeError:
        return None


def split_intents(pipe, user_input: str) -> Tuple[str, List[str]]:
    """
    Split user input into multiple intent sentences if needed.

    Always uses the LLM to decide whether to split (as in the notebook approach).
    The LLM returns a single-element list when the input has only one intent,
    so no heuristic pre-filtering is needed.

    Args:
        pipe: The LLM pipeline
        user_input: Raw user input

    Returns:
        Tuple of (current_sentence, pending_sentences)
        - current_sentence: The first intent to process now
        - pending_sentences: Remaining intents to process later (can be empty)
    """
    # Fast path: empty or very short input cannot contain multiple intents
    if not user_input or not user_input.strip():
        return user_input, []

    messages = [
        {"role": "system", "content": SPLIT_PROMPT},
        {"role": "user", "content": user_input}
    ]

    try:
        out = pipe(messages)
        generated = out[0]["generated_text"]

        if isinstance(generated, list):
            text = generated[-1].get("content", "")
        else:
            text = str(generated)

        sentences = extract_json_array(text)

        if sentences and len(sentences) >= 1:
            current = sentences[0]
            pending = sentences[1:]
            return current, pending

    except Exception as e:
        logger.error(f"Intent splitting error: {e}")

    # Fallback: return original input as single intent
    return user_input, []


class IntentQueue:
    """
    Manages a queue of pending intents for multi-turn processing.
    Queued intents are only consumed when the current booking flow is complete.
    """
    def __init__(self):
        self.pending: List[str] = []

    def add(self, sentences: List[str]) -> None:
        """Add sentences to the pending queue."""
        self.pending.extend(sentences)

    def pop(self) -> Optional[str]:
        """Get and remove the next pending sentence."""
        if self.pending:
            return self.pending.pop(0)
        return None

    def has_pending(self) -> bool:
        """Check if there are pending intents."""
        return len(self.pending) > 0

    def clear(self) -> None:
        """Clear all pending intents."""
        self.pending = []

    def peek(self) -> Optional[str]:
        """View next pending without removing."""
        if self.pending:
            return self.pending[0]
        return None

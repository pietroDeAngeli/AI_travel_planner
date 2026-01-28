"""
Intent Splitter Module

Splits user input containing multiple intents into separate sentences.
This handles over-informative users who provide multiple requests at once.

Strategy: Process only one intent at a time (the first/most important one)
and queue the rest for subsequent turns.
"""

import json
import re
from typing import List, Tuple, Optional

from schema import INTENT_SCHEMAS


# Build intent descriptions for the prompt
INTENT_DESCRIPTIONS = "\n".join([
    f"- {intent}: {schema['description']}"
    for intent, schema in INTENT_SCHEMAS.items()
    if intent not in ["OOD", "END_DIALOGUE"]
])

SPLIT_PROMPT = f"""You are an intent splitter for a travel assistant.

Given a user input, determine if it contains multiple distinct requests/intents.
If it does, split it into separate sentences, each representing a single intent.

Possible travel intents:
{INTENT_DESCRIPTIONS}

RULES:
1. If the input has only ONE intent, return it as-is in a single-element list.
2. If the input has MULTIPLE intents, split into separate sentences.
3. Keep each sentence self-contained with relevant context.
4. Preserve important details (dates, locations, numbers) in each relevant sentence.
5. Ignore greetings or filler words when splitting.

OUTPUT FORMAT: Return ONLY a JSON array of strings.
Example single intent: ["I want to book a flight to Rome"]
Example multiple intents: ["I want to book a flight to Rome", "Find me a hotel there"]

Only output the JSON array, no other text.
"""


def extract_json_array(text: str) -> Optional[List[str]]:
    """Extract a JSON array from text."""
    text = re.sub(r"```(?:json)?", "", text)
    text = text.replace("```", "").strip()
    
    # Find array brackets
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
    
    Args:
        pipe: The LLM pipeline
        user_input: Raw user input
    
    Returns:
        Tuple of (current_sentence, pending_sentences)
        - current_sentence: The first intent to process now
        - pending_sentences: Remaining intents to process later (can be empty)
    """
    # Skip splitting for very short inputs (likely single intent)
    if len(user_input.split()) <= 5:
        return user_input, []
    
    messages = [
        {"role": "system", "content": SPLIT_PROMPT},
        {"role": "user", "content": user_input}
    ]
    
    try:
        out = pipe(messages, max_new_tokens=200)
        generated = out[0]["generated_text"]
        
        if isinstance(generated, list):
            text = generated[-1].get("content", "")
        else:
            text = str(generated)
        
        sentences = extract_json_array(text)
        
        if sentences and len(sentences) > 0:
            # Return first sentence to process, rest are pending
            current = sentences[0]
            pending = sentences[1:] if len(sentences) > 1 else []
            return current, pending
        
    except Exception as e:
        print(f"Intent splitting error: {e}")
    
    # Fallback: return original input
    return user_input, []


def has_multiple_intents(pipe, user_input: str) -> bool:
    """
    Quick check if user input likely contains multiple intents.
    Uses heuristics before calling LLM.
    """
    # Heuristic checks for common multi-intent patterns
    multi_intent_markers = [
        " and also ",
        " and then ",
        ". also ",
        ". can you also ",
        "? also ",
        "? and ",
        " plus ",
        " additionally ",
        " as well as ",
    ]
    
    lower_input = user_input.lower()
    for marker in multi_intent_markers:
        if marker in lower_input:
            return True
    
    # Check for multiple sentences with question marks or periods
    sentences = re.split(r'[.?!]+', user_input)
    meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    return len(meaningful_sentences) > 1


class IntentQueue:
    """
    Manages a queue of pending intents for multi-turn processing.
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

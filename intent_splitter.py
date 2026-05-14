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

SPLIT_PROMPT = f"""You are an intent classifier and splitter for a travel booking assistant.

Your ONLY job: decide whether a user message contains MORE THAN ONE distinct travel booking request. If yes, split it; if no, return it unchanged.

The travel booking intents are:
{INTENT_DESCRIPTIONS}

═══ CRITICAL RULES ═══

RULE 1 — DEFAULT IS NO SPLIT.
When in doubt, return the full message as a single element.
Only split when you are 100% certain the user is asking for two or more DIFFERENT booking types.

RULE 2 — WHAT COUNTS AS A SPLIT.
Split ONLY when the user explicitly requests two or more DIFFERENT booking categories in the same message.
Valid split triggers: flight + hotel, flight + activity, hotel + activity, flight + hotel + activity.

RULE 3 — DO NOT SPLIT THESE CASES (very common mistakes to avoid):
- A single booking with multiple details: "I want a hotel in Rome from June 1 to June 7 for 2 people" → ONE element.
- A booking with a budget or preference: "Find me a cheap flight to Paris" → ONE element.
- Greetings, filler words, or questions added to a single booking: "Hi! Can you book me a hotel in London?" → ONE element (the booking part).
- A question or clarification about a single topic: "What hotels are available?" → ONE element.
- Compound sentences about the SAME booking type: "I need flights and also want to upgrade my seat" → ONE element.
- Adding details to an already-stated intent: "...and I'd prefer a window seat" → ONE element.

RULE 4 — COPY VERBATIM. Do NOT rephrase, summarise, add, or infer anything.
Each output string must be a literal substring or minimal paraphrase of the original input.
Never invent city names, dates, numbers, or any other slot values.

═══ EXAMPLES ═══

Input: "Book a flight from Rome to Paris and also find me a hotel in Paris"
Output: ["Book a flight from Rome to Paris", "find me a hotel in Paris"]

Input: "I want to fly to Barcelona on June 5th and book a cooking class there"
Output: ["I want to fly to Barcelona on June 5th", "book a cooking class there"]

Input: "Find me a flight to New York, a hotel for 3 nights, and a city tour"
Output: ["Find me a flight to New York", "a hotel for 3 nights", "a city tour"]

Input: "I'd like to book a hotel in Rome from July 10 to July 17 for 2 guests"
Output: ["I'd like to book a hotel in Rome from July 10 to July 17 for 2 guests"]

Input: "Can you find me a cheap flight to London?"
Output: ["Can you find me a cheap flight to London?"]

Input: "Hi! I want to visit Paris next month"
Output: ["I want to visit Paris next month"]

Input: "Book me a hotel in Milan"
Output: ["Book me a hotel in Milan"]

═══ OUTPUT FORMAT ═══
Return ONLY a JSON array of strings. No explanation, no markdown, no extra text.
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
            # Collapse consecutive duplicates produced by the splitter
            deduped = [sentences[0]]
            for s in sentences[1:]:
                if s.strip().lower() != deduped[-1].strip().lower():
                    deduped.append(s)
            current = deduped[0]
            pending = deduped[1:]
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

from typing import Dict, Any
from schema import DM_ACTIONS
from dm import DialogueState


def nlg_generate(pipe, action: str, state: DialogueState) -> str:
    """
    NLG module: generates the surface utterance
    based on the DM action and dialogue state.
    """

    if action not in DM_ACTIONS:
        action = "ASK_CLARIFICATION"

    prompts = {
        "REQUEST_MISSING_SLOT": _prompt_request_missing_slot(state),
        "ASK_CONFIRMATION": _prompt_ask_confirmation(state),
        "PROPOSE_TRIP_PLAN": _prompt_propose_trip(state),
        "PROVIDE_INFORMATION": _prompt_provide_information(state),
        "PROVIDE_COMPARISON": _prompt_provide_comparison(state),
        "ASK_CLARIFICATION": _prompt_ask_clarification(),
        "GOODBYE": _prompt_goodbye(),
    }

    prompt = prompts.get(action, _prompt_ask_clarification())

    messages = [
        {
            "role": "system",
            "content": "You are a polite and helpful travel assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    out = pipe(
        messages,
        max_new_tokens=120,
        temperature=0.7,
        do_sample=True,
    )

    return out[0]["generated_text"][-1]["content"].strip()

def _prompt_request_missing_slot(state: DialogueState) -> str:
    missing = state.info.missing_slots(state.task_intent)
    slot = missing[0] if missing else "some information"

    return f"""
You are asking the user to provide missing information.

Missing information: {slot}

Ask ONLY about this information, in a natural and polite way.
"""

def _prompt_ask_confirmation(state: DialogueState) -> str:
    return f"""
Ask the user to confirm the following trip details.

Trip details:
{state.info}

End the message with a confirmation question.
"""

def _prompt_propose_trip(state: DialogueState) -> str:
    return f"""
Generate a concise travel plan proposal based on the following information:

{state.info}

Include:
- accommodation suggestion
- 2–3 activities
- friendly tone

Do not ask questions.
"""

def _prompt_provide_information(state: DialogueState) -> str:
    return f"""
Provide useful travel information based on this context:

{state.info}

Be informative but concise.
"""

def _prompt_provide_comparison(state: DialogueState) -> str:
    return f"""
Compare the two destinations requested by the user.

Context:
{state.info}

Provide a short and clear comparison.
"""

def _prompt_ask_clarification() -> str:
    return """
Politely ask the user to clarify their request.
"""

def _prompt_goodbye() -> str:
    return """
Say goodbye in a polite and friendly way.
"""

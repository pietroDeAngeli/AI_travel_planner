from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from data import UserInformation
from schema import SLOTS, DM_ACTIONS


@dataclass
class DialogueState:
    info: UserInformation = field(default_factory=UserInformation)
    current_intent: Optional[str] = None
    task_intent: Optional[str] = None
    current_state: str = "INIT"
    confirmed: bool = False
    plan: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return (
            "DialogueState(\n"
            f"  state      = {self.current_state}\n"
            f"  intent     = {self.current_intent}\n"
            f"  task       = {self.task_intent}\n"
            f"  confirmed  = {self.confirmed}\n"
            f"  info       = {self.info}\n"
            ")"
        )


def update_info(state: DialogueState, intent: str, slots: Dict[str, Any]) -> None:
    state.current_intent = intent

    if intent == "PLAN_TRIP":
        state.task_intent = intent
        state.info.update_info(slots)
        state.confirmed = False


def dm_decide(
    pipe,
    state: DialogueState,
    nlu_output: Dict[str, Any],
    user_utterance: str,
) -> str:
    """
    LLM-based Dialogue Manager.
    The LLM selects the next abstract action.
    Symbolic guard-rails enforce coherence.
    """

    intent = nlu_output.get("intent")
    slots = nlu_output.get("slots", {})

    update_info(state, intent, slots)

    missing_slots = state.info.missing_slots(state.task_intent)

    prompt = f"""
You are a Dialogue Manager for a task-oriented travel planning system.

Your task is to choose the NEXT SYSTEM ACTION.

Available actions:
{', '.join(DM_ACTIONS)}

Current context:
- Intent: {intent}
- Task intent: {state.task_intent}
- Missing slots: {missing_slots}
- Confirmed: {state.confirmed}
- Collected info: {state.info}

Rules:
- If intent is END_DIALOGUE → GOODBYE
- If intent is OOD or unclear → ASK_CLARIFICATION
- If there are missing slots → REQUEST_MISSING_SLOT
- If no missing slots and not confirmed → ASK_CONFIRMATION
- If confirmed and intent is PLAN_TRIP → PROPOSE_TRIP_PLAN
- If intent is REQUEST_INFORMATION → PROVIDE_INFORMATION
- If intent is COMPARE_OPTIONS → PROVIDE_COMPARISON

Return ONLY the action name.
""".strip()

    messages = [
        {"role": "system", "content": "You output ONLY one action name."},
        {"role": "user", "content": prompt},
    ]

    out = pipe(
        messages,
        max_new_tokens=50,
        temperature=0.0,
        do_sample=False,
    )

    raw = out[0]["generated_text"][-1]["content"].strip().upper()

    # --- Guard rails ---
    action = raw if raw in DM_ACTIONS else "ASK_CLARIFICATION"

    if action == "PROPOSE_TRIP_PLAN" and missing_slots:
        action = "REQUEST_MISSING_SLOT"

    if action == "ASK_CONFIRMATION":
        state.current_state = "CONFIRM_DETAILS"

    if action == "PROPOSE_TRIP_PLAN":
        state.current_state = "TASK_CONFIRMED"

    if action == "GOODBYE":
        state.current_state = "INIT"

    return action

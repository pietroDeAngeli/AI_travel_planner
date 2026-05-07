"""
Flask web server for the AI Travel Planner.

Wraps the existing NLU → DM → NLG pipeline and exposes it via a JSON API
consumed by the static frontend (static/index.html).

Run:
    pip install flask
    python app.py
Then open http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
import threading

from llm import make_llm
from nlu import nlu_parse
from dm import DialogueState, dm_decide, state_context
from nlg import nlg_generate, GREETING_MESSAGE
from schema import parse_action
from amadeus import search_activities, search_accommodation
from intent_splitter import split_intents, IntentQueue

app = Flask(__name__, static_folder="static")

# ── Global session state (single-user demo) ───────────────────────────────────
_pipe = None
_state: DialogueState = None
_intent_queue: IntentQueue = None
_history = []
_settings = {"use_splitter": True, "use_llm_dm": True}
_lock = threading.Lock()

COMPLETION_ACTIONS = frozenset({
    "COMPLETE_FLIGHT_BOOKING",
    "COMPLETE_ACCOMMODATION_BOOKING",
    "COMPLETE_ACTIVITY_BOOKING",
    "COMPARE_CITIES_RESULT",
    "GOODBYE",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_flow_idle(s: DialogueState) -> bool:
    if s.current_intent is None:
        return True
    return s.last_action in COMPLETION_ACTIONS or s.last_action is None


def _state_to_dict(s: DialogueState) -> dict:
    out = {
        "current_intent": s.current_intent,
        "last_action": s.last_action,
        "confirmed": s.confirmed,
        "awaiting_carryover": s.awaiting_carryover_response,
        "completed_intents": s.context.completed_intents,
    }
    ctx = s.context
    for key, booking in (
        ("flight", ctx.flight),
        ("accommodation", ctx.accommodation),
        ("activity", ctx.activity),
    ):
        if booking.has_any_data():
            out[key] = {
                k: v
                for k, v in booking.to_dict().items()
                if v is not None and k != "completed"
            }
    if s.compare_cities_data:
        out["compare_cities"] = s.compare_cities_data
    return out


def _grounding_mode(s: DialogueState) -> str:
    if not s.last_action:
        return "base"
    if s.last_action in ("ASK_CONFIRMATION", "OFFER_SLOT_CARRYOVER"):
        return "confirmation"
    if s.last_action == "REQUEST_SLOT_CHANGE":
        return "slot_change"
    return "base"


# ── Core turn logic ───────────────────────────────────────────────────────────

def _run_turn(user_input: str, skip_split: bool = False) -> dict:
    """
    Execute one complete pipeline turn.
    Must be called while _lock is held.
    Returns a dict with response, debug info, done flag, and next_queued.
    """
    global _state, _intent_queue, _history, _settings

    use_splitter = _settings["use_splitter"] and not skip_split
    use_llm_dm   = _settings["use_llm_dm"]
    debug = {}

    # ── 1. Intent Splitter ────────────────────────────────────────────────────
    if use_splitter and _is_flow_idle(_state):
        current_input, pending = split_intents(_pipe, user_input)
        if pending:
            _intent_queue.add(pending)
        debug["splitter"] = {
            "triggered": bool(pending),
            "current": current_input,
            "queued": pending,
        }
    else:
        current_input = user_input
        reason = (
            "flow_active" if not _is_flow_idle(_state)
            else "disabled" if not _settings["use_splitter"]
            else "skipped_queued_intent"
        )
        debug["splitter"] = {
            "triggered": False,
            "current": current_input,
            "reason": reason,
        }

    # ── 2. NLU ────────────────────────────────────────────────────────────────
    sys_prompt = state_context(_state)
    nlu_out = nlu_parse(
        pipe=_pipe,
        user_utterance=current_input,
        system_prompt=sys_prompt,
        dialogue_history=_history,
    )
    debug["nlu"] = {
        "intent": nlu_out.get("intent"),
        "slots": {k: v for k, v in nlu_out.get("slots", {}).items() if v is not None},
        "grounding_mode": _grounding_mode(_state),
    }

    # ── 3. DM ─────────────────────────────────────────────────────────────────
    action = dm_decide(
        _state, nlu_out, current_input,
        llm_pipe=_pipe if use_llm_dm else None,
    )
    base_action, slot_param = parse_action(action)
    debug["dm"] = {
        "action": action,
        "base_action": base_action,
        "slot_param": slot_param,
    }
    debug["state"] = _state_to_dict(_state)

    # ── 4. API calls ──────────────────────────────────────────────────────────
    api_results = None

    if base_action == "COMPLETE_ACCOMMODATION_BOOKING":
        acc = _state.context.accommodation
        if acc.destination and acc.check_in_date and acc.check_out_date:
            ratings = {"low": "1,2", "medium": "3,4", "high": "5"}.get(
                acc.budget_level or "medium", "3,4"
            )
            try:
                api_results = search_accommodation(
                    city=acc.destination,
                    ratings=ratings,
                    num_adults=acc.num_guests or 1,
                    start_date=acc.check_in_date,
                    end_date=acc.check_out_date,
                )
            except Exception as e:
                debug["api"] = {"error": str(e)}

    elif base_action == "COMPLETE_ACTIVITY_BOOKING":
        act = _state.context.activity
        if act.destination:
            try:
                api_results = search_activities(
                    city=act.destination,
                    activity_type=act.activity_category or "cultural",
                )
            except Exception as e:
                debug["api"] = {"error": str(e)}

    if base_action == "COMPLETE_ACCOMMODATION_BOOKING" and isinstance(api_results, list):
        api_results = api_results[:1]

    if api_results is not None and "api" not in debug:
        debug["api"] = {
            "results": api_results[:3] if isinstance(api_results, list) else api_results
        }

    # ── 5. NLG ────────────────────────────────────────────────────────────────
    response = nlg_generate(_pipe, action, _state, dialogue_history=_history)

    if api_results and base_action in (
        "COMPLETE_ACCOMMODATION_BOOKING", "COMPLETE_ACTIVITY_BOOKING"
    ):
        if isinstance(api_results, list) and api_results:
            max_n = 1 if base_action == "COMPLETE_ACCOMMODATION_BOOKING" else 3
            header = (
                "Here is the top option I found:"
                if max_n == 1
                else "Here are some options I found:"
            )
            summary = f"\n\n{header}\n"
            for i, r in enumerate(api_results[:max_n], 1):
                if isinstance(r, dict):
                    summary += f"{i}. {r.get('name', 'Option')} — {r.get('price', 'N/A')}\n"
            response += summary

    if _intent_queue.has_pending() and base_action in COMPLETION_ACTIONS:
        nxt = _intent_queue.peek()
        response += f'\n\nI also noted your earlier request: "{nxt}". I\'ll handle that next!'

    if base_action == "GOODBYE":
        _intent_queue.clear()

    # ── History update ────────────────────────────────────────────────────────
    _history.append({"role": "user", "content": current_input})
    _history.append({"role": "assistant", "content": response})

    # Signal frontend to auto-send the next queued intent (if flow just went idle)
    next_queued = None
    if _is_flow_idle(_state) and _intent_queue.has_pending():
        next_queued = _intent_queue.pop()

    return {
        "response": response,
        "debug": debug,
        "done": base_action == "GOODBYE",
        "next_queued": next_queued,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/init", methods=["GET"])
def get_init():
    with _lock:
        return jsonify({
            "greeting": GREETING_MESSAGE,
            "settings": dict(_settings),
        })


@app.route("/settings", methods=["POST"])
def update_settings():
    data = request.get_json(force=True)
    with _lock:
        if "use_splitter" in data:
            _settings["use_splitter"] = bool(data["use_splitter"])
        if "use_llm_dm" in data:
            _settings["use_llm_dm"] = bool(data["use_llm_dm"])
    return jsonify({"settings": dict(_settings)})


@app.route("/reset", methods=["POST"])
def reset():
    global _state, _intent_queue, _history
    with _lock:
        _state = DialogueState()
        _intent_queue = IntentQueue()
        _history = []
    return jsonify({"greeting": GREETING_MESSAGE})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_input = (data.get("message") or "").strip()
    # skip_split=True when the frontend auto-sends a queued intent
    skip_split = bool(data.get("skip_split", False))

    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    with _lock:
        result = _run_turn(user_input, skip_split=skip_split)

    return jsonify(result)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initializing LLM pipeline — this may take a minute...")
    _pipe = make_llm()
    _state = DialogueState()
    _intent_queue = IntentQueue()
    _history = []
    print("Pipeline ready. Open http://localhost:5000")
    # threaded=False: the LLM is not thread-safe; requests are serialized by _lock
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

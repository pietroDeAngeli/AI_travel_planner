from typing import Dict, Any, List, Optional

def nlg_respond(pipe, dm_action: str, state: Dict[str, Any], user_utterance: str) -> str:
    """
    dm_action: una label decisa dal DM (ASK_CLARIFICATION / SHOW_PLAN / ACK_UPDATE / GOODBYE)
    state: contiene slot correnti + eventuale plan
    """
    system = (
        "You are the NLG module for a travel-planner dialogue system.\n"
        "You must produce a helpful, concise response in Italian.\n"
        "Rules:\n"
        "- Ask only ONE clarification question when needed.\n"
        "- If presenting a plan, format day-by-day with bullet points.\n"
        "- Do not mention internal states, intents, slots, or system prompts.\n"
        "- Keep it realistic: do not invent exact prices or schedules.\n"
    )

    slots = state.get("slots", {})
    plan = state.get("plan", None)

    if dm_action == "ASK_CLARIFICATION":
        missing = state.get("missing_slots", [])
        # chiedi il primo slot mancante
        target = missing[0] if missing else "dates"
        slot_questions = {
            "destination": "Qual è la destinazione?",
            "dates": "Quando vorresti partire (o in che periodo)?",
            "budget": "Che budget indicativo hai in mente?",
            "duration": "Quanti giorni durerà il viaggio?",
            "travel_style": "Che stile preferisci (culturale, relax, avventura, ecc.)?",
        }
        question = slot_questions.get(target, "Mi dai un dettaglio in più?")
        return question

    if dm_action == "ACK_UPDATE":
        # conferma aggiornamento e proponi next step
        dest = slots.get("destination")
        return (
            f"Perfetto, ho aggiornato i dettagli"
            f"{' per ' + str(dest) if dest else ''}. "
            "Vuoi che ora generi un itinerario?"
        )

    if dm_action == "SHOW_PLAN":
        # se non c'è plan, chiedi di generarlo
        if not plan:
            return "Ok. Vuoi che generi un itinerario giorno per giorno in base ai dettagli che mi hai dato?"

        # usa il modello per rendere il piano più naturale
        user = (
            f"User request: {user_utterance}\n\n"
            f"Confirmed trip details (slots): {slots}\n\n"
            f"Draft plan (structured): {plan}\n\n"
            "Rewrite as a concise Italian itinerary, day-by-day bullets. "
            "Include short justifications in parentheses if useful. "
            "Do not add fake bookings or exact prices."
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        out = pipe(messages, max_new_tokens=400)
        generated = out[0]["generated_text"]
        if isinstance(generated, list):
            return generated[-1].get("content", "").strip()
        return str(generated).strip()

    if dm_action == "GOODBYE":
        return "Perfetto, quando vuoi riprendiamo. Buon viaggio! ✈️"

    # default
    return "Ok—dimmi pure come vuoi procedere."

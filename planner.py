from typing import Dict, Any

def simple_plan_from_slots(slots: Dict[str, Any]) -> Dict[str, Any]:
    """
    Piano base strutturato (no LLM), così hai output deterministico.
    Poi NLG lo rende “bello”.
    """
    duration = slots.get("duration") or "3"
    try:
        days = int(str(duration).strip().split()[0])
    except Exception:
        days = 3

    dest = slots.get("destination") or "la destinazione"
    style = slots.get("travel_style") or "mix"

    plan = {}
    for d in range(1, max(1, days) + 1):
        plan[f"Day {d}"] = [
            f"Mattina: attività {style} in {dest}",
            f"Pomeriggio: esplorazione quartieri / attrazioni principali",
            f"Sera: cena tipica e passeggiata",
        ]
    return plan

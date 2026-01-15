from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from typing import Literal

# data.py
from schema import INTENT_SLOTS, SLOTS, INTENT_SCHEMAS

@dataclass
class UserInformation:
    destination: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    num_people: Optional[int] = None
    overall_budget: Optional[float] = None
    travel_style: Optional[str] = None
    travel_method: Optional[str] = None
    accommodation_type: Optional[str] = None
    budget_level: Optional[Literal["low", "medium", "high"]] = None
    leaving_time_preference: Optional[str] = None

    def __str__(self) -> str:
        fields = [
            f"destination={self.destination}",
            f"start_date={self.start_date}",
            f"end_date={self.end_date}",
            f"num_people={self.num_people}",
            f"overall_budget={self.overall_budget}",
            f"travel_style={self.travel_style}",
            f"travel_method={self.travel_method}",
            f"accommodation_type={self.accommodation_type}",
            f"budget_level={self.budget_level}",
            f"leaving_time_preference={self.leaving_time_preference}",
        ]
        return "UserInformation(" + ", ".join(fields) + ")"

    def update_info(self, new_info: Dict[str, Any]) -> None:
        for key, value in new_info.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

    def to_dict(self, intent: Optional[str]) -> Dict[str, Any]:
        if intent is None:
            relevant_slots = SLOTS
        else:
            relevant_slots = INTENT_SCHEMAS.get(intent, {}).get("slots", [])

        return {slot: getattr(self, slot) for slot in relevant_slots}
    
    def missing_slots(self, intent: Optional[str]) -> List[str]:
        required = INTENT_SLOTS.get(intent, [])
        return [s for s in required if getattr(self, s, None) is None]
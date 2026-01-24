from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from typing import Literal

# =============================================================================
# BOOKING DATA CLASSES
# =============================================================================
@dataclass
class FlightBooking:
    """Self-contained data for flight booking."""
    origin: Optional[str] = None
    destination: Optional[str] = None
    departure_date: Optional[str] = None
    return_date: Optional[str] = None
    num_passengers: Optional[int] = None
    budget_level: Optional[Literal["low", "medium", "high"]] = None
    completed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": self.origin,
            "destination": self.destination,
            "departure_date": self.departure_date,
            "return_date": self.return_date,
            "num_passengers": self.num_passengers,
            "budget_level": self.budget_level,
        }

    def missing_slots(self) -> List[str]:
        required = ["origin", "destination", "departure_date", "num_passengers", "budget_level"]
        return [s for s in required if getattr(self, s, None) is None]

    def update(self, slots: Dict[str, Any]) -> None:
        for key, value in slots.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

    def has_any_data(self) -> bool:
        return any([self.origin, self.destination, self.departure_date, 
                    self.return_date, self.num_passengers, self.budget_level])


@dataclass
class AccommodationBooking:
    """Self-contained data for accommodation booking."""
    destination: Optional[str] = None
    check_in_date: Optional[str] = None
    check_out_date: Optional[str] = None
    num_guests: Optional[int] = None
    budget_level: Optional[Literal["low", "medium", "high"]] = None
    completed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "destination": self.destination,
            "check_in_date": self.check_in_date,
            "check_out_date": self.check_out_date,
            "num_guests": self.num_guests,
            "budget_level": self.budget_level,
        }

    def missing_slots(self) -> List[str]:
        required = ["destination", "check_in_date", "check_out_date", "num_guests", "budget_level"]
        return [s for s in required if getattr(self, s, None) is None]

    def update(self, slots: Dict[str, Any]) -> None:
        for key, value in slots.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

    def has_any_data(self) -> bool:
        return any([self.destination, self.check_in_date, self.check_out_date, self.num_guests, self.budget_level])


@dataclass
class ActivityBooking:
    """Self-contained data for activity booking."""
    destination: Optional[str] = None
    activity_category: Optional[str] = None
    budget_level: Optional[Literal["low", "medium", "high"]] = None
    completed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "destination": self.destination,
            "activity_category": self.activity_category,
            "budget_level": self.budget_level,
        }

    def missing_slots(self) -> List[str]:
        required = ["destination", "activity_category", "budget_level"]
        return [s for s in required if getattr(self, s, None) is None]

    def update(self, slots: Dict[str, Any]) -> None:
        for key, value in slots.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

    def has_any_data(self) -> bool:
        return any([self.destination, self.activity_category, self.budget_level])


# =============================================================================
# TRIP CONTEXT - Holds all bookings and enables slot carryover
# =============================================================================

@dataclass
class TripContext:
    """
    Maintains context across multiple bookings.
    Enables smart slot carryover between booking types.
    """
    flight: FlightBooking = field(default_factory=FlightBooking)
    accommodation: AccommodationBooking = field(default_factory=AccommodationBooking)
    activity: ActivityBooking = field(default_factory=ActivityBooking)
    
    # Track completed bookings
    completed_intents: List[str] = field(default_factory=list)

    def get_booking(self, intent: str):
        """Get the booking object for an intent."""
        mapping = {
            "BOOK_FLIGHT": self.flight,
            "BOOK_ACCOMMODATION": self.accommodation,
            "BOOK_ACTIVITY": self.activity,
        }
        return mapping.get(intent)

    def get_carryover_values(self, from_intent: str, to_intent: str) -> Dict[str, Any]:
        """
        Get values that can be carried over from one booking to another.
        Returns a dict of {target_slot: value} for non-None values.
        """

        source_booking = self.get_booking(from_intent)
        if source_booking is None:
            return {}
        
        slot_map = get_carryover_slots(from_intent, to_intent)
        carryover = {}
        
        for source_slot, target_slot in slot_map.items():
            value = getattr(source_booking, source_slot, None)
            if value is not None:
                carryover[target_slot] = value
        
        return carryover


    def mark_completed(self, intent: str) -> None:
        """Mark a booking as completed."""
        booking = self.get_booking(intent)
        if booking and hasattr(booking, 'completed'):
            booking.completed = True
        if intent not in self.completed_intents:
            self.completed_intents.append(intent)

    def __str__(self) -> str:
        parts = []
        if self.flight.has_any_data():
            parts.append(f"Flight: {self.flight.to_dict()}")
        if self.accommodation.has_any_data():
            parts.append(f"Accommodation: {self.accommodation.to_dict()}")
        if self.activity.has_any_data():
            parts.append(f"Activity: {self.activity.to_dict()}")
        if self.completed_intents:
            parts.append(f"Completed: {self.completed_intents}")
        return "\n".join(parts) if parts else "TripContext(empty)"
    
CARRYOVER_SLOTS = {
    # From BOOK_FLIGHT to other intents
    ("BOOK_FLIGHT", "BOOK_ACCOMMODATION"): {
        "destination": "destination",
        "num_passengers": "num_guests",
        "departure_date": "check_in_date",
        "return_date": "check_out_date",
        "budget_level": "budget_level",
    },
    ("BOOK_FLIGHT", "BOOK_ACTIVITY"): {
        "destination": "destination",
        "budget_level": "budget_level",
    },
    
    # From BOOK_ACCOMMODATION to other intents
    ("BOOK_ACCOMMODATION", "BOOK_FLIGHT"): {
        "destination": "destination",
        "num_guests": "num_passengers",
        "check_in_date": "departure_date",
        "check_out_date": "return_date",
        "budget_level": "budget_level",
    },
    ("BOOK_ACCOMMODATION", "BOOK_ACTIVITY"): {
        "destination": "destination",
        "budget_level": "budget_level",
    },
    
    # From BOOK_ACTIVITY to other intents
    ("BOOK_ACTIVITY", "BOOK_FLIGHT"): {
        "destination": "destination",
        "budget_level": "budget_level",
    },
    ("BOOK_ACTIVITY", "BOOK_ACCOMMODATION"): {
        "destination": "destination",
        "budget_level": "budget_level",
    },
}

def get_carryover_slots(from_intent: str, to_intent: str) -> Dict[str, str]:
    """
    Get slot mappings for carryover between two intents.
    Returns {source_slot: target_slot} dictionary.
    """
    return CARRYOVER_SLOTS.get((from_intent, to_intent), {})
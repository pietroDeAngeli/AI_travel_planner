import os
import time
import requests
from typing import Optional, Tuple, List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from schema import ACTIVITY_CATEGORIES

BASE_URL = "https://test.api.amadeus.com"

AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET")

# Cache for access token (token, expiry_time)
_token_cache: Tuple[Optional[str], float] = (None, 0)

# Cache for geocoded cities
_geocode_cache: Dict[str, Tuple[float, float]] = {}


def get_access_token() -> str:
    """Get Amadeus access token with caching to avoid unnecessary API calls."""
    global _token_cache
    token, expiry = _token_cache
    
    # Return cached token if still valid (with 60s buffer)
    if token and time.time() < expiry - 60:
        return token
    
    url = f"{BASE_URL}/v1/security/oauth2/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": AMADEUS_API_KEY,
        "client_secret": AMADEUS_API_SECRET
    }

    try:
        response = requests.post(url, headers=headers, data=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        token = result["access_token"]
        expires_in = result.get("expires_in", 1800)  # Default 30 min
        _token_cache = (token, time.time() + expires_in)
        return token
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to get access token: {e}")


def geocode_city(city: str) -> Tuple[float, float]:
    """Get latitude/longitude for a city with caching."""
    city_lower = city.lower().strip()
    
    if city_lower in _geocode_cache:
        return _geocode_cache[city_lower]
    
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": city, "format": "json", "limit": 1}
    headers = {"User-Agent": "travel-planner-demo"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Geocoding request failed: {e}")

    if not data:
        raise ValueError(f"City not found: {city}")

    lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
    _geocode_cache[city_lower] = (lat, lon)
    return lat, lon

def search_flights():
    """Placeholder for flight search functionality."""
    return "(You can invent the flight details)"


def search_activities(city: str, radius_km: int = 3, activity_type: str = "cultural") -> List[Dict[str, Any]]:
    """Search for activities in a city, sorted by preferred activity type."""
    try:
        token = get_access_token()
        lat, lon = geocode_city(city)
    except (RuntimeError, ValueError) as e:
        print(f"[API] Setup failed: {e}")
        return []

    url = f"{BASE_URL}/v1/shopping/activities"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"latitude": lat, "longitude": lon, "radius": radius_km}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"[API] HTTP error: {e.response.status_code} - {e.response.text}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"[API] Request failed: {e}")
        return []

    activities = parse_activities(response.json())
    # Sort by preferred activity type first, then by name
    activities.sort(key=lambda a: (a["activity_type"] != activity_type, a["name"] or ""))
    return activities[:10]


def compare_options(city1: str, city2: str, compare_type: str) -> Tuple[List, List]:
    """Compare activities between two cities for a given activity type."""
    if compare_type not in ACTIVITY_CATEGORIES:
        return [], []
    
    activities1 = search_activities(city1, activity_type=compare_type)
    activities2 = search_activities(city2, activity_type=compare_type)
    return activities1, activities2


def classify_activity(name: Optional[str]) -> str:
    """Classify an activity name into a category based on keywords."""
    if not name:
        return "general"

    name_lower = name.lower()
    for category, keywords in ACTIVITY_CATEGORIES.items():
        if any(k in name_lower for k in keywords):
            return category
    return "general"


def search_accomodation(
    city: str,
    radius_km: int = 3,
    ratings: str = "1,2,3,4,5",
    num_adults: int = 1,
    start_date: str = "YYYY-MM-DD",
    end_date: str = "YYYY-MM-DD"
) -> List[Dict[str, Any]]:
    """Search for accommodation in a city with availability check."""
    try:
        token = get_access_token()
        lat, lon = geocode_city(city)
    except (RuntimeError, ValueError) as e:
        print(f"[API] Setup failed: {e}")
        return []

    headers = {"Authorization": f"Bearer {token}"}

    # Step 1: Get hotels by location
    url = f"{BASE_URL}/v1/reference-data/locations/hotels/by-geocode"
    params = {
        "latitude": lat,
        "longitude": lon,
        "radius": radius_km,
        "radiusUnit": "KM",
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"[API] HTTP error (hotels list): {e.response.status_code}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"[API] Request failed (hotels list): {e}")
        return []

    hotels = parse_hotels_list(response.json())
    if not hotels:
        return []

    hotels.sort(key=lambda h: h["distance"] or float('inf'))
    hotels = hotels[:10]

    # Step 2: Get offers for those hotels
    hotel_ids = [h["hotelId"] for h in hotels if h.get("hotelId")]
    if not hotel_ids:
        return []

    url = f"{BASE_URL}/v3/shopping/hotel-offers"
    params = {
        "hotelIds": hotel_ids,
        "adults": num_adults,
        "checkInDate": start_date,
        "checkOutDate": end_date,
        "roomQuantity": 1,
        "includeClosed": False,
        "lang": "EN",
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"[API] HTTP error (hotel offers): {e.response.status_code}")
        return hotels  # Return basic hotel info if offers fail
    except requests.exceptions.RequestException as e:
        print(f"[API] Request failed (hotel offers): {e}")
        return hotels

    rooms = parse_hotels_search(response.json())
    rooms_by_id = {r["hotelId"]: r for r in rooms if r.get("hotelId")}

    # Merge hotel info with room offers
    merged = [
        {**h, **rooms_by_id[h["hotelId"]]}
        for h in hotels
        if h["hotelId"] in rooms_by_id
    ]

    return merged if merged else hotels


def parse_activities(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse activities response from Amadeus API."""
    activities = []
    for a in data.get("data", []):
        activities.append({
            "name": a.get("name"),
            "description": a.get("shortDescription"),
            "rating": a.get("rating"),
            "price": a.get("price", {}).get("amount"),
            "currency": a.get("price", {}).get("currencyCode"),
            "activity_type": classify_activity(a.get("name"))
        })
    return activities


def parse_hotels_list(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse hotels list response from Amadeus API."""
    hotels = []
    for a in data.get("data", []):
        hotels.append({
            "name": a.get("name"),
            "hotelId": a.get("hotelId"),
            "distance": a.get("distance", {}).get("value")
        })
    return hotels


def parse_hotels_search(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse hotel offers response from Amadeus API."""
    rooms = []
    for a in data.get("data", []):
        if not a.get("available", False):
            continue

        hotel = a.get("hotel", {})
        offers = a.get("offers", [])
        offer = offers[0] if offers else {}

        obj = {
            # Hotel info
            "hotelId": hotel.get("hotelId"),
            "latitude": hotel.get("latitude"),
            "longitude": hotel.get("longitude"),
            "contact": hotel.get("contact", {}).get("phone"),
            # Room/offer info
            "price": offer.get("price", {}).get("total"),
            "currency": offer.get("price", {}).get("currency"),
            "description": offer.get("room", {}).get("description", {}).get("text"),
            "boardType": offer.get("boardType"),
            "cancellationPolicy": offer.get("policies", {}).get("cancellation", {}).get("description", {}).get("text"),
            "paymentType": offer.get("policies", {}).get("paymentType"),
        }
        rooms.append(obj)
    return rooms
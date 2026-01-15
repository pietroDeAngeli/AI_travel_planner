
BASE_URL = "https://test.api.amadeus.com"
import requests

from dotenv import load_dotenv
load_dotenv()

import os

AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET")

ACTIVITY_CATEGORIES = {

    "adventure": [
        "adventure", "hiking", "trekking", "climbing",
        "kayak", "rafting", "bike", "biking", "cycling",
        "outdoor", "jeep", "atv", "quad", "safari"
    ],

    "walk": [
        "walking", "walk", "free tour",
        "city tour", "guided walk",
        "walking tour", "on foot", "sightseeing"
    ],

    "cultural": [
        "museum", "gallery", "art", "exhibition",
        "cathedral", "church", "basilica",
        "palace", "royal", "historic", "history",
        "monument", "heritage", "archaeological",
    ],

    "food": [
        "food", "wine", "tapas", "gastronomy",
        "culinary", "tasting", "dinner", "lunch",
        "market", "cooking", "cooking class"
    ],

    "sport": [
        "stadium", "football", "soccer",
        "basketball", "tennis",
        "bernabeu", "arena", "olympic"
    ],

    "relax": [
        "spa", "wellness", "relax",
        "thermal", "bath", "cruise",
        "boat", "river", "panoramic", "sunset"
    ],

    "nature": [
        "nature", "park", "garden",
        "botanical", "natural",
        "scenic", "landscape",
        "mountain", "lake"
    ],

    "nightlife": [
        "night", "nightlife", "bar", "pub",
        "club", "show", "concert",
        "music", "live", "flamenco"
    ],

    "family": [
        "family", "kids", "children",
        "zoo", "aquarium",
        "theme park", "amusement", "park"
    ]
}


def search_activities(city, radius_km=3, activity_type="cultural"):
    token = get_access_token()
    lat, lon = geocode_city(city)

    url = f"{BASE_URL}/v1/shopping/activities"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    params = {
        "latitude": lat,
        "longitude": lon,
        "radius": radius_km
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

    except requests.exceptions.HTTPError as e:
        print("HTTP error:", e.response.status_code)
        print(e.response.text)
        return []
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return []

    activities = parse_activities(response.json())

    # Sort categories by preferred activity
    activities.sort(key=lambda a: (a["activity_type"] != activity_type, a["name"]))

    return activities[:5]

def classify_activity(name: str | None) -> str:
    if not name:
        return "other"

    name = name.lower()
    for category, keywords in ACTIVITY_CATEGORIES.items():
        if any(k in name for k in keywords):
            return category
    return "other"


def search_accomodation(city, radius_km=3, ratings="1,2,3,4,5", num_adults=1, start_date="YYYY-MM-DD", end_date="YYYY-MM-DD"):
    token = get_access_token()
    lat, lon = geocode_city(city)

    url = f"{BASE_URL}/v1/reference-data/locations/hotels/by-geocode"
    headers = {
        "Authorization": f"Bearer {token}"
    }

    params = {
        "latitude": lat,
        "longitude": lon,
        "radius": radius_km,
        "radiusUnit": "KM",
        #"ratings": ratings
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

    except requests.exceptions.HTTPError as e:
        print("HTTP error:", e.response.status_code)
        print(e.response.text)
        return []
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return []

    hotels = parse_hotels_list(response.json())

    hotels.sort(key=lambda h: h["distance"])

    hotels = hotels[:10]

    ids = [hotel["hotelId"] for hotel in hotels]

    url = f"{BASE_URL}/v3/shopping/hotel-offers"

    params = {
        "hotelIds": ids,
        "adults": num_adults,
        "checkInDate": start_date,
        "checkOutDate": end_date,
        "roomQuantity": 1,
        "includeClosed": False,
        "lang": "EN",
        #"paymentPolicy": "NONE",
        #"boardType": "ROOM_ONLY",
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

    except requests.exceptions.HTTPError as e:
        print("HTTP error:", e.response.status_code)
        print(e.response.text)
        return []
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return []

    
    rooms = parse_hotels_search(response.json())

    rooms_by_id = {r["hotelId"]: r for r in rooms}

    merged = [
        {**h, **rooms_by_id[h["hotelId"]]}
        for h in hotels
        if h["hotelId"] in rooms_by_id
    ]

    return merged

def get_access_token():
    url = f"{BASE_URL}/v1/security/oauth2/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "client_credentials",
        "client_id": AMADEUS_API_KEY,
        "client_secret": AMADEUS_API_SECRET
    }

    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()

    return response.json()["access_token"]

def geocode_city(city):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": city,
        "format": "json",
        "limit": 1
    }
    headers = {"User-Agent": "travel-planner-demo"}

    r = requests.get(url, params=params, headers=headers)
    r.raise_for_status()
    data = r.json()

    if not data:
        raise ValueError(f"City not found: {city}")

    return float(data[0]["lat"]), float(data[0]["lon"])

def parse_activities(data):
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

def parse_hotels_list(data):
    hotels = []

    for a in data.get("data", []):
        hotels.append({
            "name": a.get("name"),
            "hotelId": a.get("hotelId"),
            "distance": a.get("distance", {}).get("value")
            #"unit": a.get("distance").get("value")
        })

    return hotels

def parse_hotels_search(data):
    rooms = []

    for a in data.get("data", []):
        obj = {}

        if not a.get("available", False):
            continue

        # Hotel info
        obj["hotelId"] = a.get("hotel", {}).get("hotelId")
        obj["latitude"] = a.get("hotel", {}).get("latitude")
        obj["longitude"] = a.get("hotel", {}).get("longitude")
        obj["contact"] = a.get("hotel", {}).get("contact", {}).get("phone")

        # Room info
        offers = a.get("offers", [])
        offer = offers[0] if offers else {}

        obj["price"] = offer.get("price", {}).get("total")
        obj["currency"] = offer.get("price", {}).get("currency")
        obj["description"] = offer.get("roomInformation", {}).get("description")
        obj["boardType"] = offer.get("boardType")
        obj["cancellationPolicy"] = offer.get("policies", {}).get("refundable", {}).get("cancellationRefund")
        obj["paymentType"] = offer.get("policies", {}).get("paymentType")

        rooms.append(obj)
    return rooms
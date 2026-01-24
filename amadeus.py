
BASE_URL = "https://test.api.amadeus.com"
import requests
from schema import ACTIVITY_CATEGORIES

from dotenv import load_dotenv
load_dotenv()

import os

AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET")

def search_flights():
    return """Flight search not implemented yet."""


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

    return activities[:10]

def compare_options(city1: str, city2: str, compare_type: str):
    if compare_type not in ACTIVITY_CATEGORIES:
        return [], []
    
    activities1 = search_activities(city1, activity_type=compare_type)
    activities2 = search_activities(city2, activity_type=compare_type)
    return activities1, activities2

def request_information(city: str, entity_type: str):
    if entity_type not in ["hotels", "flights", "activities"]:
        return []

    if entity_type == "activities":
        return search_activities(city)
    elif entity_type == "hotels":
        return search_accomodation(city)
    else:
        # Flights search not implemented
        return []

def classify_activity(name: str | None) -> str:
    if not name:
        return "general"

    name = name.lower()
    for category, keywords in ACTIVITY_CATEGORIES.items():
        if any(k in name for k in keywords):
            return category
    return "general"


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
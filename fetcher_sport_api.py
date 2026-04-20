"""
Fetcher for SportAPI (sportapi7.p.rapidapi.com) via RapidAPI.
Replaces fetcher_api_football.py.

Endpoint used:
    GET /sport/football/scheduled-events/{date}  →  all football events for a given date
"""

import os
import time
import logging
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://sportapi7.p.rapidapi.com/api/v1"
HOST     = "sportapi7.p.rapidapi.com"


def _headers() -> dict:
    key = os.environ.get("RAPIDAPI_KEY", "")
    if not key:
        raise EnvironmentError("RAPIDAPI_KEY is not set")
    logger.info("RAPIDAPI_KEY chargée : %s... (longueur: %d)", key[:8], len(key))
    return {
        "x-rapidapi-key":  key,
        "x-rapidapi-host": HOST,
    }


def _get(endpoint: str, params: dict = None) -> dict | None:
    url = f"{BASE_URL}{endpoint}"
    for attempt in range(1, 4):
        try:
            time.sleep(1)
            resp = requests.get(url, headers=_headers(), params=params or {}, timeout=20)
            if resp.status_code == 429:
                logger.warning("HTTP 429 on %s (attempt %d/3) — waiting 5s", url, attempt)
                time.sleep(5)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            logger.warning("HTTP %s on %s: %s", e.response.status_code, url, e)
            return None
        except Exception as e:
            logger.error("Request error on %s: %s", url, e)
            return None
    logger.error("All 3 attempts failed for %s", url)
    return None


# --------------------------------------------------------------------------- #
#  Scheduled events for a given date                                           #
# --------------------------------------------------------------------------- #

def get_scheduled_events(date: str) -> list[dict]:
    """
    Fetch all scheduled football events for `date` (YYYY-MM-DD).
    Returns only not-started matches.
    """
    data = _get(f"/sport/football/scheduled-events/{date}")
    if not data:
        return []

    events = data.get("events", [])
    not_started = [
        e for e in events
        if e.get("status", {}).get("type") == "notstarted"
    ]
    logger.info("sport-api: %d scheduled events on %s (%d not started)",
                len(events), date, len(not_started))
    return not_started


# --------------------------------------------------------------------------- #
#  Normalize a raw event into a common fixture dict                            #
# --------------------------------------------------------------------------- #

def _normalize(event: dict) -> dict:
    home = event.get("homeTeam", {})
    away = event.get("awayTeam", {})
    tournament = event.get("tournament", {})
    unique_tournament = tournament.get("uniqueTournament", {})
    competition_name = unique_tournament.get("name") or tournament.get("name", "Unknown")

    start_ts = event.get("startTimestamp")
    if start_ts:
        match_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat()
    else:
        match_dt = None

    return {
        # Core identity
        "fixture": {
            "id":   event.get("id"),
            "date": match_dt,
        },
        # Teams
        "teams": {
            "home": {
                "id":   home.get("id"),
                "name": home.get("name", ""),
            },
            "away": {
                "id":   away.get("id"),
                "name": away.get("name", ""),
            },
        },
        # League / competition
        "league": {
            "id":     unique_tournament.get("id") or tournament.get("id"),
            "name":   competition_name,
            "season": event.get("season", {}).get("year"),
        },
        # Source metadata (mirrors api-football convention)
        "_competition_name": competition_name,
        "_competition_code": f"SA-{unique_tournament.get('id') or tournament.get('id', 'unknown')}",
        "_source": "sport-api",
        # Raw event kept for downstream use
        "_raw": event,
    }


# --------------------------------------------------------------------------- #
#  Public interface (mirrors fetcher_api_football)                             #
# --------------------------------------------------------------------------- #

def fetch_all_enriched() -> list[dict]:
    """Fetch and normalize today's not-started football matches."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    events = get_scheduled_events(today)
    enriched = []
    for e in events:
        try:
            enriched.append(_normalize(e))
        except Exception as exc:
            logger.error("Error normalizing event %s: %s", e.get("id"), exc)
    logger.info("sport-api: %d enriched fixtures returned", len(enriched))
    return enriched

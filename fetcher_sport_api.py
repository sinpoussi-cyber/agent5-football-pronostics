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
            "id":        unique_tournament.get("id") or tournament.get("id"),
            "name":      competition_name,
            "season":    event.get("season", {}).get("year"),
            "season_id": event.get("season", {}).get("id"),
        },
        # Source metadata (mirrors api-football convention)
        "_competition_name": competition_name,
        "_competition_code": f"SA-{unique_tournament.get('id') or tournament.get('id', 'unknown')}",
        "_source": "sport-api",
        # Raw event kept for downstream use
        "_raw": event,
    }


# --------------------------------------------------------------------------- #
#  Convert SofaScore event format → api-football-compatible dict               #
# --------------------------------------------------------------------------- #

def _to_af_format(event: dict) -> dict:
    """Normalize a SofaScore event to the api-football structure expected by analyzer."""
    home = event.get("homeTeam", {})
    away = event.get("awayTeam", {})
    return {
        "teams": {
            "home": {"id": home.get("id"), "name": home.get("name", "")},
            "away": {"id": away.get("id"), "name": away.get("name", "")},
        },
        "goals": {
            "home": event.get("homeScore", {}).get("current"),
            "away": event.get("awayScore", {}).get("current"),
        },
    }


# --------------------------------------------------------------------------- #
#  Enrichment — form, standings, H2H                                           #
# --------------------------------------------------------------------------- #

def get_team_form(team_id: int, last_n: int = 5) -> list[dict] | None:
    """
    Fetch last `last_n` finished events for a team.
    Returns api-football-compatible dicts, or None if data is unavailable.
    """
    data = _get(f"/team/{team_id}/events/last/0")
    if data is None:
        return None
    events = data.get("events", [])
    if not events:
        return None
    finished = [
        _to_af_format(e)
        for e in events
        if e.get("status", {}).get("type") == "finished"
        and e.get("homeScore", {}).get("current") is not None
        and e.get("awayScore", {}).get("current") is not None
    ]
    return finished[-last_n:] if finished else None


def get_standings(tournament_id: int, season_id: int) -> list[dict] | None:
    """
    Fetch standings for a tournament/season.
    Returns rows with 'rank' key (mapped from SofaScore 'position'), or None.
    """
    data = _get(f"/tournament/{tournament_id}/season/{season_id}/standings/total")
    if data is None:
        return None
    standings_list = data.get("standings", [])
    if not standings_list:
        return None
    rows = standings_list[0].get("rows", [])
    if not rows:
        return None
    return [
        {
            "team":           {"id": row.get("team", {}).get("id")},
            "rank":           row.get("position", 99),
            "points":         row.get("points", 0),
            "goalDifference": row.get("goalDifference", 0),
        }
        for row in rows
    ]


def get_h2h(event_id: int, last_n: int = 5) -> list[dict] | None:
    """
    Fetch H2H matches for a given event.
    Returns api-football-compatible dicts, or None if data is unavailable.
    """
    data = _get(f"/event/{event_id}/h2h")
    if data is None:
        return None
    # SofaScore may return H2H under 'events' or 'previousEvents'
    events = data.get("events") or data.get("previousEvents", [])
    if not events:
        return None
    finished = [
        _to_af_format(e)
        for e in events
        if e.get("status", {}).get("type") == "finished"
        and e.get("homeScore", {}).get("current") is not None
        and e.get("awayScore", {}).get("current") is not None
    ]
    return finished[-last_n:] if finished else None


def _enrich(normalized: dict) -> dict:
    """Attach form, standings, and H2H to a normalized event dict in-place."""
    home_id       = normalized["teams"]["home"]["id"]
    away_id       = normalized["teams"]["away"]["id"]
    event_id      = normalized["fixture"]["id"]
    tournament_id = normalized["league"]["id"]
    season_id     = normalized["league"].get("season_id")

    normalized["_home_form"] = get_team_form(home_id)
    normalized["_away_form"] = get_team_form(away_id)
    normalized["_h2h"]       = get_h2h(event_id)
    normalized["_standings"] = (
        get_standings(tournament_id, season_id)
        if tournament_id and season_id
        else None
    )
    return normalized


# --------------------------------------------------------------------------- #
#  Public interface (mirrors fetcher_api_football)                             #
# --------------------------------------------------------------------------- #

def fetch_all_enriched() -> list[dict]:
    """Fetch, normalize, and enrich today's not-started football matches."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    events = get_scheduled_events(today)
    enriched = []
    for e in events:
        try:
            normalized = _normalize(e)
            normalized = _enrich(normalized)
            enriched.append(normalized)
        except Exception as exc:
            logger.error("Error processing event %s: %s", e.get("id"), exc)
    logger.info("sport-api: %d enriched fixtures returned", len(enriched))
    return enriched

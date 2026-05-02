"""
Fetcher for SportAPI (sportapi7.p.rapidapi.com) via RapidAPI.

Endpoints used:
    GET /sport/football/scheduled-events/{date}                        →  events for a date
    GET /team/{team_id}/events/previous/0                              →  recent finished matches
    GET /tournament/{tournament_id}/season/{season_id}/standings/total →  league standings
    GET /event/{event_id}/h2h                                          →  head-to-head history
"""

import os
import time
import logging
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://sportapi7.p.rapidapi.com/api/v1"
HOST     = "sportapi7.p.rapidapi.com"

# In-memory cache — avoids duplicate API calls within a single run
_cache: dict = {}


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
    cache_key = url + str(sorted((params or {}).items()))
    if cache_key in _cache:
        logger.debug("cache hit: %s", url)
        return _cache[cache_key]

    for attempt in range(1, 4):
        try:
            time.sleep(1)
            resp = requests.get(url, headers=_headers(), params=params or {}, timeout=20)
            if resp.status_code == 429:
                logger.warning("HTTP 429 on %s (attempt %d/3) — waiting 5s", url, attempt)
                time.sleep(5)
                continue
            resp.raise_for_status()
            data = resp.json()
            _cache[cache_key] = data
            return data
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
    match_dt = (
        datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat()
        if start_ts else None
    )

    return {
        "fixture": {
            "id":   event.get("id"),
            "date": match_dt,
        },
        "teams": {
            "home": {"id": home.get("id"), "name": home.get("name", "")},
            "away": {"id": away.get("id"), "name": away.get("name", "")},
        },
        "league": {
            "id":        unique_tournament.get("id") or tournament.get("id"),
            "name":      competition_name,
            "season":    event.get("season", {}).get("year"),
            "season_id": event.get("season", {}).get("id"),
        },
        "_competition_name": competition_name,
        "_competition_code": f"SA-{unique_tournament.get('id') or tournament.get('id', 'unknown')}",
        "_source": "sport-api",
        "_raw": event,
    }


# --------------------------------------------------------------------------- #
#  Enrichment — form, standings, H2H                                           #
# --------------------------------------------------------------------------- #

def get_team_form(team_id: int, last_n: int = 5) -> dict | None:
    """
    Fetch last `last_n` finished matches for `team_id`.

    Returns {"avg_scored": float, "avg_conceded": float, "form": "WDLWW",
             "avg_corners": float|None}
    from the team's perspective, or None if data is unavailable.
    avg_corners is None when corner data is absent from the event payload.
    """
    data = _get(f"/team/{team_id}/events/previous/0")
    if data is None:
        return None

    events = data.get("events", [])
    finished = [
        e for e in events
        if e.get("status", {}).get("type") == "finished"
        and e.get("homeScore", {}).get("current") is not None
        and e.get("awayScore", {}).get("current") is not None
    ]
    if not finished:
        return None

    recent = finished[-last_n:]
    scored_list: list[float] = []
    conceded_list: list[float] = []
    form_chars: list[str] = []
    corners_list: list[float] = []

    for e in recent:
        is_home = e.get("homeTeam", {}).get("id") == team_id
        h_score = e["homeScore"]["current"]
        a_score = e["awayScore"]["current"]
        scored   = h_score if is_home else a_score
        conceded = a_score if is_home else h_score

        scored_list.append(scored)
        conceded_list.append(conceded)
        if scored > conceded:
            form_chars.append("W")
        elif scored == conceded:
            form_chars.append("D")
        else:
            form_chars.append("L")

        # sport-api (SofaScore) exposes corner counts in the score objects
        h_corners = e.get("homeScore", {}).get("corner")
        a_corners = e.get("awayScore", {}).get("corner")
        if h_corners is not None and a_corners is not None:
            corners_list.append(float(h_corners if is_home else a_corners))

    return {
        "avg_scored":   round(sum(scored_list) / len(scored_list), 2),
        "avg_conceded": round(sum(conceded_list) / len(conceded_list), 2),
        "form":         "".join(form_chars),
        "avg_corners":  round(sum(corners_list) / len(corners_list), 2) if corners_list else None,
    }


def get_standings(tournament_id: int, season_id: int, team_id: int) -> dict | None:
    """
    Fetch standings and return the row for `team_id`.

    Returns {"position": int, "points": int, "goal_diff": int}, or None.
    """
    data = _get(f"/tournament/{tournament_id}/season/{season_id}/standings/total")
    if data is None:
        return None

    standings_list = data.get("standings", [])
    if not standings_list:
        return None

    for row in standings_list[0].get("rows", []):
        if row.get("team", {}).get("id") == team_id:
            return {
                "position":  row.get("position", 99),
                "points":    row.get("points", 0),
                "goal_diff": row.get("goalDifference", 0),
            }

    logger.warning("standings: team %d not found in tournament %d / season %d",
                   team_id, tournament_id, season_id)
    return None


def get_h2h(event_id: int, last_n: int = 5) -> list[dict] | None:
    """
    Fetch H2H history for `event_id`.

    Returns the last `last_n` finished matches as
    [{"home_score": int, "away_score": int}, ...], or None.
    """
    data = _get(f"/event/{event_id}/h2h")
    if data is None:
        return None

    events = data.get("events") or data.get("previousEvents", [])
    finished = [
        {
            "home_score": e["homeScore"]["current"],
            "away_score": e["awayScore"]["current"],
        }
        for e in events
        if e.get("status", {}).get("type") == "finished"
        and e.get("homeScore", {}).get("current") is not None
        and e.get("awayScore", {}).get("current") is not None
    ]
    if not finished:
        return None

    return finished[-last_n:]


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

    if tournament_id and season_id:
        normalized["_standings_home"] = get_standings(tournament_id, season_id, home_id)
        normalized["_standings_away"] = get_standings(tournament_id, season_id, away_id)
    else:
        normalized["_standings_home"] = None
        normalized["_standings_away"] = None

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

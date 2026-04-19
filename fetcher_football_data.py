"""
Fetcher for football-data.org API.
Covers: PL, ELC, FL1, PD, BL1, SA, DED, PPL, CL, WC
"""

import os
import time
import requests
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"

COMPETITIONS = {
    "PL":  "Premier League",
    "ELC": "Championship",
    "FL1": "Ligue 1",
    "PD":  "LaLiga",
    "BL1": "Bundesliga",
    "SA":  "Serie A (football-data)",
    "DED": "Eredivisie (football-data)",
    "PPL": "Primeira Liga (football-data)",
    "CL":  "Champions League",
    "WC":  "Coupe du Monde",
}


def _headers() -> dict:
    key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    if not key:
        raise EnvironmentError("FOOTBALL_DATA_API_KEY is not set")
    return {"X-Auth-Token": key}


def _get(endpoint: str, params: dict = None) -> dict | list | None:
    url = f"{BASE_URL}{endpoint}"
    try:
        resp = requests.get(url, headers=_headers(), params=params or {}, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as e:
        logger.warning("HTTP %s on %s: %s", e.response.status_code, url, e)
        return None
    except Exception as e:
        logger.error("Request error on %s: %s", url, e)
        return None


# --------------------------------------------------------------------------- #
#  Upcoming matches (next 24 h)                                                #
# --------------------------------------------------------------------------- #

def get_upcoming_matches() -> list[dict]:
    now = datetime.now(timezone.utc)
    date_from = now.strftime("%Y-%m-%d")
    date_to = (now + timedelta(hours=24)).strftime("%Y-%m-%d")

    matches = []
    for code, name in COMPETITIONS.items():
        data = _get(f"/competitions/{code}/matches", {
            "dateFrom": date_from,
            "dateTo":   date_to,
            "status":   "SCHEDULED",
        })
        if not data:
            continue
        for m in data.get("matches", []):
            m["_competition_name"] = name
            m["_competition_code"] = code
            m["_source"] = "football-data"
            matches.append(m)

    logger.info("football-data: %d upcoming matches found", len(matches))
    return matches


# --------------------------------------------------------------------------- #
#  Team recent form (last 5 matches)                                           #
# --------------------------------------------------------------------------- #

def get_team_form(team_id: int, limit: int = 5) -> list[dict]:
    data = _get(f"/teams/{team_id}/matches", {"status": "FINISHED", "limit": limit})
    time.sleep(6)
    if not data:
        return []
    return data.get("matches", [])[-limit:]


# --------------------------------------------------------------------------- #
#  Head-to-head (last 5 encounters)                                            #
# --------------------------------------------------------------------------- #

def get_h2h(match_id: int, limit: int = 5) -> list[dict]:
    data = _get(f"/matches/{match_id}/head2head", {"limit": limit})
    time.sleep(6)
    if not data:
        return []
    return data.get("matches", [])[-limit:]


# --------------------------------------------------------------------------- #
#  Standing (current table)                                                    #
# --------------------------------------------------------------------------- #

def get_standings(competition_code: str) -> list[dict]:
    data = _get(f"/competitions/{competition_code}/standings")
    time.sleep(6)
    if not data:
        return []
    standings = data.get("standings", [])
    if not standings:
        return []
    # Return the TOTAL table
    for table in standings:
        if table.get("type") == "TOTAL":
            return table.get("table", [])
    return standings[0].get("table", [])


# --------------------------------------------------------------------------- #
#  Enriched match payload                                                      #
# --------------------------------------------------------------------------- #

def enrich_match(match: dict) -> dict:
    home_id = match["homeTeam"]["id"]
    away_id = match["awayTeam"]["id"]
    match_id = match["id"]
    code = match.get("_competition_code", "")

    match["_home_form"]    = get_team_form(home_id)
    match["_away_form"]    = get_team_form(away_id)
    match["_h2h"]          = get_h2h(match_id)
    match["_standings"]    = get_standings(code)
    return match


def fetch_all_enriched() -> list[dict]:
    upcoming = get_upcoming_matches()
    enriched = []
    for m in upcoming:
        try:
            enriched.append(enrich_match(m))
        except Exception as e:
            logger.error("Error enriching match %s: %s", m.get("id"), e)
    return enriched

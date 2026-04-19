"""
Fetcher for API-Football via RapidAPI.
Covers: Europa, Conférence, Serie A IT, Pro League BE, Süper Lig TR,
        Eredivisie, Liga PT, Premiership SC, Superliga DK, Eliteserien NO,
        Allsvenskan SE, Ekstraklasa PL, HNL HR, Superliga RS, PL Ukraine,
        NB I HU, First Div CY, Super Liga RO, First League CZ, Super League CH,
        Bundesliga AT, Stoiximan GR, Liga A1 IL
"""

import os
import time
import requests
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"

LEAGUES = {
    3:   "Ligue Europa",
    848: "Ligue Conférence",
    135: "Serie A Italie",
    144: "Pro League Belgique",
    203: "Süper Lig Turquie",
    88:  "Eredivisie",
    94:  "Liga Portugal",
    179: "Premiership Écosse",
    119: "Superliga Danemark",
    103: "Eliteserien Norvège",
    113: "Allsvenskan Suède",
    106: "Ekstraklasa Pologne",
    210: "HNL Croatie",
    182: "Superliga Serbie",
    333: "Premier League Ukraine",
    271: "NB I Hongrie",
    262: "First Division Chypre",
    283: "Super Liga Roumanie",
    345: "First League Tchèque",
    207: "Super League Suisse",
    218: "Bundesliga Autriche",
    197: "Stoiximan Grèce",
    384: "Liga A1 Israël",
}


def _headers() -> dict:
    key = os.environ.get("RAPIDAPI_KEY", "")
    logger.info("RAPIDAPI_KEY chargée : %s... (longueur: %d)", key[:8], len(key))
    if not key:
        raise EnvironmentError("RAPIDAPI_KEY is not set")
    return {
        "X-RapidAPI-Key":  key,
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
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
#  Upcoming fixtures (today)                                                   #
# --------------------------------------------------------------------------- #

def get_upcoming_fixtures() -> list[dict]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    fixtures = []
    for league_id, league_name in LEAGUES.items():
        data = _get("/fixtures", {"league": league_id, "date": today, "status": "NS"})
        if not data:
            continue
        for f in data.get("response", []):
            f["_competition_name"] = league_name
            f["_competition_code"] = f"AF-{league_id}"
            f["_source"] = "api-football"
            fixtures.append(f)
    logger.info("api-football: %d upcoming fixtures found", len(fixtures))
    return fixtures


# --------------------------------------------------------------------------- #
#  Team last-N matches (form)                                                  #
# --------------------------------------------------------------------------- #

def get_team_form(team_id: int, league_id: int, season: int, last: int = 5) -> list[dict]:
    data = _get("/fixtures", {
        "team":   team_id,
        "league": league_id,
        "season": season,
        "last":   last,
    })
    if not data:
        return []
    return data.get("response", [])


# --------------------------------------------------------------------------- #
#  Head-to-head                                                                #
# --------------------------------------------------------------------------- #

def get_h2h(home_id: int, away_id: int, last: int = 5) -> list[dict]:
    data = _get("/fixtures/headtohead", {
        "h2h":  f"{home_id}-{away_id}",
        "last": last,
    })
    if not data:
        return []
    return data.get("response", [])


# --------------------------------------------------------------------------- #
#  Fixture statistics                                                          #
# --------------------------------------------------------------------------- #

def get_fixture_stats(fixture_id: int) -> list[dict]:
    data = _get("/fixtures/statistics", {"fixture": fixture_id})
    if not data:
        return []
    return data.get("response", [])


# --------------------------------------------------------------------------- #
#  Standings                                                                   #
# --------------------------------------------------------------------------- #

def get_standings(league_id: int, season: int) -> list[dict]:
    data = _get("/standings", {"league": league_id, "season": season})
    if not data:
        return []
    try:
        return data["response"][0]["league"]["standings"][0]
    except (IndexError, KeyError):
        return []


# --------------------------------------------------------------------------- #
#  Enriched fixture payload                                                    #
# --------------------------------------------------------------------------- #

def _current_season(fixture: dict) -> int:
    try:
        return fixture["league"]["season"]
    except (KeyError, TypeError):
        return datetime.now(timezone.utc).year


def enrich_fixture(fixture: dict) -> dict:
    league_id = fixture["league"]["id"]
    season    = _current_season(fixture)
    home_id   = fixture["teams"]["home"]["id"]
    away_id   = fixture["teams"]["away"]["id"]

    fixture["_home_form"] = get_team_form(home_id, league_id, season)
    fixture["_away_form"] = get_team_form(away_id, league_id, season)
    fixture["_h2h"]       = get_h2h(home_id, away_id)
    fixture["_standings"] = get_standings(league_id, season)
    return fixture


def fetch_all_enriched() -> list[dict]:
    upcoming = get_upcoming_fixtures()
    enriched = []
    for f in upcoming:
        try:
            enriched.append(enrich_fixture(f))
        except Exception as e:
            logger.error("Error enriching fixture %s: %s", f.get("fixture", {}).get("id"), e)
    return enriched

"""
Normalise raw match payloads from both sources and computes statistical
features used by the pronostic engine.
"""

from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _safe(value, default=0):
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _form_results(matches: list[dict], team_id: int, source: str) -> list[str]:
    """Return list of 'W', 'D', 'L' for last matches (most recent last)."""
    results = []
    for m in matches:
        if source == "football-data":
            score  = m.get("score", {}).get("fullTime", {})
            h_id   = m.get("homeTeam", {}).get("id")
            h_g    = _safe(score.get("home"))
            a_g    = _safe(score.get("away"))
        else:  # api-football
            score  = m.get("goals", {})
            h_id   = m.get("teams", {}).get("home", {}).get("id")
            h_g    = _safe(score.get("home"))
            a_g    = _safe(score.get("away"))

        is_home = (h_id == team_id)
        if h_g > a_g:
            results.append("W" if is_home else "L")
        elif h_g < a_g:
            results.append("L" if is_home else "W")
        else:
            results.append("D")
    return results


def _avg_goals(matches: list[dict], source: str, side: str = "both") -> float:
    """Average goals per match. side: 'home'|'away'|'both'."""
    totals = []
    for m in matches:
        if source == "football-data":
            score = m.get("score", {}).get("fullTime", {})
            h_g = _safe(score.get("home"))
            a_g = _safe(score.get("away"))
        else:
            score = m.get("goals", {})
            h_g = _safe(score.get("home"))
            a_g = _safe(score.get("away"))

        if side == "home":
            totals.append(h_g)
        elif side == "away":
            totals.append(a_g)
        else:
            totals.append(h_g + a_g)
    return sum(totals) / len(totals) if totals else 0.0


def _btts_rate(matches: list[dict], source: str) -> float:
    """Fraction of matches where both teams scored."""
    if not matches:
        return 0.0
    btts = 0
    for m in matches:
        if source == "football-data":
            score = m.get("score", {}).get("fullTime", {})
            h_g = _safe(score.get("home"))
            a_g = _safe(score.get("away"))
        else:
            score = m.get("goals", {})
            h_g = _safe(score.get("home"))
            a_g = _safe(score.get("away"))
        if h_g > 0 and a_g > 0:
            btts += 1
    return btts / len(matches)


def _avg_stat_api_football(form: list[dict], stat_name: str, team_id: int) -> float:
    """Extract average of a given statistic from api-football fixture stats."""
    values = []
    for fixture in form:
        stats_list = fixture.get("statistics", [])
        for team_block in stats_list:
            if team_block.get("team", {}).get("id") == team_id:
                for s in team_block.get("statistics", []):
                    if s.get("type") == stat_name:
                        values.append(_safe(s.get("value")))
    return sum(values) / len(values) if values else 0.0


# --------------------------------------------------------------------------- #
#  Normalisation layer                                                         #
# --------------------------------------------------------------------------- #

def normalize(raw: dict) -> dict | None:
    """
    Convert a raw match from either source into a unified structure.
    Returns None if essential data is missing.
    """
    source = raw.get("_source", "")

    try:
        if source == "football-data":
            return _normalize_football_data(raw)
        elif source == "api-football":
            return _normalize_api_football(raw)
        else:
            logger.warning("Unknown source: %s", source)
            return None
    except Exception as e:
        logger.error("Normalization error: %s", e)
        return None


def _normalize_football_data(raw: dict) -> dict:
    home_id   = raw["homeTeam"]["id"]
    away_id   = raw["awayTeam"]["id"]
    home_name = raw["homeTeam"]["name"]
    away_name = raw["awayTeam"]["name"]
    source    = "football-data"

    home_form  = raw.get("_home_form", [])
    away_form  = raw.get("_away_form", [])
    h2h        = raw.get("_h2h", [])
    standings  = raw.get("_standings", [])

    home_rank = _get_rank_fbd(standings, home_id)
    away_rank = _get_rank_fbd(standings, away_id)

    return {
        "match_id":         raw.get("id"),
        "source":           source,
        "competition":      raw.get("_competition_name", ""),
        "competition_code": raw.get("_competition_code", ""),
        "utc_date":         raw.get("utcDate", ""),
        "home_id":          home_id,
        "away_id":          away_id,
        "home_name":        home_name,
        "away_name":        away_name,
        "home_rank":        home_rank,
        "away_rank":        away_rank,
        # Form
        "home_form":        _form_results(home_form, home_id, source),
        "away_form":        _form_results(away_form, away_id, source),
        # Goals averages
        "home_avg_scored":  _avg_goals(home_form, source, "home"),
        "home_avg_conceded":_avg_goals(home_form, source, "away"),
        "away_avg_scored":  _avg_goals(away_form, source, "away"),
        "away_avg_conceded":_avg_goals(away_form, source, "home"),
        # H2H
        "h2h_matches":      h2h,
        "h2h_avg_goals":    _avg_goals(h2h, source, "both"),
        "h2h_btts_rate":    _btts_rate(h2h, source),
        # Extended (not available from football-data directly)
        "home_avg_corners": 0.0,
        "away_avg_corners": 0.0,
        "home_avg_yellow":  0.0,
        "away_avg_yellow":  0.0,
    }


def _normalize_api_football(raw: dict) -> dict:
    home_id   = raw["teams"]["home"]["id"]
    away_id   = raw["teams"]["away"]["id"]
    home_name = raw["teams"]["home"]["name"]
    away_name = raw["teams"]["away"]["name"]
    source    = "api-football"

    home_form  = raw.get("_home_form", [])
    away_form  = raw.get("_away_form", [])
    h2h        = raw.get("_h2h", [])
    standings  = raw.get("_standings", [])

    home_rank = _get_rank_af(standings, home_id)
    away_rank = _get_rank_af(standings, away_id)

    return {
        "match_id":          raw.get("fixture", {}).get("id"),
        "source":            source,
        "competition":       raw.get("_competition_name", ""),
        "competition_code":  raw.get("_competition_code", ""),
        "utc_date":          raw.get("fixture", {}).get("date", ""),
        "home_id":           home_id,
        "away_id":           away_id,
        "home_name":         home_name,
        "away_name":         away_name,
        "home_rank":         home_rank,
        "away_rank":         away_rank,
        # Form
        "home_form":         _form_results(home_form, home_id, source),
        "away_form":         _form_results(away_form, away_id, source),
        # Goals averages
        "home_avg_scored":   _avg_goals(home_form, source, "home"),
        "home_avg_conceded": _avg_goals(home_form, source, "away"),
        "away_avg_scored":   _avg_goals(away_form, source, "away"),
        "away_avg_conceded": _avg_goals(away_form, source, "home"),
        # H2H
        "h2h_matches":       h2h,
        "h2h_avg_goals":     _avg_goals(h2h, source, "both"),
        "h2h_btts_rate":     _btts_rate(h2h, source),
        # Extended stats from api-football
        "home_avg_corners":  _avg_stat_api_football(home_form, "Corner Kicks", home_id),
        "away_avg_corners":  _avg_stat_api_football(away_form, "Corner Kicks", away_id),
        "home_avg_yellow":   _avg_stat_api_football(home_form, "Yellow Cards", home_id),
        "away_avg_yellow":   _avg_stat_api_football(away_form, "Yellow Cards", away_id),
    }


# --------------------------------------------------------------------------- #
#  Ranking helpers                                                             #
# --------------------------------------------------------------------------- #

def _get_rank_fbd(standings: list[dict], team_id: int) -> int:
    for row in standings:
        if row.get("team", {}).get("id") == team_id:
            return row.get("position", 99)
    return 99


def _get_rank_af(standings: list[dict], team_id: int) -> int:
    for row in standings:
        if row.get("team", {}).get("id") == team_id:
            return row.get("rank", 99)
    return 99


# --------------------------------------------------------------------------- #
#  Public entry point                                                          #
# --------------------------------------------------------------------------- #

def analyze_matches(raw_matches: list[dict]) -> list[dict]:
    normalized = []
    for raw in raw_matches:
        m = normalize(raw)
        if m:
            normalized.append(m)
    logger.info("Analyzer: %d/%d matches normalized", len(normalized), len(raw_matches))
    return normalized

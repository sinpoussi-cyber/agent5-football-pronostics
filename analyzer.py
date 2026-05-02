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


def _avg_goals(matches: list[dict], source: str,
               team_id: int | None = None, stat: str = "scored") -> float:
    """
    Average goals per match.
      team_id=None            → total goals (home + away), used for H2H averages.
      team_id set, stat="scored"   → goals scored by team_id across its form matches.
      team_id set, stat="conceded" → goals conceded by team_id across its form matches.

    For each match we detect whether team_id was home or away in that specific
    fixture before choosing the right score column, fixing the previous bug where
    side="home" always took the home-team score regardless of which side team_id
    actually played on.
    """
    totals = []
    for m in matches:
        if source == "football-data":
            score   = m.get("score", {}).get("fullTime", {})
            h_g     = _safe(score.get("home"))
            a_g     = _safe(score.get("away"))
            is_home = m.get("homeTeam", {}).get("id") == team_id
        else:  # api-football
            score   = m.get("goals", {})
            h_g     = _safe(score.get("home"))
            a_g     = _safe(score.get("away"))
            is_home = m.get("teams", {}).get("home", {}).get("id") == team_id

        if team_id is None:
            totals.append(h_g + a_g)
        elif stat == "scored":
            totals.append(h_g if is_home else a_g)
        else:  # conceded
            totals.append(a_g if is_home else h_g)
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


# --------------------------------------------------------------------------- #
#  League-level defaults for corners and yellow cards                          #
# --------------------------------------------------------------------------- #

_CORNERS_DEFAULTS: dict[str, tuple[float, float]] = {
    "premier league":   (5.8, 4.3),
    "championship":     (5.6, 4.2),
    "ligue 1":          (4.8, 3.9),
    "bundesliga":       (5.2, 4.1),
    "laliga":           (4.9, 3.8),
    "serie a":          (4.7, 3.7),
    "eredivisie":       (5.1, 4.0),
    "liga portugal":    (4.6, 3.6),
    "primeira liga":    (4.6, 3.6),
    "champions league": (5.4, 4.2),
}
_DEFAULT_CORNERS = (5.0, 4.0)

_YELLOW_DEFAULTS: dict[str, tuple[float, float]] = {
    "premier league":   (1.6, 1.5),
    "championship":     (1.7, 1.6),
    "ligue 1":          (1.9, 1.8),
    "bundesliga":       (1.7, 1.6),
    "laliga":           (2.2, 2.0),
    "serie a":          (2.0, 1.9),
    "eredivisie":       (1.6, 1.5),
    "liga portugal":    (2.0, 1.9),
    "primeira liga":    (2.0, 1.9),
    "champions league": (1.8, 1.7),
}
_DEFAULT_YELLOW = (1.8, 1.7)


def _league_corners(competition: str) -> tuple[float, float]:
    key = competition.lower()
    for name, vals in _CORNERS_DEFAULTS.items():
        if name in key:
            return vals
    return _DEFAULT_CORNERS


def _league_yellow(competition: str) -> tuple[float, float]:
    key = competition.lower()
    for name, vals in _YELLOW_DEFAULTS.items():
        if name in key:
            return vals
    return _DEFAULT_YELLOW


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
        elif source == "sport-api":
            return _normalize_sport_api(raw)
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
        # Goals averages — team_id-aware, correct regardless of home/away side played
        "home_avg_scored":  _avg_goals(home_form, source, home_id, "scored"),
        "home_avg_conceded":_avg_goals(home_form, source, home_id, "conceded"),
        "away_avg_scored":  _avg_goals(away_form, source, away_id, "scored"),
        "away_avg_conceded":_avg_goals(away_form, source, away_id, "conceded"),
        # H2H
        "h2h_matches":      h2h,
        "h2h_avg_goals":    _avg_goals(h2h, source),
        "h2h_btts_rate":    _btts_rate(h2h, source),
        # Extended: football-data free tier has no match stats — use league defaults
        "home_avg_corners": _league_corners(raw.get("_competition_name", ""))[0],
        "away_avg_corners": _league_corners(raw.get("_competition_name", ""))[1],
        "home_avg_yellow":  _league_yellow(raw.get("_competition_name", ""))[0],
        "away_avg_yellow":  _league_yellow(raw.get("_competition_name", ""))[1],
    }


def _normalize_api_football(raw: dict) -> dict:
    home_id   = raw["teams"]["home"]["id"]
    away_id   = raw["teams"]["away"]["id"]
    home_name = raw["teams"]["home"]["name"]
    away_name = raw["teams"]["away"]["name"]
    source    = "api-football"

    # None = explicitly unavailable (API returned None); list = enriched (may be empty)
    home_form  = raw.get("_home_form")
    away_form  = raw.get("_away_form")
    h2h        = raw.get("_h2h") or []
    standings  = raw.get("_standings") or []

    home_rank = _get_rank_af(standings, home_id)
    away_rank = _get_rank_af(standings, away_id)

    # Safe lists for functions that iterate (never pass None)
    _hf = home_form if isinstance(home_form, list) else []
    _af = away_form if isinstance(away_form, list) else []

    competition = raw.get("_competition_name", "")
    _corn_h, _corn_a = _league_corners(competition)
    _yell_h, _yell_a = _league_yellow(competition)

    home_corners_raw = _avg_stat_api_football(_hf, "Corner Kicks", home_id)
    away_corners_raw = _avg_stat_api_football(_af, "Corner Kicks", away_id)
    home_yellow_raw  = _avg_stat_api_football(_hf, "Yellow Cards", home_id)
    away_yellow_raw  = _avg_stat_api_football(_af, "Yellow Cards", away_id)

    return {
        "match_id":          raw.get("fixture", {}).get("id"),
        "source":            source,
        "competition":       competition,
        "competition_code":  raw.get("_competition_code", ""),
        "utc_date":          raw.get("fixture", {}).get("date", ""),
        "home_id":           home_id,
        "away_id":           away_id,
        "home_name":         home_name,
        "away_name":         away_name,
        "home_rank":         home_rank,
        "away_rank":         away_rank,
        # Form — empty list when data unavailable
        "home_form":         _form_results(_hf, home_id, source) if _hf else [],
        "away_form":         _form_results(_af, away_id, source) if _af else [],
        # Goals averages — team_id-aware, correct regardless of home/away side played
        "home_avg_scored":   _avg_goals(_hf, source, home_id, "scored")   if isinstance(home_form, list) else None,
        "home_avg_conceded": _avg_goals(_hf, source, home_id, "conceded") if isinstance(home_form, list) else None,
        "away_avg_scored":   _avg_goals(_af, source, away_id, "scored")   if isinstance(away_form, list) else None,
        "away_avg_conceded": _avg_goals(_af, source, away_id, "conceded") if isinstance(away_form, list) else None,
        # H2H
        "h2h_matches":       h2h,
        "h2h_avg_goals":     _avg_goals(h2h, source),
        "h2h_btts_rate":     _btts_rate(h2h, source),
        # Corners & cards: real stats if available, else league defaults (never 0.0)
        "home_avg_corners":  home_corners_raw if home_corners_raw > 0 else _corn_h,
        "away_avg_corners":  away_corners_raw if away_corners_raw > 0 else _corn_a,
        "home_avg_yellow":   home_yellow_raw  if home_yellow_raw  > 0 else _yell_h,
        "away_avg_yellow":   away_yellow_raw  if away_yellow_raw  > 0 else _yell_a,
    }


def _normalize_sport_api(raw: dict) -> dict:
    home_id   = raw["teams"]["home"]["id"]
    away_id   = raw["teams"]["away"]["id"]
    home_name = raw["teams"]["home"]["name"]
    away_name = raw["teams"]["away"]["name"]

    # Pre-computed dicts from fetcher_sport_api enrichment
    home_form = raw.get("_home_form")   # {"avg_scored", "avg_conceded", "form"} or None
    away_form = raw.get("_away_form")
    h2h       = raw.get("_h2h") or []  # [{"home_score", "away_score"}]
    st_home   = raw.get("_standings_home")  # {"position", "points", "goal_diff"} or None
    st_away   = raw.get("_standings_away")

    competition = raw.get("_competition_name", "")
    _corn_h, _corn_a = _league_corners(competition)
    _yell_h, _yell_a = _league_yellow(competition)

    h2h_avg_goals = (
        sum(m["home_score"] + m["away_score"] for m in h2h) / len(h2h)
        if h2h else 0.0
    )
    h2h_btts_rate = (
        sum(1 for m in h2h if m["home_score"] > 0 and m["away_score"] > 0) / len(h2h)
        if h2h else 0.0
    )

    # Use real corners when sport-api returned homeScore.corner/awayScore.corner
    # per match; fall back to league defaults when absent.
    home_avg_corners = (
        home_form["avg_corners"]
        if home_form and home_form.get("avg_corners") is not None
        else _corn_h
    )
    away_avg_corners = (
        away_form["avg_corners"]
        if away_form and away_form.get("avg_corners") is not None
        else _corn_a
    )
    # Yellow cards are not exposed in sport-api event payloads; keep league defaults.

    return {
        "match_id":          raw.get("fixture", {}).get("id"),
        "source":            "sport-api",
        "competition":       competition,
        "competition_code":  raw.get("_competition_code", ""),
        "utc_date":          raw.get("fixture", {}).get("date", ""),
        "home_id":           home_id,
        "away_id":           away_id,
        "home_name":         home_name,
        "away_name":         away_name,
        "home_rank":         st_home.get("position", 99) if st_home else 99,
        "away_rank":         st_away.get("position", 99) if st_away else 99,
        "home_form":         list(home_form["form"]) if home_form else [],
        "away_form":         list(away_form["form"]) if away_form else [],
        "home_avg_scored":   home_form.get("avg_scored")   if home_form else None,
        "home_avg_conceded": home_form.get("avg_conceded") if home_form else None,
        "away_avg_scored":   away_form.get("avg_scored")   if away_form else None,
        "away_avg_conceded": away_form.get("avg_conceded") if away_form else None,
        "h2h_matches":       h2h,
        "h2h_avg_goals":     h2h_avg_goals,
        "h2h_btts_rate":     h2h_btts_rate,
        "home_avg_corners":  home_avg_corners,
        "away_avg_corners":  away_avg_corners,
        "home_avg_yellow":   _yell_h,
        "away_avg_yellow":   _yell_a,
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

"""
Pronostic Engine — computes all betting recommendations from normalized match data.
Uses a Poisson-based model for goal distribution, plus heuristics for corners/cards.
Claude AI enriches each pronostic with a narrative analysis.
"""

from __future__ import annotations
import math
import logging
import os
from itertools import product
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

# League-specific average goals (home, away) used when team stats are unavailable
_LEAGUE_DEFAULTS: dict[str, tuple[float, float]] = {
    "premier league":   (1.5, 1.2),
    "ligue 1":          (1.3, 1.1),
    "bundesliga":       (1.7, 1.3),
    "serie a":          (1.3, 1.0),
    "laliga":           (1.4, 1.1),
    "liga portugal":    (1.4, 1.1),
    "primeira liga":    (1.4, 1.1),
    "championship":     (1.4, 1.1),
    "eredivisie":       (1.6, 1.3),
    "champions league": (1.5, 1.2),
}
_DEFAULT_GOALS = (1.35, 1.10)  # generic fallback


def _league_defaults(competition: str) -> tuple[float, float]:
    key = competition.lower()
    for name, vals in _LEAGUE_DEFAULTS.items():
        if name in key:
            return vals
    return _DEFAULT_GOALS


# --------------------------------------------------------------------------- #
#  Poisson helpers                                                             #
# --------------------------------------------------------------------------- #

def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _score_matrix(lam_home: float, lam_away: float, max_goals: int = 8) -> list[list[float]]:
    """Returns matrix[home_goals][away_goals] of probabilities."""
    return [
        [_poisson_pmf(h, lam_home) * _poisson_pmf(a, lam_away)
         for a in range(max_goals + 1)]
        for h in range(max_goals + 1)
    ]


def _expected_goals(match: dict) -> tuple[float, float]:
    """Compute expected goals for home and away teams."""
    league_avg = 1.35  # typical goals per team per match

    league_home_avg, league_away_avg = _league_defaults(match.get("competition", ""))

    # When form data is absent (avg == 0.0 or rank == 99), fall back to league averages
    def _or_league(val: float, league_val: float) -> float:
        return league_val if val == 0.0 else val

    home_att = max(_or_league(match["home_avg_scored"],   league_home_avg), 0.5)
    home_def = max(_or_league(match["home_avg_conceded"], league_away_avg), 0.5)
    away_att = max(_or_league(match["away_avg_scored"],   league_away_avg), 0.5)
    away_def = max(_or_league(match["away_avg_conceded"], league_home_avg), 0.5)

    # Home advantage modifier
    home_advantage = 1.10

    lam_home = home_att * away_def / league_avg * home_advantage
    lam_away = away_att * home_def / league_avg / home_advantage

    # Blend with H2H if available
    h2h_avg = match.get("h2h_avg_goals", 0)
    if h2h_avg > 0:
        projected_total = lam_home + lam_away
        blend = (projected_total + h2h_avg) / 2
        factor = blend / max(projected_total, 0.01)
        lam_home *= factor
        lam_away *= factor

    return round(lam_home, 3), round(lam_away, 3)


# --------------------------------------------------------------------------- #
#  Core probability computations                                               #
# --------------------------------------------------------------------------- #

def compute_1x2(matrix: list[list[float]]) -> dict:
    home = away = draw = 0.0
    for h in range(len(matrix)):
        for a in range(len(matrix[0])):
            p = matrix[h][a]
            if h > a:
                home += p
            elif h < a:
                away += p
            else:
                draw += p
    return {"1": round(home, 4), "X": round(draw, 4), "2": round(away, 4)}


def compute_double_chance(p1x2: dict) -> dict:
    return {
        "1X": round(p1x2["1"] + p1x2["X"], 4),
        "12": round(p1x2["1"] + p1x2["2"], 4),
        "X2": round(p1x2["X"] + p1x2["2"], 4),
    }


def compute_over_under(matrix: list[list[float]]) -> dict:
    thresholds = [0.5, 1.5, 2.5, 3.5]
    result = {}
    for t in thresholds:
        over = sum(
            matrix[h][a]
            for h in range(len(matrix))
            for a in range(len(matrix[0]))
            if h + a > t
        )
        result[f"O{t}"] = round(over, 4)
        result[f"U{t}"] = round(1 - over, 4)
    return result


def compute_btts(matrix: list[list[float]]) -> dict:
    yes = sum(
        matrix[h][a]
        for h in range(len(matrix))
        for a in range(len(matrix[0]))
        if h > 0 and a > 0
    )
    return {"BTTS_yes": round(yes, 4), "BTTS_no": round(1 - yes, 4)}


def compute_handicap_european(p1x2: dict) -> dict:
    """European handicap -1 / 0 / +1 for home team."""
    # Simplified: shift probability mass
    home, draw, away = p1x2["1"], p1x2["X"], p1x2["2"]
    return {
        "EH-1_home": round(home * 0.7, 4),
        "EH0_draw":  round(draw + home * 0.15 + away * 0.15, 4),
        "EH+1_home": round(home + draw * 0.5, 4),
    }


def compute_asian_handicap(lam_home: float, lam_away: float,
                           matrix: list[list[float]]) -> dict:
    """Asian handicap -0.5, -1.0, -1.5 for home team."""
    result = {}
    for hc in [-0.5, -1.0, -1.5]:
        win = sum(
            matrix[h][a]
            for h in range(len(matrix))
            for a in range(len(matrix[0]))
            if h - a > -hc
        )
        result[f"AH{hc}"] = round(win, 4)
    return result


def compute_corners(match: dict) -> dict:
    avg_home = match.get("home_avg_corners", 5.0) or 5.0
    avg_away = match.get("away_avg_corners", 4.5) or 4.5
    exp_total = avg_home + avg_away

    result = {}
    for threshold in [8.5, 9.5, 10.5]:
        # Use Poisson CDF approximation
        p_over = 1 - sum(_poisson_pmf(k, exp_total) for k in range(int(threshold) + 1))
        result[f"Corners_O{threshold}"] = round(p_over, 4)
        result[f"Corners_U{threshold}"] = round(1 - p_over, 4)
    return result


def compute_cards(match: dict) -> dict:
    avg_home = match.get("home_avg_yellow", 1.8) or 1.8
    avg_away = match.get("away_avg_yellow", 1.8) or 1.8
    exp_total = avg_home + avg_away

    result = {}
    for threshold in [3.5, 4.5]:
        p_over = 1 - sum(_poisson_pmf(k, exp_total) for k in range(int(threshold) + 1))
        result[f"Cards_O{threshold}"] = round(p_over, 4)
        result[f"Cards_U{threshold}"] = round(1 - p_over, 4)
    return result


def compute_exact_scores(matrix: list[list[float]], top_n: int = 5) -> list[dict]:
    scores = []
    for h in range(len(matrix)):
        for a in range(len(matrix[0])):
            scores.append({"score": f"{h}-{a}", "prob": matrix[h][a]})
    scores.sort(key=lambda x: x["prob"], reverse=True)
    return scores[:top_n]


def compute_halftime(lam_home: float, lam_away: float) -> dict:
    """Half-time: assume ~45% of goals scored in first half."""
    lam_h_ht = lam_home * 0.45
    lam_a_ht = lam_away * 0.45
    mat = _score_matrix(lam_h_ht, lam_a_ht, max_goals=4)
    p1x2 = compute_1x2(mat)
    ou = compute_over_under(mat)
    return {
        "HT_1": p1x2["1"],
        "HT_X": p1x2["X"],
        "HT_2": p1x2["2"],
        "HT_O0.5": ou.get("O0.5", 0),
        "HT_O1.5": ou.get("O1.5", 0),
    }


def compute_clean_sheet(matrix: list[list[float]]) -> dict:
    home_cs = sum(matrix[h][0] for h in range(len(matrix)))
    away_cs = sum(matrix[0][a] for a in range(len(matrix[0])))
    return {
        "CS_home": round(home_cs, 4),
        "CS_away": round(away_cs, 4),
    }


def compute_odd_even(matrix: list[list[float]]) -> dict:
    even_p = sum(
        matrix[h][a]
        for h in range(len(matrix))
        for a in range(len(matrix[0]))
        if (h + a) % 2 == 0
    )
    return {"Even": round(even_p, 4), "Odd": round(1 - even_p, 4)}


# --------------------------------------------------------------------------- #
#  Confidence star rating                                                      #
# --------------------------------------------------------------------------- #

def _stars(prob: float) -> str:
    if prob >= 0.70:
        return "⭐⭐⭐"
    if prob >= 0.55:
        return "⭐⭐"
    return "⭐"


def _best_1x2(p: dict) -> tuple[str, float]:
    best = max(p, key=p.get)
    return best, p[best]


def _best_dc(p: dict) -> tuple[str, float]:
    best = max(p, key=p.get)
    return best, p[best]


def _best_ou(p: dict, prefix: str = "O") -> tuple[str, float]:
    candidates = {k: v for k, v in p.items() if k.startswith(prefix)}
    if not candidates:
        return "", 0.0
    best = max(candidates, key=candidates.get)
    return best, candidates[best]


# --------------------------------------------------------------------------- #
#  Claude AI narrative                                                         #
# --------------------------------------------------------------------------- #

def _claude_narrative(match: dict, pronostics: dict) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "Analyse IA non disponible (clé manquante)."

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Tu es un expert en analyse de matchs de football. Fournis une analyse narrative concise (150 mots max) du match suivant en français.

Match : {match['home_name']} vs {match['away_name']} ({match['competition']})
Forme domicile (5 derniers) : {' '.join(match['home_form']) or 'N/A'}
Forme extérieur (5 derniers) : {' '.join(match['away_form']) or 'N/A'}
Classement domicile : {match['home_rank']} | Classement extérieur : {match['away_rank']}
Buts marqués moy. domicile : {match['home_avg_scored']:.1f} | extérieur : {match['away_avg_scored']:.1f}
Taux BTTS H2H : {match.get('h2h_btts_rate', 0):.0%}

Pronostics clés :
- 1X2 : {pronostics['p1x2']}
- Over/Under : O2.5 = {pronostics['over_under'].get('O2.5', 'N/A')}
- BTTS : {pronostics['btts']}
- Score le plus probable : {pronostics['exact_scores'][0]['score'] if pronostics['exact_scores'] else 'N/A'}

Analyse concise en mettant en avant les facteurs décisifs et le pronostic le plus fiable."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except anthropic.BadRequestError as e:
        if "credit balance too low" in str(e).lower():
            logger.warning("Anthropic credits insufficient: %s", e)
            return "Analyse IA indisponible (crédits insuffisants)"
        logger.error("Claude API error: %s", e)
        return f"Analyse IA indisponible : {e}"
    except Exception as e:
        logger.error("Claude API error: %s", e)
        return f"Analyse IA indisponible : {e}"


# --------------------------------------------------------------------------- #
#  Main engine entry point                                                     #
# --------------------------------------------------------------------------- #

def compute_pronostics(match: dict, use_ai: bool = True) -> dict:
    lam_home, lam_away = _expected_goals(match)
    matrix = _score_matrix(lam_home, lam_away)

    p1x2       = compute_1x2(matrix)
    dc         = compute_double_chance(p1x2)
    ou         = compute_over_under(matrix)
    btts       = compute_btts(matrix)
    eh         = compute_handicap_european(p1x2)
    ah         = compute_asian_handicap(lam_home, lam_away, matrix)
    corners    = compute_corners(match)
    cards      = compute_cards(match)
    exact      = compute_exact_scores(matrix)
    halftime   = compute_halftime(lam_home, lam_away)
    clean      = compute_clean_sheet(matrix)
    odd_even   = compute_odd_even(matrix)

    best_1x2_label, best_1x2_prob = _best_1x2(p1x2)
    best_dc_label,  best_dc_prob  = _best_dc(dc)
    best_ou_label,  best_ou_prob  = _best_ou(ou, "O")

    pronostics = {
        "lam_home":    lam_home,
        "lam_away":    lam_away,
        "p1x2":        p1x2,
        "double_chance": dc,
        "over_under":  ou,
        "btts":        btts,
        "handicap_eu": eh,
        "handicap_asian": ah,
        "corners":     corners,
        "cards":       cards,
        "exact_scores": exact,
        "halftime":    halftime,
        "clean_sheet": clean,
        "odd_even":    odd_even,
        # Recommendations
        "rec_1x2":     {"label": best_1x2_label, "prob": best_1x2_prob, "stars": _stars(best_1x2_prob)},
        "rec_dc":      {"label": best_dc_label,  "prob": best_dc_prob,  "stars": _stars(best_dc_prob)},
        "rec_ou":      {"label": best_ou_label,  "prob": best_ou_prob,  "stars": _stars(best_ou_prob)},
        "rec_btts":    {
            "label": "Oui" if btts["BTTS_yes"] >= btts["BTTS_no"] else "Non",
            "prob":  max(btts["BTTS_yes"], btts["BTTS_no"]),
            "stars": _stars(max(btts["BTTS_yes"], btts["BTTS_no"])),
        },
        "rec_score":   {"label": exact[0]["score"] if exact else "N/A", "prob": exact[0]["prob"] if exact else 0},
    }

    if use_ai:
        pronostics["ai_narrative"] = _claude_narrative(match, pronostics)
    else:
        pronostics["ai_narrative"] = ""

    return pronostics


def run_engine(matches: list[dict], use_ai: bool = True) -> list[dict]:
    results = []
    for match in matches:
        try:
            prono = compute_pronostics(match, use_ai=use_ai)
            results.append({"match": match, "pronostics": prono})
            logger.info("Pronostic computed: %s vs %s", match["home_name"], match["away_name"])
        except Exception as e:
            logger.error("Engine error for %s vs %s: %s",
                         match.get("home_name", "?"), match.get("away_name", "?"), e)
    return results

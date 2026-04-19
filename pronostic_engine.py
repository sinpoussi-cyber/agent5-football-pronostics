"""
Pronostic Engine — ensemble multi-model betting recommendations.
Models: Poisson, Dixon-Coles, Elo, xG-adjusted.
Claude AI enriches each pronostic with a narrative analysis.
"""

from __future__ import annotations
import math
import logging
import os
import statistics
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
_DEFAULT_GOALS = (1.35, 1.10)

DIXON_COLES_RHO = -0.13


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
    """Compute expected goals for home and away teams (base Poisson lambdas)."""
    league_avg = 1.35

    league_home_avg, league_away_avg = _league_defaults(match.get("competition", ""))

    def _or_league(val: float, league_val: float) -> float:
        return league_val if val == 0.0 else val

    home_att = max(_or_league(match["home_avg_scored"],   league_home_avg), 0.5)
    home_def = max(_or_league(match["home_avg_conceded"], league_away_avg), 0.5)
    away_att = max(_or_league(match["away_avg_scored"],   league_away_avg), 0.5)
    away_def = max(_or_league(match["away_avg_conceded"], league_home_avg), 0.5)

    home_advantage = 1.10

    lam_home = home_att * away_def / league_avg * home_advantage
    lam_away = away_att * home_def / league_avg / home_advantage

    h2h_avg = match.get("h2h_avg_goals", 0)
    if h2h_avg > 0:
        projected_total = lam_home + lam_away
        blend = (projected_total + h2h_avg) / 2
        factor = blend / max(projected_total, 0.01)
        lam_home *= factor
        lam_away *= factor

    return round(lam_home, 3), round(lam_away, 3)


# --------------------------------------------------------------------------- #
#  Model 1 — Poisson                                                           #
# --------------------------------------------------------------------------- #

def _model_poisson(lam_home: float, lam_away: float) -> dict:
    matrix = _score_matrix(lam_home, lam_away)
    p1x2 = compute_1x2(matrix)
    return {
        "p1": p1x2["1"] * 100,
        "px": p1x2["X"] * 100,
        "p2": p1x2["2"] * 100,
        "matrix": matrix,
    }


# --------------------------------------------------------------------------- #
#  Model 2 — Dixon-Coles                                                       #
# --------------------------------------------------------------------------- #

def _dixon_coles_tau(h: int, a: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Correction factor for low-scoring outcomes."""
    if h == 0 and a == 0:
        return 1.0 - lam_h * lam_a * rho
    if h == 1 and a == 0:
        return 1.0 + lam_a * rho
    if h == 0 and a == 1:
        return 1.0 + lam_h * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def _score_matrix_dixon_coles(lam_home: float, lam_away: float,
                               rho: float = DIXON_COLES_RHO,
                               max_goals: int = 8) -> list[list[float]]:
    matrix = []
    for h in range(max_goals + 1):
        row = []
        for a in range(max_goals + 1):
            p = (_poisson_pmf(h, lam_home) * _poisson_pmf(a, lam_away)
                 * _dixon_coles_tau(h, a, lam_home, lam_away, rho))
            row.append(p)
        matrix.append(row)
    # Renormalize so probabilities sum to 1
    total = sum(matrix[h][a] for h in range(max_goals + 1) for a in range(max_goals + 1))
    if total > 0:
        matrix = [[p / total for p in row] for row in matrix]
    return matrix


def _model_dixon_coles(lam_home: float, lam_away: float) -> dict:
    matrix = _score_matrix_dixon_coles(lam_home, lam_away)
    p1x2 = compute_1x2(matrix)
    return {
        "p1": p1x2["1"] * 100,
        "px": p1x2["X"] * 100,
        "p2": p1x2["2"] * 100,
        "matrix": matrix,
    }


# --------------------------------------------------------------------------- #
#  Model 3 — Elo                                                               #
# --------------------------------------------------------------------------- #

def _model_elo(match: dict) -> dict:
    home_rank = match.get("home_rank", 10)
    away_rank = match.get("away_rank", 10)

    # Fallback if rank is sentinel 99
    if home_rank == 99:
        home_rank = 12
    if away_rank == 99:
        away_rank = 12

    elo_home = 1500 - (home_rank - 1) * 15 + 100  # home advantage
    elo_away = 1500 - (away_rank - 1) * 15

    p1 = 1.0 / (1.0 + 10 ** ((elo_away - elo_home) / 400.0))
    p2 = 1.0 / (1.0 + 10 ** ((elo_home - elo_away) / 400.0))
    px = 0.28 * (1.0 - abs(p1 - p2))

    # Renormalize
    total = p1 + px + p2
    p1 /= total
    px /= total
    p2 /= total

    return {
        "p1": p1 * 100,
        "px": px * 100,
        "p2": p2 * 100,
        "elo_home": round(elo_home, 1),
        "elo_away": round(elo_away - 100, 1),  # store without home bonus for display
    }


# --------------------------------------------------------------------------- #
#  Model 4 — xG-adjusted                                                       #
# --------------------------------------------------------------------------- #

def _model_xg_adjusted(match: dict) -> dict:
    league_home_avg, league_away_avg = _league_defaults(match.get("competition", ""))

    home_scored = match.get("home_avg_scored", 0.0) or 0.0
    away_scored = match.get("away_avg_scored", 0.0) or 0.0

    xg_home = home_scored * 0.95 + league_home_avg * 0.05
    xg_away = away_scored * 0.95 + league_away_avg * 0.05

    xg_home = max(xg_home, 0.3)
    xg_away = max(xg_away, 0.3)

    # Apply defensive adjustment (same logic as base model)
    league_avg = 1.35
    home_def = max(match.get("home_avg_conceded", league_away_avg) or league_away_avg, 0.5)
    away_def = max(match.get("away_avg_conceded", league_home_avg) or league_home_avg, 0.5)

    home_advantage = 1.10
    lam_home = xg_home * away_def / league_avg * home_advantage
    lam_away = xg_away * home_def / league_avg / home_advantage

    matrix = _score_matrix(lam_home, lam_away)
    p1x2 = compute_1x2(matrix)
    return {
        "p1": p1x2["1"] * 100,
        "px": p1x2["X"] * 100,
        "p2": p1x2["2"] * 100,
        "xg_home": round(xg_home, 3),
        "xg_away": round(xg_away, 3),
        "matrix": matrix,
    }


# --------------------------------------------------------------------------- #
#  Ensemble fusion                                                              #
# --------------------------------------------------------------------------- #

def _ensemble_fusion(p_poisson: dict, p_dixon: dict, p_elo: dict, p_xg: dict) -> dict:
    """Combine the 4 models via arithmetic mean, median, and weighted average."""
    weights = {"dixon": 0.35, "poisson": 0.25, "xg": 0.25, "elo": 0.15}

    results = {}
    for outcome in ("p1", "px", "p2"):
        vals = {
            "poisson": p_poisson[outcome],
            "dixon":   p_dixon[outcome],
            "elo":     p_elo[outcome],
            "xg":      p_xg[outcome],
        }
        mean_val = statistics.mean(vals.values())
        median_val = statistics.median(vals.values())
        weighted_val = (
            vals["dixon"]   * weights["dixon"]
            + vals["poisson"] * weights["poisson"]
            + vals["xg"]      * weights["xg"]
            + vals["elo"]     * weights["elo"]
        )
        results[outcome] = {
            "mean":     round(mean_val, 2),
            "median":   round(median_val, 2),
            "weighted": round(weighted_val, 2),
            "by_model": {k: round(v, 2) for k, v in vals.items()},
        }

    # Renormalize weighted so sum = 100
    w_total = sum(results[o]["weighted"] for o in ("p1", "px", "p2"))
    if w_total > 0:
        for o in ("p1", "px", "p2"):
            results[o]["weighted"] = round(results[o]["weighted"] / w_total * 100, 2)

    return results


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
    home, draw, away = p1x2["1"], p1x2["X"], p1x2["2"]
    return {
        "EH-1_home": round(home * 0.7, 4),
        "EH0_draw":  round(draw + home * 0.15 + away * 0.15, 4),
        "EH+1_home": round(home + draw * 0.5, 4),
    }


def compute_asian_handicap(lam_home: float, lam_away: float,
                           matrix: list[list[float]]) -> dict:
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

def _format_form(form: list) -> str:
    return " ".join(form) if form else "N/A"


def _h2h_summary(match: dict) -> str:
    rate = match.get("h2h_btts_rate", 0)
    avg = match.get("h2h_avg_goals", 0)
    if avg == 0 and rate == 0:
        return "Pas de données H2H"
    return f"Moy. buts={avg:.1f} | BTTS={rate:.0%}"


def _claude_narrative(match: dict, pronostics: dict) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "Analyse IA non disponible (clé manquante)."

    client = anthropic.Anthropic(api_key=api_key)

    ensemble = pronostics.get("ensemble", {})
    p_poisson = pronostics.get("model_poisson", {})
    p_dixon   = pronostics.get("model_dixon", {})
    p_elo     = pronostics.get("model_elo", {})
    p_xg      = pronostics.get("model_xg", {})

    def _w(outcome: str) -> float:
        return ensemble.get(outcome, {}).get("weighted", 0.0)

    date_str = match.get("date", "N/A")
    if hasattr(date_str, "strftime"):
        date_str = date_str.strftime("%d/%m/%Y %H:%M")

    prompt = f"""Tu es un data scientist expert en modélisation de matchs de football et value betting.

MATCH : {match['home_name']} vs {match['away_name']}
COMPÉTITION : {match['competition']}
DATE : {date_str}

DONNÉES DISPONIBLES :
- Forme domicile ({match['home_name']}) : {_format_form(match['home_form'])} | Moy. buts marqués : {match['home_avg_scored']:.1f} | Moy. buts encaissés : {match['home_avg_conceded']:.1f}
- Forme extérieur ({match['away_name']}) : {_format_form(match['away_form'])} | Moy. buts marqués : {match['away_avg_scored']:.1f} | Moy. buts encaissés : {match['away_avg_conceded']:.1f}
- Rang domicile : {match['home_rank']} | Rang extérieur : {match['away_rank']}
- H2H (5 derniers) : {_h2h_summary(match)}

RÉSULTATS DES MODÈLES :
- Poisson     : P1={p_poisson.get('p1', 0):.1f}% | PX={p_poisson.get('px', 0):.1f}% | P2={p_poisson.get('p2', 0):.1f}%
- Dixon-Coles : P1={p_dixon.get('p1', 0):.1f}%  | PX={p_dixon.get('px', 0):.1f}%  | P2={p_dixon.get('p2', 0):.1f}%
- Elo         : P1={p_elo.get('p1', 0):.1f}%    | PX={p_elo.get('px', 0):.1f}%    | P2={p_elo.get('p2', 0):.1f}%
- xG ajusté   : P1={p_xg.get('p1', 0):.1f}%    | PX={p_xg.get('px', 0):.1f}%     | P2={p_xg.get('p2', 0):.1f}%
- FUSION FINALE (pondérée) : P1={_w('p1'):.1f}% | PX={_w('px'):.1f}% | P2={_w('p2'):.1f}%

ANALYSE REQUISE (sois concis et exploitable) :
1. Convergence ou divergence des modèles ? Pourquoi ?
2. Facteurs contextuels importants (domicile, forme, rang)
3. Score exact le plus probable avec justification
4. 2 paris sécurisés (confiance élevée)
5. 1 pari à forte value si les modèles divergent des cotes standard
6. Niveau de confiance global : Faible / Moyen / Élevé + justification

CONTRAINTE : Raisonnement mathématique, concis, pas de phrases génériques.
Si les stats sont 0.0 ou rang=99 : le dire explicitement et baisser le niveau de confiance à Faible."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
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

    # Run all 4 models
    m_poisson = _model_poisson(lam_home, lam_away)
    m_dixon   = _model_dixon_coles(lam_home, lam_away)
    m_elo     = _model_elo(match)
    m_xg      = _model_xg_adjusted(match)

    # Ensemble fusion
    ensemble = _ensemble_fusion(m_poisson, m_dixon, m_elo, m_xg)

    # Use Dixon-Coles matrix as primary for downstream markets (most accurate)
    matrix = m_dixon["matrix"]

    p1x2_final = {
        "1": round(ensemble["p1"]["weighted"] / 100, 4),
        "X": round(ensemble["px"]["weighted"] / 100, 4),
        "2": round(ensemble["p2"]["weighted"] / 100, 4),
    }

    dc         = compute_double_chance(p1x2_final)
    ou         = compute_over_under(matrix)
    btts       = compute_btts(matrix)
    eh         = compute_handicap_european(p1x2_final)
    ah         = compute_asian_handicap(lam_home, lam_away, matrix)
    corners    = compute_corners(match)
    cards      = compute_cards(match)
    exact      = compute_exact_scores(matrix)
    halftime   = compute_halftime(lam_home, lam_away)
    clean      = compute_clean_sheet(matrix)
    odd_even   = compute_odd_even(matrix)

    best_1x2_label, best_1x2_prob = _best_1x2(p1x2_final)
    best_dc_label,  best_dc_prob  = _best_dc(dc)
    best_ou_label,  best_ou_prob  = _best_ou(ou, "O")

    pronostics = {
        "lam_home":    lam_home,
        "lam_away":    lam_away,
        # Individual models
        "model_poisson": m_poisson,
        "model_dixon":   m_dixon,
        "model_elo":     m_elo,
        "model_xg":      m_xg,
        "ensemble":      ensemble,
        # Final merged 1X2 (weighted ensemble)
        "p1x2":        p1x2_final,
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
        "rec_1x2":  {"label": best_1x2_label, "prob": best_1x2_prob, "stars": _stars(best_1x2_prob)},
        "rec_dc":   {"label": best_dc_label,  "prob": best_dc_prob,  "stars": _stars(best_dc_prob)},
        "rec_ou":   {"label": best_ou_label,  "prob": best_ou_prob,  "stars": _stars(best_ou_prob)},
        "rec_btts": {
            "label": "Oui" if btts["BTTS_yes"] >= btts["BTTS_no"] else "Non",
            "prob":  max(btts["BTTS_yes"], btts["BTTS_no"]),
            "stars": _stars(max(btts["BTTS_yes"], btts["BTTS_no"])),
        },
        "rec_score": {"label": exact[0]["score"] if exact else "N/A", "prob": exact[0]["prob"] if exact else 0},
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

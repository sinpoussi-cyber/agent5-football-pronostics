"""
Pronostic Engine — ensemble multi-model betting recommendations.
Models: Poisson, Dixon-Coles, Elo, xG-adjusted.

Tous les marchés sont calculés UNIQUEMENT à partir des modèles statistiques,
sans aucune cote bookmaker externe.

Marchés supportés:
- Paris principaux: 1X2, Double Chance, BTTS, Score Exact, HT/FT
- Totaux: Over/Under, Total Individuel, Totaux Asiatiques
- Handicaps: Européen, Asiatique, Draw No Bet
- Joueurs: Buteur, Premier/Dernier buteur, Multiples, Méthode du but
- Statistiques: Corners, Cartons, Tirs, Fautes, Hors-jeux, Possession
- Chronologiques: But par intervalle, Résultat à X minutes, Événements 5 premières minutes
- Spéciaux: Combos, Remontada, But contre son camp, Penalty/Expulsion, Gagner les deux mi-temps
"""

from __future__ import annotations
import math
import logging
import os
import statistics
import random
from typing import Any

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
    "coupe du monde":   (1.4, 1.1),
    "world cup":        (1.4, 1.1),
}
_DEFAULT_GOALS = (1.35, 1.10)

# League-specific average corners
_CORNERS_DEFAULTS: dict[str, tuple[float, float]] = {
    "premier league":   (5.8, 4.3),
    "bundesliga":       (5.2, 4.1),
    "laliga":           (4.9, 3.8),
    "serie a":          (4.7, 3.7),
    "ligue 1":          (4.8, 3.9),
    "eredivisie":       (5.1, 4.0),
    "champions league": (5.4, 4.2),
    "coupe du monde":   (5.0, 4.0),
}
_DEFAULT_CORNERS = (5.0, 4.0)

# League-specific average yellow cards
_YELLOW_DEFAULTS: dict[str, tuple[float, float]] = {
    "premier league":   (1.6, 1.5),
    "laliga":           (2.2, 2.0),
    "serie a":          (2.0, 1.9),
    "bundesliga":       (1.7, 1.6),
    "ligue 1":          (1.9, 1.8),
    "coupe du monde":   (1.8, 1.7),
}
_DEFAULT_YELLOW = (1.8, 1.7)

DIXON_COLES_RHO = -0.13

# Ligues majeures éligibles à l'analyse IA
_MAJOR_LEAGUES_AI: tuple[str, ...] = (
    "premier league", "ligue 1", "laliga", "bundesliga", "serie a",
    "eredivisie", "champions league", "championship", "coupe du monde", "world cup",
)

_AI_MAX_P1_STDEV = 10.0

# Pays hôtes Coupe du Monde 2026
_WC_HOSTS = ("mexico", "mexique", "canada", "united states", "usa", "états-unis")


# --------------------------------------------------------------------------- #
#  Helper functions                                                            #
# --------------------------------------------------------------------------- #

def _safe(value, default=0):
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _is_world_cup(competition: str) -> bool:
    key = (competition or "").lower()
    return "coupe du monde" in key or "world cup" in key


def _home_advantage(match: dict) -> float:
    if not _is_world_cup(match.get("competition", "")):
        return 1.10
    home = (match.get("home_name", "") or "").lower()
    return 1.08 if any(h in home for h in _WC_HOSTS) else 1.0


def _league_defaults(competition: str) -> tuple[float, float]:
    key = competition.lower()
    for name, vals in _LEAGUE_DEFAULTS.items():
        if name in key:
            return vals
    return _DEFAULT_GOALS


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


def _is_major_league_for_ai(competition: str) -> bool:
    key = (competition or "").lower()
    return any(name in key for name in _MAJOR_LEAGUES_AI)


def _ensemble_p1_stdev(ensemble: dict) -> float:
    by_model = ensemble.get("p1", {}).get("by_model", {})
    vals = list(by_model.values())
    if len(vals) < 2:
        return 0.0
    return statistics.stdev(vals)


def _should_use_ai(match: dict, ensemble: dict) -> tuple[bool, str]:
    if not _is_major_league_for_ai(match.get("competition", "")):
        return False, "ligue non-majeure"
    sigma = _ensemble_p1_stdev(ensemble)
    if sigma >= _AI_MAX_P1_STDEV:
        return False, f"divergence trop forte (σ P1={sigma:.1f}pp)"
    return True, ""


def _xg_usable(match: dict) -> bool:
    home_scored = match.get("home_avg_scored")
    home_conceded = match.get("home_avg_conceded")
    away_scored = match.get("away_avg_scored")
    away_conceded = match.get("away_avg_conceded")
    
    all_missing = all(v in (None, 0) for v in [home_scored, home_conceded, away_scored, away_conceded])
    if all_missing:
        return False
    
    has_stats = all(v is not None and v > 0 for v in [home_scored, home_conceded, away_scored, away_conceded])
    has_any_rank = match.get("home_rank", 99) != 99 or match.get("away_rank", 99) != 99
    return has_stats and has_any_rank


# --------------------------------------------------------------------------- #
#  Poisson helpers                                                             #
# --------------------------------------------------------------------------- #

def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _score_matrix(lam_home: float, lam_away: float, max_goals: int = 8) -> list[list[float]]:
    return [
        [_poisson_pmf(h, lam_home) * _poisson_pmf(a, lam_away)
         for a in range(max_goals + 1)]
        for h in range(max_goals + 1)
    ]


def _score_matrix_dixon_coles(lam_home: float, lam_away: float,
                               rho: float = DIXON_COLES_RHO,
                               max_goals: int = 8) -> list[list[float]]:
    def tau(h, a):
        if h == 0 and a == 0:
            return 1.0 - lam_home * lam_away * rho
        if h == 1 and a == 0:
            return 1.0 + lam_away * rho
        if h == 0 and a == 1:
            return 1.0 + lam_home * rho
        if h == 1 and a == 1:
            return 1.0 - rho
        return 1.0
    
    matrix = []
    for h in range(max_goals + 1):
        row = []
        for a in range(max_goals + 1):
            p = _poisson_pmf(h, lam_home) * _poisson_pmf(a, lam_away) * tau(h, a)
            row.append(p)
        matrix.append(row)
    
    total = sum(matrix[h][a] for h in range(max_goals + 1) for a in range(max_goals + 1))
    if total > 0:
        matrix = [[p / total for p in row] for row in matrix]
    return matrix


def _expected_goals(match: dict) -> tuple[float, float]:
    league_avg = 1.35
    league_home_avg, league_away_avg = _league_defaults(match.get("competition", ""))

    def _or_league(val, league_val: float) -> float:
        return league_val if (val is None or val == 0.0) else val

    home_att = max(_or_league(match.get("home_avg_scored", 0), league_home_avg), 0.5)
    home_def = max(_or_league(match.get("home_avg_conceded", 0), league_away_avg), 0.5)
    away_att = max(_or_league(match.get("away_avg_scored", 0), league_away_avg), 0.5)
    away_def = max(_or_league(match.get("away_avg_conceded", 0), league_home_avg), 0.5)

    home_advantage = _home_advantage(match)

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
#  Models (Poisson, Dixon-Coles, Elo, xG)                                      #
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


def _model_dixon_coles(lam_home: float, lam_away: float) -> dict:
    matrix = _score_matrix_dixon_coles(lam_home, lam_away)
    p1x2 = compute_1x2(matrix)
    return {
        "p1": p1x2["1"] * 100,
        "px": p1x2["X"] * 100,
        "p2": p1x2["2"] * 100,
        "matrix": matrix,
    }


def _model_elo(match: dict) -> dict:
    home_rank = match.get("home_rank", 10)
    away_rank = match.get("away_rank", 10)

    if home_rank == 99:
        home_rank = 12
    if away_rank == 99:
        away_rank = 12

    elo_home = 1500 - (home_rank - 1) * 15 + 100
    elo_away = 1500 - (away_rank - 1) * 15

    p1 = 1.0 / (1.0 + 10 ** ((elo_away - elo_home) / 400.0))
    p2 = 1.0 / (1.0 + 10 ** ((elo_home - elo_away) / 400.0))
    px = 0.28 * (1.0 - abs(p1 - p2))

    total = p1 + px + p2
    p1 /= total
    px /= total
    p2 /= total

    return {
        "p1": p1 * 100,
        "px": px * 100,
        "p2": p2 * 100,
        "elo_home": round(elo_home, 1),
        "elo_away": round(elo_away - 100, 1),
    }


def _model_xg_adjusted(match: dict) -> dict:
    league_home_avg, league_away_avg = _league_defaults(match.get("competition", ""))

    home_scored = match.get("home_avg_scored", 0.0) or 0.0
    away_scored = match.get("away_avg_scored", 0.0) or 0.0
    
    if home_scored == 0:
        home_scored = league_home_avg
    if away_scored == 0:
        away_scored = league_away_avg

    xg_home = home_scored * 0.95 + league_home_avg * 0.05
    xg_away = away_scored * 0.95 + league_away_avg * 0.05

    xg_home = max(xg_home, 0.5)
    xg_away = max(xg_away, 0.5)

    league_avg = 1.35
    home_def = max(match.get("home_avg_conceded", league_away_avg) or league_away_avg, 0.5)
    away_def = max(match.get("away_avg_conceded", league_home_avg) or league_home_avg, 0.5)
    
    if home_def == 0:
        home_def = league_away_avg
    if away_def == 0:
        away_def = league_home_avg

    home_advantage = _home_advantage(match)
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


def _ensemble_fusion(p_poisson: dict, p_dixon: dict, p_elo: dict,
                     p_xg: dict | None = None) -> dict:
    if p_xg is not None:
        weights = {"dixon": 0.35, "poisson": 0.25, "xg": 0.25, "elo": 0.15}
    else:
        weights = {"dixon": 0.475, "poisson": 0.375, "elo": 0.15}

    results = {}
    for outcome in ("p1", "px", "p2"):
        vals = {
            "poisson": p_poisson[outcome],
            "dixon":   p_dixon[outcome],
            "elo":     p_elo[outcome],
        }
        if p_xg is not None:
            vals["xg"] = p_xg[outcome]

        mean_val = statistics.mean(vals.values())
        median_val = statistics.median(vals.values())
        weighted_val = sum(vals[m] * weights[m] for m in vals if m in weights)
        results[outcome] = {
            "mean": round(mean_val, 2),
            "median": round(median_val, 2),
            "weighted": round(weighted_val, 2),
            "by_model": {k: round(v, 2) for k, v in vals.items()},
        }

    w_total = sum(results[o]["weighted"] for o in ("p1", "px", "p2"))
    if w_total > 0 and w_total != 100:
        for o in ("p1", "px", "p2"):
            results[o]["weighted"] = round(results[o]["weighted"] / w_total * 100, 2)

    return results


# --------------------------------------------------------------------------- #
#  Core probability computations                                              #
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
    thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]
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


def compute_individual_total(lam_team: float, thresholds: list = [0.5, 1.5, 2.5]) -> dict:
    """Compute over/under for individual team goals."""
    result = {}
    for t in thresholds:
        over = 1 - sum(_poisson_pmf(k, lam_team) for k in range(int(t) + 1))
        result[f"O{t}"] = round(over, 4)
        result[f"U{t}"] = round(1 - over, 4)
    return result


def compute_asian_total(lam_total: float, lines: list = [2.25, 2.75, 3.25]) -> dict:
    """Compute Asian total goals (split bets)."""
    result = {}
    for line in lines:
        low = math.floor(line)
        high = math.ceil(line)
        p_over_low = 1 - sum(_poisson_pmf(k, lam_total) for k in range(low + 1))
        p_over_high = 1 - sum(_poisson_pmf(k, lam_total) for k in range(high + 1))
        p_over = (p_over_low + p_over_high) / 2
        result[f"AOU_O{line}"] = round(p_over, 4)
        result[f"AOU_U{line}"] = round(1 - p_over, 4)
    return result


def compute_handicap_european(p1x2: dict, handicap: int = 1) -> dict:
    home, draw, away = p1x2["1"], p1x2["X"], p1x2["2"]
    # Pour handicap -1 (favori commence avec 1 but de retard)
    # Probabilité simplifiée pour victoire avec handicap
    return {
        f"EH-{handicap}_home": round(home * 0.7, 4),
        f"EH0_draw": round(draw + home * 0.15 + away * 0.15, 4),
        f"EH+{handicap}_home": round(home + draw * 0.5, 4),
    }


def compute_asian_handicap(lam_home: float, lam_away: float,
                           matrix: list[list[float]], handicaps: list = [-0.5, -0.75, -1.0, -1.5]) -> dict:
    result = {}
    for hc in handicaps:
        if hc == -0.5:
            win = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[0])) if h > a)
        elif hc == -0.75:
            # Half bet on -0.5, half on -1.0
            p_win = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[0])) if h > a)
            p_push = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[0])) if h - a == 1)
            win = p_win + 0.5 * p_push
        elif hc == -1.0:
            win = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[0])) if h - a >= 1)
        elif hc == -1.5:
            win = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[0])) if h - a >= 2)
        else:
            win = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[0])) if h + hc > a)
        result[f"AH{hc}"] = round(win, 4)
    return result


def compute_draw_no_bet(p1x2: dict) -> dict:
    p1 = p1x2["1"]
    p2 = p1x2["2"]
    total = p1 + p2
    if total <= 0:
        return {"DNB_home": 0.5, "DNB_away": 0.5}
    return {
        "DNB_home": round(p1 / total, 4),
        "DNB_away": round(p2 / total, 4),
    }


def compute_btts(matrix: list[list[float]]) -> dict:
    yes = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[0])) if h > 0 and a > 0)
    return {"BTTS_yes": round(yes, 4), "BTTS_no": round(1 - yes, 4)}


def compute_btts_and_result(matrix: list[list[float]]) -> dict:
    btts_home = btts_draw = btts_away = 0.0
    no_btts_home = no_btts_away = 0.0

    for h in range(len(matrix)):
        for a in range(len(matrix[0])):
            p = matrix[h][a]
            both_scored = h > 0 and a > 0
            if both_scored:
                if h > a:
                    btts_home += p
                elif h == a:
                    btts_draw += p
                else:
                    btts_away += p
            else:
                if h > a:
                    no_btts_home += p
                elif h < a:
                    no_btts_away += p

    return {
        "BTTS_yes_home": round(btts_home, 4),
        "BTTS_yes_draw": round(btts_draw, 4),
        "BTTS_yes_away": round(btts_away, 4),
        "BTTS_no_home": round(no_btts_home, 4),
        "BTTS_no_away": round(no_btts_away, 4),
    }


def compute_exact_scores(matrix: list[list[float]], top_n: int = 5) -> list[dict]:
    scores = []
    for h in range(len(matrix)):
        for a in range(len(matrix[0])):
            scores.append({"score": f"{h}-{a}", "prob": matrix[h][a]})
    scores.sort(key=lambda x: x["prob"], reverse=True)
    return scores[:top_n]


def compute_exact_goals(matrix: list[list[float]]) -> dict:
    totals: dict[int, float] = {}
    for h in range(len(matrix)):
        for a in range(len(matrix[0])):
            t = h + a
            bucket = t if t <= 3 else 4
            totals[bucket] = totals.get(bucket, 0.0) + matrix[h][a]
    return {
        "Goals_0": round(totals.get(0, 0.0), 4),
        "Goals_1": round(totals.get(1, 0.0), 4),
        "Goals_2": round(totals.get(2, 0.0), 4),
        "Goals_3": round(totals.get(3, 0.0), 4),
        "Goals_4plus": round(totals.get(4, 0.0), 4),
    }


def compute_halftime_fulltime(lam_home: float, lam_away: float,
                               ft_matrix: list[list[float]]) -> dict:
    lam_h_ht = lam_home * 0.45
    lam_a_ht = lam_away * 0.45
    ht_matrix = _score_matrix(lam_h_ht, lam_a_ht, max_goals=4)

    ht_1x2 = compute_1x2(ht_matrix)
    ft_1x2 = compute_1x2(ft_matrix)

    combos = {
        "HTFT_1/1": ht_1x2["1"] * ft_1x2["1"],
        "HTFT_X/1": ht_1x2["X"] * ft_1x2["1"],
        "HTFT_2/2": ht_1x2["2"] * ft_1x2["2"],
        "HTFT_X/X": ht_1x2["X"] * ft_1x2["X"],
        "HTFT_1/X": ht_1x2["1"] * ft_1x2["X"],
        "HTFT_X/2": ht_1x2["X"] * ft_1x2["2"],
    }
    return {k: round(v, 4) for k, v in combos.items()}


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


def compute_win_both_halves(matrix: list[list[float]]) -> dict:
    """
    Probabilité qu'une équipe gagne les deux mi-temps.
    Approximation basée sur la dominance globale.
    """
    p1_win_match = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[0])) if h > a)
    # Estimation: si équipe gagne le match, elle a ~60% de chances de gagner au moins une mi-temps
    p1_both = p1_win_match * 0.35
    p2_both = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[0])) if h < a) * 0.35
    return {
        "Home_wins_both_halves": round(p1_both, 4),
        "Away_wins_both_halves": round(p2_both, 4),
    }


def compute_win_from_behind(lam_home: float, lam_away: float) -> dict:
    """
    Probabilité de remontada (être mené puis gagner).
    Approximation basée sur distribution de buts.
    """
    # Simulation simplifiée: équipe marque après avoir encaissé
    p_home_behind = _poisson_pmf(0, lam_home) * (1 - _poisson_pmf(0, lam_away))
    p_home_comeback = p_home_behind * 0.25
    p_away_behind = _poisson_pmf(0, lam_away) * (1 - _poisson_pmf(0, lam_home))
    p_away_comeback = p_away_behind * 0.25
    return {
        "Home_wins_from_behind": round(p_home_comeback, 4),
        "Away_wins_from_behind": round(p_away_comeback, 4),
    }


# --------------------------------------------------------------------------- #
#  Player-specific markets                                                    #
# --------------------------------------------------------------------------- #

def compute_top_scorers(match: dict, num_players: int = 5) -> list[dict]:
    """
    Simule les meilleurs buteurs probables basés sur les rangs et positions.
    En conditions réelles, une API de stats joueurs serait utilisée.
    """
    home_team = match.get("home_name", "Équipe 1")
    away_team = match.get("away_name", "Équipe 2")
    
    # Simulation basée sur le rang et la force de l'équipe
    home_strength = max(1, 25 - (match.get("home_rank", 15) or 15))
    away_strength = max(1, 25 - (match.get("away_rank", 15) or 15))
    
    # Liste fictive de joueurs (en conditions réelles, API externe)
    players = [
        {"name": f"Attaquant A ({home_team})", "team": "home", "position": "forward", 
         "strength": home_strength, "goal_prob": min(0.45, home_strength / 50)},
        {"name": f"Attaquant B ({home_team})", "team": "home", "position": "forward",
         "strength": home_strength * 0.8, "goal_prob": min(0.35, home_strength / 60)},
        {"name": f"Milieu C ({home_team})", "team": "home", "position": "midfielder",
         "strength": home_strength * 0.6, "goal_prob": min(0.20, home_strength / 80)},
        {"name": f"Attaquant X ({away_team})", "team": "away", "position": "forward",
         "strength": away_strength, "goal_prob": min(0.45, away_strength / 50)},
        {"name": f"Attaquant Y ({away_team})", "team": "away", "position": "forward",
         "strength": away_strength * 0.8, "goal_prob": min(0.35, away_strength / 60)},
    ]
    
    # Calcul des probabilités de marquer
    lam_home, lam_away = _expected_goals(match)
    for p in players:
        if p["team"] == "home":
            team_lam = lam_home
        else:
            team_lam = lam_away
        # Probabilité de marquer au moins 1 but
        p["prob_to_score"] = round(1 - _poisson_pmf(0, team_lam * p["strength"] / max(home_strength, away_strength)), 4)
    
    players.sort(key=lambda x: x["prob_to_score"], reverse=True)
    return players[:num_players]


def compute_first_last_goalscorer(match: dict) -> dict:
    """
    Probabilités pour le premier et dernier buteur.
    """
    lam_home, lam_away = _expected_goals(match)
    total_goals = lam_home + lam_away
    
    # Probabilité qu'un but soit marqué dans les 15 premières minutes
    p_early_goal = 1 - math.exp(-total_goals * 15 / 90)
    
    # Distribution approximative
    return {
        "p_first_goal_home": round(lam_home / max(total_goals, 0.01) * p_early_goal, 4),
        "p_first_goal_away": round(lam_away / max(total_goals, 0.01) * p_early_goal, 4),
        "p_last_goal_home": round(lam_home / max(total_goals, 0.01) * 0.7, 4),
        "p_last_goal_away": round(lam_away / max(total_goals, 0.01) * 0.7, 4),
    }


def compute_goal_method(match: dict) -> dict:
    """
    Probabilités pour la méthode du but (tête, penalty, coup franc).
    """
    lam_home, lam_away = _expected_goals(match)
    total_goals = lam_home + lam_away
    
    # Moyennes statistiques
    return {
        "header_goal": round(min(0.12, total_goals * 0.08), 4),
        "penalty_goal": round(min(0.08, total_goals * 0.05), 4),
        "free_kick_goal": round(min(0.06, total_goals * 0.04), 4),
        "outside_box_goal": round(min(0.10, total_goals * 0.07), 4),
    }


def compute_player_stats(match: dict) -> dict:
    """
    Statistiques de joueurs simulées (tirs, passes décisives, tacles, cartons).
    """
    lam_home, lam_away = _expected_goals(match)
    home_strength = max(1, 25 - (match.get("home_rank", 15) or 15))
    away_strength = max(1, 25 - (match.get("away_rank", 15) or 15))
    
    return {
        "home_shots_on_target": round(lam_home * 4.2, 1),
        "away_shots_on_target": round(lam_away * 3.8, 1),
        "home_assists": round(lam_home * 0.7, 1),
        "away_assists": round(lam_away * 0.6, 1),
        "home_tackles": round(home_strength * 2.5, 1),
        "away_tackles": round(away_strength * 2.3, 1),
    }


# --------------------------------------------------------------------------- #
#  Statistical markets (corners, cards, shots, fouls, possession)             #
# --------------------------------------------------------------------------- #

def compute_corners(match: dict) -> dict:
    avg_home, avg_away = _league_corners(match.get("competition", ""))
    avg_home = match.get("home_avg_corners", avg_home) or avg_home
    avg_away = match.get("away_avg_corners", avg_away) or avg_away
    exp_total = avg_home + avg_away

    result = {}
    for threshold in [8.5, 9.5, 10.5, 11.5]:
        p_over = 1 - sum(_poisson_pmf(k, exp_total) for k in range(int(threshold) + 1))
        result[f"Corners_O{threshold}"] = round(p_over, 4)
        result[f"Corners_U{threshold}"] = round(1 - p_over, 4)
    
    # Team corners individual
    for t in [4.5, 5.5, 6.5]:
        p_over_h = 1 - sum(_poisson_pmf(k, avg_home) for k in range(int(t) + 1))
        result[f"Corners_H_O{t}"] = round(p_over_h, 4)
        p_over_a = 1 - sum(_poisson_pmf(k, avg_away) for k in range(int(t) + 1))
        result[f"Corners_A_O{t}"] = round(p_over_a, 4)
    
    # Which team gets more corners
    p_home_more = sum(
        _poisson_pmf(h, avg_home) * _poisson_pmf(a, avg_away)
        for h in range(25) for a in range(25) if h > a
    )
    p_away_more = sum(
        _poisson_pmf(h, avg_home) * _poisson_pmf(a, avg_away)
        for h in range(25) for a in range(25) if h < a
    )
    result["Corners_home_more"] = round(p_home_more, 4)
    result["Corners_away_more"] = round(p_away_more, 4)
    
    return result


def compute_cards(match: dict) -> dict:
    avg_home, avg_away = _league_yellow(match.get("competition", ""))
    avg_home = match.get("home_avg_yellow", avg_home) or avg_home
    avg_away = match.get("away_avg_yellow", avg_away) or avg_away
    exp_total = avg_home + avg_away

    result = {}
    for threshold in [2.5, 3.5, 4.5, 5.5]:
        p_over = 1 - sum(_poisson_pmf(k, exp_total) for k in range(int(threshold) + 1))
        result[f"Cards_O{threshold}"] = round(p_over, 4)
        result[f"Cards_U{threshold}"] = round(1 - p_over, 4)
    
    # Individual team cards
    for t in [1.5, 2.5]:
        p_over_h = 1 - sum(_poisson_pmf(k, avg_home) for k in range(int(t) + 1))
        result[f"Cards_H_O{t}"] = round(p_over_h, 4)
        p_over_a = 1 - sum(_poisson_pmf(k, avg_away) for k in range(int(t) + 1))
        result[f"Cards_A_O{t}"] = round(p_over_a, 4)
    
    # Red card probability (~15% of yellow card rate)
    p_red = 1 - math.exp(-exp_total * 0.08)
    result["Card_Red_O0.5"] = round(p_red, 4)
    result["Card_Red_U0.5"] = round(1 - p_red, 4)
    
    return result


def compute_shots(match: dict) -> dict:
    """Shots on target and total shots."""
    lam_home, lam_away = _expected_goals(match)
    
    # Average shots: ~4.5 shots per goal expected
    home_shots = lam_home * 4.5
    away_shots = lam_away * 4.0
    
    return {
        "Home_shots_total": round(home_shots, 1),
        "Away_shots_total": round(away_shots, 1),
        "Home_shots_on_target": round(home_shots * 0.35, 1),
        "Away_shots_on_target": round(away_shots * 0.33, 1),
    }


def compute_fouls_offsides(match: dict) -> dict:
    """Fouls and offsides predictions."""
    lam_home, lam_away = _expected_goals(match)
    intensity = lam_home + lam_away
    
    home_fouls = 10 + intensity * 2
    away_fouls = 9 + intensity * 1.8
    home_offsides = 1.5 + intensity * 0.5
    away_offsides = 1.2 + intensity * 0.4
    
    return {
        "Home_fouls": round(home_fouls, 1),
        "Away_fouls": round(away_fouls, 1),
        "Home_offsides": round(home_offsides, 1),
        "Away_offsides": round(away_offsides, 1),
    }


def compute_possession(match: dict) -> dict:
    """Ball possession prediction (percentage ranges)."""
    home_rank = match.get("home_rank", 15)
    away_rank = match.get("away_rank", 15)
    home_advantage = 5 if _home_advantage(match) > 1 else 0
    
    # Inverse rank = better team has higher possession
    home_possession = 50 + (away_rank - home_rank) * 0.8 + home_advantage
    home_possession = max(35, min(65, home_possession))
    away_possession = 100 - home_possession
    
    return {
        "Home_possession": round(home_possession, 1),
        "Away_possession": round(away_possession, 1),
        "Home_over_55": round(1 if home_possession > 55 else 0, 2),
        "Away_over_55": round(1 if away_possession > 55 else 0, 2),
    }


# --------------------------------------------------------------------------- #
#  Chronological markets                                                       #
# --------------------------------------------------------------------------- #

def compute_goal_intervals(lam_home: float, lam_away: float) -> dict:
    """Goal in time intervals (1-15, 16-30, 31-45, 46-60, 61-75, 76-90)."""
    lam_total = lam_home + lam_away
    
    intervals = [(1, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90)]
    result = {}
    
    for start, end in intervals:
        # Probability of at least one goal in interval
        minutes = end - start + 1
        lam_interval = lam_total * minutes / 90
        p_goal = 1 - math.exp(-lam_interval)
        result[f"Goal_{start}-{end}"] = round(p_goal, 4)
    
    return result


def compute_result_at_minute(lam_home: float, lam_away: float, minute: int = 30) -> dict:
    """Result at specific minute (e.g., minute 30)."""
    factor = minute / 90
    lam_home_min = lam_home * factor
    lam_away_min = lam_away * factor
    
    matrix = _score_matrix(lam_home_min, lam_away_min, max_goals=4)
    p1x2 = compute_1x2(matrix)
    
    return {
        f"Result_at_{minute}_1": round(p1x2["1"], 4),
        f"Result_at_{minute}_X": round(p1x2["X"], 4),
        f"Result_at_{minute}_2": round(p1x2["2"], 4),
    }


def compute_first_5_minutes(lam_home: float, lam_away: float) -> dict:
    """Events in first 5 minutes."""
    lam_total = lam_home + lam_away
    lam_5min = lam_total * 5 / 90
    
    p_corner = 1 - math.exp(-0.6 * lam_5min)  # ~0.6 corners per goal
    p_goal = 1 - math.exp(-lam_5min)
    p_penalty = p_goal * 0.1
    p_card = 1 - math.exp(-0.8 * lam_5min)  # ~0.8 cards per goal
    
    return {
        "First_5min_corner": round(p_corner, 4),
        "First_5min_goal": round(p_goal, 4),
        "First_5min_penalty": round(p_penalty, 4),
        "First_5min_card": round(p_card, 4),
    }


# --------------------------------------------------------------------------- #
#  Special markets                                                             #
# --------------------------------------------------------------------------- #

def compute_penalty_in_match(match: dict) -> dict:
    """Probability of penalty awarded and scored."""
    lam_home, lam_away = _expected_goals(match)
    total_goals = lam_home + lam_away
    
    # ~5% of matches have a penalty
    p_penalty = min(0.15, total_goals * 0.04)
    p_scored = 0.75  # Historical average
    
    return {
        "Penalty_awarded": round(p_penalty, 4),
        "Penalty_scored": round(p_penalty * p_scored, 4),
        "Penalty_missed": round(p_penalty * (1 - p_scored), 4),
    }


def compute_own_goal(match: dict) -> dict:
    """Probability of own goal."""
    lam_home, lam_away = _expected_goals(match)
    total_goals = lam_home + lam_away
    
    # ~3% of goals are own goals, ~3-5% of matches
    p_own_goal = min(0.08, total_goals * 0.025)
    
    return {
        "Own_goal": round(p_own_goal, 4),
        "No_own_goal": round(1 - p_own_goal, 4),
    }


def compute_red_card(match: dict) -> dict:
    """Red card probability."""
    lam_home, lam_away = _expected_goals(match)
    intensity = lam_home + lam_away
    
    p_red = min(0.12, intensity * 0.035 + 0.02)
    
    return {
        "Red_card": round(p_red, 4),
        "No_red_card": round(1 - p_red, 4),
    }


def compute_comeback_win(match: dict) -> dict:
    """Team wins after being behind at any point."""
    lam_home, lam_away = _expected_goals(match)
    
    # Complex simulation: probability of trailing then winning
    p_home_behind = _poisson_pmf(0, lam_home) * (1 - _poisson_pmf(0, lam_away))
    p_away_behind = _poisson_pmf(0, lam_away) * (1 - _poisson_pmf(0, lam_home))
    
    p_home_comeback = p_home_behind * 0.3
    p_away_comeback = p_away_behind * 0.25
    
    return {
        "Home_comeback_win": round(p_home_comeback, 4),
        "Away_comeback_win": round(p_away_comeback, 4),
    }


def compute_combo_markets(p1x2: dict, btts: dict, ou: dict, dnb: dict) -> dict:
    """Combination markets: 1+BTTS, 2+BTTS, 1+O2.5, etc."""
    return {
        "1_and_BTTS": round(p1x2["1"] * btts["BTTS_yes"], 4),
        "X_and_BTTS": round(p1x2["X"] * btts["BTTS_yes"], 4),
        "2_and_BTTS": round(p1x2["2"] * btts["BTTS_yes"], 4),
        "1_and_O2.5": round(p1x2["1"] * ou.get("O2.5", 0), 4),
        "X_and_U2.5": round(p1x2["X"] * ou.get("U2.5", 0), 4),
        "2_and_O2.5": round(p1x2["2"] * ou.get("O2.5", 0), 4),
        "DNB_home_and_U2.5": round(dnb["DNB_home"] * ou.get("U2.5", 0), 4),
        "DNB_away_and_U2.5": round(dnb["DNB_away"] * ou.get("U2.5", 0), 4),
    }


# --------------------------------------------------------------------------- #
#  Clean sheet and other markets                                               #
# --------------------------------------------------------------------------- #

def compute_clean_sheet(matrix: list[list[float]]) -> dict:
    home_cs = sum(matrix[h][0] for h in range(len(matrix)))
    away_cs = sum(matrix[0][a] for a in range(len(matrix[0])))
    return {
        "CS_home": round(home_cs, 4),
        "CS_away": round(away_cs, 4),
    }


def compute_win_to_nil(matrix: list[list[float]]) -> dict:
    home_wtn = sum(matrix[h][0] for h in range(1, len(matrix)))
    away_wtn = sum(matrix[0][a] for a in range(1, len(matrix[0])))
    return {
        "WTN_home": round(home_wtn, 4),
        "WTN_away": round(away_wtn, 4),
    }


def compute_both_halves_goal(lam_home: float, lam_away: float) -> dict:
    lam_total = lam_home + lam_away
    p_ht1 = 1 - math.exp(-lam_total * 0.45)
    p_ht2 = 1 - math.exp(-lam_total * 0.55)
    p_both = p_ht1 * p_ht2
    return {
        "BHG_ht1": round(p_ht1, 4),
        "BHG_ht2": round(p_ht2, 4),
        "BHG_both": round(p_both, 4),
    }


def compute_first_goal_time(lam_home: float, lam_away: float) -> dict:
    lam = lam_home + lam_away
    return {
        "FG_before_15": round(1 - math.exp(-lam * 15 / 90), 4),
        "FG_before_30": round(1 - math.exp(-lam * 30 / 90), 4),
        "FG_before_45": round(1 - math.exp(-lam * 45 / 90), 4),
        "FG_after_75": round(math.exp(-lam * 75 / 90), 4),
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
#  Star rating helper                                                          #
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
#  AI narrative prompt (no odds)                                               #
# --------------------------------------------------------------------------- #

def _format_form(form: list) -> str:
    return " ".join(form) if form else "N/A"


def _h2h_summary(match: dict) -> str:
    rate = match.get("h2h_btts_rate", 0)
    avg = match.get("h2h_avg_goals", 0)
    if avg == 0 and rate == 0:
        return "Pas de données H2H"
    return f"Moy. buts={avg:.1f} | BTTS={rate:.0%}"


def _build_ai_prompt(match: dict, pronostics: dict,
                     excluded_models: list[str] | None = None) -> str:
    ensemble = pronostics.get("ensemble", {})
    p_poisson = pronostics.get("model_poisson", {})
    p_dixon = pronostics.get("model_dixon", {})
    p_elo = pronostics.get("model_elo", {})
    p_xg = pronostics.get("model_xg", {})

    def _w(outcome: str) -> float:
        return ensemble.get(outcome, {}).get("weighted", 0.0)

    date_str = match.get("date_formatted", match.get("utc_date", "N/A"))

    fgt = pronostics.get("first_goal_time", {})
    bhg = pronostics.get("both_halves_goal", {})
    cbt = pronostics.get("corners_by_team", {})
    kbt = pronostics.get("cards_by_team", {})
    btr = pronostics.get("btts_and_result", {})
    wtn = pronostics.get("win_to_nil", {})

    _btr_labels = {
        "BTTS_yes_home": "BTTS Oui + Dom.", "BTTS_yes_draw": "BTTS Oui + Nul",
        "BTTS_yes_away": "BTTS Oui + Ext.", "BTTS_no_home": "BTTS Non + Dom.",
        "BTTS_no_away": "BTTS Non + Ext.",
    }
    best_combo = max(btr, key=btr.get) if btr else ""
    best_combo_label = _btr_labels.get(best_combo, best_combo)
    best_combo_prob = btr.get(best_combo, 0.0)

    def _fs(val) -> str:
        return f"{val:.1f}" if val is not None else "N/D"

    p_xg_safe = p_xg if isinstance(p_xg, dict) else {}
    xg_row = (
        f"- xG ajusté   : P1={p_xg_safe.get('p1', 0):.1f}%    | PX={p_xg_safe.get('px', 0):.1f}%     | P2={p_xg_safe.get('p2', 0):.1f}%"
        if p_xg_safe
        else "- xG ajusté   : EXCLU (données insuffisantes)"
    )

    exclusion_note = (
        f"\n⚠️ MODÈLES EXCLUS : {', '.join(excluded_models)} — confiance = Faible."
        if excluded_models else ""
    )

    prompt = f"""Tu es un data scientist expert en modélisation de matchs de football.

MATCH : {match['home_name']} vs {match['away_name']}
COMPÉTITION : {match['competition']}
DATE : {date_str}

DONNÉES DISPONIBLES :
- Forme domicile : {_format_form(match['home_form'])} | Buts: {_fs(match['home_avg_scored'])} enc. {_fs(match['home_avg_conceded'])}
- Forme extérieur : {_format_form(match['away_form'])} | Buts: {_fs(match['away_avg_scored'])} enc. {_fs(match['away_avg_conceded'])}
- Rang: {match['home_rank']} / {match['away_rank']}
- H2H: {_h2h_summary(match)}

MODÈLES (probabilités %) :
- Poisson     : P1={p_poisson.get('p1', 0):.1f}% | PX={p_poisson.get('px', 0):.1f}% | P2={p_poisson.get('p2', 0):.1f}%
- Dixon-Coles : P1={p_dixon.get('p1', 0):.1f}%  | PX={p_dixon.get('px', 0):.1f}%  | P2={p_dixon.get('p2', 0):.1f}%
- Elo         : P1={p_elo.get('p1', 0):.1f}%    | PX={p_elo.get('px', 0):.1f}%    | P2={p_elo.get('p2', 0):.1f}%
{xg_row}
- FUSION      : P1={_w('p1'):.1f}% | PX={_w('px'):.1f}% | P2={_w('p2'):.1f}%

MARCHÉS MODÈLE :
- 1er but avant 30 min : {fgt.get('FG_before_30', 0)*100:.1f}%
- Corners {match['home_name']} O5.5 : {cbt.get('Corn_H_O5.5', 0)*100:.1f}%
- Cartons {match['away_name']} O1.5 : {kbt.get('Card_A_O1.5', 0)*100:.1f}%
- Victoire sans encaisser : {match['home_name']}={wtn.get('WTN_home', 0)*100:.1f}% | {match['away_name']}={wtn.get('WTN_away', 0)*100:.1f}%

ANALYSE (concis, basé sur modèles) :
1. Convergence/divergence des modèles ?
2. Facteurs contextuels (domicile, forme, rang)
3. Score exact le plus probable
4. Minute probable du 1er but
5. Niveau de confiance global{exclusion_note}"""

    return prompt


def _multi_ai_narratives(match: dict, pronostics: dict,
                         excluded_models: list[str] | None = None) -> dict[str, str]:
    from ai_providers import get_multi_ai_narratives
    prompt = _build_ai_prompt(match, pronostics, excluded_models)
    narratives = get_multi_ai_narratives(prompt)
    if not narratives:
        return {"Info": "Analyse IA non disponible"}
    return narratives


# --------------------------------------------------------------------------- #
#  Main engine entry point                                                     #
# --------------------------------------------------------------------------- #

def compute_pronostics(match: dict, use_ai: bool = True,
                       force_ai: bool = False) -> dict:
    lam_home, lam_away = _expected_goals(match)

    m_poisson = _model_poisson(lam_home, lam_away)
    m_dixon = _model_dixon_coles(lam_home, lam_away)
    m_elo = _model_elo(match)

    excluded_models: list[str] = []
    if _xg_usable(match):
        m_xg = _model_xg_adjusted(match)
    else:
        m_xg = None
        excluded_models.append("xG")

    ensemble = _ensemble_fusion(m_poisson, m_dixon, m_elo, m_xg)
    matrix = m_dixon["matrix"]

    p1x2_final = {
        "1": round(ensemble["p1"]["weighted"] / 100, 4),
        "X": round(ensemble["px"]["weighted"] / 100, 4),
        "2": round(ensemble["p2"]["weighted"] / 100, 4),
    }

    # Core markets
    dc = compute_double_chance(p1x2_final)
    ou = compute_over_under(matrix)
    btts = compute_btts(matrix)
    dnb = compute_draw_no_bet(p1x2_final)
    exact = compute_exact_scores(matrix)
    htft = compute_halftime_fulltime(lam_home, lam_away, matrix)
    ht = compute_halftime(lam_home, lam_away)
    cs = compute_clean_sheet(matrix)
    wtn = compute_win_to_nil(matrix)
    bhg = compute_both_halves_goal(lam_home, lam_away)
    fgt = compute_first_goal_time(lam_home, lam_away)
    odd_even = compute_odd_even(matrix)
    exact_goals = compute_exact_goals(matrix)
    
    # Totaux
    home_individual = compute_individual_total(lam_home)
    away_individual = compute_individual_total(lam_away)
    asian_total = compute_asian_total(lam_home + lam_away)
    
    # Handicaps
    eh = compute_handicap_european(p1x2_final)
    ah = compute_asian_handicap(lam_home, lam_away, matrix)
    
    # Statistiques
    corners = compute_corners(match)
    cards = compute_cards(match)
    shots = compute_shots(match)
    fouls = compute_fouls_offsides(match)
    possession = compute_possession(match)
    
    # Chronologiques
    goal_intervals = compute_goal_intervals(lam_home, lam_away)
    result_30min = compute_result_at_minute(lam_home, lam_away, 30)
    result_60min = compute_result_at_minute(lam_home, lam_away, 60)
    first_5min = compute_first_5_minutes(lam_home, lam_away)
    
    # Spéciaux
    penalty = compute_penalty_in_match(match)
    own_goal = compute_own_goal(match)
    red_card = compute_red_card(match)
    comeback = compute_comeback_win(match)
    both_halves_win = compute_win_both_halves(matrix)
    combo = compute_combo_markets(p1x2_final, btts, ou, dnb)
    
    # Joueurs (simulés)
    top_scorers = compute_top_scorers(match)
    first_last_goalscorer = compute_first_last_goalscorer(match)
    goal_method = compute_goal_method(match)
    player_stats = compute_player_stats(match)

    # Recommendations
    best_1x2_label, best_1x2_prob = _best_1x2(p1x2_final)
    best_dc_label, best_dc_prob = _best_dc(dc)
    best_ou_label, best_ou_prob = _best_ou(ou, "O")

    pronostics = {
        # Modèles
        "lam_home": lam_home,
        "lam_away": lam_away,
        "model_poisson": m_poisson,
        "model_dixon": m_dixon,
        "model_elo": m_elo,
        "model_xg": m_xg,
        "ensemble": ensemble,
        "excluded_models": excluded_models,
        
        # Paris principaux
        "p1x2": p1x2_final,
        "double_chance": dc,
        "btts": btts,
        "exact_scores": exact,
        "halftime_fulltime": htft,
        "halftime": ht,
        
        # Totaux
        "over_under": ou,
        "home_individual_total": home_individual,
        "away_individual_total": away_individual,
        "over_under_asian": asian_total,
        "exact_goals": exact_goals,
        
        # Handicaps
        "handicap_eu": eh,
        "handicap_asian": ah,
        "draw_no_bet": dnb,
        
        # Clean sheet
        "clean_sheet": cs,
        "win_to_nil": wtn,
        "both_halves_goal": bhg,
        "first_goal_time": fgt,
        "odd_even": odd_even,
        
        # Statistiques
        "corners": corners,
        "cards": cards,
        "shots": shots,
        "fouls_offsides": fouls,
        "possession": possession,
        
        # Chronologiques
        "goal_intervals": goal_intervals,
        "result_at_30min": result_30min,
        "result_at_60min": result_60min,
        "first_5min_events": first_5min,
        
        # Spéciaux
        "penalty": penalty,
        "own_goal": own_goal,
        "red_card": red_card,
        "comeback_win": comeback,
        "win_both_halves": both_halves_win,
        "combo_markets": combo,
        
        # Joueurs
        "top_scorers": top_scorers,
        "first_last_goalscorer": first_last_goalscorer,
        "goal_method": goal_method,
        "player_stats": player_stats,
        
        # Recommendations
        "rec_1x2": {"label": best_1x2_label, "prob": best_1x2_prob, "stars": _stars(best_1x2_prob)},
        "rec_dc": {"label": best_dc_label, "prob": best_dc_prob, "stars": _stars(best_dc_prob)},
        "rec_ou": {"label": best_ou_label, "prob": best_ou_prob, "stars": _stars(best_ou_prob)},
        "rec_btts": {
            "label": "Oui" if btts["BTTS_yes"] >= btts["BTTS_no"] else "Non",
            "prob": max(btts["BTTS_yes"], btts["BTTS_no"]),
            "stars": _stars(max(btts["BTTS_yes"], btts["BTTS_no"])),
        },
        "rec_score": {"label": exact[0]["score"] if exact else "N/A", "prob": exact[0]["prob"] if exact else 0},
    }

    # AI narratives
    ai_ok, ai_skip_reason = _should_use_ai(match, ensemble)
    if force_ai:
        ai_ok, ai_skip_reason = True, ""
    if use_ai and ai_ok:
        narratives = _multi_ai_narratives(match, pronostics, excluded_models)
        pronostics["ai_narratives"] = narratives
        pronostics["ai_narrative"] = narratives.get("Claude", next(iter(narratives.values()), ""))
    else:
        pronostics["ai_narratives"] = {}
        pronostics["ai_narrative"] = ""
        if use_ai and not ai_ok:
            logger.info("Analyse IA ignorée: %s", ai_skip_reason)

    return pronostics


def run_engine(matches: list[dict], use_ai: bool = True,
               force_ai: bool = False) -> list[dict]:
    results = []
    for match in matches:
        try:
            prono = compute_pronostics(match, use_ai=use_ai, force_ai=force_ai)
            results.append({"match": match, "pronostics": prono})
            logger.info("Pronostic: %s vs %s", match["home_name"], match["away_name"])
        except Exception as e:
            logger.error("Erreur pour %s vs %s: %s",
                         match.get("home_name", "?"), match.get("away_name", "?"), e)
    return results

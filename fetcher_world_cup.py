"""
Fetcher Coupe du Monde 2026 — phase de groupes complète (72 matchs).

S'appuie sur football-data.org (compétition WC), comme fetcher_football_data,
mais récupère TOUS les matchs de la phase de groupes en une fois
(stage=GROUP_STAGE), quel que soit le statut (SCHEDULED, TIMED).

Le classement est extrait par groupe (Groupe A → L) pour produire un rang
1–4 intra-groupe utilisé par le modèle Elo.

NOTE rate-limit : football-data.org free tier = 10 requêtes/minute.
En mode complet (--enrich), 72 matchs × (2 formes + 1 H2H) ≈ 220 requêtes,
soit ~22 minutes de collecte. Le mode rapide (par défaut ici désactivable)
saute l'enrichissement par équipe et utilise les moyennes H2H/ligue.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any

import fetcher_football_data as fbd

logger = logging.getLogger(__name__)

WC_CODE = "WC"
WC_NAME = "Coupe du Monde 2026"

# Fenêtre par défaut de la phase de groupes (11 juin → 27 juin 2026)
GROUP_STAGE_FROM = "2026-06-11"
GROUP_STAGE_TO   = "2026-06-28"   # inclusif côté API, marge d'un jour


# --------------------------------------------------------------------------- #
#  Matchs de la phase de groupes                                               #
# --------------------------------------------------------------------------- #

def get_group_stage_matches(date_from: str = GROUP_STAGE_FROM,
                            date_to: str = GROUP_STAGE_TO,
                            only_upcoming: bool = True) -> list[dict]:
    """
    Tous les matchs de la phase de groupes de la Coupe du Monde.
    only_upcoming=True : exclut les matchs déjà terminés ou en cours.
    """
    data = fbd._get(f"/competitions/{WC_CODE}/matches", {
        "dateFrom": date_from,
        "dateTo":   date_to,
        "stage":    "GROUP_STAGE",
    })
    if not data:
        logger.error("Coupe du Monde : aucune donnée reçue de football-data.org")
        return []

    matches = []
    skipped = 0
    for m in data.get("matches", []):
        status = m.get("status", "")
        if only_upcoming and status not in ("SCHEDULED", "TIMED"):
            skipped += 1
            continue
        m["_competition_name"] = WC_NAME
        m["_competition_code"] = WC_CODE
        m["_source"] = "football-data"
        
        # Extraire le groupe (ex: "GROUP_A") depuis le stage du match
        group = m.get("group", "")
        if not group:
            # Alternative : déduire depuis le nom du groupe dans le match
            group = m.get("stage", "")
        m["group"] = group
        
        matches.append(m)

    logger.info("Coupe du Monde : %d matchs de poules récupérés (%d ignorés — déjà joués/en cours)",
                len(matches), skipped)
    return matches


# --------------------------------------------------------------------------- #
#  Classements par groupe → rang 1-4 intra-groupe                              #
# --------------------------------------------------------------------------- #

_rank_map_cache: dict[int, int] | None = None
_group_rank_maps: dict[str, dict[int, int]] = {}  # par groupe


def _build_group_rank_maps() -> dict[str, dict[int, int]]:
    """Construit {groupe: {team_id: position}} pour tous les groupes."""
    global _group_rank_maps
    if _group_rank_maps:
        return _group_rank_maps

    data = fbd._get(f"/competitions/{WC_CODE}/standings")
    if not data:
        logger.warning("Coupe du Monde : classements non disponibles")
        return {}

    for table in data.get("standings", []):
        if table.get("type") != "TOTAL":
            continue
        # Le nom du groupe est dans 'group' (ex: "GROUP_A")
        group_name = table.get("group", "Groupe inconnu")
        group_rank_map: dict[int, int] = {}
        for row in table.get("table", []):
            team_id = row.get("team", {}).get("id")
            if team_id is not None:
                group_rank_map[team_id] = row.get("position", 99)
        if group_rank_map:
            _group_rank_maps[group_name] = group_rank_map

    logger.info("Coupe du Monde : classements chargés pour %d groupes", len(_group_rank_maps))
    return _group_rank_maps


def get_team_rank_in_group(team_id: int, group: str) -> int:
    """Retourne le rang (1-4) d'une équipe dans son groupe."""
    maps = _build_group_rank_maps()
    group_map = maps.get(group, {})
    return group_map.get(team_id, 99)


# --------------------------------------------------------------------------- #
#  Enrichissement                                                              #
# --------------------------------------------------------------------------- #

def enrich_match(match: dict, with_form: bool = True) -> dict:
    """
    Enrichit un match CDM :
      - classement intra-groupe (toujours, 1 seul appel API mutualisé)
      - forme des équipes + H2H (optionnel : 3 appels/match, ~6s chacun)
    """
    # Récupération du groupe du match
    group = match.get("group", "")
    home_id = match["homeTeam"]["id"]
    away_id = match["awayTeam"]["id"]

    # Classement intra-groupe
    home_rank = get_team_rank_in_group(home_id, group)
    away_rank = get_team_rank_in_group(away_id, group)

    # Tableau synthétique compatible avec _get_rank_fbd de l'analyzer
    match["_standings"] = [
        {"team": {"id": home_id}, "position": home_rank},
        {"team": {"id": away_id}, "position": away_rank},
    ]

    if with_form:
        logger.debug("Enrichissement forme pour %s vs %s", match["homeTeam"]["name"], match["awayTeam"]["name"])
        match["_home_form"] = fbd.get_team_form(home_id)
        match["_away_form"] = fbd.get_team_form(away_id)
        match["_h2h"] = fbd.get_h2h(match["id"])
    else:
        # Mode rapide : formes vides, pas de H2H
        match["_home_form"] = []
        match["_away_form"] = []
        match["_h2h"] = []

    return match


def fetch_all_enriched(with_form: bool = True,
                       only_upcoming: bool = True) -> list[dict]:
    """
    Récupère et enrichit tous les matchs de la phase de groupes.
    with_form=False : mode rapide (~2 min au lieu de ~25 min),
    les modèles s'appuient alors sur les moyennes de ligue par défaut.
    """
    matches = get_group_stage_matches(only_upcoming=only_upcoming)
    if not matches:
        return []

    if with_form:
        est_minutes = round(len(matches) * 3 * 6 / 60)
        logger.info("Enrichissement complet activé : ~%d min de collecte "
                    "(rate limit football-data 10 req/min)", est_minutes)

    enriched = []
    for i, m in enumerate(matches, 1):
        try:
            enriched.append(enrich_match(m, with_form=with_form))
            if with_form and i % 10 == 0:
                logger.info("Enrichissement : %d/%d matchs", i, len(matches))
        except Exception as e:
            logger.error("Erreur enrichissement match %s : %s", m.get("id"), e)

    logger.info("Coupe du Monde : %d matchs enrichis", len(enriched))
    return enriched

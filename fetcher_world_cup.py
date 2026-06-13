"""
Fetcher Coupe du Monde 2026 — phase de groupes complète (72 matchs).
Seuls les matchs non joués (SCHEDULED ou TIMED) sont retournés.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone

import fetcher_football_data as fbd

logger = logging.getLogger(__name__)

WC_CODE = "WC"
WC_NAME = "Coupe du Monde 2026"

# Fenêtre par défaut de la phase de groupes (11 juin → 27 juin 2026)
GROUP_STAGE_FROM = "2026-06-11"
GROUP_STAGE_TO   = "2026-06-28"


# --------------------------------------------------------------------------- #
#  Matchs de la phase de groupes (uniquement non joués)                        #
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
    skipped_finished = 0
    skipped_other = 0
    
    for m in data.get("matches", []):
        status = m.get("status", "")
        
        # Statuts possibles selon football-data.org:
        # SCHEDULED, TIMED → match à venir
        # IN_PLAY, PAUSED, LIVE → match en cours
        # FINISHED → match terminé
        
        if only_upcoming:
            if status in ("FINISHED", "IN_PLAY", "PAUSED", "LIVE"):
                skipped_finished += 1
                logger.debug("Match ignoré (déjà joué ou en cours): %s vs %s (%s)", 
                             m.get("homeTeam", {}).get("name"),
                             m.get("awayTeam", {}).get("name"),
                             status)
                continue
            if status not in ("SCHEDULED", "TIMED"):
                skipped_other += 1
                continue
        
        # Ajouter les métadonnées
        m["_competition_name"] = WC_NAME
        m["_competition_code"] = WC_CODE
        m["_source"] = "football-data"
        
        # Extraire le groupe
        group = m.get("group", "")
        if not group:
            group = m.get("stage", "")
        m["group"] = group
        
        # Ajouter le statut pour vérification
        m["_match_status"] = status
        
        matches.append(m)

    logger.info("Coupe du Monde : %d matchs à venir récupérés (%d matchs déjà joués ignorés, %d autres ignorés)",
                len(matches), skipped_finished, skipped_other)
    return matches


# --------------------------------------------------------------------------- #
#  Classements par groupe                                                      #
# --------------------------------------------------------------------------- #

_group_rank_maps: dict[str, dict[int, int]] = {}


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
        group_name = table.get("group", "Groupe inconnu")
        group_map: dict[int, int] = {}
        for row in table.get("table", []):
            team_id = row.get("team", {}).get("id")
            if team_id is not None:
                group_map[team_id] = row.get("position", 99)
        if group_map:
            _group_rank_maps[group_name] = group_map

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
      - classement intra-groupe
      - forme des équipes + H2H (optionnel)
    """
    group = match.get("group", "")
    home_id = match["homeTeam"]["id"]
    away_id = match["awayTeam"]["id"]

    # Classement intra-groupe
    home_rank = get_team_rank_in_group(home_id, group)
    away_rank = get_team_rank_in_group(away_id, group)

    match["_standings"] = [
        {"team": {"id": home_id}, "position": home_rank},
        {"team": {"id": away_id}, "position": away_rank},
    ]

    if with_form:
        logger.debug("Enrichissement forme pour %s vs %s", 
                     match["homeTeam"]["name"], match["awayTeam"]["name"])
        match["_home_form"] = fbd.get_team_form(home_id)
        match["_away_form"] = fbd.get_team_form(away_id)
        match["_h2h"] = fbd.get_h2h(match["id"])
    else:
        match["_home_form"] = []
        match["_away_form"] = []
        match["_h2h"] = []

    return match


def fetch_all_enriched(with_form: bool = True,
                       only_upcoming: bool = True) -> list[dict]:
    """
    Récupère et enrichit tous les matchs de la phase de groupes NON ENCORE JOUÉS.
    with_form=False : mode rapide
    only_upcoming=True : ignore les matchs déjà joués
    """
    matches = get_group_stage_matches(only_upcoming=only_upcoming)
    if not matches:
        logger.warning("Aucun match à venir trouvé pour la Coupe du Monde")
        return []

    if with_form:
        est_minutes = round(len(matches) * 3 * 6 / 60)
        logger.info("Enrichissement complet activé : ~%d min de collecte pour %d matchs", 
                    est_minutes, len(matches))

    enriched = []
    for i, m in enumerate(matches, 1):
        try:
            enriched.append(enrich_match(m, with_form=with_form))
            if with_form and i % 10 == 0:
                logger.info("Enrichissement : %d/%d matchs", i, len(matches))
        except Exception as e:
            logger.error("Erreur enrichissement match %s : %s", m.get("id"), e)

    logger.info("Coupe du Monde : %d matchs à venir enrichis", len(enriched))
    return enriched


# --------------------------------------------------------------------------- #
#  Fonction utilitaire pour vérifier si un match est joué                      #
# --------------------------------------------------------------------------- #

def is_match_played(match: dict) -> bool:
    """Retourne True si le match est déjà joué ou en cours."""
    status = match.get("status", "")
    return status in ("FINISHED", "IN_PLAY", "PAUSED", "LIVE")


def filter_upcoming_matches(matches: list[dict]) -> list[dict]:
    """Filtre une liste de matchs pour ne garder que ceux non joués."""
    return [m for m in matches if not is_match_played(m)]

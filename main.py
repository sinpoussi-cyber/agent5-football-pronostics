"""
Agent 5 Football Pronostics — point d'entrée principal.

Usage:
    python main.py                          # rapport quotidien (24h à venir)
    python main.py --type avant             # matchs du jour (9h)
    python main.py --type hebdo             # rapport hebdomadaire (lundi)
    python main.py --type coupe-du-monde    # TOUTE la phase de groupes CDM 2026 (uniquement matchs non joués)
    python main.py --type coupe-du-monde --fast   # idem, sans enrichissement
    python main.py --type coupe-du-monde --include-played  # inclut AUSSI les matchs déjà joués (déconseillé)
    python main.py --no-ai                  # désactive les analyses IA
    python main.py --dry-run                # génère le rapport sans envoyer l'email
"""

import argparse
import logging
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agent5")

import fetcher_football_data as fbd
import fetcher_sport_api     as faf
import fetcher_world_cup     as fwc
from analyzer         import analyze_matches
from pronostic_engine import run_engine
from report_generator import generate_html_report, generate_subject
from email_sender     import send_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 5 Football Pronostics")
    parser.add_argument("--type",    default="quotidien",
                        choices=["quotidien", "avant", "hebdo", "coupe-du-monde"],
                        help="Type de rapport")
    parser.add_argument("--no-ai",  action="store_true",
                        help="Désactiver les analyses IA")
    parser.add_argument("--fast",   action="store_true",
                        help="(CDM) Sauter l'enrichissement forme/H2H — collecte plus rapide")
    parser.add_argument("--include-played", action="store_true",
                        help="(CDM) Inclure aussi les matchs déjà joués ou en cours (par défaut: uniquement les matchs à venir)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Ne pas envoyer l'email, sauvegarder en HTML")
    args = parser.parse_args()

    use_ai = not args.no_ai
    report_type = args.type
    is_world_cup = report_type == "coupe-du-monde"
    
    # Par défaut, on ignore les matchs joués sauf si --include-played est spécifié
    only_upcoming = not args.include_played

    if is_world_cup:
        logger.info("Mode Coupe du Monde 2026 — phase de groupes (matchs à venir uniquement: %s)", 
                    "OUI" if only_upcoming else "NON (inclut les joués)")
    else:
        logger.info("=== Agent 5 Football Pronostics démarré (type=%s, ai=%s) ===",
                    report_type, use_ai)

    if use_ai:
        from ai_providers import available_providers
        providers = available_providers()
        logger.info("Providers IA configurés : %s",
                    ", ".join(providers) if providers else "AUCUN (clés manquantes)")

    # ------------------------------------------------------------------ #
    #  1. Collecte des données                                            #
    # ------------------------------------------------------------------ #
    if is_world_cup:
        logger.info("Mode Coupe du Monde 2026 — phase de groupes complète …")
        all_raw = []
        try:
            all_raw = fwc.fetch_all_enriched(
                with_form=not args.fast,
                only_upcoming=only_upcoming,
            )
        except Exception as e:
            logger.error("Coupe du Monde fetch failed: %s", e)
        
        # Filtre de sécurité supplémentaire : ne garder que les matchs non joués
        if only_upcoming:
            all_raw = fwc.filter_upcoming_matches(all_raw)
            logger.info("Après filtrage final: %d matchs à venir", len(all_raw))
    else:
        logger.info("Fetching data from football-data.org …")
        raw_fbd = []
        try:
            raw_fbd = fbd.fetch_all_enriched()
        except Exception as e:
            logger.error("football-data fetch failed: %s", e)

        logger.info("Fetching data from SportAPI …")
        raw_faf = []
        try:
            raw_faf = faf.fetch_all_enriched()
        except Exception as e:
            logger.error("sport-api fetch failed: %s", e)

        all_raw = raw_fbd + raw_faf

    logger.info("Total raw matches collected: %d", len(all_raw))

    if not all_raw:
        logger.warning("No upcoming matches found — exiting.")
        return

    # ------------------------------------------------------------------ #
    #  2. Normalisation                                                   #
    # ------------------------------------------------------------------ #
    logger.info("Normalizing match data …")
    matches = analyze_matches(all_raw)
    logger.info("Normalized: %d matches", len(matches))

    # ------------------------------------------------------------------ #
    #  3. Calcul des pronostics                                           #
    # ------------------------------------------------------------------ #
    logger.info("Computing pronostics (AI=%s) …", use_ai)
    results = run_engine(matches, use_ai=use_ai, force_ai=is_world_cup)
    logger.info("Pronostics computed: %d", len(results))

    # ------------------------------------------------------------------ #
    #  4. Génération du rapport HTML                                      #
    # ------------------------------------------------------------------ #
    logger.info("Generating HTML report …")
    html    = generate_html_report(results, report_type=report_type)
    subject = generate_subject(len(results), report_type=report_type)

    # ------------------------------------------------------------------ #
    #  5. Envoi ou sauvegarde                                             #
    # ------------------------------------------------------------------ #
    if args.dry_run:
        output_file = f"rapport_{report_type}.html"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("Dry-run: rapport sauvegardé dans %s", output_file)
    else:
        logger.info("Sending email: %s", subject)
        attachment = html if is_world_cup else None
        ok = send_report(subject, html,
                         attachment_html=attachment,
                         attachment_name="pronostics_cdm2026_phase_de_groupes.html")
        if ok:
            logger.info("Email envoyé avec succès.")
        else:
            logger.error("Échec de l'envoi de l'email.")
            sys.exit(1)

    logger.info("=== Agent 5 terminé ===")


if __name__ == "__main__":
    main()

"""
Agent 5 Football Pronostics — point d'entrée principal.

Usage:
    python main.py                  # rapport quotidien (24h à venir)
    python main.py --type avant     # matchs du jour (9h)
    python main.py --type hebdo     # rapport hebdomadaire (lundi)
    python main.py --no-ai          # désactive l'analyse Claude
    python main.py --dry-run        # génère le rapport sans envoyer l'email
"""

import argparse
import logging
import os
import sys
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agent5")

import fetcher_football_data as fbd
import fetcher_sport_api     as faf
from analyzer         import analyze_matches
from pronostic_engine import run_engine
from report_generator import generate_html_report, generate_subject
from email_sender     import send_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 5 Football Pronostics")
    parser.add_argument("--type",    default="quotidien",
                        choices=["quotidien", "avant", "hebdo"],
                        help="Type de rapport")
    parser.add_argument("--no-ai",  action="store_true",
                        help="Désactiver l'analyse Claude AI")
    parser.add_argument("--dry-run", action="store_true",
                        help="Ne pas envoyer l'email, sauvegarder en HTML")
    args = parser.parse_args()

    use_ai = not args.no_ai
    report_type = args.type

    logger.info("=== Agent 5 Football Pronostics démarré (type=%s, ai=%s) ===",
                report_type, use_ai)

    # ------------------------------------------------------------------ #
    #  1. Collecte des données                                            #
    # ------------------------------------------------------------------ #
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
        logger.warning("No matches found — exiting.")
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
    results = run_engine(matches, use_ai=use_ai)
    logger.info("Pronostics computed: %d", len(results))

    # ------------------------------------------------------------------ #
    #  4. Génération du rapport HTML                                      #
    # ------------------------------------------------------------------ #
    logger.info("Generating HTML report …")
    html   = generate_html_report(results, report_type=report_type)
    subject = generate_subject(len(results))

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
        ok = send_report(subject, html)
        if ok:
            logger.info("Email envoyé avec succès.")
        else:
            logger.error("Échec de l'envoi de l'email.")
            sys.exit(1)

    logger.info("=== Agent 5 terminé ===")


if __name__ == "__main__":
    main()

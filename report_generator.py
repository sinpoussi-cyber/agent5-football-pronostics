"""
Generates the HTML email report from pronostic results.
"""

from __future__ import annotations
import logging
from collections import defaultdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Formatting helpers                                                          #
# --------------------------------------------------------------------------- #

def _pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _badge(stars: str) -> str:
    colors = {"⭐⭐⭐": "#2ecc71", "⭐⭐": "#f39c12", "⭐": "#e74c3c"}
    bg = colors.get(stars, "#95a5a6")
    return f'<span style="background:{bg};color:#fff;padding:2px 8px;border-radius:12px;font-size:12px;">{stars}</span>'


def _prob_bar(prob: float) -> str:
    pct = int(prob * 100)
    color = "#2ecc71" if pct >= 70 else "#f39c12" if pct >= 55 else "#e74c3c"
    return (
        f'<div style="background:#ecf0f1;border-radius:4px;height:8px;width:100%;margin-top:3px;">'
        f'<div style="width:{pct}%;background:{color};height:8px;border-radius:4px;"></div>'
        f'</div>'
    )


def _match_time(utc_date: str) -> str:
    try:
        dt = datetime.fromisoformat(utc_date.replace("Z", "+00:00"))
        return dt.strftime("%H:%M UTC")
    except Exception:
        return utc_date


# --------------------------------------------------------------------------- #
#  Ensemble model section                                                      #
# --------------------------------------------------------------------------- #

def _render_ensemble_section(p: dict) -> str:
    ensemble = p.get("ensemble")
    if not ensemble:
        return ""

    m_poisson = p.get("model_poisson", {})
    m_dixon   = p.get("model_dixon", {})
    m_elo     = p.get("model_elo", {})
    m_xg      = p.get("model_xg", {})

    def row(label: str, m: dict, weight: str) -> str:
        return (
            f'<tr style="border-bottom:1px solid #ecf0f1;">'
            f'<td style="padding:4px 8px;font-weight:bold;">{label}</td>'
            f'<td style="padding:4px 8px;text-align:center;">{weight}</td>'
            f'<td style="padding:4px 8px;text-align:center;">{m.get("p1", 0):.1f}%</td>'
            f'<td style="padding:4px 8px;text-align:center;">{m.get("px", 0):.1f}%</td>'
            f'<td style="padding:4px 8px;text-align:center;">{m.get("p2", 0):.1f}%</td>'
            f'</tr>'
        )

    p1_w  = ensemble.get("p1", {}).get("weighted", 0)
    px_w  = ensemble.get("px", {}).get("weighted", 0)
    p2_w  = ensemble.get("p2", {}).get("weighted", 0)

    # Confidence: std deviation across models for the dominant outcome
    import statistics as _stats
    vals_p1 = [m_poisson.get("p1",0), m_dixon.get("p1",0), m_elo.get("p1",0), m_xg.get("p1",0)]
    std_dev = _stats.stdev(vals_p1) if len(vals_p1) > 1 else 0
    if std_dev < 4:
        conf_label, conf_color = "Élevée", "#2ecc71"
    elif std_dev < 9:
        conf_label, conf_color = "Moyenne", "#f39c12"
    else:
        conf_label, conf_color = "Faible", "#e74c3c"

    return f"""
  <div style="margin:0 16px 12px;border:1px solid #d5e8d4;border-radius:6px;overflow:hidden;">
    <div style="background:#1a6b3c;color:#fff;padding:8px 12px;font-weight:bold;font-size:13px;">
      🔬 Méthode — Comparatif 4 modèles
    </div>
    <table width="100%" style="font-size:12px;border-collapse:collapse;">
      <thead style="background:#f8f9fa;">
        <tr>
          <th style="padding:4px 8px;text-align:left;">Modèle</th>
          <th style="padding:4px 8px;text-align:center;">Poids</th>
          <th style="padding:4px 8px;text-align:center;">P1 (%)</th>
          <th style="padding:4px 8px;text-align:center;">PX (%)</th>
          <th style="padding:4px 8px;text-align:center;">P2 (%)</th>
        </tr>
      </thead>
      <tbody>
        {row("Poisson", m_poisson, "25%")}
        {row("Dixon-Coles", m_dixon, "35%")}
        {row("Elo", m_elo, "15%")}
        {row("xG ajusté", m_xg, "25%")}
        <tr style="background:#eafaf1;font-weight:bold;">
          <td style="padding:5px 8px;">FUSION PONDÉRÉE</td>
          <td style="padding:5px 8px;text-align:center;">100%</td>
          <td style="padding:5px 8px;text-align:center;">{p1_w:.1f}%</td>
          <td style="padding:5px 8px;text-align:center;">{px_w:.1f}%</td>
          <td style="padding:5px 8px;text-align:center;">{p2_w:.1f}%</td>
        </tr>
      </tbody>
    </table>
    <div style="padding:6px 12px;font-size:12px;color:#555;">
      Convergence modèles : <span style="color:{conf_color};font-weight:bold;">{conf_label}</span>
      <span style="color:#999;margin-left:8px;">(σ P1 = {std_dev:.1f}pp)</span>
    </div>
  </div>"""


# --------------------------------------------------------------------------- #
#  Match card HTML                                                             #
# --------------------------------------------------------------------------- #

def _section_header(title: str) -> str:
    return (
        f'<div style="grid-column:1/-1;background:#1a6b3c;color:#fff;'
        f'padding:6px 12px;border-radius:4px;font-size:12px;font-weight:bold;'
        f'letter-spacing:.5px;text-transform:uppercase;">{title}</div>'
    )


def _card(title: str, body: str) -> str:
    return (
        f'<div style="border:1px solid #eee;border-radius:6px;padding:10px;">'
        f'<div style="font-weight:bold;margin-bottom:6px;font-size:13px;">{title}</div>'
        f'{body}</div>'
    )


def _render_match_card(entry: dict) -> str:
    match = entry["match"]
    p     = entry["pronostics"]

    home = match["home_name"]
    away = match["away_name"]
    time = _match_time(match.get("utc_date", ""))

    home_form_str = " ".join(
        f'<span style="color:{"#2ecc71" if r=="W" else "#e74c3c" if r=="L" else "#f39c12"}">{r}</span>'
        for r in match.get("home_form", [])
    ) or "—"
    away_form_str = " ".join(
        f'<span style="color:{"#2ecc71" if r=="W" else "#e74c3c" if r=="L" else "#f39c12"}">{r}</span>'
        for r in match.get("away_form", [])
    ) or "—"

    # ── market data ──────────────────────────────────────────────────────────
    p1x2   = p["p1x2"]
    dc     = p["double_chance"]
    dnb    = p.get("draw_no_bet", {})
    ou     = p["over_under"]
    btts   = p["btts"]
    bhg    = p.get("both_halves_goal", {})
    eg     = p.get("exact_goals", {})
    exact  = p["exact_scores"]
    ht     = p["halftime"]
    htft   = p.get("halftime_fulltime", {})
    eh     = p.get("handicap_eu", {})
    ah     = p["handicap_asian"]
    oua    = p.get("over_under_asian", {})
    fgt    = p.get("first_goal_time", {})
    corn   = p["corners"]
    cbt    = p.get("corners_by_team", {})
    cards  = p["cards"]
    kbt    = p.get("cards_by_team", {})
    btr    = p.get("btts_and_result", {})
    wtn    = p.get("win_to_nil", {})

    rec_1x2   = p["rec_1x2"]
    rec_dc    = p["rec_dc"]
    rec_ou    = p["rec_ou"]
    rec_btts  = p["rec_btts"]
    rec_score = p["rec_score"]

    narrative = p.get("ai_narrative", "")

    exact_rows = "".join(
        f'<tr><td style="padding:2px 6px;">{e["score"]}</td>'
        f'<td style="padding:2px 6px;color:#555;">{_pct(e["prob"])}</td></tr>'
        for e in exact
    )

    # ── Section 1 : Paris principaux ─────────────────────────────────────────
    s1 = f"""
    {_section_header("1 · Paris principaux")}

    {_card("1X2", f'''
      <table width="100%" style="font-size:12px;"><tr>
        <td>1 ({_pct(p1x2["1"])})</td>
        <td>X ({_pct(p1x2["X"])})</td>
        <td>2 ({_pct(p1x2["2"])})</td>
      </tr></table>
      <div style="margin-top:6px;">Rec : <b>{rec_1x2["label"]}</b> {_badge(rec_1x2["stars"])}</div>
      {_prob_bar(rec_1x2["prob"])}''')}

    {_card("Double Chance", f'''
      <table width="100%" style="font-size:12px;"><tr>
        <td>1X ({_pct(dc["1X"])})</td>
        <td>12 ({_pct(dc["12"])})</td>
        <td>X2 ({_pct(dc["X2"])})</td>
      </tr></table>
      <div style="margin-top:6px;">Rec : <b>{rec_dc["label"]}</b> {_badge(rec_dc["stars"])}</div>
      {_prob_bar(rec_dc["prob"])}''')}

    {_card("Draw No Bet (remboursé si nul)", f'''
      <div style="font-size:12px;">
        Dom. ({home}) : <b>{_pct(dnb.get("DNB_home", 0))}</b>
        &nbsp;|&nbsp; Ext. ({away}) : <b>{_pct(dnb.get("DNB_away", 0))}</b>
      </div>''')}
    """

    # ── Section 2 : Buts ─────────────────────────────────────────────────────
    s2 = f"""
    {_section_header("2 · Buts")}

    {_card("Over / Under Buts", f'''
      <table width="100%" style="font-size:12px;"><tr>
        <td>O0.5: {_pct(ou.get("O0.5",0))}</td>
        <td>O1.5: {_pct(ou.get("O1.5",0))}</td>
        <td>O2.5: {_pct(ou.get("O2.5",0))}</td>
        <td>O3.5: {_pct(ou.get("O3.5",0))}</td>
      </tr></table>
      <div style="margin-top:6px;">Rec : <b>{rec_ou["label"]}</b> {_badge(rec_ou["stars"])}</div>
      {_prob_bar(rec_ou["prob"])}''')}

    {_card("BTTS (Les deux marquent)", f'''
      <div style="font-size:12px;">
        Oui : <b>{_pct(btts["BTTS_yes"])}</b> &nbsp;|&nbsp; Non : {_pct(btts["BTTS_no"])}
      </div>
      <div style="margin-top:6px;">Rec : <b>{rec_btts["label"]}</b> {_badge(rec_btts["stars"])}</div>
      {_prob_bar(rec_btts["prob"])}''')}

    {_card("But dans les 2 mi-temps", f'''
      <div style="font-size:12px;">
        MT1 : <b>{_pct(bhg.get("BHG_ht1",0))}</b>
        &nbsp;|&nbsp; MT2 : <b>{_pct(bhg.get("BHG_ht2",0))}</b>
        &nbsp;|&nbsp; Les deux : <b>{_pct(bhg.get("BHG_both",0))}</b>
      </div>''')}

    {_card("Nombre exact de buts", f'''
      <table width="100%" style="font-size:12px;"><tr>
        <td>0 but: {_pct(eg.get("Goals_0",0))}</td>
        <td>1 but: {_pct(eg.get("Goals_1",0))}</td>
        <td>2 buts: {_pct(eg.get("Goals_2",0))}</td>
        <td>3 buts: {_pct(eg.get("Goals_3",0))}</td>
        <td>4+: {_pct(eg.get("Goals_4plus",0))}</td>
      </tr></table>''')}
    """

    # ── Section 3 : Scores ───────────────────────────────────────────────────
    # halftime_fulltime top combos sorted by prob
    htft_sorted = sorted(htft.items(), key=lambda x: x[1], reverse=True)[:4] if htft else []
    htft_rows = "".join(
        f'<tr><td style="padding:2px 6px;">{k.replace("HTFT_","")}</td>'
        f'<td style="padding:2px 6px;color:#555;">{_pct(v)}</td></tr>'
        for k, v in htft_sorted
    )

    s3 = f"""
    {_section_header("3 · Scores")}

    {_card("Top 5 Scores Exacts", f'''
      <table style="font-size:12px;">{exact_rows}</table>
      <div style="margin-top:4px;">Rec : <b>{rec_score["label"]}</b></div>''')}

    {_card("Mi-temps / Fin de match", f'''
      <div style="font-size:12px;margin-bottom:6px;">
        <b>Mi-temps :</b>
        1={_pct(ht.get("HT_1",0))} | X={_pct(ht.get("HT_X",0))} | 2={_pct(ht.get("HT_2",0))}
        &nbsp;|&nbsp; O0.5={_pct(ht.get("HT_O0.5",0))} O1.5={_pct(ht.get("HT_O1.5",0))}
      </div>
      <div style="font-size:12px;"><b>MT/FM (top 4) :</b></div>
      <table style="font-size:12px;">{htft_rows}</table>''')}
    """

    # ── Section 4 : Handicaps ────────────────────────────────────────────────
    s4 = f"""
    {_section_header("4 · Handicaps")}

    {_card("Handicap Européen", f'''
      <table width="100%" style="font-size:12px;"><tr>
        <td>EH-1 dom.: {_pct(eh.get("EH-1_home",0))}</td>
        <td>EH0 nul: {_pct(eh.get("EH0_draw",0))}</td>
        <td>EH+1 dom.: {_pct(eh.get("EH+1_home",0))}</td>
      </tr></table>''')}

    {_card("Handicap Asiatique (domicile)", f'''
      <table width="100%" style="font-size:12px;"><tr>
        <td>-0.5: {_pct(ah.get("AH-0.5",0))}</td>
        <td>-1.0: {_pct(ah.get("AH-1.0",0))}</td>
        <td>-1.5: {_pct(ah.get("AH-1.5",0))}</td>
      </tr></table>''')}

    {_card("O/U Asiatique Buts", f'''
      <table width="100%" style="font-size:12px;"><tr>
        <td>O2.25: {_pct(oua.get("AOU_O2.25",0))}</td>
        <td>O2.75: {_pct(oua.get("AOU_O2.75",0))}</td>
        <td>O3.25: {_pct(oua.get("AOU_O3.25",0))}</td>
      </tr></table>''')}
    """

    # ── Section 5 : Timing ───────────────────────────────────────────────────
    s5 = f"""
    {_section_header("5 · Timing — Minute du premier but")}

    {_card("Probabilité du premier but", f'''
      <div style="font-size:12px;">
        Avant 15 min : <b>{_pct(fgt.get("FG_before_15",0))}</b>
        &nbsp;|&nbsp; Avant 30 min : <b>{_pct(fgt.get("FG_before_30",0))}</b>
        &nbsp;|&nbsp; Avant 45 min : <b>{_pct(fgt.get("FG_before_45",0))}</b>
        &nbsp;|&nbsp; Après 75 min : <b>{_pct(fgt.get("FG_after_75",0))}</b>
      </div>
      {_prob_bar(fgt.get("FG_before_30",0))}''')}
    """

    # ── Section 6 : Statistiques ─────────────────────────────────────────────
    s6 = f"""
    {_section_header("6 · Statistiques match")}

    {_card("Corners Total", f'''
      <table width="100%" style="font-size:12px;"><tr>
        <td>O8.5: {_pct(corn.get("Corners_O8.5",0))}</td>
        <td>O9.5: {_pct(corn.get("Corners_O9.5",0))}</td>
        <td>O10.5: {_pct(corn.get("Corners_O10.5",0))}</td>
      </tr></table>''')}

    {_card("Corners par équipe", f'''
      <div style="font-size:12px;margin-bottom:4px;">
        <b>{home} :</b>
        O4.5={_pct(cbt.get("Corn_H_O4.5",0))}
        O5.5={_pct(cbt.get("Corn_H_O5.5",0))}
        O6.5={_pct(cbt.get("Corn_H_O6.5",0))}
      </div>
      <div style="font-size:12px;margin-bottom:4px;">
        <b>{away} :</b>
        O3.5={_pct(cbt.get("Corn_A_O3.5",0))}
        O4.5={_pct(cbt.get("Corn_A_O4.5",0))}
        O5.5={_pct(cbt.get("Corn_A_O5.5",0))}
      </div>
      <div style="font-size:12px;">
        HC dom. -1.5 : <b>{_pct(cbt.get("Corn_HC_home_-1.5",0))}</b>
      </div>''')}

    {_card("Cartons Total", f'''
      <div style="font-size:12px;">
        O3.5: <b>{_pct(cards.get("Cards_O3.5",0))}</b>
        &nbsp;|&nbsp; O4.5: <b>{_pct(cards.get("Cards_O4.5",0))}</b>
      </div>''')}

    {_card("Cartons par équipe", f'''
      <div style="font-size:12px;margin-bottom:4px;">
        <b>{home} :</b>
        O1.5={_pct(kbt.get("Card_H_O1.5",0))}
        O2.5={_pct(kbt.get("Card_H_O2.5",0))}
      </div>
      <div style="font-size:12px;margin-bottom:4px;">
        <b>{away} :</b>
        O1.5={_pct(kbt.get("Card_A_O1.5",0))}
        O2.5={_pct(kbt.get("Card_A_O2.5",0))}
      </div>
      <div style="font-size:12px;">
        Carton rouge O0.5 : <b>{_pct(kbt.get("Card_Red_O0.5",0))}</b>
      </div>''')}
    """

    # ── Section 7 : Combinés ─────────────────────────────────────────────────
    s7 = f"""
    {_section_header("7 · Paris combinés")}

    {_card("BTTS + Résultat", f'''
      <table width="100%" style="font-size:12px;">
        <tr>
          <td>BTTS Oui + Dom.: <b>{_pct(btr.get("BTTS_yes_home",0))}</b></td>
          <td>BTTS Oui + Nul: <b>{_pct(btr.get("BTTS_yes_draw",0))}</b></td>
          <td>BTTS Oui + Ext.: <b>{_pct(btr.get("BTTS_yes_away",0))}</b></td>
        </tr>
        <tr>
          <td>BTTS Non + Dom.: <b>{_pct(btr.get("BTTS_no_home",0))}</b></td>
          <td></td>
          <td>BTTS Non + Ext.: <b>{_pct(btr.get("BTTS_no_away",0))}</b></td>
        </tr>
      </table>''')}

    {_card("Victoire à zéro (clean sheet + victoire)", f'''
      <div style="font-size:12px;">
        {home} gagne sans encaisser : <b>{_pct(wtn.get("WTN_home",0))}</b>
        &nbsp;|&nbsp;
        {away} gagne sans encaisser : <b>{_pct(wtn.get("WTN_away",0))}</b>
      </div>''')}
    """

    return f"""
<div style="border:1px solid #dfe6e9;border-radius:8px;margin:16px 0;overflow:hidden;font-family:Arial,sans-serif;">

  <!-- Match header -->
  <div style="background:#1a6b3c;color:#fff;padding:12px 16px;display:flex;justify-content:space-between;align-items:center;">
    <div style="font-size:16px;font-weight:bold;">{home} <span style="opacity:.7">vs</span> {away}</div>
    <div style="font-size:13px;opacity:.85;">{time}</div>
  </div>

  <!-- Form & ranks -->
  <div style="background:#f8f9fa;padding:8px 16px;font-size:13px;border-bottom:1px solid #dfe6e9;">
    <span style="font-weight:bold;">{home}</span>: {home_form_str} &nbsp;|&nbsp;
    <span style="font-weight:bold;">{away}</span>: {away_form_str}
    &nbsp;|&nbsp; Rang: {match.get("home_rank","?")} / {match.get("away_rank","?")}
  </div>

  <!-- Pronostics grid -->
  <div style="padding:16px;display:grid;grid-template-columns:1fr 1fr;gap:12px;">
    {s1}
    {s2}
    {s3}
    {s4}
    {s5}
    {s6}
    {s7}
  </div>

  <!-- Ensemble model comparison -->
  {_render_ensemble_section(p)}

  <!-- AI narrative -->
  {"" if not narrative else f'''
  <div style="margin:0 16px 16px;padding:12px;background:#eafaf1;border-left:4px solid #1a6b3c;border-radius:4px;font-size:13px;color:#2c3e50;">
    <div style="font-weight:bold;margin-bottom:4px;">🤖 Analyse IA</div>
    {narrative.replace(chr(10), "<br>")}
  </div>'''}
</div>
"""


# --------------------------------------------------------------------------- #
#  Full report                                                                 #
# --------------------------------------------------------------------------- #

def generate_html_report(results: list[dict], report_type: str = "quotidien") -> str:
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%d/%m/%Y")
    nb_matches = len(results)

    # Group by competition
    by_competition: dict[str, list[dict]] = defaultdict(list)
    for entry in results:
        comp = entry["match"].get("competition", "Autre")
        by_competition[comp].append(entry)

    competition_sections = ""
    for comp, entries in sorted(by_competition.items()):
        cards = "".join(_render_match_card(e) for e in entries)
        competition_sections += f"""
<div style="margin-bottom:32px;">
  <h2 style="color:#1a6b3c;border-bottom:2px solid #1a6b3c;padding-bottom:6px;">{comp}</h2>
  {cards}
</div>"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>Pronostics Football {date_str}</title>
</head>
<body style="margin:0;padding:0;background:#f4f6f8;font-family:Arial,sans-serif;">

  <!-- Header -->
  <div style="background:linear-gradient(135deg,#1a6b3c 0%,#27ae60 100%);color:#fff;padding:32px;text-align:center;">
    <div style="font-size:14px;letter-spacing:2px;text-transform:uppercase;opacity:.8;">Agent 5 Football Pronostics</div>
    <div style="font-size:32px;font-weight:bold;margin:8px 0;">🎯 Pronostics Football</div>
    <div style="font-size:18px;">{date_str} — {nb_matches} match{"s" if nb_matches > 1 else ""} analysé{"s" if nb_matches > 1 else ""}</div>
    <div style="font-size:13px;margin-top:8px;opacity:.75;">Rapport {report_type}</div>
  </div>

  <!-- Body -->
  <div style="max-width:900px;margin:0 auto;padding:24px;">
    {competition_sections if competition_sections else
      '<p style="text-align:center;color:#7f8c8d;">Aucun match à analyser pour cette période.</p>'}
  </div>

  <!-- Footer -->
  <div style="background:#2c3e50;color:#bdc3c7;text-align:center;padding:24px;font-size:12px;">
    <div style="margin-bottom:8px;font-size:14px;color:#ecf0f1;">⚽ Agent 5 Football Pronostics</div>
    <div style="max-width:600px;margin:0 auto;line-height:1.6;">
      <strong>⚠️ Avertissement — Jeu Responsable</strong><br>
      Ces pronostics sont fournis à titre indicatif uniquement et sont basés sur des modèles statistiques.
      Ils ne constituent pas une garantie de résultats. Les paris sportifs comportent des risques.
      Jouez de manière responsable et ne misez jamais plus que ce que vous pouvez vous permettre de perdre.
      En cas de problème avec le jeu, contactez <a href="https://www.joueurs-info-service.fr" style="color:#3498db;">Joueurs Info Service</a> au 09 74 75 13 13.
    </div>
    <div style="margin-top:12px;color:#7f8c8d;">Généré le {now.strftime("%d/%m/%Y à %H:%M")} UTC par Claude Sonnet</div>
  </div>

</body>
</html>"""
    return html


def generate_subject(nb_matches: int) -> str:
    date_str = datetime.now(timezone.utc).strftime("%d/%m/%Y")
    return f"🎯 Pronostics Football — {date_str} — {nb_matches} match{'s' if nb_matches > 1 else ''} analysé{'s' if nb_matches > 1 else ''}"

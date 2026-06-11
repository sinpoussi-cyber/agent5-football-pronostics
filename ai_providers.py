"""
Multi-AI providers — Claude (Anthropic), Gemini (Google), DeepSeek.

Chaque provider reçoit le même prompt d'analyse et retourne un texte.
Les clés API sont lues dans l'environnement :
    ANTHROPIC_API_KEY   (existante)
    GEMINI_API_KEY      (nouvelle)
    DEEPSEEK_API_KEY    (nouvelle)

Un provider sans clé est simplement ignoré (log INFO, pas d'erreur).
"""

from __future__ import annotations
import os
import time
import logging
import requests

logger = logging.getLogger(__name__)

# Modèles utilisés (modifiable via env si besoin)
CLAUDE_MODEL   = os.environ.get("CLAUDE_MODEL",   "claude-sonnet-4-20250514")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL",   "gemini-2.5-flash")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

MAX_TOKENS = 600
_TIMEOUT   = 60


# --------------------------------------------------------------------------- #
#  Claude (Anthropic)                                                          #
# --------------------------------------------------------------------------- #

def claude_analyze(prompt: str) -> str | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        if "credit balance too low" in str(e).lower():
            logger.warning("Claude: crédits insuffisants — %s", e)
            return "Analyse Claude indisponible (crédits insuffisants)"
        logger.error("Claude API error: %s", e)
        return f"Analyse Claude indisponible : {e}"


# --------------------------------------------------------------------------- #
#  Gemini (Google) — REST, aucune dépendance supplémentaire                    #
# --------------------------------------------------------------------------- #

def gemini_analyze(prompt: str) -> str | None:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent")
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": MAX_TOKENS, "temperature": 0.4},
    }
    for attempt in range(1, 4):
        try:
            resp = requests.post(
                url,
                params={"key": api_key},
                json=payload,
                timeout=_TIMEOUT,
            )
            if resp.status_code == 429:
                logger.warning("Gemini 429 (tentative %d/3) — pause 10s", attempt)
                time.sleep(10)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            logger.error("Gemini: réponse inattendue — %s", e)
            return "Analyse Gemini indisponible (réponse inattendue)"
        except Exception as e:
            logger.error("Gemini API error: %s", e)
            return f"Analyse Gemini indisponible : {e}"
    return "Analyse Gemini indisponible (rate limit)"


# --------------------------------------------------------------------------- #
#  DeepSeek — API compatible OpenAI                                            #
# --------------------------------------------------------------------------- #

def deepseek_analyze(prompt: str) -> str | None:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return None
    url = "https://api.deepseek.com/chat/completions"
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.4,
    }
    for attempt in range(1, 4):
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=_TIMEOUT,
            )
            if resp.status_code == 429:
                logger.warning("DeepSeek 429 (tentative %d/3) — pause 10s", attempt)
                time.sleep(10)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error("DeepSeek: réponse inattendue — %s", e)
            return "Analyse DeepSeek indisponible (réponse inattendue)"
        except Exception as e:
            logger.error("DeepSeek API error: %s", e)
            return f"Analyse DeepSeek indisponible : {e}"
    return "Analyse DeepSeek indisponible (rate limit)"


# --------------------------------------------------------------------------- #
#  Point d'entrée multi-IA                                                     #
# --------------------------------------------------------------------------- #

_PROVIDERS = [
    ("Claude",   claude_analyze),
    ("Gemini",   gemini_analyze),
    ("DeepSeek", deepseek_analyze),
]


def available_providers() -> list[str]:
    """Liste des providers dont la clé API est configurée."""
    keys = {
        "Claude":   "ANTHROPIC_API_KEY",
        "Gemini":   "GEMINI_API_KEY",
        "DeepSeek": "DEEPSEEK_API_KEY",
    }
    return [name for name, env in keys.items() if os.environ.get(env)]


def get_multi_ai_narratives(prompt: str) -> dict[str, str]:
    """
    Envoie le même prompt à tous les providers configurés.
    Retourne {"Claude": "...", "Gemini": "...", "DeepSeek": "..."} —
    seuls les providers avec clé API présents dans le dict.
    """
    results: dict[str, str] = {}
    for name, fn in _PROVIDERS:
        text = fn(prompt)
        if text is None:
            logger.info("%s ignoré (clé API absente)", name)
            continue
        results[name] = text
    return results

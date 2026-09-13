"""
Client minimal pour l'API MLB Stats officielle (statsapi.mlb.com).

Cette API est publique et gratuite, aucune clé n'est nécessaire. On ne fait
ici que ce dont le modèle a besoin : calendrier + scores + lanceurs
probables. Documentation communautaire (non officielle) :
https://github.com/toddrob99/MLB-StatsAPI/wiki

NB pédagogique : ce module n'est volontairement PAS appelé depuis ce
sandbox de développement (pas d'accès réseau sortant vers statsapi.mlb.com
ici) — il est écrit pour tourner tel quel une fois le repo cloné/déployé,
où l'accès internet est normal. Voir tests/test_elo.py pour ce qui, lui,
est testé sans réseau.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import requests

from src.config import MLB_SPORT_ID

BASE_URL = "https://statsapi.mlb.com/api/v1"
TIMEOUT = 15  # secondes


def _get(path: str, params: Optional[dict] = None) -> dict:
    resp = requests.get(f"{BASE_URL}{path}", params=params or {}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_teams(season: Optional[int] = None) -> list[dict]:
    """Retourne la liste des 30 équipes MLB (id, nom, abréviation, venue)."""
    params = {"sportId": MLB_SPORT_ID}
    if season:
        params["season"] = season
    data = _get("/teams", params=params)
    teams = []
    for t in data.get("teams", []):
        teams.append(
            {
                "team_id": t["id"],
                "name": t["name"],
                "abbreviation": t.get("abbreviation"),
                "venue_id": t.get("venue", {}).get("id"),
                "venue_name": t.get("venue", {}).get("name"),
                "league": t.get("league", {}).get("name"),
                "division": t.get("division", {}).get("name"),
            }
        )
    return teams


def get_schedule(
    start_date: dt.date,
    end_date: Optional[dt.date] = None,
    include_probable_pitchers: bool = True,
) -> list[dict]:
    """
    Récupère le calendrier (et les scores si les matchs sont terminés) entre
    start_date et end_date (inclus). Si end_date est omis, un seul jour est
    interrogé.

    Retourne une liste de dicts "aplatis", un par match, avec :
    game_pk, game_date, status, home/away team id+name, home/away score
    (None si pas encore joué), et l'id/nom du lanceur probable si demandé et
    disponible (souvent annoncé seulement 24-48h avant le match).
    """
    end_date = end_date or start_date
    params = {
        "sportId": MLB_SPORT_ID,
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
    }
    if include_probable_pitchers:
        params["hydrate"] = "probablePitcher,team"

    data = _get("/schedule", params=params)

    games = []
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            teams = g.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})

            def _pitcher(side: dict) -> dict:
                p = side.get("probablePitcher") or {}
                return {"id": p.get("id"), "name": p.get("fullName")}

            games.append(
                {
                    "game_pk": g.get("gamePk"),
                    "game_date": g.get("officialDate"),
                    "status": g.get("status", {}).get("detailedState"),
                    "venue": g.get("venue", {}).get("name"),
                    "home_team_id": home.get("team", {}).get("id"),
                    "home_team_name": home.get("team", {}).get("name"),
                    "home_score": home.get("score"),
                    "away_team_id": away.get("team", {}).get("id"),
                    "away_team_name": away.get("team", {}).get("name"),
                    "away_score": away.get("score"),
                    "home_probable_pitcher": _pitcher(home),
                    "away_probable_pitcher": _pitcher(away),
                }
            )
    return games


def get_completed_games_for_season(season: int) -> list[dict]:
    """
    Récupère tous les matchs terminés ("Final") d'une saison régulière, dans
    l'ordre chronologique. Utilisé pour construire/mettre à jour l'historique
    Elo. Peut représenter ~2430 matchs -> un seul appel par saison suffit
    (l'API accepte une plage de dates large).
    """
    start = dt.date(season, 3, 1)
    end = dt.date(season, 11, 15)
    all_games = get_schedule(start, end, include_probable_pitchers=False)
    finals = [g for g in all_games if g["status"] == "Final" and g["home_score"] is not None]
    finals.sort(key=lambda g: (g["game_date"], g["game_pk"]))
    return finals

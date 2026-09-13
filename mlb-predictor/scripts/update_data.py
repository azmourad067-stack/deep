"""
Script de rafraîchissement des données. Deux usages :

1. En local, une première fois, pour construire l'historique Elo initial
   (backfill sur quelques saisons passées).
2. Quotidiennement via GitHub Actions (.github/workflows/update_data.yml),
   pour ajouter les matchs de la veille aux ratings et rafraîchir les FIP
   des lanceurs.

Écrit ses résultats dans data/cache/ :
- elo_ratings.csv          (team_id, team_name, elo)
- games_history.csv        (tous les matchs "Final" vus, pour audit/backtest)
- pitcher_fip_lookup.csv   (mlbam_id, fip_to_use)
- league_avg_fip.txt       (un seul float)
- last_update.txt          (horodatage ISO)

Ce script fait des appels réseau (API MLB Stats + pybaseball/FanGraphs) : il
n'est PAS exécuté dans l'environnement de développement utilisé pour écrire
ce projet (réseau restreint), mais est conçu pour tourner tel quel une fois
le repo cloné ou déployé sur GitHub Actions / une machine avec accès internet
normal.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from src import mlb_api, pitcher_stats
from src.elo import EloRatings, build_ratings_from_history

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Nombre de saisons passées à utiliser pour construire l'historique Elo la
# toute première fois (les mises à jour suivantes ne repartent pas de zéro,
# voir logique ci-dessous).
BACKFILL_SEASONS = [dt.date.today().year - 2, dt.date.today().year - 1, dt.date.today().year]


def _team_names_map() -> dict[int, str]:
    teams = mlb_api.get_teams()
    return {t["team_id"]: t["name"] for t in teams}


def update_elo_ratings() -> EloRatings:
    games_path = DATA_DIR / "games_history.csv"

    if games_path.exists():
        # mise à jour incrémentale : on ne récupère que les matchs de la
        # saison en cours, on les fusionne avec l'historique déjà connu, et
        # on rejoue tout (le calcul est rapide, pas besoin d'optimiser).
        existing = pd.read_csv(games_path)
        current_season_games = mlb_api.get_completed_games_for_season(dt.date.today().year)
        new_games_df = pd.DataFrame(current_season_games)
        all_games_df = (
            pd.concat([existing, new_games_df])
            .drop_duplicates(subset=["game_pk"])
            .sort_values(["game_date", "game_pk"])
        )
    else:
        # premier lancement : backfill sur plusieurs saisons
        all_games = []
        for season in BACKFILL_SEASONS:
            all_games.extend(mlb_api.get_completed_games_for_season(season))
        all_games_df = pd.DataFrame(all_games).sort_values(["game_date", "game_pk"])

    all_games_df.to_csv(games_path, index=False)

    season_boundaries = [f"{y}-01-01" for y in range(BACKFILL_SEASONS[0], dt.date.today().year + 1)]
    ratings = build_ratings_from_history(
        all_games_df.to_dict("records"), season_boundaries=season_boundaries
    )

    names = _team_names_map()
    ratings.as_dataframe(names).to_csv(DATA_DIR / "elo_ratings.csv", index=False)
    return ratings


def update_pitcher_stats() -> None:
    current_year = dt.date.today().year
    crosswalk = pitcher_stats.get_id_crosswalk()
    current = pitcher_stats.get_season_pitching_stats(current_year)
    prior = pitcher_stats.get_season_pitching_stats(current_year - 1)

    lookup = pitcher_stats.build_pitcher_fip_lookup(current, prior, crosswalk)
    lookup.to_csv(DATA_DIR / "pitcher_fip_lookup.csv", index=False)

    avg_fip = pitcher_stats.league_average_fip(current)
    (DATA_DIR / "league_avg_fip.txt").write_text(str(avg_fip))


def main() -> None:
    print("Mise à jour des ratings Elo...")
    ratings = update_elo_ratings()
    print(f"  {len(ratings.ratings)} équipes notées.")

    print("Mise à jour des stats de lanceurs (FIP)...")
    update_pitcher_stats()

    (DATA_DIR / "last_update.txt").write_text(dt.datetime.utcnow().isoformat())
    print("Terminé.")


if __name__ == "__main__":
    main()

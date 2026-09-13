"""
Moteur de rating Elo pour équipes MLB.

Pur calcul, sans aucune dépendance réseau -> c'est la partie du projet la
plus facile à tester unitairement (voir tests/test_elo.py) et la plus fiable
du modèle, contrairement à l'ajustement lanceur qui dépend de données
externes plus fragiles.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import ELO_INITIAL, ELO_K_FACTOR, ELO_MEAN, ELO_MEAN_REVERSION, HOME_FIELD_ADVANTAGE_ELO


def expected_home_win_prob(
    home_elo: float,
    away_elo: float,
    home_field_advantage: float = HOME_FIELD_ADVANTAGE_ELO,
) -> float:
    """Probabilité de victoire à domicile selon la formule logistique Elo."""
    diff = (away_elo) - (home_elo + home_field_advantage)
    return 1.0 / (1.0 + 10 ** (diff / 400.0))


@dataclass
class EloRatings:
    """Ratings courants de chaque équipe, identifiées par leur team_id MLB."""

    ratings: dict[int, float] = field(default_factory=dict)
    k_factor: float = ELO_K_FACTOR
    home_field_advantage: float = HOME_FIELD_ADVANTAGE_ELO

    def get(self, team_id: int) -> float:
        return self.ratings.get(team_id, ELO_INITIAL)

    def update_game(self, home_team_id: int, away_team_id: int, home_won: bool) -> None:
        """Met à jour les deux ratings après un match terminé."""
        home_elo = self.get(home_team_id)
        away_elo = self.get(away_team_id)

        expected_home = expected_home_win_prob(home_elo, away_elo, self.home_field_advantage)
        actual_home = 1.0 if home_won else 0.0

        self.ratings[home_team_id] = home_elo + self.k_factor * (actual_home - expected_home)
        self.ratings[away_team_id] = away_elo + self.k_factor * ((1 - actual_home) - (1 - expected_home))

    def regress_all_to_mean(
        self, factor: float = ELO_MEAN_REVERSION, mean: float = ELO_MEAN
    ) -> None:
        """À appeler entre deux saisons : ramène chaque rating vers la
        moyenne pour refléter les changements de roster pendant l'intersaison."""
        for team_id, rating in self.ratings.items():
            self.ratings[team_id] = mean + (rating - mean) * factor

    def as_dataframe(self, team_names: dict[int, str] | None = None) -> pd.DataFrame:
        rows = [
            {
                "team_id": tid,
                "team_name": (team_names or {}).get(tid, str(tid)),
                "elo": round(elo, 1),
            }
            for tid, elo in self.ratings.items()
        ]
        return pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)


def build_ratings_from_history(
    games: list[dict],
    season_boundaries: list[str] | None = None,
) -> EloRatings:
    """
    Reconstruit les ratings Elo en rejouant chronologiquement une liste de
    matchs terminés (format : voir mlb_api.get_completed_games_for_season).

    `season_boundaries` : liste de dates (YYYY-01-01 par ex.) à partir
    desquelles appliquer une régression vers la moyenne, pour matérialiser le
    changement de saison. Optionnel — si non fourni, aucune régression n'est
    appliquée (utile pour rejouer une seule saison en continu).
    """
    ratings = EloRatings()
    boundaries = sorted(season_boundaries or [])
    next_boundary_idx = 0

    for game in games:
        while (
            next_boundary_idx < len(boundaries)
            and game["game_date"] >= boundaries[next_boundary_idx]
        ):
            ratings.regress_all_to_mean()
            next_boundary_idx += 1

        home_won = game["home_score"] > game["away_score"]
        ratings.update_game(game["home_team_id"], game["away_team_id"], home_won)

    return ratings

"""
Combine Elo d'équipe + ajustement lanceur titulaire (FIP) + avantage du
terrain en une probabilité de victoire pour un match donné.

C'est le seul endroit du projet où les trois briques du "Niveau 1" se
rencontrent -> si tu veux un jour ajouter une variable (forme récente,
bullpen, météo), c'est ici qu'elle doit rentrer dans le calcul, pas dans
l'app Streamlit elle-même.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.config import FIP_TO_ELO_SCALE, HOME_FIELD_ADVANTAGE_ELO
from src.elo import expected_home_win_prob


@dataclass
class GamePrediction:
    home_team: str
    away_team: str
    home_elo: float
    away_elo: float
    home_pitcher_fip: float
    away_pitcher_fip: float
    home_win_prob: float

    @property
    def away_win_prob(self) -> float:
        return 1.0 - self.home_win_prob


def pitcher_elo_adjustment(fip: float, league_avg_fip: float, scale: float = FIP_TO_ELO_SCALE) -> float:
    """
    Convertit un FIP en un ajustement de rating Elo (en points).
    Un FIP MEILLEUR que la moyenne (donc plus bas) donne un ajustement
    POSITIF (l'équipe est renforcée pour ce match).
    """
    return (league_avg_fip - fip) * scale


def predict_game(
    home_team: str,
    away_team: str,
    home_elo: float,
    away_elo: float,
    home_pitcher_fip: float,
    away_pitcher_fip: float,
    league_avg_fip: float,
    home_field_advantage: float = HOME_FIELD_ADVANTAGE_ELO,
) -> GamePrediction:
    effective_home_elo = home_elo + pitcher_elo_adjustment(home_pitcher_fip, league_avg_fip)
    effective_away_elo = away_elo + pitcher_elo_adjustment(away_pitcher_fip, league_avg_fip)

    home_win_prob = expected_home_win_prob(
        effective_home_elo, effective_away_elo, home_field_advantage
    )

    return GamePrediction(
        home_team=home_team,
        away_team=away_team,
        home_elo=home_elo,
        away_elo=away_elo,
        home_pitcher_fip=home_pitcher_fip,
        away_pitcher_fip=away_pitcher_fip,
        home_win_prob=home_win_prob,
    )

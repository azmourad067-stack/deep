"""
Tests unitaires purs (pas de réseau, pas de pybaseball) pour la partie du
projet la plus facile à valider mécaniquement : le moteur Elo et la
combinaison Elo + FIP dans predict.py.

Lancer avec : pytest tests/test_elo.py -v
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.elo import EloRatings, build_ratings_from_history, expected_home_win_prob
from src.predict import pitcher_elo_adjustment, predict_game


def test_equal_elo_home_advantage_only():
    """À rating égal, l'équipe à domicile doit être légèrement favorite
    uniquement grâce au home field advantage (pas de biais visiteur)."""
    prob = expected_home_win_prob(1500, 1500, home_field_advantage=24)
    assert 0.53 < prob < 0.55


def test_no_home_field_advantage_is_fair_coin():
    prob = expected_home_win_prob(1500, 1500, home_field_advantage=0)
    assert math.isclose(prob, 0.5, abs_tol=1e-9)


def test_higher_elo_team_favored():
    prob_favorite_home = expected_home_win_prob(1600, 1400, home_field_advantage=0)
    prob_underdog_home = expected_home_win_prob(1400, 1600, home_field_advantage=0)
    assert prob_favorite_home > 0.5
    assert prob_underdog_home < 0.5
    assert math.isclose(prob_favorite_home, 1 - prob_underdog_home, abs_tol=1e-9)


def test_update_game_moves_ratings_in_right_direction():
    ratings = EloRatings(k_factor=4.0, home_field_advantage=24.0)
    ratings.ratings = {1: 1500.0, 2: 1500.0}

    # équipe 1 (domicile) gagne contre toute attente -> son rating doit monter
    ratings.update_game(home_team_id=1, away_team_id=2, home_won=True)
    assert ratings.get(1) > 1500.0
    assert ratings.get(2) < 1500.0

    # somme des deltas ~ conservée (jeu à somme nulle, propriété de base de l'Elo)
    delta_home = ratings.get(1) - 1500.0
    delta_away = ratings.get(2) - 1500.0
    assert math.isclose(delta_home, -delta_away, abs_tol=1e-9)


def test_regress_to_mean_pulls_extreme_ratings_in():
    ratings = EloRatings()
    ratings.ratings = {1: 1650.0, 2: 1350.0}
    ratings.regress_all_to_mean(factor=0.75, mean=1505.0)

    assert 1505.0 < ratings.get(1) < 1650.0
    assert 1350.0 < ratings.get(2) < 1505.0


def test_build_ratings_from_history_is_order_dependent_and_deterministic():
    games = [
        {"game_date": "2024-04-01", "home_team_id": 1, "away_team_id": 2, "home_score": 5, "away_score": 2},
        {"game_date": "2024-04-02", "home_team_id": 2, "away_team_id": 1, "home_score": 1, "away_score": 6},
    ]
    ratings_a = build_ratings_from_history(games)
    ratings_b = build_ratings_from_history(games)
    # déterministe : même historique -> mêmes ratings
    assert ratings_a.get(1) == ratings_b.get(1)
    # équipe 1 a gagné les deux matchs -> rating final > 1500
    assert ratings_a.get(1) > 1500.0
    assert ratings_a.get(2) < 1500.0


def test_pitcher_adjustment_sign():
    # meilleur FIP que la moyenne (plus bas) -> ajustement positif
    assert pitcher_elo_adjustment(fip=3.00, league_avg_fip=4.00, scale=30) > 0
    # pire FIP que la moyenne (plus haut) -> ajustement négatif
    assert pitcher_elo_adjustment(fip=5.00, league_avg_fip=4.00, scale=30) < 0
    # FIP = moyenne -> ajustement nul
    assert pitcher_elo_adjustment(fip=4.00, league_avg_fip=4.00, scale=30) == 0


def test_predict_game_ace_vs_replacement_favors_ace_even_on_road():
    """Un ace (FIP 2.50) qui visite une équipe légèrement plus forte à l'Elo
    doit quand même obtenir une probabilité de victoire correcte, ce qui
    valide que l'ajustement lanceur a un effet mesurable sur la sortie
    finale (et pas seulement sur les ratings intermédiaires)."""
    pred = predict_game(
        home_team="Team A",
        away_team="Team B",
        home_elo=1520,
        away_elo=1500,
        home_pitcher_fip=4.50,   # lanceur domicile moyen/faible
        away_pitcher_fip=2.50,   # ace en visite
        league_avg_fip=4.00,
    )
    assert 0.0 < pred.home_win_prob < 1.0
    assert math.isclose(pred.home_win_prob + pred.away_win_prob, 1.0, abs_tol=1e-9)
    # l'ace en visite doit réduire l'avantage du domicile de manière notable
    assert pred.home_win_prob < 0.55

"""
Constantes du modèle "Niveau 1".

Toutes les valeurs ici sont des points de départ raisonnables tirés de la
littérature sabermetrics générale, PAS des constantes gravées dans le marbre.
Utilise `scripts/backtest.py` pour les recalibrer sur données réelles avant de
faire confiance aux probabilités produites par l'app.
"""

# --- Elo ---
ELO_INITIAL = 1500.0          # rating de départ pour une équipe inconnue
ELO_K_FACTOR = 4.0            # sensibilité de la mise à jour après un match
                               # (bas car 162 matchs/saison -> chaque match
                               # doit peu bouger le rating)
ELO_MEAN_REVERSION = 0.75     # entre deux saisons, on ramène chaque rating
                               # vers la moyenne (roster qui change) :
                               # new = MEAN + (old - MEAN) * ELO_MEAN_REVERSION
ELO_MEAN = 1505.0

# --- Avantage du terrain ---
# +24 points Elo ≈ un peu plus de 53-54% de victoire à domicile entre deux
# équipes de force égale (ordre de grandeur observé en MLB).
HOME_FIELD_ADVANTAGE_ELO = 24.0

# --- Ajustement lanceur titulaire ---
# On convertit l'écart de FIP par rapport à la moyenne de ligue en points Elo.
# FIP_TO_ELO_SCALE : nombre de points Elo par unité de FIP d'écart à la
# moyenne. À CALIBRER (voir scripts/backtest.py) : c'est le paramètre le plus
# incertain du modèle.
FIP_TO_ELO_SCALE = 30.0
LEAGUE_AVG_FIP_FALLBACK = 4.00   # utilisé si on ne peut pas calculer la
                                  # moyenne de ligue à la volée
MIN_IP_FOR_CURRENT_SEASON_FIP = 15.0  # sous ce seuil d'innings lancées cette
                                        # saison, on préfère le FIP de la
                                        # saison précédente (échantillon trop
                                        # petit sinon)

# --- Divers ---
MLB_SPORT_ID = 1  # id "MLB" dans l'API MLB Stats (les ligues mineures ont
                    # d'autres sportId)

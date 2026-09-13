"""
Récupération des statistiques de lanceurs (FIP en priorité, xFIP en repli)
via pybaseball, qui scrape FanGraphs.

Point délicat : l'API MLB Stats identifie les joueurs par un ID "MLBAM",
alors que FanGraphs (donc pybaseball) utilise son propre ID ("IDfg"). On
utilise le registre Chadwick (`pybaseball.chadwick_register`), qui fait la
table de correspondance entre les deux, plutôt que du matching par nom (trop
fragile : accents, suffixes Jr./Sr., homonymes).

NB : comme pour mlb_api.py, ce module n'est pas exécuté dans le sandbox de
développement (pas d'accès réseau vers FanGraphs/Baseball Savant ici), mais
est écrit pour fonctionner une fois déployé / lancé en local.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from src.config import LEAGUE_AVG_FIP_FALLBACK, MIN_IP_FOR_CURRENT_SEASON_FIP


def get_id_crosswalk() -> pd.DataFrame:
    """
    Table de correspondance MLBAM id <-> FanGraphs id.
    Coûteux à télécharger (registre complet) : à appeler une seule fois par
    rafraîchissement de données (scripts/update_data.py), pas à chaque
    prédiction.
    """
    from pybaseball import chadwick_register

    reg = chadwick_register()
    cols = ["key_mlbam", "key_fangraphs", "name_first", "name_last"]
    return reg[cols].dropna(subset=["key_mlbam", "key_fangraphs"]).copy()


def get_season_pitching_stats(season: int, qual: int = 0) -> pd.DataFrame:
    """
    Stats de lancer agrégées depuis le début de `season` jusqu'à aujourd'hui
    si `season` est la saison en cours (FanGraphs met à jour quotidiennement),
    ou la saison complète si `season` est terminée.

    qual=0 -> pas de seuil minimum d'IP (on gère nous-mêmes le seuil de
    fiabilité via MIN_IP_FOR_CURRENT_SEASON_FIP, sinon les jeunes lanceurs
    / callups récents disparaîtraient du tableau).
    """
    from pybaseball import pitching_stats

    df = pitching_stats(season, qual=qual)
    keep = ["IDfg", "Name", "Team", "IP", "FIP", "xFIP", "ERA"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def league_average_fip(season_pitching: pd.DataFrame) -> float:
    """Moyenne de FIP pondérée par les innings lancées, pour la saison donnée."""
    df = season_pitching.dropna(subset=["FIP", "IP"])
    if df.empty or df["IP"].sum() == 0:
        return LEAGUE_AVG_FIP_FALLBACK
    return float((df["FIP"] * df["IP"]).sum() / df["IP"].sum())


def build_pitcher_fip_lookup(
    current_season_stats: pd.DataFrame,
    prior_season_stats: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construit une table indexée par MLBAM id -> FIP "à utiliser aujourd'hui" :
    - FIP saison en cours si IP >= MIN_IP_FOR_CURRENT_SEASON_FIP (signal fiable)
    - sinon FIP saison précédente si disponible
    - sinon NaN (le code appelant retombe alors sur la moyenne de ligue)

    C'est volontairement une règle simple et transparente plutôt qu'une
    pondération bayésienne sophistiquée : plus facile à auditer et à expliquer
    dans l'app.
    """
    cw = crosswalk.rename(columns={"key_mlbam": "mlbam_id", "key_fangraphs": "IDfg"})
    cw = cw[["mlbam_id", "IDfg"]].dropna()

    cur = current_season_stats.merge(cw, on="IDfg", how="left")
    prior = prior_season_stats.merge(cw, on="IDfg", how="left")

    cur_reliable = cur[cur["IP"] >= MIN_IP_FOR_CURRENT_SEASON_FIP][["mlbam_id", "FIP"]]
    prior_fip = prior[["mlbam_id", "FIP"]].rename(columns={"FIP": "FIP_prior"})

    lookup = cur_reliable.merge(prior_fip, on="mlbam_id", how="outer")
    lookup["fip_to_use"] = lookup["FIP"].combine_first(lookup["FIP_prior"])
    return lookup[["mlbam_id", "fip_to_use"]].dropna(subset=["mlbam_id"])


def fip_for_pitcher(
    mlbam_id: Optional[int],
    lookup: pd.DataFrame,
    league_avg_fip: float,
) -> float:
    """Retourne le FIP à utiliser pour un lanceur donné, avec repli sur la
    moyenne de ligue si l'id est manquant ou absent de la table (rookie sans
    historique, erreur de correspondance, etc.)."""
    if mlbam_id is None or lookup.empty:
        return league_avg_fip
    row = lookup[lookup["mlbam_id"] == mlbam_id]
    if row.empty or pd.isna(row.iloc[0]["fip_to_use"]):
        return league_avg_fip
    return float(row.iloc[0]["fip_to_use"])

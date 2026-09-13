"""
MLB Predictor — Niveau 1 (Elo + avantage du terrain + lanceur titulaire)

Point d'entrée Streamlit. L'app est volontairement légère : elle ne fait pas
de calcul lourd elle-même, elle lit les fichiers pré-calculés dans
data/cache/ (rafraîchis par scripts/update_data.py / GitHub Actions) et
n'appelle l'API MLB Stats en direct que pour le calendrier du jour (léger et
rapide, mis en cache 1h).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st

from src import mlb_api
from src.predict import predict_game

DATA_DIR = Path(__file__).resolve().parent / "data" / "cache"

st.set_page_config(page_title="MLB Predictor — Niveau 1", page_icon="⚾", layout="wide")


# ---------- Chargement des données pré-calculées ----------

@st.cache_data(ttl=3600)
def load_elo_ratings() -> pd.DataFrame:
    path = DATA_DIR / "elo_ratings.csv"
    if not path.exists():
        return pd.DataFrame(columns=["team_id", "team_name", "elo"])
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_pitcher_lookup() -> pd.DataFrame:
    path = DATA_DIR / "pitcher_fip_lookup.csv"
    if not path.exists():
        return pd.DataFrame(columns=["mlbam_id", "fip_to_use"])
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_league_avg_fip() -> float:
    path = DATA_DIR / "league_avg_fip.txt"
    if not path.exists():
        return 4.00
    return float(path.read_text())


@st.cache_data(ttl=1800)
def load_schedule(target_date: dt.date) -> list[dict]:
    return mlb_api.get_schedule(target_date)


def fip_for(mlbam_id: int | None, lookup: pd.DataFrame, league_avg: float) -> float:
    if mlbam_id is None or lookup.empty:
        return league_avg
    row = lookup[lookup["mlbam_id"] == mlbam_id]
    if row.empty or pd.isna(row.iloc[0]["fip_to_use"]):
        return league_avg
    return float(row.iloc[0]["fip_to_use"])


# ---------- UI ----------

st.title("⚾ MLB Predictor")
st.caption(
    "Modèle Elo + avantage du terrain + FIP du lanceur titulaire. "
    "Outil pédagogique — pas un conseil de pari."
)

elo_df = load_elo_ratings()
pitcher_lookup = load_pitcher_lookup()
league_avg_fip = load_league_avg_fip()

if elo_df.empty:
    st.warning(
        "Aucune donnée en cache pour le moment. Lance `python scripts/update_data.py` "
        "en local (ou attends le prochain rafraîchissement automatique) avant de "
        "voir des prédictions."
    )

tab_today, tab_rankings, tab_methodo = st.tabs(
    ["📅 Matchs du jour", "📊 Classement Elo", "ℹ️ Méthodologie"]
)

with tab_today:
    selected_date = st.date_input("Date", value=dt.date.today())

    try:
        games = load_schedule(selected_date)
    except Exception as exc:  # réseau indisponible, API en maintenance, etc.
        games = []
        st.error(f"Impossible de récupérer le calendrier MLB pour cette date : {exc}")

    if not games:
        st.info("Aucun match trouvé pour cette date (hors-saison, jour off, ou pas encore de calendrier disponible).")

    elo_map = dict(zip(elo_df["team_id"], elo_df["elo"])) if not elo_df.empty else {}

    for g in games:
        home_id, away_id = g["home_team_id"], g["away_team_id"]
        home_elo = elo_map.get(home_id, 1500.0)
        away_elo = elo_map.get(away_id, 1500.0)

        home_pitcher = g["home_probable_pitcher"]
        away_pitcher = g["away_probable_pitcher"]
        home_fip = fip_for(home_pitcher["id"], pitcher_lookup, league_avg_fip)
        away_fip = fip_for(away_pitcher["id"], pitcher_lookup, league_avg_fip)

        pred = predict_game(
            home_team=g["home_team_name"],
            away_team=g["away_team_name"],
            home_elo=home_elo,
            away_elo=away_elo,
            home_pitcher_fip=home_fip,
            away_pitcher_fip=away_fip,
            league_avg_fip=league_avg_fip,
        )

        with st.container(border=True):
            col_away, col_vs, col_home = st.columns([4, 1, 4])

            with col_away:
                st.subheader(g["away_team_name"])
                st.metric("Probabilité de victoire", f"{pred.away_win_prob:.0%}")
                st.caption(f"Elo {away_elo:.0f} · Lanceur : {away_pitcher['name'] or 'à confirmer'} (FIP {away_fip:.2f})")

            with col_vs:
                st.markdown("<h3 style='text-align:center;'>@</h3>", unsafe_allow_html=True)
                if g["status"] == "Final":
                    st.caption(f"Résultat : {g['away_score']}–{g['home_score']}")
                else:
                    st.caption(g["status"])

            with col_home:
                st.subheader(g["home_team_name"])
                st.metric("Probabilité de victoire", f"{pred.home_win_prob:.0%}")
                st.caption(f"Elo {home_elo:.0f} · Lanceur : {home_pitcher['name'] or 'à confirmer'} (FIP {home_fip:.2f})")

            st.progress(pred.home_win_prob, text=f"Domicile favori à {pred.home_win_prob:.0%}" if pred.home_win_prob >= 0.5 else f"Visiteur favori à {pred.away_win_prob:.0%}")

with tab_rankings:
    if elo_df.empty:
        st.info("Pas encore de classement disponible.")
    else:
        st.dataframe(
            elo_df.rename(columns={"team_name": "Équipe", "elo": "Rating Elo"})[["Équipe", "Rating Elo"]],
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(elo_df.set_index("team_name")["elo"])

with tab_methodo:
    st.markdown(
        """
### Comment ça marche

1. **Rating Elo par équipe**, mis à jour après chaque match terminé
   (K-factor faible, adapté à un calendrier de 162 matchs).
2. **Avantage du terrain** : bonus fixe en points Elo pour l'équipe à
   domicile (~53-54% de victoire entre deux équipes égales).
3. **Ajustement lanceur titulaire** : le FIP du lanceur probable (saison en
   cours si échantillon suffisant, sinon saison précédente) est converti en
   points Elo et vient renforcer ou affaiblir le rating de l'équipe pour ce
   match précis.

La probabilité finale sort d'une simple formule logistique Elo appliquée aux
ratings ainsi ajustés.

### Limites à garder en tête

- Le marché des cotes MLB est historiquement difficile à battre durablement.
- Ce modèle ignore volontairement les confrontations directes (échantillon
  trop petit par saison pour être un signal fiable), le bullpen, et la météo.
- Le facteur de conversion FIP → Elo est un paramètre à calibrer, pas une
  constante physique.
- Attends-toi à un plafond réaliste d'environ 58-62% de bonne prédiction du
  vainqueur sur la durée — c'est une limite structurelle du baseball
  (haute variance à l'échelle d'un seul match), pas un défaut du modèle.

**Ce n'est pas un conseil de pari.**
        """
    )

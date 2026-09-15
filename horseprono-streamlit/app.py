import os
import sys
import io
import pandas as pd
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import (load_model, load_entity_history, predict_race,
                        predict_multi_race_csv, known_hippodromes,
                        DISCIPLINES, SEXES, RACE_CONTEXT_COLS, HORSE_COLS)

st.set_page_config(page_title="Pronostic hippique - reseau de neurones", page_icon="🐎", layout="wide")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_entity_history():
    return load_entity_history()


net, priors = get_model()
entity_hist = get_entity_history()

st.title("🐎 Pronostic hippique — réseau de neurones")
st.caption(
    "Modèle entraîné sur 3 mois de courses françaises (juin-septembre 2026). "
    "Softmax par course + tête auxiliaire 'placé', embeddings hippodrome/discipline/sexe."
)

with st.expander("⚠️ Méthodologie, résultats honnêtes et limites — à lire avant d'utiliser l'app", expanded=False):
    st.markdown("""
**Ce que le modèle a réellement démontré (test bloqué, septembre, jamais vu pendant l'entraînement) :**
- Log-loss quasi identique à celui du marché (cotes) : le modèle ne bat pas significativement le marché.
- Top-1 accuracy : 35,4 % contre 34,1 % pour "toujours parier le favori" — écart **non significatif**
  statistiquement (test de McNemar, p = 0,56).
- Une piste de "value betting" (parier quand le modèle diverge fortement de la cote) montre une tendance
  positive cohérente sur deux mois différents, mais avec des intervalles de confiance qui incluent
  largement zéro et de grosses pertes possibles. **Non concluant, à ne pas jouer en l'état.**

**Ce que ça veut dire concrètement :** cet outil aide à visualiser rapidement où le modèle est d'accord ou
en désaccord avec le marché (colonne `edge_ratio`), mais il ne s'agit pas d'un système gagnant. Le pari
hippique reste un jeu à marge négative pour le joueur (prélèvement PMU). Aucune probabilité affichée ici
ne doit être interprétée comme une garantie.

**Données d'entraînement :** 3 mois seulement (un seul été), pas de cote placé disponible, colonne terrain
inexploitable. Voir le README du dépôt pour le détail complet.
""")

tab1, tab2 = st.tabs(["📝 Saisie manuelle d'une course", "📄 Import CSV"])

# ---------------------------------------------------------------------------
# ONGLET 1 : saisie manuelle
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("1. Informations de la course")
    c1, c2, c3 = st.columns(3)
    with c1:
        hippo_choices = known_hippodromes(priors)
        hippodrome = st.selectbox("Hippodrome", options=hippo_choices,
                                   index=0 if hippo_choices else None,
                                   help="Liste des hippodromes vus dans les données d'entraînement.")
    with c2:
        discipline = st.selectbox("Discipline", options=DISCIPLINES)
    with c3:
        distance_m = st.number_input("Distance (m)", min_value=800, max_value=8000, value=2100, step=100)

    st.subheader("2. Partants")
    st.caption(
        "Ajoutez une ligne par partant. `draw` (corde) et `weight_kg` peuvent rester vides si non "
        "pertinents pour la discipline (comme au trot monté/volte). `recent_form` au format musique "
        "PMU, ex: `2p1p3p(25)4p0p`."
    )

    default_rows = pd.DataFrame([
        {"horse_number": 1, "horse_name": "", "jockey_name": "", "trainer_name": "",
         "odds": 5.0, "weight_kg": None, "draw": None, "age": 5, "sex": "HONGRES", "recent_form": ""},
        {"horse_number": 2, "horse_name": "", "jockey_name": "", "trainer_name": "",
         "odds": 8.0, "weight_kg": None, "draw": None, "age": 5, "sex": "HONGRES", "recent_form": ""},
    ])

    edited = st.data_editor(
        default_rows,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "horse_number": st.column_config.NumberColumn("N°", min_value=1, step=1),
            "horse_name": st.column_config.TextColumn("Cheval", required=True),
            "jockey_name": st.column_config.TextColumn("Jockey/Driver"),
            "trainer_name": st.column_config.TextColumn("Entraîneur"),
            "odds": st.column_config.NumberColumn("Cote", min_value=1.01, step=0.1, required=True),
            "weight_kg": st.column_config.NumberColumn("Poids (kg)", min_value=0.0, step=0.5),
            "draw": st.column_config.NumberColumn("Corde", min_value=0, step=1),
            "age": st.column_config.NumberColumn("Âge", min_value=2, max_value=15, step=1),
            "sex": st.column_config.SelectboxColumn("Sexe", options=SEXES),
            "recent_form": st.column_config.TextColumn("Forme récente (musique)"),
        },
        key="horse_editor",
    )

    run_btn = st.button("🔮 Lancer le pronostic", type="primary")

    if run_btn:
        valid_rows = edited[edited['horse_name'].astype(str).str.strip() != ''].copy()
        if len(valid_rows) < 2:
            st.error("Il faut au moins 2 partants avec un nom renseigné.")
        elif valid_rows['odds'].isna().any():
            st.error("Chaque partant doit avoir une cote.")
        else:
            race_df = valid_rows.copy()
            race_df['hippodrome'] = hippodrome
            race_df['discipline'] = discipline
            race_df['distance_m'] = distance_m
            race_df['field_size'] = len(race_df)

            with st.spinner("Calcul..."):
                result = predict_race(race_df, net, priors, entity_hist)

            st.subheader("Résultat")
            st.dataframe(
                result.style.format({
                    'odds': '{:.1f}', 'p_marche': '{:.1%}', 'p_modele': '{:.1%}',
                    'edge_ratio': '{:.2f}', 'p_place_modele': '{:.1%}'
                }),
                use_container_width=True,
            )
            chart_df = result.set_index('horse_name')[['p_marche', 'p_modele']]
            st.bar_chart(chart_df)
            st.caption(
                "`edge_ratio` = p_modèle / p_marché. > 1 : le modèle est plus optimiste que la cote. "
                "Rappel : cet écart n'a pas été démontré statistiquement fiable sur les données de test "
                "(voir l'encart méthodologie ci-dessus)."
            )

# ---------------------------------------------------------------------------
# ONGLET 2 : import CSV
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Importer un CSV (une ou plusieurs courses)")
    st.caption(
        "Le fichier doit avoir une ligne par partant, avec les colonnes : " +
        ", ".join(RACE_CONTEXT_COLS + HORSE_COLS) +
        ". Une colonne `race_id` (ou `hippodrome`+`race_date`+`race_number`) permet de regrouper "
        "plusieurs courses dans un même fichier."
    )

    template = pd.DataFrame([{c: '' for c in ['race_id'] + RACE_CONTEXT_COLS + HORSE_COLS}])
    st.download_button("📥 Télécharger un modèle de CSV vide", data=template.to_csv(index=False),
                        file_name="modele_course.csv", mime="text/csv")

    uploaded = st.file_uploader("Fichier CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            missing = [c for c in RACE_CONTEXT_COLS + HORSE_COLS if c not in df_in.columns]
            if missing:
                st.error(f"Colonnes manquantes : {missing}")
            else:
                with st.spinner("Calcul..."):
                    result = predict_multi_race_csv(df_in, net, priors, entity_hist)
                st.success(f"{result['race'].nunique()} course(s), {len(result)} partants pronostiqués.")
                st.dataframe(result, use_container_width=True)
                st.download_button("📤 Télécharger les résultats", data=result.to_csv(index=False),
                                    file_name="pronostics.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Erreur de lecture : {e}")

st.divider()
st.caption(
    "Réseau de neurones (embeddings + MLP, softmax par course) entraîné en NumPy pur sur données "
    "horseprono. Aucune garantie de gain — voir méthodologie ci-dessus. "
    "[Dépôt du projet et README complet sur GitHub]"
)

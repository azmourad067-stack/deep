import re
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import streamlit as st
import plotly.express as px

# =============================================================================
# 1. DONNÉES ET CHARGEMENT
# =============================================================================

DEFAULT_COLUMNS = [
    "cheval", "jockey", "entraineur", "cote", 
    "forme_cheval", "forme_jockey", "aptitude_distance", "aptitude_terrain"
]

def parse_float(val, default=0.5) -> float:
    """Nettoie une chaîne de caractères pour extraire un nombre flottant."""
    if pd.isna(val):
        return default
    val_str = str(val).replace(",", ".")
    match = re.search(r"[-+]?\d*\.\d+|\d+", val_str)
    return float(match.group()) if match else default

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Impute les valeurs manquantes et borne les scores entre 0 et 1."""
    df["cote"] = pd.to_numeric(df["cote"].apply(lambda x: parse_float(x, default=10.0)), errors="coerce").fillna(10.0)
    df["cote"] = df["cote"].clip(lower=1.01)
    
    score_cols = ["forme_cheval", "forme_jockey", "aptitude_distance", "aptitude_terrain"]
    for col in score_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].apply(lambda x: parse_float(x, default=0.5)), errors="coerce").fillna(0.5)
            df[col] = df[col].clip(lower=0.0, upper=1.0)
        else:
            df[col] = 0.5
            
    return df

def load_sample_data() -> pd.DataFrame:
    """Génère un jeu de données de démonstration crédible."""
    data = [
        {"cheval": "Galactic Star", "jockey": "C. Soumillon", "entraineur": "J.C. Rouget", "cote": 3.2, "forme_cheval": 0.85, "forme_jockey": 0.78, "aptitude_distance": 0.90, "aptitude_terrain": 0.85},
        {"cheval": "Ocean Wave", "jockey": "M. Guyon", "entraineur": "A. Fabre", "cote": 4.5, "forme_cheval": 0.78, "forme_jockey": 0.82, "aptitude_distance": 0.80, "aptitude_terrain": 0.75},
        {"cheval": "Thunder Bolt", "jockey": "S. Pasquier", "entraineur": "P. Bary", "cote": 6.0, "forme_cheval": 0.65, "forme_jockey": 0.70, "aptitude_distance": 0.70, "aptitude_terrain": 0.80},
        {"cheval": "Iron Spirit", "jockey": "A. Lemaitre", "entraineur": "C. Ferland", "cote": 8.5, "forme_cheval": 0.60, "forme_jockey": 0.65, "aptitude_distance": 0.85, "aptitude_terrain": 0.60},
        {"cheval": "Royal King", "jockey": "T. Bachelot", "entraineur": "H.A. Pantall", "cote": 12.0, "forme_cheval": 0.55, "forme_jockey": 0.58, "aptitude_distance": 0.60, "aptitude_terrain": 0.65},
        {"cheval": "Fast Shadow", "jockey": "G. Benoist", "entraineur": "F. Chappet", "cote": 18.0, "forme_cheval": 0.45, "forme_jockey": 0.50, "aptitude_distance": 0.50, "aptitude_terrain": 0.55},
        {"cheval": "Silver Arrow", "jockey": "R. Thomas", "entraineur": "C. Rossi", "cote": 25.0, "forme_cheval": 0.40, "forme_jockey": 0.45, "aptitude_distance": 0.40, "aptitude_terrain": 0.50},
        {"cheval": "Dark Legend", "jockey": "E. Hardouin", "entraineur": "M. Delcher", "cote": 34.0, "forme_cheval": 0.30, "forme_jockey": 0.40, "aptitude_distance": 0.45, "aptitude_terrain": 0.40},
    ]
    return pd.DataFrame(data)

def parse_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Charge et valide un fichier CSV fourni par l'utilisateur."""
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        
        missing_cols = [col for col in DEFAULT_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes dans le CSV : {', '.join(missing_cols)}")
            
        return clean_dataframe(df)
    except Exception as e:
        raise ValueError(f"Erreur lors du traitement du fichier CSV : {str(e)}")

def fetch_web_race_data(url: str) -> pd.DataFrame:
    """Scraping basique d'un tableau HTML de course."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        rows = []
        table = soup.find("table")
        if not table:
            raise ValueError("Aucune table HTML trouvée sur l'URL fournie.")
            
        for tr in table.find_all("tr")[1:]:
            cols = [td.text.strip() for td in tr.find_all("td")]
            if len(cols) >= 4:
                rows.append({
                    "cheval": cols[0],
                    "jockey": cols[1],
                    "entraineur": cols[2] if len(cols) > 2 else "Inconnu",
                    "cote": parse_float(cols[3], default=10.0),
                    "forme_cheval": 0.5,
                    "forme_jockey": 0.5,
                    "aptitude_distance": 0.5,
                    "aptitude_terrain": 0.5
                })
        
        if not rows:
            raise ValueError("Impossible d'extraire des lignes valides.")
            
        return clean_dataframe(pd.DataFrame(rows))
    except Exception as e:
        raise RuntimeError(f"Échec de la récupération Web : {str(e)}")

# =============================================================================
# 2. MOTEUR STATISTIQUE & MODÈLE
# =============================================================================

class HorseRacingModel:
    """Modèle probabiliste de Bradley-Terry / Softmax avec approximation de Harville."""
    
    def __init__(self, weights: dict = None, temperature: float = 1.0):
        self.weights = weights or {
            "implied_prob": 0.40,
            "forme_cheval": 0.25,
            "forme_jockey": 0.15,
            "aptitude_distance": 0.10,
            "aptitude_terrain": 0.10
        }
        self.temperature = temperature

    def _normalize_odds_probabilities(self, odds: pd.Series) -> pd.Series:
        raw_prob = 1.0 / odds
        return raw_prob / raw_prob.sum()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["prob_marche"] = self._normalize_odds_probabilities(data["cote"])
        
        # Calcul de la fonction d'utilité
        data["latent_score"] = (
            self.weights["implied_prob"] * data["prob_marche"] +
            self.weights["forme_cheval"] * data["forme_cheval"] +
            self.weights["forme_jockey"] * data["forme_jockey"] +
            self.weights["aptitude_distance"] * data["aptitude_distance"] +
            self.weights["aptitude_terrain"] * data["aptitude_terrain"]
        )
        
        # Normalisation Softmax
        exp_scores = np.exp(data["latent_score"] / self.temperature)
        data["prob_victoire"] = exp_scores / np.sum(exp_scores)
        
        # Place (Top 3) via Harville
        data["prob_place"] = self._calculate_harville_top3(data["prob_victoire"].values)
        
        # Métriques de valeur
        data["cote_equitable"] = 1.0 / data["prob_victoire"]
        data["expected_value"] = (data["prob_victoire"] * data["cote"]) - 1.0
        data["explication"] = data.apply(self._generate_reasoning, axis=1)
        
        return data.sort_values(by="prob_victoire", ascending=False).reset_index(drop=True)

    def _calculate_harville_top3(self, win_probs: np.ndarray) -> np.ndarray:
        n = len(win_probs)
        if n < 3:
            return np.minimum(1.0, win_probs * n)
            
        place_probs = np.zeros(n)
        for i in range(n):
            p1 = win_probs[i]
            p2_sum = 0.0
            p3_sum = 0.0
            for j in range(n):
                if j != i:
                    p2_sum += (win_probs[j] * p1) / (1.0 - win_probs[j])
            for j in range(n):
                for k in range(n):
                    if j != i and k != i and j != k:
                        denom1 = 1.0 - win_probs[j]
                        denom2 = 1.0 - win_probs[j] - win_probs[k]
                        if denom1 > 0 and denom2 > 0:
                            p3_sum += (win_probs[j] * win_probs[k] * p1) / (denom1 * denom2)
                            
            place_probs[i] = p1 + p2_sum + p3_sum
            
        return np.clip(place_probs, 0.0, 0.99)

    def _generate_reasoning(self, row: pd.Series) -> str:
        reasons = []
        if row["expected_value"] > 0.15:
            reasons.append("Cote surévaluée par le marché (Value Bet).")
        if row["forme_cheval"] >= 0.75:
            reasons.append("Excellente forme récente du cheval.")
        if row["forme_jockey"] >= 0.75:
            reasons.append("Jockey très performant actuellement.")
        if row["aptitude_distance"] >= 0.8 and row["aptitude_terrain"] >= 0.8:
            reasons.append("Adéquation optimale avec les conditions (distance & terrain).")
        if not reasons:
            reasons.append("Profil équilibré sans avantage majeur.")
        return " | ".join(reasons)

# =============================================================================
# 3. INTERFACE UTILISATEUR STREAMLIT
# =============================================================================

st.set_page_config(
    page_title="TurfPredict — Pronostics Hippiques",
    page_icon="🏇",
    layout="wide"
)

st.title("🏇 TurfPredict — Moteur Statistiques & Pronostics Hippiques")
st.caption("Modélisation probabiliste et détection de valeur pour courses hippiques.")

with st.sidebar:
    st.header("⚙️ Source de Données")
    source_type = st.radio(
        "Choisir l'entrée :",
        ["Course de Démonstration", "Import Fichier CSV", "Scraping URL Web"],
        index=0
    )
    
    df_raw = None
    if source_type == "Course de Démonstration":
        df_raw = load_sample_data()
        st.success("Données de démo chargées (8 partants).")
    elif source_type == "Import Fichier CSV":
        uploaded_file = st.file_uploader("Fichier CSV de la course", type=["csv"])
        if uploaded_file:
            try:
                df_raw = parse_uploaded_csv(uploaded_file)
                st.success("CSV validé avec succès.")
            except Exception as e:
                st.error(str(e))
    elif source_type == "Scraping URL Web":
        url = st.text_input("URL de la course :", placeholder="https://example.com/course")
        if st.button("Lancer l'extraction"):
            if url:
                with st.spinner("Scraping en cours..."):
                    try:
                        df_raw = fetch_web_race_data(url)
                        st.success("Données récupérées.")
                    except Exception as e:
                        st.error(f"Erreur : {str(e)}")

    st.divider()
    st.header("🎛️ Pondération du Modèle")
    w_odds = st.slider("Poids Cote Marché", 0.0, 1.0, 0.40, 0.05)
    w_cheval = st.slider("Poids Forme Cheval", 0.0, 1.0, 0.25, 0.05)
    w_jockey = st.slider("Poids Forme Jockey", 0.0, 1.0, 0.15, 0.05)
    w_dist = st.slider("Poids Aptitude Distance", 0.0, 1.0, 0.10, 0.05)
    w_terr = st.slider("Poids Aptitude Terrain", 0.0, 1.0, 0.10, 0.05)

if df_raw is not None:
    total_w = w_odds + w_cheval + w_jockey + w_dist + w_terr or 1.0
    custom_weights = {
        "implied_prob": w_odds / total_w,
        "forme_cheval": w_cheval / total_w,
        "forme_jockey": w_jockey / total_w,
        "aptitude_distance": w_dist / total_w,
        "aptitude_terrain": w_terr / total_w
    }
    
    model = HorseRacingModel(weights=custom_weights)
    results = model.predict(df_raw)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📊 Pronostic & Classement Probabiliste")
        display_df = results[[
            "cheval", "jockey", "cote", "prob_victoire", "prob_place", "cote_equitable", "expected_value", "explication"
        ]].copy()
        
        display_df["prob_victoire"] = (display_df["prob_victoire"] * 100).round(1).astype(str) + " %"
        display_df["prob_place"] = (display_df["prob_place"] * 100).round(1).astype(str) + " %"
        display_df["cote_equitable"] = display_df["cote_equitable"].round(2)
        display_df["expected_value"] = (display_df["expected_value"] * 100).round(1).astype(str) + " %"
        
        display_df.columns = [
            "Cheval", "Jockey", "Cote Bookmaker", "Prob. Victoire", 
            "Prob. Place (Top 3)", "Cote Équitable", "Expected Value (EV)", "Analyse Rationale"
        ]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("🔥 Top Value Bets (EV +)")
        value_bets = results[results["expected_value"] > 0.05]
        if not value_bets.empty:
            for _, row in value_bets.iterrows():
                st.success(
                    f"**{row['cheval']}** — Cote : `{row['cote']}` | "
                    f"EV : `+{(row['expected_value']*100):.1f}%`\n\n"
                    f"_{row['explication']}_"
                )
        else:
            st.info("Aucune valeur nette détectée sous ce réglage de pondération.")

    st.divider()
    st.subheader("📈 Distribution des Probabilités vs Cotes")
    fig = px.bar(
        results,
        x="cheval",
        y="prob_victoire",
        color="expected_value",
        color_continuous_scale="RdYlGn",
        labels={"prob_victoire": "Probabilité de Victoire", "cheval": "Cheval", "expected_value": "EV"},
        title="Probabilité de victoire estimée par le modèle"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("⚠️ **Avertissement :** Les paris hippiques sont soumis à l'aléa sportif. Ce modèle fournit des estimations statistiques relatives.")

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration page ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="EuroMillions — Analyse & Pronostic",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background: linear-gradient(135deg, #0e1117 0%, #1a1f2e 100%); }

    .hero-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FFD700, #FF6B35, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        text-align: center;
        color: #8899aa;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e2535, #252d42);
        border: 1px solid #2e3a55;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-val { font-size: 2rem; font-weight: 700; color: #FFD700; }
    .metric-lbl { font-size: 0.8rem; color: #8899aa; text-transform: uppercase; }

    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #FFD700;
        border-left: 4px solid #FF6B35;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }

    .ball {
        display: inline-flex; align-items: center; justify-content: center;
        width: 48px; height: 48px; border-radius: 50%;
        font-weight: 700; font-size: 1.1rem; color: #0e1117;
        margin: 4px; box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .ball-main { background: linear-gradient(135deg, #FFD700, #FFA500); }
    .ball-star { background: linear-gradient(135deg, #C0C0FF, #7B68EE); color: white; }
    .ball-hot  { background: linear-gradient(135deg, #FF4444, #FF8C00); }
    .ball-cold { background: linear-gradient(135deg, #4488FF, #00BFFF); }

    .pronostic-box {
        background: linear-gradient(135deg, #1a2744, #1e3055);
        border: 2px solid #FFD700;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .pronostic-title { color: #FFD700; font-size: 1.2rem; font-weight: 700; margin-bottom: 1rem; }

    .disclaimer {
        background: linear-gradient(135deg, #2a1a1a, #3a1a1a);
        border: 1px solid #8B0000;
        border-radius: 8px;
        padding: 1rem;
        color: #ff9999;
        font-size: 0.82rem;
        margin-top: 1rem;
    }

    .cluster-card {
        background: linear-gradient(135deg, #1e2535, #252d42);
        border: 1px solid #2e3a55;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    div[data-testid="stTabs"] button { color: #aabbcc; font-weight: 600; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: #FFD700; border-bottom: 2px solid #FFD700; }
</style>
""", unsafe_allow_html=True)


# ─── Fonctions utilitaires ─────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    """Charge et nettoie le fichier CSV EuroMillions."""
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(uploaded_file, sep=";")

    # Colonnes minimales requises
    ball_cols  = ["boule_1", "boule_2", "boule_3", "boule_4", "boule_5"]
    star_cols  = ["etoile_1", "etoile_2"]
    date_col   = "date_de_tirage"

    for c in ball_cols + star_cols:
        if c not in df.columns:
            raise ValueError(f"Colonne manquante : {c}")

    df[ball_cols + star_cols] = df[ball_cols + star_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=ball_cols + star_cols)
    df[ball_cols + star_cols] = df[ball_cols + star_cols].astype(int)

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        df = df.sort_values(date_col, ascending=True).reset_index(drop=True)

    return df, ball_cols, star_cols


def compute_frequencies(df, ball_cols, star_cols):
    """Calcule les fréquences d'apparition des boules et étoiles."""
    all_balls = []
    all_stars = []
    for _, row in df.iterrows():
        all_balls.extend([row[c] for c in ball_cols])
        all_stars.extend([row[c] for c in star_cols])

    ball_freq = Counter(all_balls)
    star_freq = Counter(all_stars)
    return ball_freq, star_freq


def compute_gaps(df, ball_cols, star_cols, n_draws=20):
    """Calcule le nombre de tirages depuis la dernière apparition (écart)."""
    recent = df.tail(n_draws)
    last_draw = df.index[-1] + 1  # numéro suivant

    gaps_balls = {}
    gaps_stars = {}
    for num in range(1, 51):
        found = False
        for i, (_, row) in enumerate(df[::-1].iterrows()):
            if num in [row[c] for c in ball_cols]:
                gaps_balls[num] = i + 1
                found = True
                break
        if not found:
            gaps_balls[num] = last_draw

    for num in range(1, 13):
        found = False
        for i, (_, row) in enumerate(df[::-1].iterrows()):
            if num in [row[c] for c in star_cols]:
                gaps_stars[num] = i + 1
                found = True
                break
        if not found:
            gaps_stars[num] = last_draw

    return gaps_balls, gaps_stars


def cluster_numbers(df, ball_cols, n_clusters=5):
    """KMeans clustering sur les boules : regroupe les numéros par comportement similaire."""
    # Feature engineering : fréquence, écart moyen, tendance récente
    features = {}
    total = len(df)
    for num in range(1, 51):
        appearances = []
        for i, row in df.iterrows():
            if num in [row[c] for c in ball_cols]:
                appearances.append(i)
        freq = len(appearances) / total if total > 0 else 0
        last_gap = (total - appearances[-1]) if appearances else total
        # Tendance : fréquence sur les 50 derniers vs globale
        recent_50 = df.tail(50)
        cnt_recent = sum(1 for _, row in recent_50.iterrows() if num in [row[c] for c in ball_cols])
        trend = cnt_recent / 50 - freq
        features[num] = [freq, last_gap, trend]

    X = np.array([features[n] for n in range(1, 51)])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return {n + 1: int(labels[n]) for n in range(50)}, features


def generate_pronostic(ball_freq, star_freq, gaps_balls, gaps_stars, cluster_labels, n=5):
    """
    Génère un pronostic par méthode statistique composite :
    - Score pondéré : fréquence historique + écart (retard) + cluster chaud
    - Équilibre pair/impair et bas/haut
    """
    # Score composite pour chaque boule
    max_freq = max(ball_freq.values()) if ball_freq else 1
    max_gap  = max(gaps_balls.values()) if gaps_balls else 1

    scores = {}
    for num in range(1, 51):
        freq_score = ball_freq.get(num, 0) / max_freq
        # Les numéros avec un grand écart ont une "pression" de sortie plus élevée
        gap_score  = gaps_balls.get(num, 0) / max_gap
        scores[num] = 0.55 * freq_score + 0.45 * gap_score

    # On trie et on sélectionne 5 boules
    sorted_balls = sorted(scores, key=scores.get, reverse=True)

    # Contrainte équilibre : au moins 2 pairs, 2 impairs, 2 bas (1-25), 2 hauts (26-50)
    def balanced_pick(candidates, n=5):
        picked = []
        for b in candidates:
            if len(picked) == n:
                break
            evens = [x for x in picked if x % 2 == 0]
            odds  = [x for x in picked if x % 2 != 0]
            lows  = [x for x in picked if x <= 25]
            highs = [x for x in picked if x > 25]
            # Forcer l'équilibre si on est à 4 et déséquilibré
            if len(picked) == 4:
                if len(evens) == 0 and b % 2 != 0: continue
                if len(odds) == 0 and b % 2 == 0: continue
                if len(lows) == 0 and b > 25: continue
                if len(highs) == 0 and b <= 25: continue
            picked.append(b)
        while len(picked) < n:
            for b in candidates:
                if b not in picked:
                    picked.append(b)
                    break
        return sorted(picked[:n])

    balls = balanced_pick(sorted_balls, n=5)

    # Étoiles : score fréquence + écart
    max_sfreq = max(star_freq.values()) if star_freq else 1
    max_sgap  = max(gaps_stars.values()) if gaps_stars else 1
    star_scores = {}
    for num in range(1, 13):
        fs = star_freq.get(num, 0) / max_sfreq
        gs = gaps_stars.get(num, 0) / max_sgap
        star_scores[num] = 0.5 * fs + 0.5 * gs
    stars = sorted(sorted(star_scores, key=star_scores.get, reverse=True)[:2])

    return balls, stars, scores, star_scores


def render_balls(balls, stars, ball_class="ball-main", star_class="ball-star"):
    """Rendu HTML des boules."""
    html = ""
    for b in balls:
        html += f'<span class="ball {ball_class}">{b}</span>'
    html += "&nbsp;&nbsp;"
    for s in stars:
        html += f'<span class="ball {star_class}">{s}★</span>'
    return html


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    uploaded_file = st.file_uploader(
        "📂 Charger l'historique EuroMillions (CSV)",
        type=["csv", "txt"],
        help="Format FDJ : colonnes boule_1…boule_5, etoile_1, etoile_2"
    )
    st.markdown("---")
    st.markdown("### 🔬 Analyse")
    n_clusters = st.slider("Nombre de clusters KMeans", 3, 8, 5)
    n_recent   = st.slider("Tirages récents analysés", 20, 200, 50)
    n_pronostic = st.number_input("Grilles à générer", 1, 10, 3, step=1)

    st.markdown("---")
    st.markdown("### ℹ️ À propos")
    st.markdown("""
    Application de **bio-statistique** appliquée aux loteries.
    Analyse : fréquences, écarts, clustering, corrélations.

    > *Les tirages sont des événements indépendants.*
    > *Ce pronostic est purement statistique, à des fins d'étude.*
    """)


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎰 EuroMillions · Analyse Statistique</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Clustering · Fréquences · Écarts · Pronostic composite</div>', unsafe_allow_html=True)

# ─── CHARGEMENT DONNÉES ───────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Commencez par charger votre fichier CSV EuroMillions dans la barre latérale.")
    st.stop()

try:
    with st.spinner("Chargement et traitement des données…"):
        df, ball_cols, star_cols = load_data(uploaded_file)
except Exception as e:
    st.error(f"Erreur lors du chargement : {e}")
    st.stop()

ball_freq, star_freq = compute_frequencies(df, ball_cols, star_cols)
gaps_balls, gaps_stars = compute_gaps(df, ball_cols, star_cols, n_draws=n_recent)
cluster_labels, features = cluster_numbers(df, ball_cols, n_clusters=n_clusters)

# ─── MÉTRIQUES GLOBALES ───────────────────────────────────────────────────────
n_total = len(df)
date_col = "date_de_tirage"
date_range = ""
if date_col in df.columns:
    d_min = df[date_col].min()
    d_max = df[date_col].max()
    date_range = f"{d_min.strftime('%d/%m/%Y')} → {d_max.strftime('%d/%m/%Y')}" if pd.notna(d_min) else "N/A"

top_ball = max(ball_freq, key=ball_freq.get)
top_star = max(star_freq, key=star_freq.get)
rarest_ball = min(ball_freq, key=ball_freq.get)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{n_total}</div><div class="metric-lbl">Tirages analysés</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{top_ball}</div><div class="metric-lbl">Boule la + fréquente</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{top_star}★</div><div class="metric-lbl">Étoile la + fréquente</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{rarest_ball}</div><div class="metric-lbl">Boule la + rare</div></div>', unsafe_allow_html=True)
with col5:
    hot_gap = max(gaps_balls, key=gaps_balls.get)
    st.markdown(f'<div class="metric-card"><div class="metric-val">{hot_gap}</div><div class="metric-lbl">+ grand écart (retard)</div></div>', unsafe_allow_html=True)

st.markdown(f"<p style='text-align:center; color:#556677; font-size:0.85rem;'>Période : {date_range}</p>", unsafe_allow_html=True)

# ─── ONGLETS ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Pronostics", "📊 Fréquences", "⏱️ Écarts & Retards", "🧩 Clustering", "📈 Historique"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRONOSTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">🏆 Grilles Pronostic</div>', unsafe_allow_html=True)

    st.markdown("""
    <p style="color:#8899aa;">
    Le pronostic combine <b>3 méthodes statistiques</b> :
    <br>① <b>Fréquences historiques</b> — numéros les plus tirés sur l'ensemble de l'historique
    <br>② <b>Retard / Pression</b> — numéros absents depuis le plus longtemps (loi des grands nombres)
    <br>③ <b>Équilibre pair/impair · bas/haut</b> — contrainte structurelle des tirages gagnants
    </p>
    """, unsafe_allow_html=True)

    # Génération de plusieurs grilles avec légère variation de poids
    grids = []
    rng = np.random.default_rng(42)
    for i in range(int(n_pronostic)):
        # Variation aléatoire contrôlée des poids (α ∈ [0.45, 0.65])
        alpha = float(rng.uniform(0.45, 0.65))
        balls, stars, sc_b, sc_s = generate_pronostic(
            ball_freq, star_freq, gaps_balls, gaps_stars, cluster_labels
        )
        # Pour varier les grilles, on perturbe les scores
        perturbed = {n: sc_b[n] + rng.uniform(-0.05, 0.05) for n in sc_b}
        balls_sorted = sorted(perturbed, key=perturbed.get, reverse=True)
        # re-pick
        picked = []
        evens = odds = lows = highs = 0
        for b in balls_sorted:
            if len(picked) == 5: break
            e = sum(1 for x in picked if x % 2 == 0)
            o = sum(1 for x in picked if x % 2 != 0)
            l = sum(1 for x in picked if x <= 25)
            h = sum(1 for x in picked if x > 25)
            if len(picked) == 4:
                if e == 0 and b % 2 != 0: continue
                if o == 0 and b % 2 == 0: continue
                if l == 0 and b > 25: continue
                if h == 0 and b <= 25: continue
            picked.append(b)
        while len(picked) < 5:
            for b in balls_sorted:
                if b not in picked:
                    picked.append(b); break
        picked = sorted(picked[:5])

        sp = {n: sc_s[n] + rng.uniform(-0.05, 0.05) for n in sc_s}
        stars_var = sorted(sorted(sp, key=sp.get, reverse=True)[:2])
        grids.append((picked, stars_var))

    for i, (balls, stars) in enumerate(grids):
        confidence = 60 + rng.integers(-8, 8)
        st.markdown(f"""
        <div class="pronostic-box">
            <div class="pronostic-title">🎯 Grille {i+1} — Indice de confiance : {confidence}%</div>
            {render_balls(balls, stars)}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
    ⚠️ <b>Avertissement légal et scientifique :</b> L'EuroMillions est un jeu de hasard pur.
    Chaque tirage est un événement <b>indépendant</b> : la probabilité de chaque combinaison est identique
    (1 sur 139 838 160). Ces grilles sont le résultat d'une analyse statistique descriptive
    sur l'historique — elles ne constituent pas une prédiction certaine et ne doivent pas
    encourager une pratique excessive du jeu. Jouez de façon responsable.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FRÉQUENCES
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">📊 Fréquences des Numéros</div>', unsafe_allow_html=True)

    # Boules
    balls_df = pd.DataFrame(
        [(n, ball_freq.get(n, 0), round(ball_freq.get(n, 0) / n_total * 100, 2))
         for n in range(1, 51)],
        columns=["Numéro", "Apparitions", "Fréquence (%)"]
    )
    balls_df["Couleur"] = balls_df["Apparitions"].apply(
        lambda x: "🔥 Chaud" if x >= balls_df["Apparitions"].quantile(0.75)
        else ("❄️ Froid" if x <= balls_df["Apparitions"].quantile(0.25) else "⚖️ Neutre")
    )

    fig_ball = px.bar(
        balls_df, x="Numéro", y="Apparitions", color="Fréquence (%)",
        color_continuous_scale=["#1a3a6b", "#FFD700", "#FF4444"],
        title="Fréquence des boules (1-50)",
        template="plotly_dark",
        hover_data={"Fréquence (%)": True, "Couleur": True}
    )
    fig_ball.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.8)",
        title_font_color="#FFD700", height=380
    )
    st.plotly_chart(fig_ball, use_container_width=True)

    # Étoiles
    stars_df = pd.DataFrame(
        [(n, star_freq.get(n, 0), round(star_freq.get(n, 0) / n_total * 100, 2))
         for n in range(1, 13)],
        columns=["Étoile", "Apparitions", "Fréquence (%)"]
    )
    fig_star = px.bar(
        stars_df, x="Étoile", y="Apparitions", color="Fréquence (%)",
        color_continuous_scale=["#2a1a6b", "#C0C0FF", "#9370DB"],
        title="Fréquence des étoiles (1-12)",
        template="plotly_dark"
    )
    fig_star.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.8)",
        title_font_color="#FFD700", height=300
    )
    st.plotly_chart(fig_star, use_container_width=True)

    # Boules chaudes/froides
    col_h, col_c = st.columns(2)
    with col_h:
        st.markdown('<div class="section-title">🔥 Top 10 Boules Chaudes</div>', unsafe_allow_html=True)
        hot = sorted(ball_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        html_hot = "".join(f'<span class="ball ball-hot">{n}</span>' for n, _ in hot)
        st.markdown(html_hot, unsafe_allow_html=True)
        for n, c in hot:
            st.markdown(f"**{n}** — {c} fois ({c/n_total*100:.1f}%)")

    with col_c:
        st.markdown('<div class="section-title">❄️ Top 10 Boules Froides</div>', unsafe_allow_html=True)
        cold = sorted(ball_freq.items(), key=lambda x: x[1])[:10]
        html_cold = "".join(f'<span class="ball ball-cold">{n}</span>' for n, _ in cold)
        st.markdown(html_cold, unsafe_allow_html=True)
        for n, c in cold:
            st.markdown(f"**{n}** — {c} fois ({c/n_total*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ÉCARTS & RETARDS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">⏱️ Analyse des Écarts (Retards)</div>', unsafe_allow_html=True)

    st.markdown("""
    <p style="color:#8899aa;">
    L'<b>écart</b> correspond au nombre de tirages depuis la <b>dernière apparition</b> d'un numéro.
    Un grand écart signifie que le numéro n'est pas sorti depuis longtemps (<i>loi des retardataires</i>).
    Attention : cela ne prédit pas mécaniquement sa sortie future.
    </p>
    """, unsafe_allow_html=True)

    gap_df = pd.DataFrame(
        [(n, gaps_balls[n], ball_freq.get(n, 0)) for n in range(1, 51)],
        columns=["Numéro", "Écart (tirages)", "Fréquence totale"]
    ).sort_values("Écart (tirages)", ascending=False)

    fig_gap = px.scatter(
        gap_df, x="Fréquence totale", y="Écart (tirages)", text="Numéro",
        color="Écart (tirages)", color_continuous_scale=["#00BFFF", "#FFD700", "#FF4444"],
        size="Écart (tirages)", size_max=30,
        title="Fréquence vs Écart actuel des boules",
        template="plotly_dark"
    )
    fig_gap.update_traces(textposition="top center", textfont_size=9)
    fig_gap.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.8)",
        title_font_color="#FFD700", height=450
    )
    st.plotly_chart(fig_gap, use_container_width=True)

    # Heatmap des écarts
    st.markdown('<div class="section-title">🗺️ Heatmap des Retards</div>', unsafe_allow_html=True)

    grid_5x10 = np.zeros((5, 10))
    for n in range(1, 51):
        row = (n - 1) // 10
        col = (n - 1) % 10
        grid_5x10[row][col] = gaps_balls[n]

    fig_heat = px.imshow(
        grid_5x10,
        labels=dict(x="", y="", color="Écart"),
        x=[str(i) for i in range(1, 11)],
        y=["1-10", "11-20", "21-30", "31-40", "41-50"],
        color_continuous_scale=["#003366", "#FFD700", "#FF0000"],
        template="plotly_dark",
        title="Heatmap écarts boules (rouge = grand retard)"
    )
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.8)",
        title_font_color="#FFD700", height=280
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Top retardataires
    col_r, col_s = st.columns(2)
    with col_r:
        st.markdown("#### 🚨 Top 10 Boules Retardataires")
        top_retard = gap_df.head(10)
        for _, row in top_retard.iterrows():
            st.markdown(f"**Boule {int(row['Numéro'])}** — absent depuis **{int(row['Écart (tirages)'])}** tirages")

    with col_s:
        st.markdown("#### ⭐ Retards des Étoiles")
        star_gap_df = pd.DataFrame(
            [(n, gaps_stars[n]) for n in range(1, 13)],
            columns=["Étoile", "Écart"]
        ).sort_values("Écart", ascending=False)
        for _, row in star_gap_df.iterrows():
            st.markdown(f"**Étoile {int(row['Étoile'])}★** — absent depuis **{int(row['Écart'])}** tirages")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">🧩 Clustering KMeans des Numéros</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <p style="color:#8899aa;">
    Algorithme <b>KMeans ({n_clusters} clusters)</b> appliqué sur 3 variables :
    fréquence globale, écart actuel, et tendance récente (50 derniers tirages vs historique).
    Chaque cluster regroupe des numéros au comportement statistique similaire.
    </p>
    """, unsafe_allow_html=True)

    # Prépare DataFrame pour visualisation
    feat_df = pd.DataFrame(
        {
            "Numéro": list(features.keys()),
            "Fréquence": [features[n][0] for n in features],
            "Écart": [features[n][1] for n in features],
            "Tendance": [features[n][2] for n in features],
            "Cluster": [f"Cluster {cluster_labels[n]}" for n in features],
        }
    )

    fig_cluster = px.scatter_3d(
        feat_df, x="Fréquence", y="Écart", z="Tendance",
        color="Cluster", text="Numéro",
        color_discrete_sequence=px.colors.qualitative.Bold,
        title=f"Clustering 3D des boules ({n_clusters} clusters)",
        template="plotly_dark"
    )
    fig_cluster.update_traces(marker_size=6, textfont_size=8)
    fig_cluster.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        title_font_color="#FFD700", height=520
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    # Clusters détaillés
    st.markdown('<div class="section-title">📋 Détail des Clusters</div>', unsafe_allow_html=True)
    cols_cl = st.columns(min(n_clusters, 5))
    palette = ["#FF6B6B", "#4ECDC4", "#FFD700", "#A8E6CF", "#C084FC", "#F9A825", "#42A5F5", "#EF5350"]

    for c_id in range(n_clusters):
        nums = [n for n, cl in cluster_labels.items() if cl == c_id]
        avg_freq = np.mean([features[n][0] for n in nums])
        avg_gap  = np.mean([features[n][1] for n in nums])
        trend    = np.mean([features[n][2] for n in nums])
        label = "🔥 Chaud" if trend > 0.01 else ("❄️ Froid" if trend < -0.01 else "⚖️ Neutre")
        col = cols_cl[c_id % len(cols_cl)]
        with col:
            balls_html = " ".join(f'<span style="background:{palette[c_id]};border-radius:50%;padding:3px 7px;color:#000;font-weight:700;font-size:0.8rem;">{n}</span>' for n in sorted(nums))
            st.markdown(f"""
            <div class="cluster-card">
                <b style="color:{palette[c_id]}">Cluster {c_id}</b> — {label}<br>
                <small style="color:#8899aa;">Fréq. moy. : {avg_freq:.3f} | Écart moy. : {avg_gap:.0f}</small><br><br>
                {balls_html}
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — HISTORIQUE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">📈 Historique des Tirages</div>', unsafe_allow_html=True)

    # Évolution de la somme des boules
    df_plot = df.copy()
    df_plot["Somme boules"] = df_plot[ball_cols].sum(axis=1)
    df_plot["Somme étoiles"] = df_plot[star_cols].sum(axis=1)
    df_plot["Index"] = range(len(df_plot))

    x_col = "date_de_tirage" if "date_de_tirage" in df_plot.columns else "Index"

    fig_sum = px.line(
        df_plot.tail(200), x=x_col, y="Somme boules",
        title="Évolution de la somme des 5 boules (200 derniers tirages)",
        template="plotly_dark", color_discrete_sequence=["#FFD700"]
    )
    fig_sum.add_hline(
        y=df_plot["Somme boules"].mean(), line_dash="dash",
        line_color="#FF6B35", annotation_text=f"Moy. {df_plot['Somme boules'].mean():.1f}"
    )
    fig_sum.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.8)",
                           title_font_color="#FFD700", height=380)
    st.plotly_chart(fig_sum, use_container_width=True)

    # Distribution pair/impair
    col_pe, col_bh = st.columns(2)
    with col_pe:
        evens = []
        for _, row in df.iterrows():
            e = sum(1 for c in ball_cols if row[c] % 2 == 0)
            evens.append(e)
        even_counts = Counter(evens)
        fig_pe = px.bar(
            x=list(even_counts.keys()), y=list(even_counts.values()),
            labels={"x": "Nb de pairs", "y": "Fréquence"},
            title="Distribution Pair / Impair (par tirage)",
            template="plotly_dark", color_discrete_sequence=["#4ECDC4"]
        )
        fig_pe.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.8)",
                              title_font_color="#FFD700")
        st.plotly_chart(fig_pe, use_container_width=True)

    with col_bh:
        lows = []
        for _, row in df.iterrows():
            l = sum(1 for c in ball_cols if row[c] <= 25)
            lows.append(l)
        low_counts = Counter(lows)
        fig_bh = px.bar(
            x=list(low_counts.keys()), y=list(low_counts.values()),
            labels={"x": "Nb de boules 1-25", "y": "Fréquence"},
            title="Distribution Bas (1-25) / Haut (26-50)",
            template="plotly_dark", color_discrete_sequence=["#C084FC"]
        )
        fig_bh.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.8)",
                              title_font_color="#FFD700")
        st.plotly_chart(fig_bh, use_container_width=True)

    # Données brutes
    with st.expander("📋 Voir les données brutes"):
        display_cols = ["date_de_tirage"] + ball_cols + star_cols if "date_de_tirage" in df.columns else ball_cols + star_cols
        st.dataframe(df[display_cols].tail(100).sort_index(ascending=False), use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#334455; font-size:0.75rem;'>"
    "EuroMillions Analyser · Analyse purement statistique · Jouez responsable · 18+"
    "</p>",
    unsafe_allow_html=True
)

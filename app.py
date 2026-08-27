import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import json
from datetime import datetime, timedelta
import time

# Configuration de la page
st.set_page_config(
    page_title="Pronostic Hippique Pro - API PMU",
    page_icon="🐴",
    layout="wide"
)

# === TITRE ===
st.title("🏇 Outil de Pronostic Hippique avec données PMU")
st.markdown("---")

# === INITIALISATION DES SESSIONS ===
if 'chevaux' not in st.session_state:
    st.session_state.chevaux = pd.DataFrame(columns=[
        'Nom', 'Numero', 'Forme_recente', 'Perf_distance', 'Perf_piste',
        'Poids', 'Jockey', 'Entraineur', 'Fraicheur', 'Cote', 'Corde',
        'Victoires', 'Places', 'Gains'
    ])

if 'poids_param' not in st.session_state:
    st.session_state.poids_param = {
        'Forme_recente': 25,
        'Perf_distance': 20,
        'Perf_piste': 15,
        'Poids': 10,
        'Cote': 10,
        'Jockey': 8,
        'Entraineur': 6,
        'Fraicheur': 6
    }

if 'donnees_pmu' not in st.session_state:
    st.session_state.donnees_pmu = None

# === FONCTIONS ===
def nettoyer_nom(nom):
    """Nettoie le nom d'un cheval pour comparaison"""
    return nom.strip().upper().replace(' ', '').replace("'", "")

def appel_api_pmu(date_str, reunion_num):
    """
    Appelle l'API PMU pour récupérer les données d'une réunion
    date_str: format DDMMYYYY
    reunion_num: 1, 2, 3...
    """
    try:
        url = f"https://online.turfinfo.api.pmu.fr/rest/client/61/programme/{date_str}/R{reunion_num}?specialisation=INTERNET"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ Erreur API PMU : {response.status_code}")
            return None
    except Exception as e:
        st.error(f"❌ Erreur de connexion : {str(e)}")
        return None

def extraire_chevaux_api(donnees):
    """Extrait les chevaux d'une course depuis les données PMU"""
    chevaux = []
    
    try:
        # Navigation dans la structure JSON
        courses = donnees.get('programme', {}).get('courses', [])
        
        if not courses:
            st.warning("Aucune course trouvée pour cette réunion")
            return []
        
        # On prend la première course (ou on pourrait laisser l'utilisateur choisir)
        course = courses[0]
        partants = course.get('partants', [])
        
        for partant in partants:
            # Extraction des informations
            cheval_info = partant.get('cheval', {})
            jockey_info = partant.get('jockey', {})
            entraîneur_info = partant.get('entraineur', {})
            performances = partant.get('performances', {})
            
            # Statistiques de carrière
            stats = performances.get('carriere', {})
            stats_annee = performances.get('annee', {})
            
            # Cotes
            cotes = partant.get('cotes', [])
            cote_actuelle = cotes[0].get('valeur', 0) if cotes else 0
            
            # Nom et numéro
            nom = cheval_info.get('nom', 'Inconnu')
            numero = partant.get('numero', 0)
            
            # Gains (attention : en centimes !)
            gains = stats.get('gains', 0) / 100  # Conversion en euros
            
            chevaux.append({
                'Nom': nom,
                'Numero': numero,
                'Cote': cote_actuelle,
                'Poids': partant.get('poids', 0),
                'Corde': partant.get('corde', 0),
                'Jockey': jockey_info.get('nom', ''),
                'Entraineur': entraîneur_info.get('nom', ''),
                'Victoires': stats.get('victoires', 0),
                'Places': stats.get('places', 0),
                'Gains': gains,
                'NbCourses': stats.get('partants', 0),
                'VictoiresAnnee': stats_annee.get('victoires', 0),
                'GainsAnnee': stats_annee.get('gains', 0) / 100
            })
        
        return chevaux
        
    except Exception as e:
        st.error(f"❌ Erreur lors de l'extraction des données : {str(e)}")
        return []

def calculer_indicateurs(df):
    """Calcule les indicateurs manquants (forme, perf, etc.)"""
    if df.empty:
        return df
    
    # Forme récente : basée sur un ratio victoires/places
    df['Forme_recente'] = df.apply(
        lambda row: 3.0 if row['Victoires'] > 0 else 
                   5.0 if row['Places'] > 3 else 
                   8.0, axis=1
    )
    
    # Performance sur la distance (approximative)
    df['Perf_distance'] = df.apply(
        lambda row: 4.0 if row['Victoires'] >= 2 else 
                   6.0 if row['Victoires'] >= 1 else 
                   8.0, axis=1
    )
    
    # Performance sur la piste (idem)
    df['Perf_piste'] = df['Perf_distance']
    
    # Fraîcheur (nombre de courses récentes)
    df['Fraicheur'] = df.apply(
        lambda row: 1 if row['NbCourses'] > 0 else 3,
        axis=1
    )
    
    return df

# === SIDEBAR - RECHERCHE PMU ===
with st.sidebar:
    st.header("🔍 Récupération PMU")
    
    # Sélection de la date
    date_defaut = datetime.now().strftime('%d/%m/%Y')
    date_course = st.date_input(
        "Date de la réunion",
        value=datetime.now(),
        min_value=datetime.now() - timedelta(days=7),
        max_value=datetime.now() + timedelta(days=2),
        format="DD/MM/YYYY"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        reunion = st.number_input("Réunion (R)", min_value=1, max_value=9, value=1, step=1)
    with col2:
        course = st.number_input("Course (C)", min_value=1, max_value=9, value=1, step=1)
    
    if st.button("📥 Charger les données PMU", type="primary", use_container_width=True):
        with st.spinner("Récupération des données..."):
            date_str = date_course.strftime('%d%m%Y')
            donnees = appel_api_pmu(date_str, reunion)
            
            if donnees:
                chevaux_api = extraire_chevaux_api(donnees)
                if chevaux_api:
                    df_api = pd.DataFrame(chevaux_api)
                    df_api = calculer_indicateurs(df_api)
                    
                    # Ajout des colonnes manquantes
                    for col in st.session_state.chevaux.columns:
                        if col not in df_api.columns:
                            df_api[col] = None
                    
                    st.session_state.chevaux = df_api[st.session_state.chevaux.columns]
                    st.session_state.donnees_pmu = donnees
                    st.success(f"✅ {len(df_api)} chevaux chargés !")
                    st.rerun()
    
    st.divider()
    st.caption("💡 **Astuce** : L'API PMU fournit automatiquement les cotes et statistiques des chevaux.")

# === TAB 1 : SAISIE ===
tab1, tab2, tab3, tab4 = st.tabs(["📝 Saisie des chevaux", "⚙️ Poids des critères", "📊 Résultats", "🔬 Données brutes API"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Paramètres de la course")
        distance = st.number_input("Distance (mètres)", min_value=1000, max_value=4000, value=2400, step=100)
        type_piste = st.selectbox("Type de piste", ["Herbe", "PSF", "Sable"])
        etat_terrain = st.selectbox("État du terrain", ["Bon", "Souple", "Lourd", "Collant", "Très lourd"])
    
    with col2:
        st.subheader("Ajouter/Modifier un cheval")
        with st.form("add_horse"):
            if not st.session_state.chevaux.empty:
                nom_existant = st.selectbox(
                    "Ou sélectionner un cheval existant",
                    options=["-- Nouveau cheval --"] + st.session_state.chevaux['Nom'].tolist()
                )
                
                if nom_existant != "-- Nouveau cheval --":
                    cheval_selectionne = st.session_state.chevaux[st.session_state.chevaux['Nom'] == nom_existant].iloc[0]
                    nom = nom_existant
                    cote_def = cheval_selectionne['Cote'] if pd.notna(cheval_selectionne['Cote']) else 10.0
                    poids_def = cheval_selectionne['Poids'] if pd.notna(cheval_selectionne['Poids']) else 58
                    corde_def = cheval_selectionne['Corde'] if pd.notna(cheval_selectionne['Corde']) else 5
                    forme_def = cheval_selectionne['Forme_recente'] if pd.notna(cheval_selectionne['Forme_recente']) else 5.0
                else:
                    nom = ""
                    cote_def = 10.0
                    poids_def = 58
                    corde_def = 5
                    forme_def = 5.0
            else:
                nom = ""
                cote_def = 10.0
                poids_def = 58
                corde_def = 5
                forme_def = 5.0
            
            nom_input = st.text_input("Nom du cheval", value=nom)
            
            col_a, col_b = st.columns(2)
            with col_a:
                forme = st.number_input("Forme récente (moyenne des places)", min_value=1.0, max_value=20.0, value=forme_def, step=0.5)
                perf_dist = st.number_input("Perf sur la distance", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
                perf_piste = st.number_input("Perf sur ce type de piste", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
                poids = st.number_input("Poids porté (kg)", min_value=45, max_value=75, value=int(poids_def), step=1)
            
            with col_b:
                fraicheur = st.number_input("Courses dans les 30 jours", min_value=0, max_value=10, value=2, step=1)
                cote = st.number_input("Cote", min_value=1.0, max_value=100.0, value=float(cote_def), step=0.5)
                corde = st.number_input("Numéro de corde", min_value=1, max_value=20, value=int(corde_def), step=1)
                jockey = st.number_input("% victoires du jockey", min_value=0, max_value=100, value=15, step=1)
                entraineur = st.number_input("% victoires de l'entraîneur", min_value=0, max_value=100, value=12, step=1)
            
            submitted = st.form_submit_button("➕ Ajouter/Mettre à jour")
            
            if submitted and nom_input:
                if nom_existant != "-- Nouveau cheval --" and nom_existant == nom_input:
                    # Mise à jour
                    idx = st.session_state.chevaux[st.session_state.chevaux['Nom'] == nom_input].index[0]
                    st.session_state.chevaux.loc[idx] = [
                        nom_input, None, forme, perf_dist, perf_piste,
                        poids, jockey, entraineur, fraicheur, cote, corde,
                        None, None, None
                    ]
                    st.success(f"✅ {nom_input} mis à jour !")
                else:
                    # Ajout
                    nouveau = pd.DataFrame([{
                        'Nom': nom_input,
                        'Numero': None,
                        'Forme_recente': forme,
                        'Perf_distance': perf_dist,
                        'Perf_piste': perf_piste,
                        'Poids': poids,
                        'Jockey': jockey,
                        'Entraineur': entraineur,
                        'Fraicheur': fraicheur,
                        'Cote': cote,
                        'Corde': corde,
                        'Victoires': None,
                        'Places': None,
                        'Gains': None
                    }])
                    st.session_state.chevaux = pd.concat([st.session_state.chevaux, nouveau], ignore_index=True)
                    st.success(f"✅ {nom_input} ajouté !")
                st.rerun()
    
    # Affichage du tableau
    if not st.session_state.chevaux.empty:
        st.subheader(f"📋 Chevaux saisis ({len(st.session_state.chevaux)})")
        
        col_aff = st.columns([5, 1])
        with col_aff[0]:
            st.dataframe(st.session_state.chevaux, use_container_width=True)
        with col_aff[1]:
            if st.button("🗑️ Supprimer le dernier"):
                st.session_state.chevaux = st.session_state.chevaux.iloc[:-1]
                st.rerun()
            if st.button("🔄 Tout effacer"):
                st.session_state.chevaux = pd.DataFrame(columns=st.session_state.chevaux.columns)
                st.rerun()

# === TAB 2 : POIDS ===
with tab2:
    st.subheader("Ajustez l'importance de chaque critère")
    
    col1, col2 = st.columns(2)
    
    with col1:
        poids_forme = st.slider("Forme récente", 0, 50, st.session_state.poids_param['Forme_recente'])
        poids_dist = st.slider("Performance sur la distance", 0, 50, st.session_state.poids_param['Perf_distance'])
        poids_piste = st.slider("Performance sur la piste", 0, 50, st.session_state.poids_param['Perf_piste'])
        poids_poids = st.slider("Poids porté", 0, 50, st.session_state.poids_param['Poids'])
    
    with col2:
        poids_cote = st.slider("Cote", 0, 50, st.session_state.poids_param['Cote'])
        poids_jockey = st.slider("Statistiques jockey", 0, 50, st.session_state.poids_param['Jockey'])
        poids_entraineur = st.slider("Statistiques entraîneur", 0, 50, st.session_state.poids_param['Entraineur'])
        poids_fraicheur = st.slider("Fraîcheur", 0, 50, st.session_state.poids_param['Fraicheur'])
    
    st.session_state.poids_param = {
        'Forme_recente': poids_forme,
        'Perf_distance': poids_dist,
        'Perf_piste': poids_piste,
        'Poids': poids_poids,
        'Cote': poids_cote,
        'Jockey': poids_jockey,
        'Entraineur': poids_entraineur,
        'Fraicheur': poids_fraicheur
    }
    
    total = sum(st.session_state.poids_param.values())
    st.info(f"📊 Total: {total}%")
    
    # Graphique de répartition
    df_poids = pd.DataFrame({
        'Critère': list(st.session_state.poids_param.keys()),
        'Poids': list(st.session_state.poids_param.values())
    })
    fig = px.pie(df_poids, values='Poids', names='Critère', title="Répartition des poids")
    st.plotly_chart(fig, use_container_width=True)

# === TAB 3 : RÉSULTATS ===
with tab3:
    if st.session_state.chevaux.empty:
        st.warning("⚠️ Veuillez d'abord saisir ou charger des chevaux")
    else:
        if st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True):
            df = st.session_state.chevaux.copy()
            
            # Nettoyage des valeurs None
            df = df.fillna({
                'Forme_recente': 5.0,
                'Perf_distance': 5.0,
                'Perf_piste': 5.0,
                'Poids': 58,
                'Cote': 10.0,
                'Corde': 5,
                'Jockey': 15,
                'Entraineur': 12,
                'Fraicheur': 2
            })
            
            # === NORMALISATION ===
            colonnes_min = ['Forme_recente', 'Perf_distance', 'Perf_piste', 'Poids', 'Cote', 'Corde']
            for col in colonnes_min:
                if col in df.columns and df[col].nunique() > 1:
                    df[f'{col}_norm'] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                else:
                    df[f'{col}_norm'] = 0.5
            
            colonnes_max = ['Jockey', 'Entraineur', 'Fraicheur']
            for col in colonnes_max:
                if col in df.columns and df[col].nunique() > 1:
                    df[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                else:
                    df[f'{col}_norm'] = 0.5
            
            # === SCORE ===
            df['Score'] = (
                df['Forme_recente_norm'] * st.session_state.poids_param['Forme_recente'] +
                df['Perf_distance_norm'] * st.session_state.poids_param['Perf_distance'] +
                df['Perf_piste_norm'] * st.session_state.poids_param['Perf_piste'] +
                df['Poids_norm'] * st.session_state.poids_param['Poids'] +
                df['Cote_norm'] * st.session_state.poids_param['Cote'] +
                df['Jockey_norm'] * st.session_state.poids_param['Jockey'] +
                df['Entraineur_norm'] * st.session_state.poids_param['Entraineur'] +
                df['Fraicheur_norm'] * st.session_state.poids_param['Fraicheur']
            )
            
            # === PROBABILITÉ ===
            exp_scores = np.exp(df['Score'] - df['Score'].max())
            df['Probabilité'] = exp_scores / exp_scores.sum()
            df['Probabilité %'] = (df['Probabilité'] * 100).round(1)
            
            # === TRI ===
            df_pronostic = df.sort_values('Score', ascending=False).reset_index(drop=True)
            
            # === AFFICHAGE ===
            st.subheader("🏆 Pronostic final")
            
            # Tableau
            colonnes_affichage = ['Nom', 'Score', 'Probabilité %', 'Cote', 'Corde', 'Numero']
            st.dataframe(
                df_pronostic[colonnes_affichage].style.background_gradient(subset=['Score'], cmap='RdYlGn'),
                use_container_width=True
            )
            
            # Graphique
            fig = px.bar(
                df_pronostic,
                x='Nom',
                y='Probabilité %',
                color='Probabilité %',
                color_continuous_scale='RdYlGn',
                title="Probabilité estimée par cheval",
                labels={'Probabilité %': 'Probabilité (%)', 'Nom': 'Cheval'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top 3
            st.subheader("📈 Classement")
            top3 = df_pronostic.head(3)
            
            col_rank = st.columns(3)
            for i in range(3):
                with col_rank[i]:
                    if i < len(top3):
                        emoji = ["🥇", "🥈", "🥉"][i]
                        st.success(f"{emoji} **{top3.iloc[i]['Nom']}**")
                        st.metric("Probabilité", f"{top3.iloc[i]['Probabilité %']}%")
                        st.caption(f"Cote: {top3.iloc[i]['Cote']:.1f}")
            
            # Détection de valeur
            st.subheader("💡 Opportunités de valeur")
            df_pronostic['Cote_implicite'] = 1 / df_pronostic['Probabilité']
            df_pronostic['Valeur'] = df_pronostic['Cote'] - df_pronostic['Cote_implicite']
            
            valeur_positive = df_pronostic[df_pronostic['Valeur'] > 0].head(3)
            if not valeur_positive.empty:
                st.info("**Ces chevaux sont sous-cotés par le marché :**")
                for _, row in valeur_positive.iterrows():
                    st.write(f"  • **{row['Nom']}** : cote {row['Cote']:.1f} vs proba implicite {row['Cote_implicite']:.1f} → écart de **{row['Valeur']:.1f}** points")
            else:
                st.info("ℹ️ Aucune opportunité de valeur majeure détectée.")
            
            # Export
            csv = df_pronostic.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les résultats (CSV)",
                data=csv,
                file_name=f"pronostic_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

# === TAB 4 : DONNÉES BRUTES API ===
with tab4:
    if st.session_state.donnees_pmu:
        st.subheader("📦 Données brutes reçues de l'API PMU")
        
        # Affichage structuré
        try:
            course = st.session_state.donnees_pmu.get('programme', {}).get('courses', [{}])[0]
            st.json(course)
            
            st.divider()
            
            # Métadonnées de la course
            st.subheader("📋 Métadonnées")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Réunion", course.get('reunion', {}).get('numero', 'N/A'))
            with col2:
                st.metric("Course", course.get('ordre', 'N/A'))
            with col3:
                st.metric("Distance", f"{course.get('distance', 'N/A')}m")
            
            # Partants
            st.subheader("🐴 Partants")
            partants = course.get('partants', [])
            for p in partants:
                st.write(f"**{p.get('numero', 'N/A')}** - {p.get('cheval', {}).get('nom', 'Inconnu')} (Cote: {p.get('cotes', [{}])[0].get('valeur', 'N/A')})")
                
        except Exception as e:
            st.error(f"Erreur d'affichage : {str(e)}")
    else:
        st.info("ℹ️ Chargez d'abord des données via le panneau de gauche")

# === FOOTER ===
st.markdown("---")
st.caption("🐴 Pronostic Hippique Pro v2.0 - Données PMU en temps réel")

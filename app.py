 

"""
═══════════════════════════════════════════════════════════════════════════════
 QuantTurf Pro v4.4.0 — "BENTER EDITION + DATA-CALIBRATED (recherche 2026)"
═══════════════════════════════════════════════════════════════════════════════
 Améliorations majeures par rapport à v4.3.0 :
 ─────────────────────────────────────────────
 ✅ Modèle Plackett-Luce (Harville) pour ordres d'arrivée exacts
 ✅ Benter Blend (log-log fusion modèle/marché, formule Benter 1994)
 ✅ Débiaisage rigoureux de l'overround (favori-outsider bias correction)
 ✅ Gestion corde TROT AUTOSTART (numéros 4-5-6 favorisés vs 1-2-3 en plat)
 ✅ Shrinkage bayésien sur la musique (régression vers moyenne empirique)
 ✅ État du terrain (bon, souple, lourd) + poids + jours de repos
 ✅ Kelly dynamique (ajusté par incertitude/volatilité)
 ✅ Paris exotiques rigoureux : Couplé / Trio / Quarté+ / Quinté+ ordre & désordre
 ✅ Détection de value avec seuil dynamique selon overround
 ✅ Backtester intégré (mode validation)
 ✅ Architecture modulaire en classes
 ✅ Diagnostic complet (calibration, divergence, edge expected)
 ───────────────── NOUVEAU EN v4.4 (recalibrage documenté) ─────────────────
 🆕 Correction Benter/Henery du biais de la formule de Harville brute pour les
    places 2-3-4-5 (le PL "pur" surestime les favoris en place et sous-estime
    les outsiders — Benter 1994 §"First, Second and Third", Henery 1981)
 🆕 Cotes PMU exotiques recalées sur les taux de redistribution (TRJ) OFFICIELS
    2026 publiés par le PMU, par type de pari et par canal (guichet / en ligne)
 🆕 Facteur "jours de repos" désormais calibré PAR DISCIPLINE sur une étude
    portant sur >1,25M chevaux (Turfmining.fr), au lieu d'une courbe unique
 🆕 Facteur poids modulé par la distance (méthodologie officielle BHA
    "Performance Figures" : le poids pénalise proportionnellement plus sur
    les longues distances que sur les sprints)
 🆕 Shrinkage bayésien appliqué aux coefficients de Platt scaling eux-mêmes
    (le calibrage d'origine reposait sur ~12 courses/discipline, échantillon
    trop restreint pour estimer 2 paramètres sans sur-ajustement massif)
 🆕 Méthode de débiaisage des cotes "Shin (1993)" proposée en option, en
    complément de la méthode "power" (les deux sont documentées dans la
    littérature académique comme alternatives valables)
 🆕 Mise en garde documentée sur le biais de sélection des courses Quinté+
    (courses volontairement choisies pour leur incertitude par le PMU —
    les taux de victoire du favori y sont structurellement plus bas que sur
    l'ensemble des courses françaises)
═══════════════════════════════════════════════════════════════════════════════
Sources scientifiques et empiriques (voir commentaires inline pour le détail) :
- Benter, W. (1994). "Computer Based Horse Race Handicapping and Wagering
  Systems: A Report." In Efficiency of Racetrack Betting Markets.
  Reproduction annotée avec code : Leung & Leung (2023), Acta Machina,
  actamachina.com/posts/annotated-benter-paper
- Harville, D. (1973). "Assigning Probabilities to the Outcomes of
  Multi-Entry Competitions." JASA 68(342).
- Henery, R.J. (1981). "Permutation probabilities as models for horse races."
  JRSS B, 43(1) — dampening des probabilités de place successives.
- Plackett, R. (1975). "The Analysis of Permutations." JRSS C, 24(2).
- Kelly, J.L. (1956). "A New Interpretation of Information Rate." Bell
  System Technical Journal.
- Snowberg, E. & Wolfers, J. (2010). "Explaining the Favorite-Longshot
  Bias: Is it Risk-Love or Misperceptions?" JPE 118(4).
- Shin, H.S. (1993). "Measuring the Incidence of Insider Trading in a
  Market for State-Contingent Claims." Economic Journal 103(420).
- British Horseracing Authority — "Performance Figures" methodology,
  britishhorseracing.com/regulation/performance-figures/
- Turfmining.fr (2019/2020) — analyse de la récence sur >1,25M partants
  (Plat, Trot attelé, Trot monté, Haies, Steeple, Cross), données PMU.
- PMU — Taux de redistribution (TRJ) officiels par pari, guichet et en
  ligne, en vigueur au 1er trimestre 2026 (documents PMU / turf.bzh).
═══════════════════════════════════════════════════════════════════════════════
AVERTISSEMENT MÉTHODOLOGIQUE IMPORTANT (v4.4) :
Les paramètres marqués "CALIBRÉ v4.2 sur 175 quintés 2026" dans ce fichier
proviennent d'un échantillon de courses SUPPORT DU QUINTÉ+. Ces courses sont
sélectionnées par le PMU précisément pour leur caractère ouvert et incertain
(grands pelotons, handicaps, chevaux de niveau proche), ce qui les rend
structurellement moins prévisibles qu'une course PMU "moyenne" (Simple Gagnant
ordinaire, petit peloton, etc.). Les taux de victoire du favori observés sur
cet échantillon (ex. "5.5% favoris gagnent au Plat") ne doivent donc PAS être
extrapolés tels quels à toute course non-Quinté+ : ils décrivent un régime de
course particulier. Ce script reste utilisable pour toute course, mais son
étalonnage de marché (Platt scaling) est le plus fiable sur des courses
Quinté+ ou similaires en profil d'incertitude.
═══════════════════════════════════════════════════════════════════════════════
"""
 
from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from itertools import combinations, permutations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache
import logging
import time
import warnings
 
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
# =============================================================================
# 1.  CONFIGURATION GLOBALE
# =============================================================================
@dataclass
class Config:
    # --- App ---
    APP_VERSION: str = "4.3.0"
    APP_NAME: str = "QuantTurf Pro"
    APP_TAG: str = "Quarté Coverage Edition (20 combinaisons)"
 
    # --- Monte Carlo / Plackett-Luce ---
    MC_ITERATIONS: int = 8000
    TEMPERATURE: float = 1.0
    NOISE_BASE: float = 0.20           # v4.1: +incertitude reconnue
 
    # --- Marché (CALIBRÉ v4.2 sur 175 quintés 2026 - janv-juin) ---
    # Observations dataset réel (NB : échantillon de courses SUPPORT QUINTÉ+,
    # donc structurellement plus ouvertes que la moyenne des courses PMU —
    # voir avertissement méthodologique en tête de fichier) :
    # - Cote médiane gagnant = 8.10€ (Trot 5.55 / Plat 9.30 / Haies 9.70)
    # - Seulement 9.8% des courses gagnées par favori cote<4
    # - 28.7% des gagnants ont cote >= 12€
    # - La synthèse de presse rend +19.4% ROI → BASELINE TO BEAT
    MARKET_WEIGHT: float = 0.55        # ↑ (le marché reste roi)
    # Benter (1994) : c_i ∝ exp(α·f_i + β·π_i) où f=log(proba modèle),
    # π=log(proba marché). Sur son échantillon HKJC 1986-93, Benter estime
    # ces poids par MLE ; leur valeur absolue dépend fortement de la qualité
    # relative du modèle fondamental, donc pas de constante universelle
    # transférable — α/β doivent rester calibrés localement (ici : backtest
    # 175 quintés). Cf. actamachina.com/posts/annotated-benter-paper.
    BENTER_ALPHA: float = 0.50
    BENTER_BETA: float = 1.30
    OVERROUND_CORRECTION: bool = True
    # Méthode de débiaisage des cotes marché : "power" (Shin/Cain/Vaughan
    # Williams-style transform, p ∝ (1/cote)^γ) ou "shin" (Shin 1993, modèle
    # d'insider trading — souvent mieux calibré sur les gros pelotons selon
    # la littérature comparative, ex. Štrumbelj 2014). Les deux sont
    # implémentées ; "power" reste le défaut car déjà intégrée au Benter Blend.
    OVERROUND_METHOD: str = "power"    # "power" | "shin"
 
    # --- Platt Scaling par discipline (calibré sur backtest, PUIS RÉTRÉCI
    # BAYÉSIENNEMENT vers un prior neutre — v4.4) ---
    # ATTENTION STATISTIQUE (déjà signalée dans le script d'origine) : ces
    # coefficients (a, b) étaient estimés sur seulement ~12 courses PAR
    # DISCIPLINE. Avec un tel n, l'erreur-type sur 2 paramètres ajustés est
    # énorme (le calcul brut ci-dessous montre une compression extrême pour
    # le Plat : un cheval à 35% de proba modèle se retrouvait ramené à ~15%
    # après Platt). Par cohérence avec le shrinkage bayésien déjà appliqué
    # à la musique ailleurs dans ce script (§2), on applique le MÊME principe
    # ici : on rétrécit (a, b) vers le prior neutre (a=1, b=0, "on fait
    # confiance au modèle tel quel") avec un poids K équivalent à ~40 courses
    # fantômes. Justification générale : un estimateur à si faible n a une
    # variance bien supérieure à son biais espéré, et le rétrécir vers un
    # prior raisonnable réduit l'erreur quadratique moyenne hors échantillon
    # (principe d'estimation par rétrécissement / empirical Bayes,
    # cf. la même logique que Stein 1956 en plus simple). Les valeurs
    # RAW_* ci-dessous sont les coefficients bruts issus du backtest ;
    # PLATT_* (calculés dans __post_init__) sont leur version rétrécie,
    # effectivement utilisée par le moteur.
    PLATT_SHRINKAGE_N_OBS: float = 12.0   # taille d'échantillon du backtest d'origine
    PLATT_SHRINKAGE_K: float = 40.0       # "courses fantômes" tirant vers (1, 0)
    PLATT_RAW_GLOBAL: Tuple[float, float] = (0.80, -0.40)
    # Plat = imprévisible sur l'échantillon Quinté+ (5.5% favoris gagnent)
    PLATT_RAW_PLAT:   Tuple[float, float] = (0.40, -1.50)
    # Trot = plus prévisible sur l'échantillon Quinté+ (38.2% favoris gagnent)
    PLATT_RAW_TROT:   Tuple[float, float] = (1.30, +0.50)
    # Obstacle = entre les deux (19% favoris gagnent)
    PLATT_RAW_OBSTACLE: Tuple[float, float] = (0.80, -0.50)
    # Champs calculés (rétrécis) — remplis dans __post_init__
    PLATT_GLOBAL: Tuple[float, float] = None
    PLATT_PLAT: Tuple[float, float] = None
    PLATT_TROT: Tuple[float, float] = None
    PLATT_OBSTACLE: Tuple[float, float] = None
    USE_PLATT_CALIBRATION: bool = True
 
    # --- Benter Blend par discipline (calibré) ---
    BENTER_AB_PLAT:     Tuple[float, float] = (0.25, 1.20)  # Plat : marche>modele
    BENTER_AB_TROT:     Tuple[float, float] = (0.55, 1.70)  # Trot : marche tres predictif
    BENTER_AB_OBSTACLE: Tuple[float, float] = (0.40, 1.30)
    USE_DISCIPLINE_BLEND: bool = True
 
    # --- Value / Kelly (CALIBRÉ v4.2 sur 175 quintés) ---
    # Cote médiane gagnant = 8.10€, P25 = 4.33€, P75 = 13.47€
    # → Sweet spot Simple Gagnant : entre P25 et P75 = [4.5, 13]
    VALUE_THRESHOLD: float = 1.20      # ↑ de 1.15 → 1.20 (plus strict)
    VALUE_COTE_MIN: float = 4.5        # ↓ de 5.0 → 4.5 (P25 observé)
    VALUE_COTE_MAX: float = 13.0       # ↑ de 10.0 → 13.0 (P75 observé)
    # Kelly fractionnaire : Benter (1994) lui-même déconseille le Kelly plein
    # ("betting the full amount recommended by the Kelly formula is unwise")
    # et recommande une fraction 1/2 à 1/3 lorsque l'edge est bien estimé sur
    # de gros échantillons (opération HKJC, données sur des milliers de
    # courses). Ici, l'edge du modèle est estimé sur un échantillon
    # nettement plus restreint (175 quintés) : l'incertitude d'estimation
    # est donc plus grande, ce qui justifie de rester EN DEÇÀ de la
    # fourchette 1/2-1/3 de Benter. KELLY_FRACTION=0.20 (~1/5) est donc
    # cohérent avec la littérature, en plus prudent. Voir aussi MacLean,
    # Ziemba & Blazenko (1992) sur le compromis risque/rendement du Kelly
    # fractionnaire.
    KELLY_FRACTION: float = 0.20
    MIN_KELLY_ODDS: float = 4.50
    MAX_KELLY_STAKE: float = 0.025     # ↓ cap plus prudent : 2.5%
    PLACE_ODDS_FACTOR: Dict[str, float] = None
 
    # --- Correction Benter/Henery du biais de la formule de Harville pure
    #     (v4.4) ---
    # Benter (1994, "First, Second, and Third") montre empiriquement (Tables
    # 9-10 de son papier, données HKJC) que la formule de Harville (1973)
    # brute — tirages séquentiels strictement proportionnels aux forces
    # restantes — SURESTIME la probabilité des favoris de finir 2e/3e et
    # SOUS-ESTIME celle des outsiders. Il corrige ce biais en amortissant
    # l'exposant des forces à chaque position successive : force_pos_k =
    # force^exposant_k, avec des exposants estimés par MLE = 1.0 (1re place,
    # non biaisée), 0.81 (2e place), 0.65 (3e place) sur son échantillon.
    # Henery (1981, JRSS B) et Stern (1990) documentent le même phénomène
    # indépendamment. Le modèle Plackett-Luce séquentiel de ce script étant
    # mathématiquement identique au processus générateur de la formule de
    # Harville (tirages proportionnels aux forces restantes), il hérite du
    # MÊME biais — d'où l'intérêt d'appliquer la même correction ici, à la
    # fois dans la simulation Monte-Carlo (plackett_luce_simulate) et dans
    # le calcul exact utilisé pour la couverture Quarté (_pl_prob_top4).
    # Benter n'a estimé que 2 positions (2e, 3e) ; les exposants pour les
    # positions 4 et 5 (nécessaires ici pour Quarté+/Quinté+) ne sont PAS
    # dans la littérature académique connue et sont donc EXTRAPOLÉS par
    # continuité de la décroissance observée (~0.16/rang) : à calibrer sur
    # données réelles françaises si un historique suffisant est disponible.
    HARVILLE_DAMPING: Tuple[float, float, float, float, float] = (
        1.00,   # 1re place — formule non biaisée (proba modèle/marché brute)
        0.81,   # 2e place  — Benter (1994), MLE sur données HKJC
        0.65,   # 3e place  — Benter (1994), MLE sur données HKJC
        0.50,   # 4e place  — EXTRAPOLÉ (non estimé par Benter)
        0.35,   # 5e place  — EXTRAPOLÉ (non estimé par Benter)
    )
    USE_HARVILLE_DAMPING: bool = True
 
    # --- v4.2 : Simple Gagnant settings ---
    SG_USE_PRESSE_AS_BASELINE: bool = True   # +19.4% ROI sur baseline presse
    SG_AVOID_PURE_FAVORITES: bool = True     # éviter cotes <3 en Plat (5.5% win)
    SG_AVOID_EXTREME_OUTSIDERS: bool = True  # éviter cotes >20 (rares)
 
    # --- Empirique (corde, expérience) ---
    EMPIRICAL_WEIGHT: float = 0.25
    USE_EXPERIENCE_FACTOR: bool = True
 
    # --- Shrinkage bayésien ---
    SHRINKAGE_K: float = 4.0           # nb "courses fantômes" vers moyenne
    POPULATION_MEAN_SCORE: float = 4.0 # moyenne pop. des scores musique
    POPULATION_MEAN_WIN: float = 0.10  # 10% victoires moyennes pop.
 
    # --- Paris ---
    RACE_TYPES: List[str] = None
    TRACK_CONDITIONS: List[str] = None
    DEPART_TYPES: List[str] = None
 
    # --- Musique parsing ---
    MUSIC_POSITION_SCORES: Dict[str, float] = None
    MUSIC_RACE_TYPE_WEIGHTS: Dict[str, float] = None
 
    # --- Tables empiriques corde ---
    DRAW_WIN_PROB_PLAT: Dict[int, float] = None
    DRAW_PLACE_PROB_PLAT: Dict[int, float] = None
    DRAW_WIN_PROB_AUTOSTART: Dict[int, float] = None
    DRAW_PLACE_PROB_AUTOSTART: Dict[int, float] = None
 
    def __post_init__(self):
        if self.MUSIC_POSITION_SCORES is None:
            self.MUSIC_POSITION_SCORES = {
                "1": 10.0, "2": 7.5, "3": 5.5, "4": 4.0, "5": 3.0,
                "6": 2.0, "7": 1.5, "8": 1.0, "9": 0.5, "0": 0.2,
                "D": -2.0, "A": -1.5, "T": -1.5, "R": -1.0, "P": 0.3,
            }
        if self.MUSIC_RACE_TYPE_WEIGHTS is None:
            self.MUSIC_RACE_TYPE_WEIGHTS = {
                "a": 1.00, "m": 0.90, "p": 1.00, "h": 0.95,
                "s": 0.90, "c": 0.85, "x": 1.00,
            }
        if self.RACE_TYPES is None:
            self.RACE_TYPES = ["Plat", "Attelé", "Monté", "Haies",
                               "Steeple-chase", "Cross-country"]
        if self.TRACK_CONDITIONS is None:
            self.TRACK_CONDITIONS = ["Bon", "Bon souple", "Souple",
                                    "Très souple", "Collant", "Lourd",
                                    "Très lourd"]
        if self.DEPART_TYPES is None:
            self.DEPART_TYPES = ["Stalles (Plat)", "Autostart (Trot)",
                                "Volte (Trot)", "Élastique (Obstacle)"]
        if self.PLACE_ODDS_FACTOR is None:
            # Rapports cote_placé/cote_gagnant empiriques selon nb partants
            self.PLACE_ODDS_FACTOR = {
                "small": 0.50,   # ≤ 7 partants : place uniquement sur 2 premiers
                "medium": 0.40,  # 8-15
                "large": 0.32,   # ≥16
            }
 
        # --- v4.4 : rétrécissement bayésien des coefficients Platt ---
        # (a_shrunk, b_shrunk) = (n·a_obs + K·1, n·b_obs + K·0) / (n+K)
        # Même formule que le shrinkage de la musique (§2), appliquée ici
        # aux paramètres de calibration eux-mêmes car n=12 courses/discipline
        # est insuffisant pour estimer 2 paramètres sans sur-ajustement.
        n_obs = self.PLATT_SHRINKAGE_N_OBS
        K = self.PLATT_SHRINKAGE_K
 
        def _shrink(raw: Tuple[float, float]) -> Tuple[float, float]:
            a_obs, b_obs = raw
            a_shrunk = (n_obs * a_obs + K * 1.0) / (n_obs + K)
            b_shrunk = (n_obs * b_obs + K * 0.0) / (n_obs + K)
            return (round(a_shrunk, 4), round(b_shrunk, 4))
 
        if self.PLATT_GLOBAL is None:
            self.PLATT_GLOBAL = _shrink(self.PLATT_RAW_GLOBAL)
        if self.PLATT_PLAT is None:
            self.PLATT_PLAT = _shrink(self.PLATT_RAW_PLAT)
        if self.PLATT_TROT is None:
            self.PLATT_TROT = _shrink(self.PLATT_RAW_TROT)
        if self.PLATT_OBSTACLE is None:
            self.PLATT_OBSTACLE = _shrink(self.PLATT_RAW_OBSTACLE)
 
        # --- TABLES EMPIRIQUES BASÉES SUR ÉTUDES PUBLIQUES (Turf.bzh, PMU) ---
        # PLAT : corde 1-4 favorisée, surtout < 1800m. Cohérent avec la
        # littérature générale sur le "draw bias" (avantage du rail sur les
        # sprints, effet qui s'estompe sur les distances longues où les
        # partants ont le temps de se replacer) — cf. draw_factor_v4 plus
        # bas, qui module déjà ce facteur par distance.
        if self.DRAW_WIN_PROB_PLAT is None:
            self.DRAW_WIN_PROB_PLAT = {
                1: 11.8, 2: 11.5, 3: 11.0, 4: 10.5, 5: 9.5,
                6: 8.5, 7: 7.5, 8: 6.5, 9: 5.5, 10: 4.8,
                11: 4.2, 12: 3.6, 13: 3.2, 14: 2.8, 15: 2.5,
                16: 2.2, 17: 1.9, 18: 1.6, 19: 1.3, 20: 1.0,
            }
        if self.DRAW_PLACE_PROB_PLAT is None:
            self.DRAW_PLACE_PROB_PLAT = {
                1: 31.0, 2: 30.0, 3: 29.0, 4: 27.5, 5: 25.0,
                6: 22.5, 7: 20.0, 8: 17.5, 9: 15.5, 10: 14.0,
                11: 12.5, 12: 11.0, 13: 10.0, 14: 9.0, 15: 8.0,
                16: 7.0, 17: 6.0, 18: 5.5, 19: 5.0, 20: 4.5,
            }
        # AUTOSTART (Trot) : numéros 4-5-6 favorisés, 1-2-3 risquent l'enfermement.
        # Confirmé par plusieurs sources indépendantes de statistiques PMU
        # (Turf.bzh "Stat Autostart", Turfomania — étude 15 ans à Vincennes,
        # Betschool, Dataturf) : le numéro 5 ressort systématiquement comme
        # le meilleur numéro isolé, 4 et 6 juste derrière, 1 pénalisé par le
        # risque d'enfermement à la corde malgré la plus courte distance à
        # parcourir, 8-9 pénalisés par l'effort centrifuge à l'extérieur, et
        # toute la 2e ligne (10-20) nettement défavorisée. La forme en
        # cloche centrée sur 4-5-6 ci-dessous reproduit fidèlement ce
        # consensus ; conservée telle quelle (déjà conforme à la littérature).
        if self.DRAW_WIN_PROB_AUTOSTART is None:
            self.DRAW_WIN_PROB_AUTOSTART = {
                1: 9.0,  2: 9.5,  3: 10.0, 4: 11.5, 5: 12.0, 6: 11.0,
                7: 9.5,  8: 8.0,  9: 6.5,  10: 5.0,
                # 2ème ligne (handicap derrière)
                11: 3.5, 12: 2.8, 13: 2.3, 14: 1.9, 15: 1.6,
                16: 1.3, 17: 1.1, 18: 0.9, 19: 0.7, 20: 0.5,
            }
        if self.DRAW_PLACE_PROB_AUTOSTART is None:
            self.DRAW_PLACE_PROB_AUTOSTART = {
                1: 24.0, 2: 25.0, 3: 27.0, 4: 30.0, 5: 30.5, 6: 28.5,
                7: 24.5, 8: 21.0, 9: 18.0, 10: 14.5,
                11: 11.0, 12: 9.0, 13: 7.5, 14: 6.0, 15: 5.0,
                16: 4.2, 17: 3.5, 18: 3.0, 19: 2.5, 20: 2.0,
            }
 
 
CONFIG = Config()
 
 
# =============================================================================
# 2.  PARSING DE LA MUSIQUE (avec shrinkage bayésien)
# =============================================================================
@dataclass
class MusicMetrics:
    score: float
    regularity: float
    races_count: int
    avg_position: float
    best_position: int
    recent_form: float
    trend: float
    is_debutant: bool
    win_ratio: float
    podium_ratio: float
    consistency: float = 0.0
    shrunk_score: float = 0.0    # score après shrinkage bayésien
    shrunk_win_ratio: float = 0.0
 
 
@lru_cache(maxsize=1024)
def parse_music_v4(music_str: str) -> MusicMetrics:
    """
    Parse la musique d'un cheval/driver/entraîneur.
    Format type : '1a2a3a(23)4aDa5a' (chiffres + type de course)
    Applique un shrinkage bayésien vers la moyenne population.
    """
    if (not music_str or
            music_str.strip().upper() in ("", "-", "INEDIT", "INÉDIT", "N/A", "0")):
        return MusicMetrics(
            score=CONFIG.POPULATION_MEAN_SCORE,
            regularity=0.50, races_count=0, avg_position=5.0,
            best_position=10, recent_form=CONFIG.POPULATION_MEAN_SCORE,
            trend=0.0, is_debutant=True,
            win_ratio=CONFIG.POPULATION_MEAN_WIN,
            podium_ratio=0.30,
            shrunk_score=CONFIG.POPULATION_MEAN_SCORE,
            shrunk_win_ratio=CONFIG.POPULATION_MEAN_WIN,
        )
    try:
        clean = re.sub(r"[()\s]", "", music_str.strip().upper())
        tokens = re.findall(r"([0-9DATRP])([AMPHSC]?)", clean)
        if not tokens:
            return parse_music_v4("")
 
        raw_scores, numeric_positions = [], []
        for pos_char, rtype_char in tokens:
            rtype = rtype_char.lower() if rtype_char else "x"
            pos_score = CONFIG.MUSIC_POSITION_SCORES.get(pos_char, 0.3)
            type_weight = CONFIG.MUSIC_RACE_TYPE_WEIGHTS.get(rtype, 1.0)
            raw_scores.append(pos_score * type_weight)
            if pos_char.isdigit():
                numeric_positions.append(int(pos_char) if pos_char != "0" else 10)
 
        n = len(raw_scores)
        raw_scores_arr = np.array(raw_scores)
 
        # --- Decay exponentiel : courses récentes pèsent plus ---
        decay = np.exp(-0.30 * np.arange(n))
        decay /= decay.sum()
        weighted_score = float(np.dot(raw_scores_arr, decay))
 
        # --- Forme récente (3 dernières) ---
        recent_n = min(3, n)
        rd = decay[:recent_n] / decay[:recent_n].sum()
        recent_form = float(np.dot(raw_scores_arr[:recent_n], rd))
 
        # --- Régularité ---
        if len(numeric_positions) >= 2:
            pos_std = float(np.std(numeric_positions))
            regularity = max(0.0, 1.0 - pos_std / 5.0)
        else:
            pos_std = 3.0
            regularity = 0.50
 
        # --- Tendance (forme récente vs ancienne) ---
        if n >= 4:
            recent_avg = np.mean(raw_scores_arr[: n // 2])
            old_avg = np.mean(raw_scores_arr[n // 2:])
            trend = (recent_avg - old_avg) / (abs(old_avg) + 1e-9)
        else:
            trend = 0.0
 
        # --- Ratios ---
        win_count = sum(1 for p in numeric_positions if p == 1)
        podium_count = sum(1 for p in numeric_positions if p <= 3)
        win_ratio = win_count / max(n, 1)
        podium_ratio = podium_count / max(n, 1)
 
        # --- Consistance ---
        consistency = max(0.0, min(1.0, 1.0 - pos_std / 10.0))
 
        # ──────────────────────────────────────────────────────────
        # SHRINKAGE BAYÉSIEN
        # ──────────────────────────────────────────────────────────
        # Formule : score_shrunk = (n*score + K*μ_pop) / (n+K)
        # Plus n est petit, plus on tire vers la moyenne population
        K = CONFIG.SHRINKAGE_K
        shrunk_score = (n * weighted_score + K * CONFIG.POPULATION_MEAN_SCORE) / (n + K)
        shrunk_win = (n * win_ratio + K * CONFIG.POPULATION_MEAN_WIN) / (n + K)
 
        return MusicMetrics(
            score=weighted_score,
            regularity=regularity,
            races_count=n,
            avg_position=float(np.mean(numeric_positions)) if numeric_positions else 5.0,
            best_position=int(min(numeric_positions)) if numeric_positions else 10,
            recent_form=recent_form,
            trend=float(trend),
            is_debutant=False,
            win_ratio=win_ratio,
            podium_ratio=podium_ratio,
            consistency=consistency,
            shrunk_score=float(shrunk_score),
            shrunk_win_ratio=float(shrunk_win),
        )
    except Exception as e:
        logger.warning(f"Music parsing error '{music_str}': {e}")
        return parse_music_v4("")
 
 
# =============================================================================
# 3.  FACTEURS CONTEXTUELS
# =============================================================================
def experience_factor(races_count: int) -> float:
    """Coefficient multiplicateur 0.7-1.2 selon expérience."""
    if not CONFIG.USE_EXPERIENCE_FACTOR:
        return 1.0
    if races_count <= 0:   return 0.70
    if races_count <= 3:   return 0.82
    if races_count <= 10:  return 1.00
    if races_count <= 30:  return 1.10
    return 1.18
 
 
def draw_factor_v4(draw: int, race_type: str, distance: int,
                   depart_type: str = "Stalles (Plat)",
                   track: str = "Bon") -> float:
    """
    Facteur de corde RAFFINÉ — gère plat ET autostart trot.
    Retourne un score [-1.5, +1.5] à fusionner dans le composite.
    """
    if not draw or draw <= 0:
        return 0.0
    draw = min(int(draw), 20)
 
    # ────────── PLAT (stalles) ──────────
    if race_type == "Plat":
        # 1-4 nettement favorisés, 5-7 OK, 8+ pénalisés
        if draw <= 2:    base = 1.0
        elif draw <= 4:  base = 0.7
        elif draw <= 6:  base = 0.3
        elif draw <= 9:  base = -0.2
        elif draw <= 12: base = -0.6
        else:            base = -1.0
 
        # Modulation par distance
        if distance <= 1300:   dist_mult = 1.6   # sprint : corde décisive
        elif distance <= 1600: dist_mult = 1.3
        elif distance <= 2000: dist_mult = 1.0
        elif distance <= 2400: dist_mult = 0.7
        else:                  dist_mult = 0.4
 
        # Modulation terrain : sur terrain lourd, la corde peut devenir piège
        if track in ("Lourd", "Très lourd", "Collant"):
            base *= 0.3  # neutralise quasi l'effet corde
        elif track in ("Souple", "Très souple"):
            base *= 0.7
 
        return base * dist_mult
 
    # ────────── TROT AUTOSTART ──────────
    if depart_type == "Autostart (Trot)" and race_type in ("Attelé", "Monté"):
        # Premier rang (1-10), centre privilégié
        if draw in (4, 5, 6):     base = 0.9
        elif draw in (3, 7):      base = 0.5
        elif draw in (2, 8):      base = 0.2
        elif draw in (1, 9):      base = -0.2
        elif draw == 10:          base = -0.5
        elif draw <= 14:          base = -0.7
        else:                     base = -1.0      # 2e ligne handicap
 
        # Effet réduit sur longues distances
        if distance >= 2700:
            base *= 0.7
        return base
 
    # ────────── OBSTACLE / autres : effet quasi nul ──────────
    return 0.0
 
 
def track_factor(track: str, race_type: str) -> float:
    """Facteur multiplicateur global selon état du terrain (~1.0 neutre)."""
    # Sur terrain lourd, la régularité prime sur la pointe de vitesse
    if track in ("Lourd", "Très lourd"):  return 0.92
    if track == "Collant":                return 0.95
    if track in ("Souple", "Très souple"): return 0.98
    return 1.0
 
 
def weight_factor(weight_kg: float, distance: int = 1600, ref_weight: float = 56.0) -> float:
    """
    Plat uniquement : un cheval avec poids élevé est désavantagé.
 
    v4.4 : coefficient MODULÉ PAR LA DISTANCE, au lieu d'un -2%/kg fixe.
    Source : British Horseracing Authority, méthodologie officielle des
    "Performance Figures" (britishhorseracing.com/regulation/performance-
    figures/), qui publie l'équivalence poids/longueur observée par
    distance : ~3.41 lb/longueur sur 5 furlongs (≈1000m, terrain bon) contre
    ~1.22 lb/longueur sur 1m6f (≈2800m). Autrement dit, il faut BEAUCOUP
    MOINS de poids pour "valoir" une longueur sur une course longue que sur
    un sprint : le poids pénalise donc PROPORTIONNELLEMENT PLUS sur les
    longues distances (fatigue cumulée sur la durée de l'effort), ce que
    confirment aussi des sources de vulgarisation US (strideodds.ai,
    horseracingsense.com : "the weight-to-performance relationship is
    stronger at longer distances than at sprints").
    On interpole linéairement le coefficient %perf/kg entre 1000m (-1.25%/kg)
    et 2800m (-3.50%/kg), avec un point d'ancrage à 1600m (-2.0%/kg) qui
    reproduit exactement le coefficient fixe utilisé dans les versions
    précédentes du script (pas de régression de précision au mile).
    """
    if weight_kg <= 0:
        return 1.0
    delta = weight_kg - ref_weight
    coef_1000, coef_2800 = -0.0125, -0.0350
    d = max(1000, min(int(distance) if distance else 1600, 2800))
    coef = coef_1000 + (coef_2800 - coef_1000) * (d - 1000) / 1800
    return max(0.85, min(1.15, 1.0 + coef * delta))
 
 
def rest_factor(days_since_last_race: int, race_type: str = "Plat") -> float:
    """
    Facteur "jours de repos" (récence), CALIBRÉ PAR DISCIPLINE — v4.4.
 
    Source : Turfmining.fr, "Comment la récence influe-t-elle sur la
    performance du cheval ?" — analyse empirique sur des bases PMU réelles :
    477 343 partants Plat, 575 982 Trot attelé, 97 105 Trot monté,
    63 378 Haies, 38 706 Steeple, 5 438 Cross
    (turfmining.fr/comment-le-repos-influe-t-il-sur-la-performance-du-cheval/).
 
    Constats clés (courbes de performance observée par tranche de récence),
    qui remplacent l'ancienne courbe unique (14-30j optimum, <7j "fatigue"),
    laquelle ne collait en réalité qu'approximativement au Trot :
    - PLAT : les chevaux les plus performants ont une récence de 1 jour à
      2 mois ; au-delà, légère baisse mais SANS décrochage franc. Il n'y a
      donc PAS de pénalité "fatigue" pour un cheval qui repart vite au Plat,
      contrairement à l'idée reçue reprise dans les versions précédentes.
    - TROT (attelé/monté) : résultat contre-intuitif — les chevaux à
      récence < 1 semaine ne sont PAS les plus performants. Le pic se situe
      entre 1 semaine et 1 mois, avec déclin ensuite, et un regain net pour
      les rentrées après plus d'un an d'arrêt (effet "fraîcheur").
    - HAIES : pic entre 2 semaines et 1 mois, déclin léger et progressif.
    - STEEPLE : tolère un repos plus long, la performance reste maximale
      jusqu'à 3 mois de récence.
    - CROSS-COUNTRY : récence quasi sans impact observable sur l'échantillon
      (n=5438, à interpréter avec prudence — faible taille d'échantillon).
    """
    d = days_since_last_race
    if d < 0:
        return 1.0  # inconnu
 
    if race_type == "Plat":
        if d <= 60:   return 1.00
        if d <= 180:  return 0.95
        return 0.90
 
    if race_type in ("Attelé", "Monté"):
        if d < 7:     return 0.93   # pas optimal, contrairement à l'intuition
        if d <= 30:   return 1.00   # pic observé (1 semaine à 1 mois)
        if d <= 60:   return 0.95
        if d <= 180:  return 0.90
        if d <= 365:  return 0.85
        return 0.90                  # regain après une longue coupure (>1 an)
 
    if race_type == "Haies":
        if d <= 30:   return 1.00
        if d <= 60:   return 0.96
        if d <= 120:  return 0.90
        return 0.85
 
    if race_type == "Steeple-chase":
        if d <= 90:   return 1.00
        if d <= 180:  return 0.93
        return 0.85
 
    # Cross-country (et types non listés) : impact quasi nul dans l'étude
    return 1.0
 
 
# =============================================================================
# 4.  SCORE COMPOSITE (entrée du modèle softmax)
# =============================================================================
def get_weights_v4(race_type: str) -> Dict[str, float]:
    """Poids normalisés par discipline. Total ≈ 1.0."""
    if race_type == "Plat":
        return {
            # Cheval (45%)
            "horse_score": 0.22, "horse_form": 0.10, "horse_regularity": 0.05,
            "horse_trend": 0.04, "horse_win": 0.04,
            # Jockey (20%)
            "driver_score": 0.10, "driver_form": 0.05, "driver_win": 0.05,
            # Entraîneur (15%)
            "trainer_score": 0.08, "trainer_form": 0.04, "trainer_win": 0.03,
            # Corde + contexte (20%)
            "draw_factor": 0.12, "synergy": 0.03, "weight_adj": 0.03, "rest_adj": 0.02,
        }
    elif race_type in ("Attelé", "Monté"):
        return {
            # Cheval (35%)
            "horse_score": 0.18, "horse_form": 0.08, "horse_regularity": 0.04,
            "horse_trend": 0.03, "horse_win": 0.02,
            # Driver/jockey (32%) — TRÈS important au trot
            "driver_score": 0.16, "driver_form": 0.09, "driver_win": 0.07,
            # Entraîneur (18%)
            "trainer_score": 0.10, "trainer_form": 0.05, "trainer_win": 0.03,
            # Corde autostart + contexte (15%)
            "draw_factor": 0.08, "synergy": 0.03, "weight_adj": 0.00, "rest_adj": 0.04,
        }
    else:  # Obstacle (Haies, Steeple, Cross)
        return {
            "horse_score": 0.24, "horse_form": 0.12, "horse_regularity": 0.06,
            "horse_trend": 0.04, "horse_win": 0.03,
            "driver_score": 0.12, "driver_form": 0.06, "driver_win": 0.04,
            "trainer_score": 0.12, "trainer_form": 0.06, "trainer_win": 0.04,
            "draw_factor": 0.00, "synergy": 0.03, "weight_adj": 0.02, "rest_adj": 0.02,
        }
 
 
def composite_score_v4(feat: Dict, weights: Dict) -> float:
    """Score linéaire pondéré. Sera ensuite passé en softmax."""
    s = 0.0
 
    s += weights["horse_score"]      * np.clip(feat["horse_score"], 0, 12)
    s += weights["horse_form"]       * np.clip(feat["horse_form"], 0, 12)
    s += weights["horse_regularity"] * np.clip(feat["horse_regularity"], 0, 1) * 10
    s += weights["horse_trend"]      * (np.clip(feat["horse_trend"], -1, 1) + 1) * 5
    s += weights["horse_win"]        * np.clip(feat["horse_win"], 0, 1) * 20
 
    s += weights["driver_score"] * np.clip(feat["driver_score"], 0, 12)
    s += weights["driver_form"]  * np.clip(feat["driver_form"], 0, 12)
    s += weights["driver_win"]   * np.clip(feat["driver_win"], 0, 1) * 20
 
    s += weights["trainer_score"] * np.clip(feat["trainer_score"], 0, 12)
    s += weights["trainer_form"]  * np.clip(feat["trainer_form"], 0, 12)
    s += weights["trainer_win"]   * np.clip(feat["trainer_win"], 0, 1) * 20
 
    # Corde
    if weights.get("draw_factor", 0) > 0:
        s += weights["draw_factor"] * feat.get("draw_factor", 0) * 5
 
    # Synergie cheval/jockey/entraîneur
    h = np.clip(feat["horse_score"], 0.1, 12)
    d = np.clip(feat["driver_score"], 0.1, 12)
    t = np.clip(feat["trainer_score"], 0.1, 12)
    syn = min(h, d, t) / max(h, d, t)
    s += weights.get("synergy", 0) * syn * 10
 
    # Ajustements multiplicatifs
    s += weights.get("weight_adj", 0) * (feat.get("weight_factor", 1.0) - 1.0) * 50
    s += weights.get("rest_adj",   0) * (feat.get("rest_factor",   1.0) - 1.0) * 50
 
    # Bruit minimal pour briser les égalités
    return max(0.05, s)
 
 
# =============================================================================
# 5.  MOTEUR PROBABILISTE — Softmax + Benter Blend + Plackett-Luce
# =============================================================================
def softmax_temp(scores: np.ndarray, T: float = 1.0) -> np.ndarray:
    s = np.asarray(scores, dtype=float) / max(T, 0.05)
    s -= s.max()
    e = np.exp(np.clip(s, -50, 50))
    p = e / (e.sum() + 1e-12)
    return p
 
 
def remove_overround_power(odds: np.ndarray, gamma: float = 1.12) -> np.ndarray:
    """
    Débiaise les cotes par la méthode "power" : normalisation + correction
    favori-outsider bias (favourite-longshot bias). Selon la littérature
    (Snowberg & Wolfers 2010 ; Sobel & Raines 2003 ; Wikipedia "Favourite-
    longshot bias" pour une revue des méthodes Power/Shin/goto_conversion),
    les favoris sont systématiquement sous-cotés et les outsiders sur-cotés
    par le marché. La transformation power (p_true ∝ p_raw^γ) corrige cela.
 
    Calibration de γ : une étude comparative multi-bookmakers (5 opérateurs)
    trouve un exposant optimal γ ∈ [1.06, 1.15] selon l'opérateur, avec une
    moyenne proche de 1.10-1.12 ("Forecast Sports Outcomes under Efficient
    Market Hypothesis", arXiv:2604.17194). La valeur γ=1.12 utilisée ici
    est donc bien à l'intérieur de la fourchette documentée dans la
    littérature — conservée sans changement, mais avec cette source ajoutée
    pour traçabilité (elle n'était auparavant justifiée que par
    "ajusté empiriquement", sans référence).
    """
    eps = 1e-9
    valid = odds > 1.01
    if not valid.any():
        return np.ones(len(odds)) / max(len(odds), 1)
    p_raw = np.where(valid, 1.0 / np.maximum(odds, 1.01), eps)
    if CONFIG.OVERROUND_CORRECTION:
        p_corr = np.power(p_raw, gamma)
        p_corr = p_corr / p_corr.sum()
    else:
        p_corr = p_raw / p_raw.sum()
    return p_corr
 
 
def remove_overround_shin(odds: np.ndarray) -> np.ndarray:
    """
    Débiaise les cotes par la méthode de Shin (1993), alternative à la
    méthode "power" ci-dessus. Shin modélise l'overround comme provenant
    d'une fraction z de parieurs informés ("insiders"), ce qui donne une
    justification théorique (et non purement empirique) au débiaisage.
    Formule (cas à N issues) : on résout z tel que
        Σ_i [ (z² + 4(1-z)·p_i²/S) ** 0.5 - z ] / (2(1-z)) = 1
    où p_i = 1/cote_i (probabilités brutes, overround inclus) et S = Σ p_i.
    Une fois z estimé, la probabilité débiaisée de l'issue i est :
        p_shin_i = [ (z² + 4(1-z)·p_i²/S) ** 0.5 - z ] / (2(1-z))
    Des études comparatives (ex. Štrumbelj 2014 ; Deschamps & Gergaud 2007)
    trouvent Shin comparable ou légèrement supérieur à la normalisation
    simple / power sur de grands pelotons ; le choix entre les deux méthodes
    reste débattu selon le sport et la taille du peloton. Proposée ici comme
    option (CONFIG.OVERROUND_METHOD = "shin") plutôt qu'en remplacement du
    power method, faute de backtest local tranchant en faveur de l'une ou
    l'autre sur des courses hippiques françaises spécifiquement.
    """
    eps = 1e-9
    valid = odds > 1.01
    n = len(odds)
    if not valid.any():
        return np.ones(n) / max(n, 1)
    p_raw = np.where(valid, 1.0 / np.maximum(odds, 1.01), eps)
    S = p_raw.sum()
    if S <= 1.0 + 1e-6 or n < 2:
        # Pas d'overround détectable ou trop peu d'issues : normalisation simple
        return p_raw / S
 
    def _implied_sum(z: float) -> float:
        z = min(max(z, 1e-6), 1 - 1e-6)
        inner = z ** 2 + 4 * (1 - z) * (p_raw ** 2) / S
        return float(np.sum((np.sqrt(np.maximum(inner, 0)) - z) / (2 * (1 - z))))
 
    # Recherche par bissection de z tel que _implied_sum(z) == 1
    lo, hi = 1e-6, 1 - 1e-6
    f_lo, f_hi = _implied_sum(lo) - 1.0, _implied_sum(hi) - 1.0
    if f_lo * f_hi > 0:
        # Cas dégénéré : repli sur la méthode power par défaut
        return remove_overround_power(odds)
    for _ in range(60):
        mid = (lo + hi) / 2
        f_mid = _implied_sum(mid) - 1.0
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    z = (lo + hi) / 2
    inner = z ** 2 + 4 * (1 - z) * (p_raw ** 2) / S
    p_shin = (np.sqrt(np.maximum(inner, 0)) - z) / (2 * (1 - z))
    p_shin = np.clip(p_shin, eps, None)
    return p_shin / p_shin.sum()
 
 
def remove_overround(odds: np.ndarray) -> np.ndarray:
    """Point d'entrée : sélectionne la méthode de débiaisage via CONFIG."""
    if getattr(CONFIG, "OVERROUND_METHOD", "power") == "shin":
        return remove_overround_shin(odds)
    return remove_overround_power(odds)
 
 
def benter_blend(p_model: np.ndarray, p_market: np.ndarray,
                 alpha: float = None, beta: float = None,
                 race_type: str = None) -> np.ndarray:
    """
    Fusion Benter (1994) : p_final ∝ p_model^α · p_market^β
    v4.1 : exposants par discipline si disponibles.
    """
    if alpha is None or beta is None:
        if CONFIG.USE_DISCIPLINE_BLEND and race_type:
            if race_type == "Plat":
                alpha, beta = CONFIG.BENTER_AB_PLAT
            elif race_type in ("Attelé", "Monté"):
                alpha, beta = CONFIG.BENTER_AB_TROT
            elif race_type in ("Haies", "Steeple-chase", "Cross-country"):
                alpha, beta = CONFIG.BENTER_AB_OBSTACLE
            else:
                alpha = CONFIG.BENTER_ALPHA
                beta = CONFIG.BENTER_BETA
        else:
            if alpha is None: alpha = CONFIG.BENTER_ALPHA
            if beta is None:  beta = CONFIG.BENTER_BETA
    eps = 1e-12
    log_blend = alpha * np.log(p_model + eps) + beta * np.log(p_market + eps)
    log_blend -= log_blend.max()
    p = np.exp(log_blend)
    return p / p.sum()
 
 
def platt_calibrate(probs: np.ndarray, race_type: str = None) -> np.ndarray:
    """
    Platt scaling : p_cal = sigmoid(a * logit(p) + b)
    Paramètres (a, b) calibrés sur 12 courses réelles par discipline.
 
    En Plat (a=0.45, b=-1.30) : forte compression, le modèle est sur-confiant.
    En Trot (a=1.20, b=+0.40) : légère amplification.
    """
    if not CONFIG.USE_PLATT_CALIBRATION:
        return probs
    if race_type == "Plat":
        a, b = CONFIG.PLATT_PLAT
    elif race_type in ("Attelé", "Monté"):
        a, b = CONFIG.PLATT_TROT
    elif race_type in ("Haies", "Steeple-chase", "Cross-country"):
        a, b = CONFIG.PLATT_OBSTACLE
    else:
        a, b = CONFIG.PLATT_GLOBAL
    eps = 1e-9
    p = np.clip(probs, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    p_cal = 1.0 / (1.0 + np.exp(-np.clip(a * logit_p + b, -50, 50)))
    s = p_cal.sum()
    return p_cal / s if s > 0 else probs
 
 
def plackett_luce_simulate(strengths: np.ndarray, n_iter: int,
                            noise: float = 0.18,
                            damping: Tuple[float, ...] = None) -> np.ndarray:
    """
    Simule n_iter ordres d'arrivée par modèle Plackett-Luce (Harville).
 
    v4.4 — CORRECTION BENTER/HENERY DU BIAIS DE HARVILLE :
    Un tirage Plackett-Luce "pur" (Gumbel-max en un seul coup sur toutes
    les positions, comme dans les versions précédentes) est mathématiquement
    IDENTIQUE au processus séquentiel de la formule de Harville (1973) —
    tirer la position 1 proportionnellement aux forces, puis la position 2
    proportionnellement aux forces restantes, etc. Benter (1994) démontre
    empiriquement (Tables 9-10 de son papier) que ce processus SURESTIME la
    probabilité des favoris de finir 2e/3e et SOUS-ESTIME celle des
    outsiders — la formule brute "ne reconnaît pas le caractère de plus en
    plus aléatoire des luttes pour la 2e et la 3e place". Sa correction
    consiste à tirer chaque position successive avec une force amortie
    (force^exposant), les exposants estimés par MLE valant 1.0 / 0.81 / 0.65
    pour les positions 1/2/3 (repris dans CONFIG.HARVILLE_DAMPING). On tire
    donc ici position par position (et non plus en un seul argsort global),
    avec un amortissement croissant sur les positions 2 à 5.
 
    Le bruit gaussien (incertitude du modèle, paramètre `noise`) reste
    appliqué à la composante systématique (log-force) et non à
    l'amortissement — il représente une source d'incertitude distincte
    (imprécision du modèle fondamental) du biais structurel de Harville
    corrigé par le damping.
 
    Seules les CONFIG.HARVILLE_DAMPING premières positions (par défaut 5,
    suffisant pour Quinté+) sont tirées avec précision ; le reste du peloton
    est simplement mélangé aléatoirement car aucun code de ce script
    n'exploite les rangs au-delà de la 5e place.
    """
    if damping is None:
        damping = CONFIG.HARVILLE_DAMPING if CONFIG.USE_HARVILLE_DAMPING else (1.0,) * 5
    n = len(strengths)
    orders = np.zeros((n_iter, n), dtype=np.int32)
    base_log = np.log(np.maximum(strengths, 1e-9))
    n_positions = min(n, len(damping))
    idx_all = np.arange(n)
    for it in range(n_iter):
        mask = np.ones(n, dtype=bool)
        chosen = np.empty(n, dtype=np.int32)
        for pos in range(n_positions):
            damp = damping[pos] if pos < len(damping) else damping[-1]
            avail = idx_all[mask]
            # force amortie (Benter/Henery) + bruit de modèle + Gumbel(0,1)
            damped_log = base_log[avail] * damp
            step_noise = np.random.normal(0, noise, len(avail))
            gumbel = -np.log(-np.log(np.random.uniform(1e-12, 1 - 1e-12, len(avail))))
            scores = damped_log + step_noise + gumbel
            pick_local = int(np.argmax(scores))
            pick = avail[pick_local]
            chosen[pos] = pick
            mask[pick] = False
        # Positions restantes (au-delà de n_positions) : ordre non exploité
        # en aval, mélangé aléatoirement pour ne pas biaiser artificiellement
        if n_positions < n:
            rest = idx_all[mask]
            np.random.shuffle(rest)
            chosen[n_positions:] = rest
        orders[it] = chosen
    return orders
 
 
# =============================================================================
# 6.  CORRECTION EMPIRIQUE (corde + expérience)
# =============================================================================
def empirical_win_prob(draw: int, race_type: str, distance: int,
                       depart_type: str) -> float:
    """Probabilité empirique de victoire en fraction [0, 1]."""
    if draw <= 0:
        return 0.10
    draw = min(draw, 20)
    if race_type == "Plat":
        base = CONFIG.DRAW_WIN_PROB_PLAT.get(draw, 2.0) / 100.0
        # Modulation distance
        if distance <= 1300:   m = 1.30
        elif distance <= 1600: m = 1.15
        elif distance <= 2000: m = 1.00
        elif distance <= 2400: m = 0.85
        else:                  m = 0.70
        return base * m
    elif depart_type == "Autostart (Trot)":
        base = CONFIG.DRAW_WIN_PROB_AUTOSTART.get(draw, 2.0) / 100.0
        return base
    return 0.10
 
 
def empirical_correction(p_model: np.ndarray, draws: List[int],
                         race_type: str, distance: int, depart_type: str,
                         exp_factors: np.ndarray, weight: float = None) -> np.ndarray:
    """
    Mélange convexe entre proba modèle et proba empirique pondérée par expérience.
    """
    if weight is None:
        weight = CONFIG.EMPIRICAL_WEIGHT
    n = len(p_model)
    p_emp = np.zeros(n)
    for i, d in enumerate(draws):
        p_emp[i] = empirical_win_prob(d, race_type, distance, depart_type) * exp_factors[i]
    if p_emp.sum() < 1e-9:
        return p_model
    p_emp /= p_emp.sum()
    p_blend = (1 - weight) * p_model + weight * p_emp
    return p_blend / p_blend.sum()
 
 
# =============================================================================
# 7.  KELLY & VALUE
# =============================================================================
def kelly_bet(prob: float, odds: float, volatility: float = 1.0,
              fraction: float = None) -> Tuple[float, float]:
    """
    Kelly fractionnaire dynamique :
    - Réduit la mise si volatilité élevée
    - Cap absolu à CONFIG.MAX_KELLY_STAKE
    Retourne (kelly_pur, kelly_recommandé).
    """
    if fraction is None:
        fraction = CONFIG.KELLY_FRACTION
    if odds <= CONFIG.MIN_KELLY_ODDS or prob < 0.04:
        return 0.0, 0.0
    b = odds - 1
    q = 1 - prob
    if b <= 0:
        return 0.0, 0.0
    k = (prob * b - q) / b
    k = max(0.0, k)
    # Ajustement volatilité
    vol_adj = 1.0 / (1.0 + max(0, volatility - 1.0))
    k_reco = min(k * fraction * vol_adj, CONFIG.MAX_KELLY_STAKE)
    return float(k), float(k_reco)
 
 
def expected_roi(prob: float, odds: float, stake: float = 100.0) -> float:
    if stake <= 0 or odds <= 1.0:
        return 0.0
    ev = stake * (odds * prob - 1.0)
    return (ev / stake) * 100
 
 
# =============================================================================
# 8.  PARIS EXOTIQUES (via Plackett-Luce simulations)
# =============================================================================
# ──────────────────────────────────────────────────────────────────────────
# COTES PMU RÉALISTES — calibration sur les TRJ (taux de retour aux joueurs)
# OFFICIELS publiés par le PMU, en vigueur au 1er trimestre 2026
# ──────────────────────────────────────────────────────────────────────────
# La cote PMU réelle pour un pari combiné est proche de 1/p × TRJ, où TRJ
# (= 1 - DPE, déduction proportionnelle aux enjeux) est FIXÉ RÉGLEMENTAIREMENT
# par type de pari (arrêté du 5 juin 2010 relatif aux prélèvements et
# modalités de calcul des rapports de paris hippiques mutuels, homologué
# ANJ) et publié par le PMU. Contrairement aux valeurs "estimées" des
# versions précédentes (ex. 0.71 pour Quarté+, 0.68 pour Quinté+, qui
# n'étaient pas sourcées), les valeurs ci-dessous sont les TRJ RÉELS,
# extraits des barèmes PMU en vigueur ("Taux applicables PMU Point de
# Vente" du 10/02/2026 et "Règlement PMU des Paris en Ligne" du 31/03/2026).
# On retient par défaut le taux GUICHET (point de vente), légèrement plus
# conservateur / universel que le taux en ligne (qui varie encore un peu
# selon que la course est ou non support du e-Quinté+). Les taux en ligne
# sont donnés en commentaire pour ajustement si l'utilisateur joue
# exclusivement sur PMU.fr / l'app PMU PLAY.
PMU_TAKEOUT = {
    # Couplé Gagnant/Placé/Ordre : DPE 26.00% guichet → TRJ 74.00%
    # (en ligne depuis le 01/04/2026 : DPE 18.25% → TRJ 81.75%)
    "couple_gagnant": 0.7400,
    "couple_place":   0.7400,
    # Trio / Trio Ordre : DPE 30.90% guichet → TRJ 69.10%
    # (en ligne : DPE 31.65% → TRJ 68.35%, très proche)
    "trio_ordre":     0.6910,
    "trio_desordre":  0.6910,
    # Quarté+ : DPE 36.70% guichet → TRJ 63.30%
    # (en ligne : DPE 37.30% → TRJ 62.70%)
    "quarte_desordre": 0.6330,
    # Quinté+ : DPE 37.85% guichet → TRJ 62.15%
    # (en ligne : DPE 36.00% → TRJ 64.00%, légèrement meilleur en ligne)
    "quinte_desordre": 0.6215,
}
 
def _pmu_estimated_odds(p: float, bet_type: str,
                        min_odds: float, max_odds: float) -> float:
    """Estime la cote PMU réelle pour un pari combiné."""
    if p <= 0:
        return max_odds
    payout_rate = PMU_TAKEOUT.get(bet_type, 0.72)
    raw = (1.0 / p) * payout_rate
    return float(np.clip(raw, min_odds, max_odds))
 
 
def analyze_exotics(results: List[Dict], orders: np.ndarray,
                     top_n: int = 10) -> Dict[str, List[Dict]]:
    """
    Calcule les meilleurs paris exotiques avec cotes PMU réalistes.
    Tri par ROI espéré décroissant.
    """
    n_iter, n_horses = orders.shape
    output = {"couple_gagnant": [], "couple_place": [],
              "trio_ordre": [], "trio_desordre": [],
              "quarte_desordre": [], "quinte_desordre": []}
 
    if n_horses < 3:
        return output
 
    # ──────── COUPLÉ GAGNANT (1-2 ordre exact) ────────
    cg = {}
    for it in range(n_iter):
        key = (int(orders[it, 0]), int(orders[it, 1]))
        cg[key] = cg.get(key, 0) + 1
    for (i, j), c in cg.items():
        p = c / n_iter
        if p < 0.005: continue
        est_odds = _pmu_estimated_odds(p, "couple_gagnant", 3.0, 400.0)
        output["couple_gagnant"].append({
            "combo": f"{results[i]['number']}-{results[j]['number']}",
            "names": f"{results[i]['name'][:8]} → {results[j]['name'][:8]}",
            "prob_pct": round(p * 100, 2),
            "estimated_odds": round(est_odds, 1),
            "expected_roi": round(expected_roi(p, est_odds, 10), 1),
        })
 
    # ──────── COUPLÉ PLACÉ (2 dans top 3, désordre) ────────
    cp = {}
    for it in range(n_iter):
        top3 = sorted(orders[it, :3].tolist())
        for a, b in combinations(top3, 2):
            key = (int(a), int(b))
            cp[key] = cp.get(key, 0) + 1
    for (i, j), c in cp.items():
        p = c / n_iter
        if p < 0.02: continue
        est_odds = _pmu_estimated_odds(p, "couple_place", 1.8, 80.0)
        output["couple_place"].append({
            "combo": f"{results[i]['number']}-{results[j]['number']}",
            "names": f"{results[i]['name'][:8]} & {results[j]['name'][:8]}",
            "prob_pct": round(p * 100, 2),
            "estimated_odds": round(est_odds, 1),
            "expected_roi": round(expected_roi(p, est_odds, 10), 1),
        })
 
    # ──────── TRIO ORDRE ────────
    to_dict = {}
    for it in range(n_iter):
        key = tuple(int(x) for x in orders[it, :3])
        to_dict[key] = to_dict.get(key, 0) + 1
    for key, c in to_dict.items():
        p = c / n_iter
        if p < 0.003: continue
        est_odds = _pmu_estimated_odds(p, "trio_ordre", 10.0, 2000.0)
        i, j, k = key
        output["trio_ordre"].append({
            "combo": f"{results[i]['number']}-{results[j]['number']}-{results[k]['number']}",
            "prob_pct": round(p * 100, 3),
            "estimated_odds": round(est_odds, 1),
            "expected_roi": round(expected_roi(p, est_odds, 10), 1),
        })
 
    # ──────── TRIO DÉSORDRE ────────
    td_dict = {}
    for it in range(n_iter):
        key = tuple(sorted(int(x) for x in orders[it, :3]))
        td_dict[key] = td_dict.get(key, 0) + 1
    for key, c in td_dict.items():
        p = c / n_iter
        if p < 0.01: continue
        est_odds = _pmu_estimated_odds(p, "trio_desordre", 4.0, 500.0)
        i, j, k = key
        output["trio_desordre"].append({
            "combo": f"{results[i]['number']}-{results[j]['number']}-{results[k]['number']}",
            "prob_pct": round(p * 100, 2),
            "estimated_odds": round(est_odds, 1),
            "expected_roi": round(expected_roi(p, est_odds, 10), 1),
        })
 
    # ──────── QUARTÉ+ DÉSORDRE ────────
    if n_horses >= 4:
        q4 = {}
        for it in range(n_iter):
            key = tuple(sorted(int(x) for x in orders[it, :4]))
            q4[key] = q4.get(key, 0) + 1
        for key, c in q4.items():
            p = c / n_iter
            if p < 0.005: continue
            est_odds = _pmu_estimated_odds(p, "quarte_desordre", 12.0, 5000.0)
            output["quarte_desordre"].append({
                "combo": "-".join(str(results[i]['number']) for i in key),
                "prob_pct": round(p * 100, 3),
                "estimated_odds": round(est_odds, 1),
                "expected_roi": round(expected_roi(p, est_odds, 5), 1),
            })
 
    # ──────── QUINTÉ+ DÉSORDRE ────────
    if n_horses >= 5:
        q5 = {}
        for it in range(n_iter):
            key = tuple(sorted(int(x) for x in orders[it, :5]))
            q5[key] = q5.get(key, 0) + 1
        for key, c in q5.items():
            p = c / n_iter
            if p < 0.002: continue
            est_odds = _pmu_estimated_odds(p, "quinte_desordre", 25.0, 30000.0)
            output["quinte_desordre"].append({
                "combo": "-".join(str(results[i]['number']) for i in key),
                "prob_pct": round(p * 100, 4),
                "estimated_odds": round(est_odds, 1),
                "expected_roi": round(expected_roi(p, est_odds, 2), 1),
            })
 
    # Tri : (a) ROI positifs d'abord, (b) puis par probabilité décroissante
    # Cela évite d'afficher des combos peu probables même si ROI équivalent
    for k in output:
        # Plafonner les ROI affichés pour rester réaliste (max +300%)
        for r in output[k]:
            if r["expected_roi"] > 300:
                r["expected_roi_raw"] = r["expected_roi"]
                r["expected_roi"] = 300.0
                r["flag"] = "⚠️ ROI très élevé (cap +300%)"
        output[k].sort(
            key=lambda x: (x["expected_roi"], x["prob_pct"]),
            reverse=True
        )
        output[k] = output[k][:top_n]
        for i, r in enumerate(output[k]):
            r["rank"] = i + 1
    return output
 
 
# ==========================================================================
# QUARTÉ COVERAGE — v4.3 : génération 20 combos couvrant seuil de proba
# ==========================================================================
def _pl_prob_top4(strengths: np.ndarray, top4_idx: Tuple[int, int, int, int],
                   damping: Tuple[float, ...] = None) -> float:
    """
    Probabilité EXACTE d'un top-4 dans l'ordre selon Plackett-Luce, avec la
    même correction Benter/Henery (v4.4) que plackett_luce_simulate — voir
    la docstring de cette dernière pour la justification. Sans cette
    correction, cette fonction fermée était incohérente avec la simulation
    Monte-Carlo utilisée ailleurs dans le moteur (deux modèles de
    probabilité différents pour la même quantité), ce qui est corrigé ici.
    """
    if damping is None:
        damping = CONFIG.HARVILLE_DAMPING if CONFIG.USE_HARVILLE_DAMPING else (1.0,) * 4
    s = strengths
    if s.sum() <= 0:
        return 0.0
    p = 1.0
    used_mask = np.zeros(len(s), dtype=bool)
    for pos, idx in enumerate(top4_idx):
        damp = damping[pos] if pos < len(damping) else damping[-1]
        avail = ~used_mask
        damped = np.power(np.maximum(s[avail], 1e-12), damp)
        remaining_total = damped.sum()
        if remaining_total <= 0:
            return 0.0
        s_idx_damped = np.power(max(s[idx], 1e-12), damp)
        p *= s_idx_damped / remaining_total
        used_mask[idx] = True
    return p
 
 
def _pl_prob_top4_unordered(strengths: np.ndarray, combo: Tuple[int, int, int, int]) -> float:
    """Probabilité que 4 chevaux (peu importe l'ordre) soient exactement le top-4."""
    return sum(_pl_prob_top4(strengths, perm) for perm in permutations(combo))
 
 
def generate_quarte_coverage(
    results: List[Dict],
    strengths: np.ndarray,
    n_combos: int = 20,
    coverage_target: float = 0.50,
    mode: str = "coverage"
) -> Dict[str, Any]:
    """
    Génère N combinaisons Quarté (désordre) intelligentes.
 
    Modes :
    - 'coverage'   : sélectionne les N combos qui maximisent la couverture cumulée.
                     Retourne aussi la proba cumulée = "chance que le vrai quarté
                     soit parmi nos N tickets".
    - 'top_horses' : identifie d'abord les chevaux à >= coverage_target% de chance
                     d'être dans le top 4, puis génère N combos parmi eux.
    """
    n = len(strengths)
    if n < 4:
        return {"combos": [], "total_coverage_pct": 0, "mode": mode,
                "error": "moins de 4 partants"}
 
    total_s = strengths.sum()
    if total_s <= 0:
        return {"combos": [], "total_coverage_pct": 0, "mode": mode,
                "error": "forces nulles"}
 
    # 1) Proba marginale P(cheval i ∈ top 4)
    # Formule Harville exacte : 1 - Π_k (1 - s_i / (Σs - déjà pris))
    # On calcule via simulation partielle exacte pour top 4
    p_top4_marginal = np.zeros(n)
    for i in range(n):
        # Exact : proba d'être dans les 4 premiers = somme sur toutes les positions 1..4
        # de la proba d'y arriver. Formule fermée : intégration compliquée.
        # Approximation via 1 - (1 - s_i/S) itératif sur 4 positions :
        p_not_yet = 1.0
        s_remaining = total_s
        p_in_top4 = 0.0
        for pos in range(4):
            # P(sortir à cette position | pas encore sorti)
            p_at_pos = strengths[i] / s_remaining if s_remaining > 0 else 0
            p_in_top4 += p_not_yet * p_at_pos
            p_not_yet *= (1 - p_at_pos)
            # Réduction moyenne du dénominateur (approximation)
            s_remaining -= (total_s - strengths[i]) / (n - 1) if n > 1 else 0
        p_top4_marginal[i] = min(1.0, p_in_top4)
 
    # 2) Génération des combinaisons candidates
    # On limite au pool des 12 meilleurs pour rester tractable (C(12,4)=495)
    ranking = np.argsort(-strengths)
    n_pool = min(12, n)
    pool = ranking[:n_pool].tolist()
 
    all_combos = []
    for combo in combinations(pool, 4):
        p = _pl_prob_top4_unordered(strengths, combo)
        all_combos.append({"indices": combo, "prob": p})
    all_combos.sort(key=lambda x: -x["prob"])
 
    # 3) Sélection selon mode
    if mode == "top_horses":
        strong = [i for i in range(n) if p_top4_marginal[i] >= coverage_target]
        if len(strong) < 4:
            # Fallback : les 6 meilleurs
            strong = ranking[:6].tolist()
        selected = [c for c in all_combos if all(idx in strong for idx in c["indices"])]
        selected = selected[:n_combos]
    else:  # 'coverage'
        selected = all_combos[:n_combos]
 
    # 4) Couverture cumulée
    cum = 0.0
    for c in selected:
        cum += c["prob"]
        c["cum_prob"] = cum
 
    # 5) Format lisible
    combos_out = []
    for i, c in enumerate(selected):
        nums = sorted(results[idx]["number"] for idx in c["indices"])
        names = tuple(results[idx]["name"][:12] for idx in c["indices"])
        combos_out.append({
            "rank": i + 1,
            "numbers": tuple(nums),
            "combo": "-".join(str(n) for n in nums),
            "names": names,
            "prob_pct": round(c["prob"] * 100, 3),
            "cum_prob_pct": round(c["cum_prob"] * 100, 2),
        })
 
    # 6) Estimation ROI (Quarté+ base 1.30€)
    stake_per_combo = 1.30
    total_stake = len(combos_out) * stake_per_combo
    # Cote Quarté+ typique : (1/proba_gagnante) × TRJ officiel Quarté+
    # (v4.4 : réutilise PMU_TAKEOUT["quarte_desordre"] = 0.6330, au lieu
    # d'une constante 0.71 non sourcée — cohérence avec analyze_exotics)
    # ROI = cum × cote_moy - 1
    if cum > 0 and len(combos_out) > 0:
        p_moy = cum / len(combos_out)
        cote_moy = PMU_TAKEOUT["quarte_desordre"] / p_moy
        expected_win = cum * cote_moy * stake_per_combo
        roi_pct = ((expected_win - total_stake) / total_stake) * 100
    else:
        cote_moy = 0
        roi_pct = -100
 
    return {
        "combos": combos_out,
        "total_coverage_pct": round(cum * 100, 2),
        "coverage_target_pct": round(coverage_target * 100, 1),
        "n_combos": len(combos_out),
        "total_stake_eur": round(total_stake, 2),
        "avg_estimated_odds": round(cote_moy, 1),
        "roi_estimate_pct": round(roi_pct, 1),
        "mode": mode,
        "strong_horses_count": int(sum(1 for p in p_top4_marginal
                                        if p >= coverage_target)),
        "p_top4_marginal": {results[i]["number"]: round(float(p) * 100, 1)
                             for i, p in enumerate(p_top4_marginal)},
    }
 
 
def best_place_bet(results: List[Dict], n_runners: int) -> Optional[Dict]:
    """Trouve le meilleur cheval pour le pari Placé."""
    if n_runners <= 4:
        place_factor = CONFIG.PLACE_ODDS_FACTOR["small"]
    elif n_runners <= 7:
        place_factor = 0.45
    elif n_runners <= 15:
        place_factor = CONFIG.PLACE_ODDS_FACTOR["medium"]
    else:
        place_factor = CONFIG.PLACE_ODDS_FACTOR["large"]
 
    best = None
    best_roi = -np.inf
    for r in results:
        pp = r["place_prob"] / 100
        if pp < 0.12: continue
        wo = r["odds"]
        if wo < 1.5: continue
        place_odds = max(1.20, wo * place_factor)
        roi = expected_roi(pp, place_odds, 100)
        if roi > best_roi:
            best_roi = roi
            k_pur, k_reco = kelly_bet(pp, place_odds, volatility=1.0)
            best = {
                "number": r["number"],
                "name": r["name"],
                "win_prob": r["win_prob"],
                "place_prob": r["place_prob"],
                "estimated_place_odds": round(place_odds, 2),
                "expected_roi_place": round(roi, 1),
                "kelly_pure": round(k_pur, 4),
                "kelly_recommended": round(k_reco, 4),
            }
    return best
 
 
# =============================================================================
# 9.  MOTEUR PRINCIPAL — RaceEngine v4
# =============================================================================
class RaceEngine:
    """Encapsule toute la logique de prédiction pour une course."""
 
    def __init__(self, race_info: Dict, horses: List[Dict]):
        self.race_info = race_info
        self.horses = horses
        self.n = len(horses)
        self.race_type = race_info.get("race_type", "Plat")
        self.distance = int(race_info.get("distance", 1600))
        self.track = race_info.get("track", "Bon")
        self.depart_type = race_info.get("depart_type", "Stalles (Plat)")
 
    # ── 9.1 Préparation des features ───────────────────────────────────
    def _build_features(self) -> Tuple[List[Dict], List[int], np.ndarray]:
        feats, draws, exp_factors = [], [], []
        for h in self.horses:
            m_h = parse_music_v4(h.get("horse_music", ""))
            m_d = parse_music_v4(h.get("driver_music", ""))
            m_t = parse_music_v4(h.get("trainer_music", ""))
 
            exp_h = experience_factor(m_h.races_count)
            exp_d = experience_factor(m_d.races_count)
            exp_t = experience_factor(m_t.races_count)
            combined_exp = (exp_h * exp_d * exp_t) ** (1/3)
            exp_factors.append(combined_exp)
 
            draw = h.get("draw", 0)
            draws.append(draw)
 
            df = draw_factor_v4(draw, self.race_type, self.distance,
                                 self.depart_type, self.track)
            wf = (weight_factor(h.get("weight", 0), self.distance)
                  if self.race_type == "Plat" else 1.0)
            rf = rest_factor(h.get("days_rest", -1), self.race_type)
            tf = track_factor(self.track, self.race_type)
 
            # On utilise les scores SHRUNK (régression vers moyenne)
            feats.append({
                "number": h.get("number", 0),
                "name": h.get("name", ""),
                "odds": float(h.get("odds", 0)),
                "horse_score": m_h.shrunk_score * exp_h * tf,
                "horse_form": m_h.recent_form,
                "horse_regularity": m_h.regularity,
                "horse_trend": m_h.trend,
                "horse_win": m_h.shrunk_win_ratio,
                "horse_is_debutant": m_h.is_debutant,
                "driver_score": m_d.shrunk_score * exp_d,
                "driver_form": m_d.recent_form,
                "driver_win": m_d.shrunk_win_ratio,
                "trainer_score": m_t.shrunk_score * exp_t,
                "trainer_form": m_t.recent_form,
                "trainer_win": m_t.shrunk_win_ratio,
                "draw_factor": df,
                "weight_factor": wf,
                "rest_factor": rf,
            })
        return feats, draws, np.array(exp_factors)
 
    # ── 9.2 Prédiction complète ────────────────────────────────────────
    def predict(self, mc_iter: int = None, market_weight: float = None,
                value_threshold: float = None) -> Dict[str, Any]:
        t0 = time.time()
        if mc_iter is None:        mc_iter = CONFIG.MC_ITERATIONS
        if market_weight is None:  market_weight = CONFIG.MARKET_WEIGHT
        if value_threshold is None: value_threshold = CONFIG.VALUE_THRESHOLD
 
        feats, draws, exp_factors = self._build_features()
        weights = get_weights_v4(self.race_type)
        scores = np.array([composite_score_v4(f, weights) for f in feats])
        if scores.std() < 1e-6:
            scores += np.random.normal(0, 0.05, self.n)
 
        # === ÉTAPE 1 : Probabilité modèle pure (softmax) ===
        p_model_raw = softmax_temp(scores, T=CONFIG.TEMPERATURE)
 
        # === ÉTAPE 2 : Correction empirique (corde + expérience) ===
        p_model = empirical_correction(p_model_raw, draws, self.race_type,
                                         self.distance, self.depart_type,
                                         exp_factors)
 
        # === ÉTAPE 3 : Marché débiaisé ===
        odds_arr = np.array([f["odds"] for f in feats])
        has_market = (odds_arr > 1.5).sum() >= self.n * 0.5
        if has_market:
            p_market = remove_overround(odds_arr)
        else:
            p_market = np.ones(self.n) / self.n
 
        # === ÉTAPE 3.5 (v4.1) : Platt scaling du modèle ===
        p_model = platt_calibrate(p_model, race_type=self.race_type)
 
        # === ÉTAPE 4 : Benter Blend (v4.1 : discipline-aware) ===
        if has_market and market_weight > 0:
            p_final = benter_blend(p_model, p_market,
                                    race_type=self.race_type)
            if abs(market_weight - 0.50) > 0.05:
                p_final = (1 - market_weight) * p_model + market_weight * p_final
                p_final /= p_final.sum()
        else:
            p_final = p_model
 
        # === ÉTAPE 5 : Simulation Plackett-Luce pour exotiques + place ===
        # On reconstruit des forces compatibles avec p_final
        strengths = p_final * 100  # échelle arbitraire
        orders = plackett_luce_simulate(strengths, mc_iter, noise=CONFIG.NOISE_BASE)
 
        # Probabilités de place (top 3) via PL
        place_counts = np.zeros(self.n)
        win_counts = np.zeros(self.n)
        for it in range(mc_iter):
            win_counts[orders[it, 0]] += 1
            for k in range(3):
                place_counts[orders[it, k]] += 1
        p_place_mc = place_counts / mc_iter
        p_win_mc = win_counts / mc_iter
 
        # Volatilité : écart entre p_final et p_win_mc
        volatility = np.abs(p_final - p_win_mc) / (p_final + 1e-9)
 
        # === ÉTAPE 6 : Construction des résultats ===
        results = []
        # Overround
        if has_market:
            raw_or = sum(1.0 / o for o in odds_arr if o > 1.01)
            overround_pct = round((raw_or - 1.0) * 100, 1)
        else:
            overround_pct = None
 
        # Seuil de value dynamique
        if overround_pct is not None and overround_pct > 0:
            dyn_value_th = max(value_threshold, 1.0 + overround_pct / 100 * 1.2)
        else:
            dyn_value_th = value_threshold
 
        for i, (feat, horse) in enumerate(zip(feats, self.horses)):
            ratio = p_final[i] / (p_market[i] + 1e-9)
            cote = horse.get("odds", 2.0)
            # v4.1 : filtre value bet par cote (sweet spot [5, 10])
            is_value = (
                ratio >= dyn_value_th
                and p_final[i] >= 0.04
                and CONFIG.VALUE_COTE_MIN <= cote <= CONFIG.VALUE_COTE_MAX
            )
            k_pur, k_reco = kelly_bet(p_final[i], cote,
                                       volatility=1 + volatility[i])
            roi = expected_roi(p_final[i], cote)
 
            results.append({
                "rank": 0,
                "number": horse.get("number", i + 1),
                "name": horse.get("name", f"Cheval {i+1}"),
                "odds": float(horse.get("odds", 0)),
                "win_prob": round(float(p_final[i]) * 100, 2),
                "win_prob_model": round(float(p_model[i]) * 100, 2),
                "win_prob_market": round(float(p_market[i]) * 100, 2),
                "place_prob": round(float(p_place_mc[i]) * 100, 2),
                "composite_score": round(float(scores[i]), 3),
                "value_ratio": round(float(ratio), 2),
                "is_value_bet": bool(is_value),
                "kelly_pure": round(k_pur, 4),
                "kelly_recommended": round(k_reco, 4),
                "expected_roi": round(roi, 2),
                "volatility": round(float(volatility[i]), 3),
                "draw": draws[i],
                "draw_factor": round(feat["draw_factor"], 3),
            })
 
        results.sort(key=lambda x: x["win_prob"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
 
        # === ÉTAPE 7 : Exotiques + Place + Quarté Coverage ===
        exotics = analyze_exotics(results, orders)
        bp = best_place_bet(results, self.n)
        quarte_coverage_20 = generate_quarte_coverage(
            results, strengths,
            n_combos=20,
            coverage_target=0.50,
            mode="coverage"
        )
        quarte_top_horses_20 = generate_quarte_coverage(
            results, strengths,
            n_combos=20,
            coverage_target=0.50,
            mode="top_horses"
        )
 
        # === Diagnostic ===
        sorted_p = sorted([r["win_prob"] for r in results], reverse=True)
        if len(sorted_p) >= 2:
            gap = sorted_p[0] - sorted_p[1]
            conf_idx = min(100, round(45 + gap * 2.5, 1))
        else:
            conf_idx = 50
        vol_idx = min(100, round(volatility.mean() * 60, 1))
 
        # KL divergence modèle / marché (mesure de désaccord)
        if has_market:
            eps = 1e-12
            kl = float(np.sum(p_final * np.log((p_final + eps) / (p_market + eps))))
        else:
            kl = None
 
        return {
            "results": results,
            "exotics": exotics,
            "best_place": bp,
            "quarte_coverage_20": quarte_coverage_20,
            "quarte_top_horses_20": quarte_top_horses_20,
            "confidence_idx": conf_idx,
            "volatility_idx": vol_idx,
            "overround_pct": overround_pct,
            "dynamic_value_threshold": round(dyn_value_th, 3),
            "kl_divergence": round(kl, 3) if kl else None,
            "execution_time": round(time.time() - t0, 2),
            "n_simulations": mc_iter,
        }
 
 
def run_engine_v4(race_info: Dict, horses: List[Dict], **kwargs) -> Dict:
    """API publique compatible avec l'ancienne v3."""
    engine = RaceEngine(race_info, horses)
    return engine.predict(**kwargs)
 
 
# =============================================================================
# 10.  INTERFACE STREAMLIT
# =============================================================================
def apply_css():
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg,#07071a 0%,#0d1b2a 40%,#12192b 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#0d1b2a,#07071a); }
    h1, h2, h3 { color:#e8e8e8 !important; }
    div[data-testid="metric-container"] {
        background: rgba(0,180,216,0.08);
        border: 1px solid rgba(0,255,136,0.15);
        border-radius: 12px;
        padding: 10px;
    }
    .value-bet { color:#00ff88; font-weight:bold; }
    </style>
    """, unsafe_allow_html=True)
 
 
def render_header():
    st.markdown(f"""
    <div style="text-align:center; padding: 18px 0;">
      <h1 style="font-size:2.6em;
                 background: linear-gradient(90deg,#00ff88,#00b4d8,#7b2ff7);
                 -webkit-background-clip:text;
                 -webkit-text-fill-color:transparent;">
        🏇 {CONFIG.APP_NAME} v{CONFIG.APP_VERSION}
      </h1>
      <p style="color:#7b9ec4; font-size:1.05em; margin-top:-10px;">
        <em>{CONFIG.APP_TAG}</em> — Plackett-Luce · Benter Blend · Kelly dynamique
      </p>
    </div>
    """, unsafe_allow_html=True)
 
 
def init_session_state():
    if "horses_data" not in st.session_state:
        st.session_state.horses_data = pd.DataFrame({
            "N°": list(range(1, 11)),
            "Nom": [f"Cheval {i+1}" for i in range(10)],
            "Cote": [5.0] * 10,
            "Musique Cheval": [""] * 10,
            "Musique Driver": [""] * 10,
            "Musique Entraîneur": [""] * 10,
            "Corde": [0] * 10,
            "Poids": [56.0] * 10,
            "Jours repos": [21] * 10,
        })
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
 
 
def main():
    st.set_page_config(page_title=f"🏇 {CONFIG.APP_NAME} v{CONFIG.APP_VERSION}",
                       layout="wide", initial_sidebar_state="expanded")
    init_session_state()
    apply_css()
    render_header()
 
    # ============= SIDEBAR =============
    with st.sidebar:
        st.markdown("### ⚙️ Paramètres du moteur")
 
        with st.expander("🔬 Monte Carlo / Plackett-Luce", expanded=True):
            mc_iter = st.slider("Itérations PL", 1000, 15000,
                                CONFIG.MC_ITERATIONS, 500)
            noise = st.slider("Bruit log-normal", 0.05, 0.40,
                              CONFIG.NOISE_BASE, 0.01)
            CONFIG.NOISE_BASE = noise
 
        with st.expander("🎯 Marché & Benter Blend", expanded=True):
            mw = st.slider("Poids du marché", 0.0, 0.70,
                           CONFIG.MARKET_WEIGHT, 0.05)
            alpha = st.slider("α (exposant modèle)", 0.5, 2.0,
                              CONFIG.BENTER_ALPHA, 0.05)
            beta = st.slider("β (exposant marché)", 0.0, 2.0,
                             CONFIG.BENTER_BETA, 0.05)
            CONFIG.BENTER_ALPHA = alpha
            CONFIG.BENTER_BETA = beta
            CONFIG.OVERROUND_CORRECTION = st.checkbox(
                "Débiaiser favori/outsider", value=True,
                help="Correction power du biais favori-outsider")
 
        with st.expander("🧠 Empirique & shrinkage"):
            emp_w = st.slider("Poids empirisme (corde+exp.)", 0.0, 0.70,
                               CONFIG.EMPIRICAL_WEIGHT, 0.05)
            CONFIG.EMPIRICAL_WEIGHT = emp_w
            CONFIG.USE_EXPERIENCE_FACTOR = st.checkbox(
                "Facteur expérience", value=CONFIG.USE_EXPERIENCE_FACTOR)
            K = st.slider("Shrinkage K (courses fantômes)", 0.0, 15.0,
                          CONFIG.SHRINKAGE_K, 0.5,
                          help="Plus K est élevé, plus on tire vers la moyenne population")
            CONFIG.SHRINKAGE_K = K
 
        with st.expander("💰 Value & Kelly"):
            vt = st.slider("Seuil de value (ratio)", 1.05, 1.80,
                            CONFIG.VALUE_THRESHOLD, 0.05)
            kf = st.slider("Kelly fractionnaire", 0.05, 0.50,
                            CONFIG.KELLY_FRACTION, 0.05)
            CONFIG.KELLY_FRACTION = kf
            max_stake = st.slider("Cap max bankroll (%)", 1.0, 15.0,
                                  CONFIG.MAX_KELLY_STAKE * 100, 0.5) / 100
            CONFIG.MAX_KELLY_STAKE = max_stake
 
        st.markdown("---")
        st.caption(f"v{CONFIG.APP_VERSION} — {CONFIG.APP_TAG}")
        st.caption("Inspiré de Benter (1994), Harville (1973)")
 
    # ============= TABS =============
    tab1, tab2, tab3 = st.tabs(["📥 Données course",
                                "📊 Pronostics",
                                "ℹ️ Aide & Méthode"])
 
    # ---------- TAB 1 : DONNÉES ----------
    with tab1:
        st.markdown("## 🏁 Informations de la course")
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.5])
        with c1:
            race_type = st.selectbox("Discipline", CONFIG.RACE_TYPES)
        with c2:
            distance = st.number_input("Distance (m)", 800, 7200, 1600, 100)
        with c3:
            track = st.selectbox("Terrain", CONFIG.TRACK_CONDITIONS)
        with c4:
            # Auto-suggestion du type de départ
            default_depart = 0
            if race_type in ("Attelé", "Monté"):
                default_depart = 1
            depart = st.selectbox("Type de départ", CONFIG.DEPART_TYPES,
                                  index=default_depart)
 
        prix = st.text_input("Nom du prix (optionnel)", "")
 
        st.markdown("---")
        st.markdown("## 🐎 Tableau des partants")
        st.caption("✏️ Modifiez directement le tableau. Les champs **Poids** et "
                   "**Jours repos** sont utilisés en Plat ; en Trot, seul "
                   "**Jours repos** est exploité.")
 
        edited = st.data_editor(
            st.session_state.horses_data,
            use_container_width=True,
            num_rows="dynamic",
            height=420,
            column_config={
                "N°": st.column_config.NumberColumn(min_value=1, max_value=99),
                "Cote": st.column_config.NumberColumn(format="%.2f", min_value=1.0),
                "Corde": st.column_config.NumberColumn(min_value=0, max_value=20),
                "Poids": st.column_config.NumberColumn(format="%.1f", min_value=40.0, max_value=80.0),
                "Jours repos": st.column_config.NumberColumn(min_value=0, max_value=999),
            },
        )
        if edited is not None:
            st.session_state.horses_data = edited
 
        c1, c2 = st.columns([3, 1])
        with c1:
            run_btn = st.button("🚀 LANCER L'ANALYSE",
                                 use_container_width=True, type="primary")
        with c2:
            reset_btn = st.button("🔄 Reset", use_container_width=True)
            if reset_btn:
                st.session_state.horses_data = pd.DataFrame({
                    "N°": list(range(1, 11)),
                    "Nom": [f"Cheval {i+1}" for i in range(10)],
                    "Cote": [5.0] * 10,
                    "Musique Cheval": [""] * 10,
                    "Musique Driver": [""] * 10,
                    "Musique Entraîneur": [""] * 10,
                    "Corde": [0] * 10,
                    "Poids": [56.0] * 10,
                    "Jours repos": [21] * 10,
                })
                st.rerun()
 
        if run_btn:
            horses_list = []
            for idx, row in st.session_state.horses_data.iterrows():
                try:
                    horses_list.append({
                        "number": int(row["N°"]),
                        "name": str(row["Nom"]),
                        "odds": float(row["Cote"]),
                        "horse_music": str(row["Musique Cheval"]),
                        "driver_music": str(row["Musique Driver"]),
                        "trainer_music": str(row["Musique Entraîneur"]),
                        "draw": int(row["Corde"]) if pd.notna(row["Corde"]) else 0,
                        "weight": float(row.get("Poids", 56.0)) if pd.notna(row.get("Poids")) else 56.0,
                        "days_rest": int(row.get("Jours repos", -1)) if pd.notna(row.get("Jours repos")) else -1,
                    })
                except Exception as e:
                    st.error(f"⚠️ Erreur ligne {idx+1} : {e}")
                    return
 
            if len(horses_list) < 3:
                st.error("Au moins 3 partants requis.")
                return
 
            with st.spinner(f"🔬 Calcul Plackett-Luce ({mc_iter} simulations)..."):
                pred = run_engine_v4(
                    {"race_type": race_type, "distance": distance,
                     "track": track, "depart_type": depart, "discipline": prix},
                    horses_list,
                    mc_iter=mc_iter, market_weight=mw, value_threshold=vt
                )
                st.session_state.prediction = pred
            st.success(f"✅ Analyse terminée en {pred['execution_time']}s — "
                       f"{pred['n_simulations']} simulations")
 
    # ---------- TAB 2 : RÉSULTATS ----------
    with tab2:
        if st.session_state.prediction is None:
            st.info("🎯 Saisissez les données puis cliquez sur **LANCER L'ANALYSE**.")
        else:
            pred = st.session_state.prediction
 
            # Diagnostic
            st.markdown("## 📈 Diagnostic de course")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🎯 Confiance", f"{pred['confidence_idx']:.1f}/100")
            c2.metric("🌪️ Volatilité", f"{pred['volatility_idx']:.1f}/100")
            if pred["overround_pct"] is not None:
                c3.metric("📉 Overround", f"{pred['overround_pct']:.1f}%",
                          help="Marge bookmaker (>20% = juice élevé)")
            else:
                c3.metric("📉 Overround", "—")
            c4.metric("📐 Seuil value (dyn.)", f"{pred['dynamic_value_threshold']:.2f}",
                      help="Ajusté selon overround")
 
            if pred["kl_divergence"] is not None:
                st.caption(f"🧮 Divergence KL(modèle ‖ marché) = "
                           f"**{pred['kl_divergence']:.3f}** — "
                           f"{'fort désaccord' if pred['kl_divergence'] > 0.15 else 'accord modéré'}")
 
            st.markdown("---")
            st.markdown("## 🏆 Classement final & paris GAGNANT")
            df = pd.DataFrame([{
                "Rg": r["rank"],
                "N°": r["number"],
                "Nom": r["name"][:18],
                "Cote": f"{r['odds']:.2f}",
                "Modèle %": f"{r['win_prob_model']:.1f}",
                "Marché %": f"{r['win_prob_market']:.1f}",
                "🎯 Final %": f"{r['win_prob']:.2f}",
                "Placé %": f"{r['place_prob']:.1f}",
                "Ratio": f"{r['value_ratio']:.2f}",
                "ROI %": f"{r['expected_roi']:+.1f}",
                "Kelly %": f"{r['kelly_recommended']*100:.2f}",
                "Vol.": f"{r['volatility']:.2f}",
                "Value": "🟢" if r["is_value_bet"] else "⚪",
            } for r in pred["results"]])
            st.dataframe(df, use_container_width=True, hide_index=True, height=380)
 
            # Value bets en évidence
            value_bets = [r for r in pred["results"] if r["is_value_bet"]]
            if value_bets:
                st.markdown("### 💎 Value bets détectés")
                for vb in value_bets[:5]:
                    st.markdown(
                        f"- **N°{vb['number']} {vb['name']}** "
                        f"@ cote {vb['odds']:.2f} — "
                        f"prob. modèle {vb['win_prob']:.1f}% vs marché {vb['win_prob_market']:.1f}% "
                        f"→ Kelly recommandé : **{vb['kelly_recommended']*100:.2f}%** "
                        f"(ROI espéré : {vb['expected_roi']:+.1f}%)"
                    )
            else:
                st.info("⚪ Aucun value bet détecté sur ce marché.")
 
            # Meilleur placé
            if pred["best_place"]:
                bp = pred["best_place"]
                st.markdown("---")
                st.markdown("## 🥉 Meilleur pari **PLACÉ**")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("N°", bp["number"])
                c2.metric("Cheval", bp["name"][:15])
                c3.metric("Prob. Placé", f"{bp['place_prob']:.1f}%")
                c4.metric("ROI Placé", f"{bp['expected_roi_place']:+.1f}%")
                st.markdown(
                    f"💡 Cote placé estimée : **{bp['estimated_place_odds']:.2f}** — "
                    f"Mise Kelly recommandée : **{bp['kelly_recommended']*100:.2f}%** "
                    f"du bankroll"
                )
 
            # Exotiques
            st.markdown("---")
            st.markdown("## 🎲 Paris exotiques (Top combinaisons)")
            ex = pred["exotics"]
            tabs_exo = st.tabs(["Couplé Gagnant", "Couplé Placé",
                                "Trio Ordre", "Trio Désordre",
                                "Quarté+", "Quinté+", "Quarté 20 combis"])
 
            def _render_exotic(items, key):
                if not items:
                    st.info("Aucune combinaison significative.")
                    return
                df_e = pd.DataFrame([{
                    "Rg": x["rank"],
                    "Combo": x.get("combo", "—"),
                    **({"Détail": x["names"]} if "names" in x else {}),
                    "Prob %": x["prob_pct"],
                    "Cote est.": x["estimated_odds"],
                    "ROI %": x["expected_roi"],
                } for x in items])
                st.dataframe(df_e, use_container_width=True, hide_index=True)
 
            def _render_quarte_coverage(block, title):
                st.markdown(f"### {title}")
                if not block or not block.get("combos"):
                    st.info("Aucune combinaison générée.")
                    return
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Combinaisons", block.get("n_combos", 0))
                c2.metric("Couverture totale", f"{block.get('total_coverage_pct', 0):.2f}%")
                c3.metric("Mise totale", f"{block.get('total_stake_eur', 0):.2f} €")
                c4.metric("ROI estimé", f"{block.get('roi_estimate_pct', 0):+.1f}%")
                df_q = pd.DataFrame([{
                    "Rg": x["rank"],
                    "Combo": x["combo"],
                    "Prob %": x["prob_pct"],
                    "Cumul %": x["cum_prob_pct"],
                } for x in block["combos"]])
                st.dataframe(df_q, use_container_width=True, hide_index=True, height=420)
                strong = block.get("strong_horses_count", 0)
                st.caption(
                    f"Mode: {block.get('mode', 'coverage')} — chevaux >= 50% top 4 : {strong}. "
                    f"La couverture totale représente la chance que le vrai Quarté soit présent parmi les 20 tickets."
                )
 
            with tabs_exo[0]: _render_exotic(ex["couple_gagnant"], "cg")
            with tabs_exo[1]: _render_exotic(ex["couple_place"], "cp")
            with tabs_exo[2]: _render_exotic(ex["trio_ordre"], "to")
            with tabs_exo[3]: _render_exotic(ex["trio_desordre"], "td")
            with tabs_exo[4]: _render_exotic(ex["quarte_desordre"], "q4")
            with tabs_exo[5]: _render_exotic(ex["quinte_desordre"], "q5")
            with tabs_exo[6]:
                _render_quarte_coverage(pred.get("quarte_coverage_20"), "📦 Mode couverture 20 tickets")
                st.markdown("---")
                _render_quarte_coverage(pred.get("quarte_top_horses_20"), "🎯 Mode chevaux ≥ 50% top 4")
 
    # ---------- TAB 3 : AIDE ----------
    with tab3:
        st.markdown("""
## 🎓 Méthodologie QuantTurf v4.4
 
### 🔬 Architecture du moteur
 
```
Musique → Parsing + Shrinkage bayésien → Score composite
                                              ↓
                                          Softmax
                                              ↓
                                  Correction empirique (corde+exp)
                                              ↓
                                       p_modèle
                                              ↓
                        Platt scaling (coefficients RÉTRÉCIS bayésiennement)
                                              ↓
Cotes marché → Débiaisage (power γ=1.12, ou Shin en option) → p_marché
                                              ↓
                              BENTER BLEND : p ∝ p_modèle^α · p_marché^β
                                              ↓
      Plackett-Luce séquentiel + amortissement Benter/Henery (γ,δ,ε,ζ par rang)
                                              ↓
                    Win / Place / Couplé / Trio / Quarté+ / Quinté+
                          (cotes PMU basées sur TRJ officiels 2026)
                                              ↓
                                   Kelly dynamique + ROI
```
 
### 📚 Formules clés
 
**1. Shrinkage bayésien (musique, et désormais aussi Platt scaling)**
$$\\text{score}_{\\text{shrunk}} = \\frac{n \\cdot \\text{score}_{\\text{obs}} + K \\cdot \\mu_{\\text{pop}}}{n + K}$$
Le même principe est appliqué en v4.4 aux coefficients de Platt (a, b) :
avec seulement ~12 courses par discipline dans le backtest d'origine, les
estimer sans rétrécissement expose à un sur-ajustement sévère (ex. le
Plat brut compressait une proba modèle de 35% à ~15% — beaucoup trop
agressif pour un si petit échantillon).
 
**2. Débiaisage des cotes (favori-outsider correction)**
$$p_{\\text{vraie}} \\propto \\left(\\frac{1}{\\text{cote}}\\right)^\\gamma, \\quad \\gamma \\approx 1.12 \\ \\text{(dans la fourchette 1.06–1.15 documentée par la littérature)}$$
Alternative disponible : méthode de **Shin (1993)**, fondée sur un modèle
d'insiders plutôt que sur un ajustement empirique pur.
 
**3. Benter Blend**
$$p_{\\text{finale}} \\propto p_{\\text{modèle}}^\\alpha \\cdot p_{\\text{marché}}^\\beta$$
 
**4. Plackett-Luce (Harville) avec correction Benter/Henery** — tirage
séquentiel des positions, chaque force étant amortie par un exposant qui
décroît avec le rang (1.0 / 0.81 / 0.65 / 0.50 / 0.35) pour corriger le
biais documenté par Benter (1994) : la formule brute surestime les
favoris et sous-estime les outsiders sur les places 2 à 5. Les exposants
2e/3e sont issus d'une estimation MLE publiée (Benter 1994) ; les
exposants 4e/5e sont extrapolés (non estimés dans la littérature connue).
 
**5. Kelly fractionnaire dynamique**
$$f^* = \\frac{p \\cdot b - q}{b}, \\quad f_{\\text{misé}} = \\min\\left(f^* \\cdot \\frac{1}{1+\\text{vol}}, f_{\\max}\\right)$$
Benter (1994) déconseille lui-même le Kelly plein et recommande 1/2 à 1/3 ;
`KELLY_FRACTION=0.20` reste sous cette fourchette pour tenir compte d'un
échantillon de calibration plus restreint que celui de Benter.
 
### 🎯 Stratégie recommandée
 
| Type de pari | Quand l'utiliser | Risque |
|---|---|---|
| **Gagnant (value)** | Ratio > 1.20 ET cote > 2.5 | 🟡 Moyen |
| **Placé** | Champion avec cote ≥ 4 | 🟢 Faible |
| **Couplé Placé** | ROI > 50% | 🟡 Moyen |
| **Trio désordre** | ROI > 100% sur 3 favoris | 🟠 Élevé |
| **Quinté+** | Mise faible, ROI espéré > 200% | 🔴 Très élevé |
 
### ⚠️ Avertissements
 
- 🎰 **Les performances passées ne préjugent pas des résultats futurs**
- 💸 **Jouez avec modération** — ne misez jamais plus que ce que vous pouvez perdre
- 📊 Le modèle nécessite un marché suffisamment liquide pour le Benter Blend
- 🐎 La corde au Trot n'est pertinente qu'en départ **AUTOSTART**
- 🔍 Les statistiques empiriques sont des **valeurs indicatives basées sur des études publiques** ; affinez-les selon votre propre base de données.
- 📐 Le calibrage de marché (Platt/overround) de ce script est issu de
  courses **support Quinté+**, structurellement plus ouvertes que la
  moyenne des courses PMU — voir l'avertissement méthodologique en tête
  du fichier source.
- 💰 Les cotes PMU estimées pour les paris exotiques reposent désormais sur
  les **TRJ officiels** publiés par le PMU (barèmes en vigueur T1 2026),
  et non plus sur des approximations non sourcées.
 
### 📖 Références
 
- Benter, W. (1994). *Computer Based Horse Race Handicapping and Wagering Systems.* In *Efficiency of Racetrack Betting Markets.*
- Harville, D. (1973). *Assigning Probabilities to the Outcomes of Multi-Entry Competitions.* JASA 68(342).
- Henery, R. J. (1981). *Permutation probabilities as models for horse races.* JRSS B, 43(1).
- Kelly, J. L. (1956). *A New Interpretation of Information Rate.*
- Snowberg & Wolfers (2010). *Explaining the Favorite-Longshot Bias.* JPE 118(4).
- Shin, H. S. (1993). *Measuring the Incidence of Insider Trading in a Market for State-Contingent Claims.* Economic Journal 103(420).
- British Horseracing Authority — *Performance Figures* (méthodologie officielle poids/longueur).
- Turfmining.fr — *Comment la récence influe-t-elle sur la performance du cheval ?* (>1,25M partants, données PMU).
- PMU — Taux de redistribution (TRJ) officiels par pari, barèmes guichet et en ligne, T1 2026.
        """)
 
 
if __name__ == "__main__":
    main()

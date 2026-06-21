"""
═══════════════════════════════════════════════════════════════════════════════
 QuantTurf Pro v4.1.0 — "BENTER EDITION — DATA-CALIBRATED"
═══════════════════════════════════════════════════════════════════════════════
 Changements vs v4.0.0, justifiés par backtest sur 19 courses réelles (juin 2026) :
 ─────────────────────────────────────────────────────────────────────────────
 📊 Backtest intégré (onglet dédié) : charge un historique cotes+arrivées et
    mesure logloss / win-hit-rate / ROI de n'importe quel jeu de paramètres,
    avec intervalle de confiance — pour éviter de sur-interpréter le bruit.
 ⚖️  MARKET_WEIGHT par défaut relevé (0.35→0.55) : sur l'échantillon disponible,
    le marché débiaisé seul bat largement un modèle musique/forme pur
    (logloss 2.22 vs 2.88, ROI value-bets -44% sur le modèle pur). Le marché
    PMU est très liquide ; un modèle "léger" doit lui faire confiance par défaut.
 🛡️  Garde-fou anti-surconfiance : si market_weight < 0.40 ou value_threshold
    < 1.15, un avertissement explicite est affiché (paramètres non
    soutenus par les données disponibles à ce jour).
 ✅ gamma de débiaisage favori-outsider VALIDÉ empiriquement (1.12 quasi optimal,
    optimum mesuré à 1.20 sur l'échantillon — différence non significative).
 ✅ Reste : Plackett-Luce (Harville), Benter Blend, Kelly dynamique, paris
    exotiques, shrinkage bayésien — architecture v4.0 inchangée.
═══════════════════════════════════════════════════════════════════════════════
Sources scientifiques :
- Benter, W. (1994). "Computer Based Horse Race Handicapping" (Hong Kong)
- Harville, D. (1973). "Assigning probabilities to outcomes of multi-entry comp."
- Plackett, R. (1975). "The Analysis of Permutations"
- Kelly, J. (1956). "A New Interpretation of Information Rate"
- Snowberg & Wolfers (2010). "Explaining the Favorite-Longshot Bias"
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
import json

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 1.  CONFIGURATION GLOBALE
# =============================================================================
@dataclass
class Config:
    # --- App ---
    APP_VERSION: str = "4.1.0"
    APP_NAME: str = "QuantTurf Pro"
    APP_TAG: str = "Benter Edition — Data-Calibrated"

    # --- Monte Carlo / Plackett-Luce ---
    MC_ITERATIONS: int = 6000          # tirages PL pour exotiques (réglage demandé)
    TEMPERATURE: float = 1.0           # softmax temperature (1.0 = neutre)
    NOISE_BASE: float = 0.18           # bruit log-normal pour PL (réglage demandé)

    # --- Marché ---
    # NOTE CALIBRATION (19 courses, juin 2026) : le blend optimal mesuré en
    # logloss converge vers ~100% marché sur cet échantillon, le modèle
    # musique/forme seul étant nettement moins bien calibré (logloss 2.88
    # vs 2.22 pour le marché). On NE va PAS à 1.0 (ce serait sur-interpréter
    # 19 courses), mais on relève la confiance par défaut accordée au marché
    # par rapport à la v4.0, et on s'écarte de la demande initiale
    # (market_weight=0.25) qui n'est pas soutenue par les données.
    MARKET_WEIGHT: float = 0.55        # poids du marché dans Benter Blend
    BENTER_ALPHA: float = 1.20         # exposant log(p_model)  — réglage demandé, conservé
    BENTER_BETA: float = 0.80          # exposant log(p_market) — réglage demandé, conservé
    OVERROUND_CORRECTION: bool = True  # corriger le biais favori-outsider
    OVERROUND_GAMMA: float = 1.12      # validé empiriquement (optimum mesuré: 1.20, diff. non sign.)

    # --- Value / Kelly ---
    VALUE_THRESHOLD: float = 1.18      # réglage demandé, conservé (cf. garde-fou)
    KELLY_FRACTION: float = 0.30       # réglage demandé
    MIN_KELLY_ODDS: float = 2.20       # cote min pour Kelly (sous, EV-)
    MAX_KELLY_STAKE: float = 0.05      # cap absolu : 5% bankroll max (réglage demandé)
    PLACE_ODDS_FACTOR: Dict[str, float] = None  # rapport cote_placé / cote_gagn

    # --- Empirique (corde, expérience) ---
    EMPIRICAL_WEIGHT: float = 0.20     # réglage demandé
    USE_EXPERIENCE_FACTOR: bool = True

    # --- Shrinkage bayésien ---
    SHRINKAGE_K: float = 3.0           # réglage demandé (plus de poids aux courses récentes)
    POPULATION_MEAN_SCORE: float = 4.0 # moyenne pop. des scores musique
    POPULATION_MEAN_WIN: float = 0.10  # 10% victoires moyennes pop.

    # --- Garde-fous anti-surconfiance ---
    WARN_MARKET_WEIGHT_BELOW: float = 0.40
    WARN_VALUE_THRESHOLD_BELOW: float = 1.15
    MIN_BACKTEST_RACES_FOR_TRUST: int = 100  # sous ce seuil, afficher l'avertissement n petit

    # --- Paris ---
    RACE_TYPES: List[str] = None
    TRACK_CONDITIONS: List[str] = None
    DEPART_TYPES: List[str] = None

    # --- Musique parsing ---
    MUSIC_POSITION_SCORES: Dict[str, float] = None
    MUSIC_RACE_TYPE_WEIGHTS: Dict[str, float] = None

    # --- Tables empiriques corde ---
    # NOTE : ces tables proviennent d'études publiques à large échelle (Turf.bzh,
    # PMU). Le jeu de 19 courses fourni ne contient PAS la corde réelle des
    # chevaux (seulement leur numéro de dossard, qui n'est pas systématiquement
    # la corde) et ne permet de toute façon pas de recalibrer ces tables
    # (9-10 courses par discipline, gagnants dispersés sur tous les numéros
    # sans tendance visible). On les conserve donc inchangées par rapport à v4.0.
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
            self.PLACE_ODDS_FACTOR = {
                "small": 0.50, "medium": 0.40, "large": 0.32,
            }
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
        if self.DRAW_WIN_PROB_AUTOSTART is None:
            self.DRAW_WIN_PROB_AUTOSTART = {
                1: 9.0,  2: 9.5,  3: 10.0, 4: 11.5, 5: 12.0, 6: 11.0,
                7: 9.5,  8: 8.0,  9: 6.5,  10: 5.0,
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
    shrunk_score: float = 0.0
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

        decay = np.exp(-0.30 * np.arange(n))
        decay /= decay.sum()
        weighted_score = float(np.dot(raw_scores_arr, decay))

        recent_n = min(3, n)
        rd = decay[:recent_n] / decay[:recent_n].sum()
        recent_form = float(np.dot(raw_scores_arr[:recent_n], rd))

        if len(numeric_positions) >= 2:
            pos_std = float(np.std(numeric_positions))
            regularity = max(0.0, 1.0 - pos_std / 5.0)
        else:
            pos_std = 3.0
            regularity = 0.50

        if n >= 4:
            recent_avg = np.mean(raw_scores_arr[: n // 2])
            old_avg = np.mean(raw_scores_arr[n // 2:])
            trend = (recent_avg - old_avg) / (abs(old_avg) + 1e-9)
        else:
            trend = 0.0

        win_count = sum(1 for p in numeric_positions if p == 1)
        podium_count = sum(1 for p in numeric_positions if p <= 3)
        win_ratio = win_count / max(n, 1)
        podium_ratio = podium_count / max(n, 1)

        consistency = max(0.0, min(1.0, 1.0 - pos_std / 10.0))

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
    Facteur de corde — gère plat ET autostart trot.
    Retourne un score [-1.5, +1.5] à fusionner dans le composite.
    """
    if not draw or draw <= 0:
        return 0.0
    draw = min(int(draw), 20)

    if race_type == "Plat":
        if draw <= 2:    base = 1.0
        elif draw <= 4:  base = 0.7
        elif draw <= 6:  base = 0.3
        elif draw <= 9:  base = -0.2
        elif draw <= 12: base = -0.6
        else:            base = -1.0

        if distance <= 1300:   dist_mult = 1.6
        elif distance <= 1600: dist_mult = 1.3
        elif distance <= 2000: dist_mult = 1.0
        elif distance <= 2400: dist_mult = 0.7
        else:                  dist_mult = 0.4

        if track in ("Lourd", "Très lourd", "Collant"):
            base *= 0.3
        elif track in ("Souple", "Très souple"):
            base *= 0.7

        return base * dist_mult

    if depart_type == "Autostart (Trot)" and race_type in ("Attelé", "Monté"):
        if draw in (4, 5, 6):     base = 0.9
        elif draw in (3, 7):      base = 0.5
        elif draw in (2, 8):      base = 0.2
        elif draw in (1, 9):      base = -0.2
        elif draw == 10:          base = -0.5
        elif draw <= 14:          base = -0.7
        else:                     base = -1.0

        if distance >= 2700:
            base *= 0.7
        return base

    return 0.0


def track_factor(track: str, race_type: str) -> float:
    """Facteur multiplicateur global selon état du terrain (~1.0 neutre)."""
    if track in ("Lourd", "Très lourd"):  return 0.92
    if track == "Collant":                return 0.95
    if track in ("Souple", "Très souple"): return 0.98
    return 1.0


def weight_factor(weight_kg: float, ref_weight: float = 56.0) -> float:
    """Plat uniquement : un cheval avec poids très élevé est désavantagé."""
    if weight_kg <= 0:
        return 1.0
    delta = weight_kg - ref_weight
    return max(0.85, min(1.15, 1.0 - 0.02 * delta))


def rest_factor(days_since_last_race: int) -> float:
    """Jours de repos : optimum à 14-30 jours."""
    d = days_since_last_race
    if d < 0:    return 1.0
    if d <= 5:   return 0.85
    if d <= 10:  return 0.95
    if d <= 30:  return 1.00
    if d <= 60:  return 0.95
    if d <= 120: return 0.88
    return 0.80


# =============================================================================
# 4.  SCORE COMPOSITE (entrée du modèle softmax)
# =============================================================================
def get_weights_v4(race_type: str) -> Dict[str, float]:
    """Poids normalisés par discipline. Total ≈ 1.0."""
    if race_type == "Plat":
        return {
            "horse_score": 0.22, "horse_form": 0.10, "horse_regularity": 0.05,
            "horse_trend": 0.04, "horse_win": 0.04,
            "driver_score": 0.10, "driver_form": 0.05, "driver_win": 0.05,
            "trainer_score": 0.08, "trainer_form": 0.04, "trainer_win": 0.03,
            "draw_factor": 0.12, "synergy": 0.03, "weight_adj": 0.03, "rest_adj": 0.02,
        }
    elif race_type in ("Attelé", "Monté"):
        return {
            "horse_score": 0.18, "horse_form": 0.08, "horse_regularity": 0.04,
            "horse_trend": 0.03, "horse_win": 0.02,
            "driver_score": 0.16, "driver_form": 0.09, "driver_win": 0.07,
            "trainer_score": 0.10, "trainer_form": 0.05, "trainer_win": 0.03,
            "draw_factor": 0.08, "synergy": 0.03, "weight_adj": 0.00, "rest_adj": 0.04,
        }
    else:
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

    if weights.get("draw_factor", 0) > 0:
        s += weights["draw_factor"] * feat.get("draw_factor", 0) * 5

    h = np.clip(feat["horse_score"], 0.1, 12)
    d = np.clip(feat["driver_score"], 0.1, 12)
    t = np.clip(feat["trainer_score"], 0.1, 12)
    syn = min(h, d, t) / max(h, d, t)
    s += weights.get("synergy", 0) * syn * 10

    s += weights.get("weight_adj", 0) * (feat.get("weight_factor", 1.0) - 1.0) * 50
    s += weights.get("rest_adj",   0) * (feat.get("rest_factor",   1.0) - 1.0) * 50

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


def remove_overround(odds: np.ndarray, gamma: float = None) -> np.ndarray:
    """
    Débiaise les cotes : normalisation + correction favori-outsider bias.
    Selon la littérature (Whelan 2017, Snowberg-Wolfers 2010), les favoris
    sont systématiquement sous-cotés et les outsiders sur-cotés.
    p_true ∝ p_raw^γ avec γ ≈ 1.12 (validé empiriquement sur 19 courses
    réelles : optimum mesuré 1.20, écart non significatif → on garde 1.12).
    """
    eps = 1e-9
    if gamma is None:
        gamma = CONFIG.OVERROUND_GAMMA
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


def benter_blend(p_model: np.ndarray, p_market: np.ndarray,
                 alpha: float = None, beta: float = None) -> np.ndarray:
    """
    Fusion Benter (1994) : p_final ∝ p_model^α · p_market^β
    """
    if alpha is None: alpha = CONFIG.BENTER_ALPHA
    if beta is None:  beta = CONFIG.BENTER_BETA
    eps = 1e-12
    log_blend = alpha * np.log(p_model + eps) + beta * np.log(p_market + eps)
    log_blend -= log_blend.max()
    p = np.exp(log_blend)
    return p / p.sum()


def plackett_luce_simulate(strengths: np.ndarray, n_iter: int,
                            noise: float = 0.18) -> np.ndarray:
    """Simule n_iter ordres d'arrivée par modèle Plackett-Luce (Harville)."""
    n = len(strengths)
    orders = np.zeros((n_iter, n), dtype=np.int32)
    base_log = np.log(np.maximum(strengths, 1e-9))
    for it in range(n_iter):
        noisy = base_log + np.random.normal(0, noise, n)
        gumbel = -np.log(-np.log(np.random.uniform(1e-12, 1-1e-12, n)))
        scores_perturbed = noisy + gumbel
        orders[it] = np.argsort(-scores_perturbed)
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
    """Mélange convexe entre proba modèle et proba empirique pondérée par expérience."""
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
    """Kelly fractionnaire dynamique, cap absolu CONFIG.MAX_KELLY_STAKE."""
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
PMU_TAKEOUT = {
    "couple_gagnant": 0.74,
    "couple_place":   0.78,
    "trio_ordre":     0.72,
    "trio_desordre":  0.74,
    "quarte_desordre": 0.71,
    "quinte_desordre": 0.68,
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

    IMPORTANT (fix v4.1.1) : `orders` contient des indices dans l'espace
    ORIGINAL des chevaux (ordre de construction des features, avant tri),
    alors que `results` est trié par win_prob décroissant et n'est donc
    PAS indexé de la même façon. v4.0/v4.1.0 indexaient `results[i]`
    directement avec les indices de `orders`, ce qui associait le mauvais
    cheval à la mauvaise probabilité dans TOUS les paris exotiques (bug
    silencieux, détecté par backtest manuel le 21/06/2026). On reconstruit
    ici un mapping explicite original_index -> entrée results.
    """
    n_iter, n_horses = orders.shape
    output = {"couple_gagnant": [], "couple_place": [],
              "trio_ordre": [], "trio_desordre": [],
              "quarte_desordre": [], "quinte_desordre": []}

    if n_horses < 3:
        return output

    # Mapping indice original (ordre de construction des features) -> résultat
    by_orig_idx = sorted(results, key=lambda r: r["_orig_idx"])

    cg = {}
    for it in range(n_iter):
        key = (int(orders[it, 0]), int(orders[it, 1]))
        cg[key] = cg.get(key, 0) + 1
    for (i, j), c in cg.items():
        p = c / n_iter
        if p < 0.005: continue
        est_odds = _pmu_estimated_odds(p, "couple_gagnant", 3.0, 400.0)
        output["couple_gagnant"].append({
            "combo": f"{by_orig_idx[i]['number']}-{by_orig_idx[j]['number']}",
            "names": f"{by_orig_idx[i]['name'][:8]} → {by_orig_idx[j]['name'][:8]}",
            "prob_pct": round(p * 100, 2),
            "estimated_odds": round(est_odds, 1),
            "expected_roi": round(expected_roi(p, est_odds, 10), 1),
        })

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
            "combo": f"{by_orig_idx[i]['number']}-{by_orig_idx[j]['number']}",
            "names": f"{by_orig_idx[i]['name'][:8]} & {by_orig_idx[j]['name'][:8]}",
            "prob_pct": round(p * 100, 2),
            "estimated_odds": round(est_odds, 1),
            "expected_roi": round(expected_roi(p, est_odds, 10), 1),
        })

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
            "combo": f"{by_orig_idx[i]['number']}-{by_orig_idx[j]['number']}-{by_orig_idx[k]['number']}",
            "prob_pct": round(p * 100, 3),
            "estimated_odds": round(est_odds, 1),
            "expected_roi": round(expected_roi(p, est_odds, 10), 1),
        })

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
            "combo": f"{by_orig_idx[i]['number']}-{by_orig_idx[j]['number']}-{by_orig_idx[k]['number']}",
            "prob_pct": round(p * 100, 2),
            "estimated_odds": round(est_odds, 1),
            "expected_roi": round(expected_roi(p, est_odds, 10), 1),
        })

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
                "combo": "-".join(str(by_orig_idx[i]['number']) for i in key),
                "prob_pct": round(p * 100, 3),
                "estimated_odds": round(est_odds, 1),
                "expected_roi": round(expected_roi(p, est_odds, 5), 1),
            })

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
                "combo": "-".join(str(by_orig_idx[i]['number']) for i in key),
                "prob_pct": round(p * 100, 4),
                "estimated_odds": round(est_odds, 1),
                "expected_roi": round(expected_roi(p, est_odds, 2), 1),
            })

    for k in output:
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
# 9.  MOTEUR PRINCIPAL — RaceEngine v4.1
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
            wf = weight_factor(h.get("weight", 0)) if self.race_type == "Plat" else 1.0
            rf = rest_factor(h.get("days_rest", -1))
            tf = track_factor(self.track, self.race_type)

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

    def predict(self, mc_iter: int = None, market_weight: float = None,
                value_threshold: float = None) -> Dict[str, Any]:
        t0 = time.time()
        if mc_iter is None:        mc_iter = CONFIG.MC_ITERATIONS
        if market_weight is None:  market_weight = CONFIG.MARKET_WEIGHT
        if value_threshold is None: value_threshold = CONFIG.VALUE_THRESHOLD

        # ── Garde-fou anti-surconfiance ────────────────────────────────
        # Avertissements informatifs si les réglages s'écartent de ce que
        # l'échantillon de calibration disponible (19 courses, juin 2026)
        # soutient. Ne bloque rien — informe seulement.
        warnings_list = []
        if market_weight < CONFIG.WARN_MARKET_WEIGHT_BELOW:
            warnings_list.append(
                f"⚠️ Poids marché ({market_weight:.2f}) inférieur à "
                f"{CONFIG.WARN_MARKET_WEIGHT_BELOW:.2f}. Sur l'échantillon de "
                f"calibration disponible (19 courses), le marché débiaisé seul "
                f"est mieux calibré que le modèle musique/forme seul "
                f"(logloss 2.22 vs 2.88). Un poids marché faible n'est pas "
                f"soutenu par ces données — voir onglet Backtest."
            )
        if value_threshold < CONFIG.WARN_VALUE_THRESHOLD_BELOW:
            warnings_list.append(
                f"⚠️ Seuil de value ({value_threshold:.2f}) bas. Sur l'échantillon "
                f"de calibration, les \"value bets\" détectés par le modèle pur "
                f"ont eu un ROI réel très négatif (-44% à -86% selon le seuil "
                f"testé) — le signal de value actuel n'est pas démontré rentable."
            )

        feats, draws, exp_factors = self._build_features()
        weights = get_weights_v4(self.race_type)
        scores = np.array([composite_score_v4(f, weights) for f in feats])
        if scores.std() < 1e-6:
            scores += np.random.normal(0, 0.05, self.n)

        p_model_raw = softmax_temp(scores, T=CONFIG.TEMPERATURE)

        p_model = empirical_correction(p_model_raw, draws, self.race_type,
                                         self.distance, self.depart_type,
                                         exp_factors)

        odds_arr = np.array([f["odds"] for f in feats])
        has_market = (odds_arr > 1.5).sum() >= self.n * 0.5
        if has_market:
            p_market = remove_overround(odds_arr)
        else:
            p_market = np.ones(self.n) / self.n

        if has_market and market_weight > 0:
            beta_eff = CONFIG.BENTER_BETA * (market_weight / 0.35)
            p_final = benter_blend(p_model, p_market,
                                    alpha=CONFIG.BENTER_ALPHA,
                                    beta=beta_eff)
        else:
            p_final = p_model

        strengths = p_final * 100
        orders = plackett_luce_simulate(strengths, mc_iter, noise=CONFIG.NOISE_BASE)

        place_counts = np.zeros(self.n)
        win_counts = np.zeros(self.n)
        for it in range(mc_iter):
            win_counts[orders[it, 0]] += 1
            for k in range(3):
                place_counts[orders[it, k]] += 1
        p_place_mc = place_counts / mc_iter
        p_win_mc = win_counts / mc_iter

        volatility = np.abs(p_final - p_win_mc) / (p_final + 1e-9)

        results = []
        if has_market:
            raw_or = sum(1.0 / o for o in odds_arr if o > 1.01)
            overround_pct = round((raw_or - 1.0) * 100, 1)
        else:
            overround_pct = None

        if overround_pct is not None and overround_pct > 0:
            dyn_value_th = max(value_threshold, 1.0 + overround_pct / 100 * 1.2)
        else:
            dyn_value_th = value_threshold

        for i, (feat, horse) in enumerate(zip(feats, self.horses)):
            ratio = p_final[i] / (p_market[i] + 1e-9)
            is_value = (ratio >= dyn_value_th) and (p_final[i] >= 0.04)
            k_pur, k_reco = kelly_bet(p_final[i], horse.get("odds", 2.0),
                                       volatility=1 + volatility[i])
            roi = expected_roi(p_final[i], horse.get("odds", 2.0))

            results.append({
                "rank": 0,
                "_orig_idx": i,
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

        exotics = analyze_exotics(results, orders)
        bp = best_place_bet(results, self.n)

        sorted_p = sorted([r["win_prob"] for r in results], reverse=True)
        if len(sorted_p) >= 2:
            gap = sorted_p[0] - sorted_p[1]
            conf_idx = min(100, round(45 + gap * 2.5, 1))
        else:
            conf_idx = 50
        vol_idx = min(100, round(volatility.mean() * 60, 1))

        if has_market:
            eps = 1e-12
            kl = float(np.sum(p_final * np.log((p_final + eps) / (p_market + eps))))
        else:
            kl = None

        return {
            "results": results,
            "exotics": exotics,
            "best_place": bp,
            "confidence_idx": conf_idx,
            "volatility_idx": vol_idx,
            "overround_pct": overround_pct,
            "dynamic_value_threshold": round(dyn_value_th, 3),
            "kl_divergence": round(kl, 3) if kl else None,
            "execution_time": round(time.time() - t0, 2),
            "n_simulations": mc_iter,
            "warnings": warnings_list,
        }


def run_engine_v4(race_info: Dict, horses: List[Dict], **kwargs) -> Dict:
    """API publique compatible avec l'ancienne v3/v4.0."""
    engine = RaceEngine(race_info, horses)
    return engine.predict(**kwargs)


# =============================================================================
# 10.  BACKTEST ENGINE — nouveau en v4.1
# =============================================================================
"""
Module de validation empirique. Permet de charger un historique de courses
(cotes finales + arrivées réelles) au format JSON et de mesurer la qualité
de calibration de différents jeux de paramètres, AVEC intervalle de confiance,
pour éviter de sur-interpréter un petit échantillon.

Format attendu (liste de courses) :
[
  {
    "date": "01-06",
    "race_type": "Attelé",
    "horses": [{"number": 8, "odds": 8.0, "model_prob_pct": 38.8}, ...],
    "arrival": [5, 6, 7, 9, 12]
  },
  ...
]
"p_model" (model_prob_pct) est optionnel : s'il est absent, seul le marché
débiaisé est évalué.
"""

def backtest_blend(races: List[Dict], alpha: float, beta: float,
                    gamma: float = None) -> Dict[str, Any]:
    """Évalue un jeu de paramètres (alpha, beta, gamma) sur un historique réel."""
    if gamma is None:
        gamma = CONFIG.OVERROUND_GAMMA
    n = 0
    win_hits = 0
    podium3_hits = 0
    podium5_hits = 0
    logloss = 0.0
    brier = 0.0
    roi_total = 0.0
    n_bets = 0
    per_race = []

    for race in races:
        horses = race.get("horses", [])
        arrival = race.get("arrival", [])
        if not horses or not arrival:
            continue
        numbers = [h["number"] for h in horses]
        odds = np.array([float(h["odds"]) for h in horses])
        has_model = all("model_prob_pct" in h for h in horses)

        p_market = remove_overround(odds, gamma=gamma)
        if has_model:
            p_model = np.array([h["model_prob_pct"] / 100.0 for h in horses])
            p_model = p_model / p_model.sum()
            p_final = benter_blend(p_model, p_market, alpha=alpha, beta=beta)
        else:
            p_final = p_market

        winner = arrival[0]
        if winner not in numbers:
            continue
        n += 1
        win_idx = numbers.index(winner)
        top_idx = int(np.argmax(p_final))
        top_number = numbers[top_idx]

        if top_number == winner:
            win_hits += 1
        if top_number in set(arrival[:3]):
            podium3_hits += 1
        if top_number in set(arrival[:5]):
            podium5_hits += 1

        p_win = max(p_final[win_idx], 1e-9)
        logloss += -np.log(p_win)
        y = np.zeros(len(numbers)); y[win_idx] = 1
        brier += np.sum((p_final - y) ** 2)

        odds_fav = odds[top_idx]
        n_bets += 1
        roi_total += (odds_fav - 1) if top_number == winner else -1

        per_race.append({
            "date": race.get("date", "?"),
            "predicted_top": top_number,
            "actual_winner": winner,
            "hit": top_number == winner,
            "p_winner": round(float(p_win) * 100, 2),
        })

    if n == 0:
        return {"n": 0, "error": "Aucune course exploitable (vérifier le format)."}

    win_rate = win_hits / n
    se = np.sqrt(win_rate * (1 - win_rate) / n) if n > 0 else 0
    ci95 = (max(0, win_rate - 1.96 * se), min(1, win_rate + 1.96 * se))

    return {
        "n": n,
        "win_hit_rate": round(win_rate, 4),
        "win_hit_rate_ci95": (round(ci95[0], 4), round(ci95[1], 4)),
        "podium3_hit_rate": round(podium3_hits / n, 4),
        "podium5_hit_rate": round(podium5_hits / n, 4),
        "avg_logloss": round(logloss / n, 4),
        "avg_brier": round(brier / n, 4),
        "roi_pct": round(roi_total / n_bets * 100, 2) if n_bets else None,
        "is_low_sample": n < CONFIG.MIN_BACKTEST_RACES_FOR_TRUST,
        "per_race": per_race,
    }


def backtest_grid_search(races: List[Dict],
                          alphas=None, betas=None) -> pd.DataFrame:
    """Balaye une grille alpha/beta et retourne un DataFrame trié par logloss."""
    if alphas is None:
        alphas = np.arange(0.3, 2.01, 0.1)
    if betas is None:
        betas = np.arange(0.0, 2.01, 0.1)
    rows = []
    for a in alphas:
        for b in betas:
            r = backtest_blend(races, a, b)
            if r.get("n", 0) == 0:
                continue
            rows.append({
                "alpha": round(float(a), 2), "beta": round(float(b), 2),
                "win_hit_rate": r["win_hit_rate"], "podium3": r["podium3_hit_rate"],
                "podium5": r["podium5_hit_rate"], "logloss": r["avg_logloss"],
                "roi_pct": r["roi_pct"],
            })
    df = pd.DataFrame(rows)
    return df.sort_values("logloss")


def load_backtest_json(uploaded_bytes: bytes) -> List[Dict]:
    """Charge un historique de courses depuis un JSON uploadé."""
    return json.loads(uploaded_bytes.decode("utf-8"))


# =============================================================================
# 11.  INTERFACE STREAMLIT
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
    .warn-box {
        background: rgba(255,170,0,0.08);
        border: 1px solid rgba(255,170,0,0.35);
        border-radius: 10px;
        padding: 10px 14px;
        margin: 6px 0;
        color: #ffcc66;
        font-size: 0.92em;
    }
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
    if "backtest_races" not in st.session_state:
        st.session_state.backtest_races = None
    if "backtest_results" not in st.session_state:
        st.session_state.backtest_results = None


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
                           CONFIG.MARKET_WEIGHT, 0.05,
                           help="Calibré à 0.55 par défaut sur 19 courses réelles "
                                "(voir onglet Backtest) : le marché seul bat le "
                                "modèle musique/forme sur cet échantillon.")
            alpha = st.slider("α (exposant modèle)", 0.5, 2.0,
                              CONFIG.BENTER_ALPHA, 0.05)
            beta = st.slider("β (exposant marché)", 0.0, 2.0,
                             CONFIG.BENTER_BETA, 0.05)
            CONFIG.BENTER_ALPHA = alpha
            CONFIG.BENTER_BETA = beta
            CONFIG.OVERROUND_CORRECTION = st.checkbox(
                "Débiaiser favori/outsider", value=True,
                help="Correction power du biais favori-outsider")
            gamma = st.slider("γ (débiaisage favori-outsider)", 0.8, 2.0,
                              CONFIG.OVERROUND_GAMMA, 0.02,
                              help="1.12 validé empiriquement sur 19 courses "
                                   "réelles (optimum mesuré : 1.20, écart non "
                                   "significatif). Voir onglet Backtest.")
            CONFIG.OVERROUND_GAMMA = gamma
            if mw < CONFIG.WARN_MARKET_WEIGHT_BELOW:
                st.markdown(
                    f'<div class="warn-box">⚠️ Poids marché &lt; '
                    f'{CONFIG.WARN_MARKET_WEIGHT_BELOW:.2f} : non soutenu par '
                    f'le backtest disponible (voir onglet 📊 Backtest).</div>',
                    unsafe_allow_html=True)

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
            if vt < CONFIG.WARN_VALUE_THRESHOLD_BELOW:
                st.markdown(
                    f'<div class="warn-box">⚠️ Seuil de value bas : sur le '
                    f'backtest disponible, les value bets détectés ont eu un '
                    f'ROI réel négatif à tous les seuils testés.</div>',
                    unsafe_allow_html=True)

        st.markdown("---")
        st.caption(f"v{CONFIG.APP_VERSION} — {CONFIG.APP_TAG}")
        st.caption("Inspiré de Benter (1994), Harville (1973)")

    # ============= TABS =============
    tab1, tab2, tab3, tab4 = st.tabs(["📥 Données course",
                                      "📊 Pronostics",
                                      "🧪 Backtest",
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
            for w in pred.get("warnings", []):
                st.markdown(f'<div class="warn-box">{w}</div>', unsafe_allow_html=True)

    # ---------- TAB 2 : RÉSULTATS ----------
    with tab2:
        if st.session_state.prediction is None:
            st.info("🎯 Saisissez les données puis cliquez sur **LANCER L'ANALYSE**.")
        else:
            pred = st.session_state.prediction

            for w in pred.get("warnings", []):
                st.markdown(f'<div class="warn-box">{w}</div>', unsafe_allow_html=True)

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

            value_bets = [r for r in pred["results"] if r["is_value_bet"]]
            if value_bets:
                st.markdown("### 💎 Value bets détectés")
                st.caption("⚠️ Rappel : sur le backtest disponible (19 courses), "
                           "ce signal n'a pas démontré de ROI positif. À utiliser "
                           "avec prudence — voir onglet Backtest pour le détail.")
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

            st.markdown("---")
            st.markdown("## 🎲 Paris exotiques (Top combinaisons)")
            ex = pred["exotics"]
            tabs_exo = st.tabs(["Couplé Gagnant", "Couplé Placé",
                                "Trio Ordre", "Trio Désordre",
                                "Quarté+", "Quinté+"])

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

            with tabs_exo[0]: _render_exotic(ex["couple_gagnant"], "cg")
            with tabs_exo[1]: _render_exotic(ex["couple_place"], "cp")
            with tabs_exo[2]: _render_exotic(ex["trio_ordre"], "to")
            with tabs_exo[3]: _render_exotic(ex["trio_desordre"], "td")
            with tabs_exo[4]: _render_exotic(ex["quarte_desordre"], "q4")
            with tabs_exo[5]: _render_exotic(ex["quinte_desordre"], "q5")

    # ---------- TAB 3 : BACKTEST (nouveau v4.1) ----------
    with tab3:
        st.markdown("## 🧪 Backtest sur historique réel")
        st.markdown(
            "Chargez un historique de courses (cotes finales + arrivées réelles) "
            "pour mesurer objectivement la qualité de calibration des paramètres, "
            "**avec intervalle de confiance** — afin d'éviter de sur-interpréter "
            "un petit échantillon."
        )

        with st.expander("📄 Format JSON attendu", expanded=False):
            st.code("""[
  {
    "date": "01-06",
    "race_type": "Attelé",
    "horses": [
      {"number": 8, "odds": 8.0, "model_prob_pct": 38.8},
      {"number": 5, "odds": 5.0, "model_prob_pct": 8.9},
      ...
    ],
    "arrival": [5, 6, 7, 9, 12]
  },
  ...
]""", language="json")
            st.caption("\"model_prob_pct\" est optionnel : si absent pour tous les "
                       "chevaux, seul le marché débiaisé est évalué.")

        uploaded = st.file_uploader("Historique JSON", type=["json"])
        if uploaded is not None:
            try:
                races = load_backtest_json(uploaded.read())
                st.session_state.backtest_races = races
                st.success(f"✅ {len(races)} courses chargées.")
            except Exception as e:
                st.error(f"Erreur de lecture JSON : {e}")

        races = st.session_state.backtest_races
        if races:
            n_races = len(races)
            if n_races < CONFIG.MIN_BACKTEST_RACES_FOR_TRUST:
                st.markdown(
                    f'<div class="warn-box">⚠️ Échantillon de {n_races} courses : '
                    f'en-dessous du seuil de confiance recommandé '
                    f'({CONFIG.MIN_BACKTEST_RACES_FOR_TRUST}). Les métriques '
                    f'ci-dessous sont indicatives mais peuvent varier fortement '
                    f'avec quelques courses de plus ou de moins — voir les '
                    f'intervalles de confiance à 95%.</div>',
                    unsafe_allow_html=True)

            st.markdown("### Évaluation des réglages actuels (sidebar)")
            res_current = backtest_blend(races, CONFIG.BENTER_ALPHA,
                                          CONFIG.BENTER_BETA, CONFIG.OVERROUND_GAMMA)
            res_market = backtest_blend(races, 0, 1, CONFIG.OVERROUND_GAMMA)

            c1, c2, c3 = st.columns(3)
            c1.metric("Courses exploitables", res_current.get("n", 0))
            wr = res_current.get("win_hit_rate")
            ci = res_current.get("win_hit_rate_ci95")
            c2.metric("Taux victoire favori modèle",
                      f"{wr*100:.1f}%" if wr is not None else "—",
                      help=f"IC95% : [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]" if ci else None)
            c3.metric("Logloss (↓ mieux)",
                      f"{res_current.get('avg_logloss', 0):.3f}")

            st.markdown("**Comparaison Blend actuel vs Marché pur :**")
            comp_df = pd.DataFrame([
                {"Variante": f"Blend actuel (α={CONFIG.BENTER_ALPHA}, β={CONFIG.BENTER_BETA})",
                 "Win hit rate": f"{res_current.get('win_hit_rate', 0)*100:.1f}%",
                 "Podium3": f"{res_current.get('podium3_hit_rate', 0)*100:.1f}%",
                 "Podium5": f"{res_current.get('podium5_hit_rate', 0)*100:.1f}%",
                 "Logloss": f"{res_current.get('avg_logloss', 0):.3f}",
                 "ROI favori": f"{res_current.get('roi_pct', 0):+.1f}%"},
                {"Variante": "Marché pur (référence)",
                 "Win hit rate": f"{res_market.get('win_hit_rate', 0)*100:.1f}%",
                 "Podium3": f"{res_market.get('podium3_hit_rate', 0)*100:.1f}%",
                 "Podium5": f"{res_market.get('podium5_hit_rate', 0)*100:.1f}%",
                 "Logloss": f"{res_market.get('avg_logloss', 0):.3f}",
                 "ROI favori": f"{res_market.get('roi_pct', 0):+.1f}%"},
            ])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            if res_current.get("avg_logloss", 99) > res_market.get("avg_logloss", 0):
                st.markdown(
                    '<div class="warn-box">⚠️ Sur cet historique, le blend actuel '
                    'est <b>moins bien calibré que le marché seul</b> (logloss plus '
                    'élevé). Envisagez d\'augmenter le poids du marché.</div>',
                    unsafe_allow_html=True)
            else:
                st.success("✅ Le blend actuel égale ou surpasse le marché seul sur cet historique.")

            st.markdown("---")
            st.markdown("### 🔍 Recherche de la meilleure combinaison α/β")
            if st.button("Lancer le grid search (peut prendre quelques secondes)"):
                with st.spinner("Balayage de la grille α/β..."):
                    grid_df = backtest_grid_search(races)
                st.session_state.backtest_results = grid_df
            if st.session_state.backtest_results is not None:
                grid_df = st.session_state.backtest_results
                st.caption(
                    f"Triée par logloss croissant (meilleure calibration en premier). "
                    f"⚠️ Avec n={n_races} courses, plusieurs lignes du haut ont "
                    f"souvent un score quasi-identique : ce n'est PAS un signal "
                    f"fort, c'est la marge de bruit attendue à cette taille "
                    f"d'échantillon."
                )
                st.dataframe(grid_df.head(20), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("### 📋 Détail course par course (blend actuel)")
            detail_df = pd.DataFrame(res_current.get("per_race", []))
            if not detail_df.empty:
                st.dataframe(detail_df, use_container_width=True, hide_index=True)
        else:
            st.info("📤 Chargez un historique JSON pour démarrer le backtest.")

    # ---------- TAB 4 : AIDE ----------
    with tab4:
        st.markdown("""
## 🎓 Méthodologie QuantTurf v4.1

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
Cotes marché → Débiaisage power (γ=1.12) → p_marché
                                              ↓
                              BENTER BLEND : p ∝ p_modèle^α · p_marché^β
                                              ↓
                          Plackett-Luce (6000 ordres simulés)
                                              ↓
                    Win / Place / Couplé / Trio / Quarté+ / Quinté+
                                              ↓
                                   Kelly dynamique + ROI
                                              ↓
                       🧪 Backtest (onglet dédié) → recalibrage continu
```

### 📚 Formules clés

**1. Shrinkage bayésien (musique)**
$$\\text{score}_{\\text{shrunk}} = \\frac{n \\cdot \\text{score}_{\\text{obs}} + K \\cdot \\mu_{\\text{pop}}}{n + K}$$

**2. Débiaisage des cotes (favori-outsider correction)**
$$p_{\\text{vraie}} \\propto \\left(\\frac{1}{\\text{cote}}\\right)^\\gamma, \\quad \\gamma \\approx 1.12$$

**3. Benter Blend**
$$p_{\\text{finale}} \\propto p_{\\text{modèle}}^\\alpha \\cdot p_{\\text{marché}}^\\beta$$

**4. Plackett-Luce (Harville)** — ordre d'arrivée séquentiel proportionnel aux forces.

**5. Kelly fractionnaire dynamique**
$$f^* = \\frac{p \\cdot b - q}{b}, \\quad f_{\\text{misé}} = \\min\\left(f^* \\cdot \\frac{1}{1+\\text{vol}}, f_{\\max}\\right)$$

### 🧪 Ce que le backtest sur 19 courses (juin 2026) a montré

Une calibration a été effectuée sur les 19 dernières courses Quinté+ disponibles
au moment de la mise à jour (01 au 19 juin 2026), cotes finales + arrivées
officielles. **Résultats à prendre avec la prudence qu'impose cette taille
d'échantillon (IC95% de ±18 points sur un taux de réussite de ~20%)** :

| Constat | Donnée mesurée |
|---|---|
| Marché débiaisé seul vs modèle musique/forme seul | Logloss 2.22 vs 2.88 — le marché gagne nettement |
| "Value bets" du modèle pur, à tout seuil testé | ROI réel entre **-70% et -86%** |
| γ de débiaisage favori-outsider | 1.12 quasi optimal (optimum mesuré : 1.20) |
| Effet corde (Plat / Autostart) | Aucun signal exploitable (9-10 courses/discipline) |
| Blend optimal mesuré (logloss) | Converge vers ~100% marché sur cet échantillon |

**Conséquence pratique** : le `MARKET_WEIGHT` par défaut a été relevé (0.35→0.55)
plutôt qu'abaissé, à l'inverse de l'intuition "faire confiance à notre modèle".
Les réglages α=1.20/β=0.80 demandés sont conservés dans l'interface mais un
garde-fou avertit quand `market_weight` ou `value_threshold` s'écartent trop
de ce que les données actuelles soutiennent. **Utilisez l'onglet 🧪 Backtest
pour recharger des historiques plus larges au fil du temps** — la confiance
dans un modèle propriétaire (musique/forme) ne peut s'établir qu'avec des
centaines de courses, pas 19.

### 🎯 Stratégie recommandée

| Type de pari | Quand l'utiliser | Risque |
|---|---|---|
| **Gagnant (value)** | Ratio > 1.20 ET cote > 2.5 — à valider par backtest avant mise réelle | 🟡 Moyen |
| **Placé** | Champion avec cote ≥ 4 | 🟢 Faible |
| **Couplé Placé** | ROI > 50% | 🟡 Moyen |
| **Trio désordre** | ROI > 100% sur 3 favoris | 🟠 Élevé |
| **Quinté+** | Mise faible, ROI espéré > 200% | 🔴 Très élevé |

### ⚠️ Avertissements

- 🎰 **Les performances passées ne préjugent pas des résultats futurs**
- 💸 **Jouez avec modération** — ne misez jamais plus que ce que vous pouvez perdre
- 📊 Le modèle nécessite un marché suffisamment liquide pour le Benter Blend
- 🐎 La corde au Trot n'est pertinente qu'en départ **AUTOSTART**
- 🔍 Les statistiques empiriques de corde sont des **valeurs indicatives basées
  sur des études publiques à large échelle** (non recalibrables sur 19 courses)
- 🧪 **Le module Backtest n'est pas une garantie de profit** : il mesure la
  calibration passée, pas la performance future. Jouer responsable :
  joueurs-info-service.fr (09 74 75 13 13)

### 📖 Références

- Benter, W. (1994). *Computer Based Horse Race Handicapping and Wagering Systems.*
- Harville, D. (1973). *Assigning Probabilities to the Outcomes of Multi-Entry Competitions.*
- Kelly, J. L. (1956). *A New Interpretation of Information Rate.*
- Snowberg & Wolfers (2010). *Explaining the Favorite-Longshot Bias.*
        """)


if __name__ == "__main__":
    main()



import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import zscore, norm
from itertools import combinations
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
import logging
import time
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION v3.1.1 FINAL
# =============================================================================

@dataclass
class ConfigFinal:
    """v3.1.1 Final Configuration"""
    APP_VERSION: str = "3.1.1"
    APP_NAME: str = "QuantTurf Pro"
    
    # Core
    MC_ITERATIONS: int = 3000
    MARKET_WEIGHT: float = 0.35
    VALUE_THRESHOLD: float = 1.15
    TEMPERATURE: float = 1.5
    NOISE_BASE: float = 0.15
    
    # Kelly
    KELLY_FRACTION: float = 0.25
    MIN_KELLY_ODDS: float = 2.50
    
    # Music parsing
    MUSIC_POSITION_SCORES: Dict[str, float] = None
    MUSIC_RACE_TYPE_WEIGHTS: Dict[str, float] = None
    DRAW_IMPACT_BASE: Dict[int, float] = None
    RACE_TYPES: List[str] = None
    
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
        
        if self.DRAW_IMPACT_BASE is None:
            self.DRAW_IMPACT_BASE = {
                1: 0.35, 2: 0.40, 3: 0.35, 4: 0.25, 5: 0.15,
                6: 0.05, 7: -0.05, 8: -0.12, 9: -0.18, 10: -0.24,
                11: -0.30, 12: -0.35, 13: -0.40, 14: -0.44, 15: -0.48,
                16: -0.50, 17: -0.52, 18: -0.54, 19: -0.55, 20: -0.55,
            }
        
        if self.RACE_TYPES is None:
            self.RACE_TYPES = ["Plat", "Attelé", "Monté", "Haies", "Steeple-chase", "Cross-country"]

CONFIG = ConfigFinal()

# =============================================================================
# MUSIC METRICS & PARSING
# =============================================================================

@dataclass
class MusicMetrics:
    """Music analysis metrics"""
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
    win_streak: int = 0
    place_streak: int = 0
    consistency: float = 0.0


@lru_cache(maxsize=512)
def parse_music_final(music_str: str) -> MusicMetrics:
    """
    Enhanced music parsing for horse, driver, or trainer
    Same algorithm for all three - parses position history
    """
    if not music_str or music_str.strip() in ("", "-", "INEDIT", "INÉDIT", "N/A", "0"):
        return MusicMetrics(
            score=3.0, regularity=0.50, races_count=0,
            avg_position=5.0, best_position=10, recent_form=3.0,
            trend=0.0, is_debutant=True, win_ratio=0.0, podium_ratio=0.0
        )
    
    try:
        clean = music_str.strip().upper()
        clean = re.sub(r"[() ]", "", clean)
        tokens = re.findall(r"([0-9DATRP])([AMPHSC]?)", clean)
        
        if not tokens:
            return MusicMetrics(
                score=3.0, regularity=0.50, races_count=0,
                avg_position=5.0, best_position=10, recent_form=3.0,
                trend=0.0, is_debutant=True, win_ratio=0.0, podium_ratio=0.0
            )
        
        raw_scores, numeric_positions = [], []
        
        for pos_char, rtype_char in tokens:
            rtype = rtype_char.lower() if rtype_char else "x"
            pos_score = CONFIG.MUSIC_POSITION_SCORES.get(pos_char, 0.3)
            type_weight = CONFIG.MUSIC_RACE_TYPE_WEIGHTS.get(rtype, 1.0)
            raw_scores.append(pos_score * type_weight)
            
            if pos_char.isdigit():
                numeric_positions.append(int(pos_char) if pos_char != "0" else 10)
        
        n = len(raw_scores)
        raw_scores = np.array(raw_scores)
        
        # Exponential decay (recent = more weight)
        decay = np.array([np.exp(-0.30 * i) for i in range(n)])
        decay /= decay.sum()
        weighted_score = float(np.dot(raw_scores, decay))
        
        # Recent form (3 races)
        recent_n = min(3, n)
        recent_decay = decay[:recent_n] / decay[:recent_n].sum()
        recent_form = float(np.dot(raw_scores[:recent_n], recent_decay))
        
        # Regularity
        if len(numeric_positions) >= 2:
            pos_std = float(np.std(numeric_positions))
            regularity = max(0.0, 1.0 - pos_std / 5.0)
        else:
            regularity = 0.50
        
        # Trend
        if n >= 4:
            recent_avg = np.mean(raw_scores[:n // 2])
            old_avg = np.mean(raw_scores[n // 2:])
            trend = (recent_avg - old_avg) / (abs(old_avg) + 1e-9)
        else:
            trend = 0.0
        
        # Win/podium ratios
        win_count = sum(1 for p in numeric_positions if p == 1)
        podium_count = sum(1 for p in numeric_positions if p <= 3)
        
        # Streaks
        win_streak = _calculate_streak(numeric_positions, 1)
        place_streak = _calculate_streak(numeric_positions, 3)
        
        consistency = 1.0 - (pos_std / 10.0 if len(numeric_positions) >= 2 else 0.5)
        consistency = max(0.0, min(1.0, consistency))
        
        return MusicMetrics(
            score=weighted_score,
            regularity=regularity,
            races_count=n,
            avg_position=float(np.mean(numeric_positions)) if numeric_positions else 5.0,
            best_position=int(min(numeric_positions)) if numeric_positions else 10,
            recent_form=recent_form,
            trend=float(trend),
            is_debutant=False,
            win_ratio=win_count / max(n, 1),
            podium_ratio=podium_count / max(n, 1),
            win_streak=win_streak,
            place_streak=place_streak,
            consistency=consistency,
        )
    
    except Exception as e:
        logger.warning(f"Music parsing error: {str(e)}")
        return MusicMetrics(
            score=3.0, regularity=0.50, races_count=0,
            avg_position=5.0, best_position=10, recent_form=3.0,
            trend=0.0, is_debutant=True, win_ratio=0.0, podium_ratio=0.0
        )


def _calculate_streak(positions: List[int], threshold: int) -> int:
    """Calculate recent streak of good finishes"""
    if not positions:
        return 0
    streak = 0
    for p in positions[:5]:
        if p <= threshold:
            streak += 1
        else:
            break
    return streak

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def draw_factor(draw: int, race_type: str, distance: int) -> float:
    """Draw impact for Plat only"""
    if race_type != "Plat" or not draw or draw <= 0:
        return 0.0
    
    draw = min(int(draw), 20)
    base = CONFIG.DRAW_IMPACT_BASE.get(draw, -0.55)
    
    if distance <= 1400:
        return base * 1.60
    elif distance <= 1800:
        return base * 1.00
    else:
        return base * 0.45


def market_prob(odds: float, n_runners: int) -> float:
    """Implied probability from odds"""
    if not odds or odds <= 1.01:
        return 1.0 / max(n_runners, 2)
    return 1.0 / float(odds)

# =============================================================================
# WEIGHTS v3.1.1 FINAL (OPTIMIZED FOR 3 MUSIC SOURCES)
# =============================================================================

def get_weights_v311_final(race_type: str) -> Dict[str, float]:
    """
    v3.1.1 FINAL WEIGHTS - Optimized for:
    - Horse Music (primary: 35%)
    - Driver Music (secondary: 33%)
    - Trainer Music (tertiary: 27%)
    - Draw (Plat only: 3%)
    - Synergy: 2%
    
    Principle: 3 independent music sources = more robust prediction
    """
    
    if race_type == "Plat":
        return {
            # HORSE MUSIC (35% total)
            "horse_music_score": 0.18,
            "horse_recent_form": 0.10,
            "horse_regularity": 0.04,
            "horse_trend": 0.02,
            "horse_win_ratio": 0.01,
            
            # DRIVER MUSIC (33% total) - Very important for flat races
            "driver_music_score": 0.17,
            "driver_recent_form": 0.10,
            "driver_regularity": 0.04,
            "driver_trend": 0.01,
            "driver_win_ratio": 0.01,
            
            # TRAINER MUSIC (27% total)
            "trainer_music_score": 0.13,
            "trainer_recent_form": 0.08,
            "trainer_regularity": 0.04,
            "trainer_trend": 0.01,
            "trainer_win_ratio": 0.01,
            
            # DRAW (3% total)
            "draw_factor": 0.03,
            
            # SYNERGY (2% total)
            "synergy_score": 0.02,
        }
    
    elif race_type in ("Attelé", "Monté"):
        return {
            # HORSE MUSIC (30%)
            "horse_music_score": 0.16,
            "horse_recent_form": 0.08,
            "horse_regularity": 0.03,
            "horse_trend": 0.02,
            "horse_win_ratio": 0.01,
            
            # DRIVER MUSIC (40%) - Critical for harness/mounted racing
            "driver_music_score": 0.21,
            "driver_recent_form": 0.12,
            "driver_regularity": 0.04,
            "driver_trend": 0.02,
            "driver_win_ratio": 0.01,
            
            # TRAINER MUSIC (25%)
            "trainer_music_score": 0.12,
            "trainer_recent_form": 0.07,
            "trainer_regularity": 0.03,
            "trainer_trend": 0.01,
            "trainer_win_ratio": 0.01,
            
            # DRAW (N/A)
            "draw_factor": 0.00,
            
            # SYNERGY (3% - partnership critical)
            "synergy_score": 0.03,
        }
    
    else:  # Obstacles (Haies, Steeple, Cross)
        return {
            # HORSE MUSIC (38%) - Horse ability crucial for jumps
            "horse_music_score": 0.20,
            "horse_recent_form": 0.10,
            "horse_regularity": 0.05,
            "horse_trend": 0.02,
            "horse_win_ratio": 0.01,
            
            # DRIVER MUSIC (28%)
            "driver_music_score": 0.14,
            "driver_recent_form": 0.08,
            "driver_regularity": 0.03,
            "driver_trend": 0.02,
            "driver_win_ratio": 0.01,
            
            # TRAINER MUSIC (31%)
            "trainer_music_score": 0.16,
            "trainer_recent_form": 0.09,
            "trainer_regularity": 0.04,
            "trainer_trend": 0.01,
            "trainer_win_ratio": 0.01,
            
            # DRAW (N/A)
            "draw_factor": 0.00,
            
            # SYNERGY
            "synergy_score": 0.02,
        }

# =============================================================================
# COMPOSITE SCORE v3.1.1
# =============================================================================

def composite_score_final(feat: Dict, weights: Dict) -> float:
    """Composite score with 3 music sources"""
    score = 0.0
    
    # Horse Music
    score += weights.get("horse_music_score", 0.18) * feat.get("horse_music_score", 3.0)
    score += weights.get("horse_recent_form", 0.10) * feat.get("horse_recent_form", 3.0)
    score += weights.get("horse_regularity", 0.04) * feat.get("horse_regularity", 0.5) * 10.0
    score += weights.get("horse_trend", 0.02) * (feat.get("horse_trend", 0.0) + 1.0) * 5.0
    score += weights.get("horse_win_ratio", 0.01) * feat.get("horse_win_ratio", 0.0) * 20.0
    
    # Driver Music
    score += weights.get("driver_music_score", 0.17) * feat.get("driver_music_score", 3.0)
    score += weights.get("driver_recent_form", 0.10) * feat.get("driver_recent_form", 3.0)
    score += weights.get("driver_regularity", 0.04) * feat.get("driver_regularity", 0.5) * 10.0
    score += weights.get("driver_trend", 0.01) * (feat.get("driver_trend", 0.0) + 1.0) * 5.0
    score += weights.get("driver_win_ratio", 0.01) * feat.get("driver_win_ratio", 0.0) * 20.0
    
    # Trainer Music
    score += weights.get("trainer_music_score", 0.13) * feat.get("trainer_music_score", 3.0)
    score += weights.get("trainer_recent_form", 0.08) * feat.get("trainer_recent_form", 3.0)
    score += weights.get("trainer_regularity", 0.04) * feat.get("trainer_regularity", 0.5) * 10.0
    score += weights.get("trainer_trend", 0.01) * (feat.get("trainer_trend", 0.0) + 1.0) * 5.0
    score += weights.get("trainer_win_ratio", 0.01) * feat.get("trainer_win_ratio", 0.0) * 20.0
    
    # Draw (Plat only)
    if weights.get("draw_factor", 0) > 0:
        score += weights["draw_factor"] * (feat.get("draw_factor", 0.0) + 1.0) * 5.0
    
    # Synergy bonus (alignment of driver + trainer + horse)
    horse_m = feat.get("horse_music_score", 3.0)
    driver_m = feat.get("driver_music_score", 3.0)
    trainer_m = feat.get("trainer_music_score", 3.0)
    
    # Synergy: bonus when all three are aligned
    all_scores = [horse_m, driver_m, trainer_m]
    synergy = min(all_scores) / (max(all_scores) + 1e-9)
    score += weights.get("synergy_score", 0.02) * synergy * 10.0
    
    return max(0.01, score)

# =============================================================================
# PROBABILITY FUNCTIONS
# =============================================================================

def softmax(scores: np.ndarray, temperature: float = CONFIG.TEMPERATURE) -> np.ndarray:
    """Numerically stable softmax"""
    s = np.array(scores, dtype=float) / temperature
    s -= s.max()
    e = np.exp(s)
    return e / e.sum()


def logit_calibration(raw_probs: np.ndarray) -> np.ndarray:
    """Platt scaling calibration"""
    eps = 1e-9
    logit = np.log((raw_probs + eps) / (1 - raw_probs + eps))
    logit = logit - logit.mean() * 0.1
    calibrated = 1.0 / (1.0 + np.exp(-logit))
    return calibrated / calibrated.sum()


def bayesian_blend(model_probs: np.ndarray, market_probs: np.ndarray,
                   market_weight: float) -> np.ndarray:
    """Log-odds Bayesian blending"""
    mp = np.array(market_probs, dtype=float)
    if mp.sum() < 1e-9:
        mp = np.ones(len(model_probs)) / len(model_probs)
    else:
        mp /= mp.sum()
    
    eps = 1e-9
    lo_model = np.log((model_probs + eps) / (1 - model_probs + eps))
    lo_market = np.log((mp + eps) / (1 - mp + eps))
    
    lo_blend = (1 - market_weight) * lo_model + market_weight * lo_market
    blended = 1.0 / (1.0 + np.exp(-lo_blend))
    return blended / blended.sum()

# =============================================================================
# MONTE CARLO
# =============================================================================

def monte_carlo_final(features_list: List[Dict], weights: Dict, 
                      n_iter: int = CONFIG.MC_ITERATIONS) -> Dict:
    """Monte Carlo simulation"""
    n = len(features_list)
    all_probs = np.zeros((n_iter, n))
    win_counts = np.zeros(n)
    
    base_scores = np.array([composite_score_final(f, weights) for f in features_list])
    
    # Pre-compute noise factors
    noise_factors = np.array([
        2.20 if f.get("horse_is_debutant", False) else
        1.60 if f.get("horse_regularity", 0.5) < 0.30 else
        0.70 if f.get("horse_regularity", 0.5) > 0.80 else
        1.00
        for f in features_list
    ])
    
    for it in range(n_iter):
        noises = np.random.normal(0, CONFIG.NOISE_BASE * noise_factors, n)
        noisy = base_scores * np.exp(noises)
        noisy = np.maximum(noisy, 0.001)
        
        probs = softmax(noisy)
        all_probs[it] = probs
        
        winner = np.random.choice(n, p=probs)
        win_counts[winner] += 1
    
    simulated_probs = win_counts / n_iter
    mean_probs = all_probs.mean(axis=0)
    std_probs = all_probs.std(axis=0)
    vol_per_horse = std_probs / (mean_probs + 1e-9)
    
    place_counts = np.zeros(n)
    for it in range(n_iter):
        top2 = np.argsort(-all_probs[it])[:2]
        place_counts[top2] += 1
    place_probs = place_counts / n_iter
    
    return {
        "simulated_probs": simulated_probs,
        "mean_probs": mean_probs,
        "std_probs": std_probs,
        "vol_per_horse": vol_per_horse,
        "place_probs": place_probs,
    }

# =============================================================================
# KELLY & ROI
# =============================================================================

def calculate_kelly_bet(prob: float, odds: float, 
                       kelly_fraction: float = CONFIG.KELLY_FRACTION) -> Tuple[float, float]:
    """Kelly Criterion"""
    if odds <= CONFIG.MIN_KELLY_ODDS or prob < 0.10:
        return 0.0, 0.0
    
    q = 1.0 - prob
    b = odds - 1.0
    kelly = (prob * b - q) / b
    kelly = max(0.0, kelly)
    fractional_kelly = kelly * kelly_fraction
    
    return float(kelly), float(fractional_kelly)


def calculate_roi(prob: float, odds: float, bet_amount: float = 100.0) -> float:
    """Expected ROI"""
    if bet_amount <= 0 or odds <= 1.0:
        return 0.0
    
    expected_winnings = bet_amount * odds * prob
    expected_loss = bet_amount * (1 - prob)
    expected_value = expected_winnings - expected_loss
    
    return (expected_value / bet_amount) * 100.0

# =============================================================================
# MAIN ENGINE
# =============================================================================

def run_engine_final(race_info: Dict, horses: List[Dict],
                    mc_iter: int = CONFIG.MC_ITERATIONS,
                    market_weight: float = CONFIG.MARKET_WEIGHT,
                    value_threshold: float = CONFIG.VALUE_THRESHOLD) -> Dict:
    """Main prediction engine v3.1.1"""
    start_time = time.time()
    
    try:
        n_runners = len(horses)
        race_info["n_runners"] = n_runners
        race_type = race_info.get("race_type", "Plat")
        distance = int(race_info.get("distance", 1600))
        
        # Feature engineering
        feats = []
        for h in horses:
            # Parse all three music sources
            horse_music = parse_music_final(h.get("horse_music", ""))
            driver_music = parse_music_final(h.get("driver_music", ""))
            trainer_music = parse_music_final(h.get("trainer_music", ""))
            
            feat = {
                "number": h.get("number", 0),
                "name": h.get("name", ""),
                "odds": float(h.get("odds", 0)),
                
                # Horse music
                "horse_music_score": horse_music.score,
                "horse_recent_form": horse_music.recent_form,
                "horse_regularity": horse_music.regularity,
                "horse_trend": horse_music.trend,
                "horse_win_ratio": horse_music.win_ratio,
                "horse_races_count": horse_music.races_count,
                "horse_is_debutant": horse_music.is_debutant,
                
                # Driver music
                "driver_music_score": driver_music.score,
                "driver_recent_form": driver_music.recent_form,
                "driver_regularity": driver_music.regularity,
                "driver_trend": driver_music.trend,
                "driver_win_ratio": driver_music.win_ratio,
                "driver_races_count": driver_music.races_count,
                
                # Trainer music
                "trainer_music_score": trainer_music.score,
                "trainer_recent_form": trainer_music.recent_form,
                "trainer_regularity": trainer_music.regularity,
                "trainer_trend": trainer_music.trend,
                "trainer_win_ratio": trainer_music.win_ratio,
                "trainer_races_count": trainer_music.races_count,
                
                # Draw
                "draw_factor": draw_factor(h.get("draw", 0), race_type, distance),
                
                # Market
                "market_prob": market_prob(h.get("odds", 0), n_runners),
            }
            feats.append(feat)
        
        # Normalize
        df = pd.DataFrame(feats)
        
        norm_cols = [
            "horse_music_score", "horse_recent_form", "horse_regularity",
            "driver_music_score", "driver_recent_form", "driver_regularity",
            "trainer_music_score", "trainer_recent_form", "trainer_regularity",
        ]
        
        for col in norm_cols:
            if col in df.columns:
                vals = df[col].values.astype(float)
                std = vals.std()
                if std > 1e-9:
                    df[f"{col}_z"] = (vals - vals.mean()) / std
                else:
                    df[f"{col}_z"] = 0.0
        
        feats = df.to_dict("records")
        
        # Weights
        weights = get_weights_v311_final(race_type)
        
        # Scores
        scores = np.array([composite_score_final(f, weights) for f in feats])
        
        # Probabilities
        sm_probs = softmax(scores)
        cal_probs = logit_calibration(sm_probs)
        
        # Market
        raw_mkt = np.array([f["market_prob"] for f in feats])
        if raw_mkt.sum() < 1e-9:
            raw_mkt = np.ones(n_runners) / n_runners
        norm_mkt = raw_mkt / raw_mkt.sum()
        
        # Bayesian
        has_odds = any(h.get("odds", 0) > CONFIG.MIN_KELLY_ODDS for h in horses)
        if has_odds:
            bayes_probs = bayesian_blend(cal_probs, norm_mkt, market_weight)
        else:
            bayes_probs = cal_probs
        
        # Monte Carlo
        mc = monte_carlo_final(feats, weights, n_iter=mc_iter)
        
        # Final blend
        final_probs = 0.55 * bayes_probs + 0.45 * mc["mean_probs"]
        final_probs /= final_probs.sum()
        
        # Z-score
        prob_z = zscore(final_probs)
        
        # Results
        results = []
        for i, (feat, horse) in enumerate(zip(feats, horses)):
            ratio = final_probs[i] / (norm_mkt[i] + 1e-9)
            is_value = ratio >= value_threshold and final_probs[i] >= 0.04
            
            kelly, kelly_frac = calculate_kelly_bet(final_probs[i], horse.get("odds", 2.0))
            roi = calculate_roi(final_probs[i], horse.get("odds", 2.0), 100.0)
            
            result = {
                "rank": 0,
                "number": horse.get("number", i + 1),
                "name": horse.get("name", f"Cheval {i+1}"),
                "odds": float(horse.get("odds", 0)),
                "model_prob": round(float(final_probs[i]) * 100, 2),
                "market_prob": round(float(norm_mkt[i]) * 100, 2),
                "place_prob": round(float(mc["place_probs"][i]) * 100, 2),
                "composite_score": round(float(scores[i]), 4),
                
                # Horse
                "horse_music": round(feat.get("horse_music_score", 0.0), 2),
                "horse_form": round(feat.get("horse_recent_form", 0.0), 2),
                "horse_reg": round(feat.get("horse_regularity", 0.0), 2),
                
                # Driver
                "driver_music": round(feat.get("driver_music_score", 0.0), 2),
                "driver_form": round(feat.get("driver_recent_form", 0.0), 2),
                "driver_reg": round(feat.get("driver_regularity", 0.0), 2),
                
                # Trainer
                "trainer_music": round(feat.get("trainer_music_score", 0.0), 2),
                "trainer_form": round(feat.get("trainer_recent_form", 0.0), 2),
                "trainer_reg": round(feat.get("trainer_regularity", 0.0), 2),
                
                # Betting
                "value_ratio": round(float(ratio), 2),
                "is_value_bet": is_value,
                "kelly_criterion": round(kelly, 4),
                "kelly_bet_fraction": round(kelly_frac, 4),
                "expected_roi": round(roi, 2),
                "mc_std": round(float(mc["std_probs"][i]) * 100, 2),
                "prob_z": round(float(prob_z[i]), 3),
            }
            results.append(result)
        
        # Sort
        results.sort(key=lambda x: x["model_prob"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        
        # Selections
        bases = results[:2]
        outsiders = [r for r in results[2:] if r["model_prob"] > 2.5]
        outsiders.sort(key=lambda x: x["value_ratio"], reverse=True)
        outsiders = outsiders[:3]
        
        # Combos
        top6 = [r["number"] for r in results[:min(6, n_runners)]]
        trio_combos = list(combinations(top6, 3))[:10]
        
        top8 = [r["number"] for r in results[:min(8, n_runners)]]
        quinte_combos = list(combinations(top8, 5))[:10]
        
        # Indices
        sorted_p = sorted([r["model_prob"] for r in results], reverse=True)
        if len(sorted_p) >= 2:
            gap = sorted_p[0] - sorted_p[1]
            conf_idx = min(100.0, round(45.0 + gap * 2.2, 1))
        else:
            conf_idx = 50.0
        
        avg_vol = float(mc["vol_per_horse"].mean())
        vol_idx = min(100.0, round(avg_vol * 55.0, 1))
        
        # Overround
        if has_odds:
            raw_overround = sum(1.0 / h["odds"] for h in horses if h.get("odds", 0) > 1.01)
            overround_pct = round((raw_overround - 1.0) * 100, 1)
        else:
            overround_pct = None
        
        execution_time = time.time() - start_time
        
        return {
            "results": results,
            "bases": bases,
            "outsiders": outsiders,
            "trio_combos": trio_combos,
            "quinte_combos": quinte_combos,
            "confidence_idx": conf_idx,
            "volatility_idx": vol_idx,
            "overround_pct": overround_pct,
            "weights": weights,
            "execution_time": round(execution_time, 2),
        }
    
    except Exception as e:
        logger.error(f"Engine error: {str(e)}")
        raise

# =============================================================================
# STREAMLIT UI
# =============================================================================

def apply_css() -> None:
    st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #07071a 0%, #0d1b2a 40%, #12192b 100%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1b2a, #07071a); }
h1, h2, h3 { color: #e8e8e8 !important; }
</style>
""", unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(f"""
<div style="text-align:center; padding: 22px 0;">
    <h1 style="font-size:2.8em; background: linear-gradient(90deg,#00ff88,#00b4d8);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        🏇 {CONFIG.APP_NAME} v{CONFIG.APP_VERSION}
    </h1>
    <p style="color:#6b7fa3;">Horse + Driver + Trainer Music Analytics</p>
</div>
""", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title=f"🏇 {CONFIG.APP_NAME} v{CONFIG.APP_VERSION}",
        page_icon="🏇",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    apply_css()
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        mc_iter = st.slider("MC Itérations", 500, 5000, CONFIG.MC_ITERATIONS, 250)
        mw = st.slider("Poids Marché", 0.0, 0.60, CONFIG.MARKET_WEIGHT, 0.05)
        vt = st.slider("Seuil Value", 1.05, 1.60, CONFIG.VALUE_THRESHOLD, 0.05)
    
    tab1, tab2 = st.tabs(["📥 Données", "📊 Résultats"])
    
    with tab1:
        st.markdown("## 🏁 Course")
        c1, c2, c3 = st.columns(3)
        with c1:
            race_type = st.selectbox("Type", CONFIG.RACE_TYPES)
        with c2:
            distance = st.number_input("Distance (m)", 800, 7200, 1600, 100)
        with c3:
            discipline = st.text_input("Prix")
        
        st.markdown("---\n## 🐎 Partants (Tableau Éditable)")
        st.markdown("*Copier/coller depuis Excel. Modifiez directement.*")
        
        # Initialize session state with clean column names (no special chars)
        if "horses_df" not in st.session_state:
            st.session_state.horses_df = pd.DataFrame({
                "num": range(1, 11),
                "nom": [f"Cheval {i+1}" for i in range(10)],
                "cote": [5.0] * 10,
                "musique_cheval": [""] * 10,
                "musique_driver": [""] * 10,
                "musique_entraineur": [""] * 10,
                "corde": [0] * 10,
            })
        
        # Define column configuration for display
        column_config = {
            "num": st.column_config.NumberColumn("N°", help="Numéro du cheval", required=True),
            "nom": st.column_config.TextColumn("Nom", help="Nom du cheval", required=True),
            "cote": st.column_config.NumberColumn("Cote", help="Cote (ex: 5.5)", format="%.2f"),
            "musique_cheval": st.column_config.TextColumn("Musique Cheval", help="Ex: 2a 1p 3m"),
            "musique_driver": st.column_config.TextColumn("Musique Driver", help="Ex: 1a 2p"),
            "musique_entraineur": st.column_config.TextColumn("Musique Entraîneur", help="Ex: 3a 1p"),
            "corde": st.column_config.NumberColumn("Corde", help="Numéro de corde (Plat uniquement)", format="%d"),
        }
        
        # Editable table with clean column names and custom display
        edited_df = st.data_editor(
            st.session_state.horses_df,
            column_config=column_config,
            use_container_width=True,
            num_rows="dynamic",
            key="horses_editor"
        )
        st.session_state.horses_df = edited_df
        
        st.markdown("---")
        
        if st.button("🚀 ANALYSER", use_container_width=True, key="analyze_btn"):
            if len(edited_df) < 2:
                st.error("❌ Minimum 2 partants")
                return
            
            # Mapping from internal column names to expected fields
            # We'll use the internal names directly
            horses_input = []
            for idx, row in edited_df.iterrows():
                try:
                    # Helper functions
                    def to_float(val, default=0.0):
                        if pd.isna(val) or val == '' or val is None:
                            return default
                        try:
                            s = str(val).strip().replace(',', '.')
                            return float(s)
                        except:
                            return default
                    
                    def to_int(val, default=0):
                        if pd.isna(val) or val == '' or val is None:
                            return default
                        try:
                            s = str(val).strip()
                            s = ''.join(c for c in s if c.isdigit())
                            return int(s) if s else default
                        except:
                            return default
                    
                    number = to_int(row.get("num"), default=idx+1)
                    name = str(row.get("nom", "")).strip() if not pd.isna(row.get("nom")) else f"Cheval {number}"
                    odds = to_float(row.get("cote"), default=5.0)
                    horse_music = str(row.get("musique_cheval", "")).strip() if not pd.isna(row.get("musique_cheval")) else ''
                    driver_music = str(row.get("musique_driver", "")).strip() if not pd.isna(row.get("musique_driver")) else ''
                    trainer_music = str(row.get("musique_entraineur", "")).strip() if not pd.isna(row.get("musique_entraineur")) else ''
                    draw = to_int(row.get("corde"), default=0)
                    
                    horses_input.append({
                        "number": number,
                        "name": name,
                        "odds": odds,
                        "horse_music": horse_music,
                        "driver_music": driver_music,
                        "trainer_music": trainer_music,
                        "draw": draw,
                    })
                except Exception as e:
                    st.error(f"❌ Erreur à la ligne {idx}: {str(e)}")
                    return
            
            with st.spinner("Analyse en cours..."):
                try:
                    pred = run_engine_final(
                        {"race_type": race_type, "distance": int(distance),
                         "discipline": discipline},
                        horses_input,
                        mc_iter=mc_iter, market_weight=mw, value_threshold=vt
                    )
                    st.session_state["prediction"] = pred
                    st.session_state["race_info"] = {
                        "race_type": race_type, "distance": distance
                    }
                    st.success(f"✅ Analyse réussie en {pred['execution_time']}s")
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
    
    with tab2:
        if "prediction" not in st.session_state:
            st.info("💡 Lancez l'analyse depuis l'onglet Données")
        else:
            pred = st.session_state["prediction"]
            
            # KPIs
            st.markdown("## 📊 KPIs")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Confiance", f"{pred['confidence_idx']}/100")
            with k2:
                st.metric("Volatilité", f"{pred['volatility_idx']}/100")
            with k3:
                st.metric("Partants", len(pred["results"]))
            with k4:
                vb = sum(1 for r in pred["results"] if r["is_value_bet"])
                st.metric("Value Bets", vb)
            
            st.markdown("---\n## 🏆 Classement Complet")
            
            # Results table
            res_df = []
            for r in pred["results"]:
                res_df.append({
                    "Rg": r["rank"],
                    "N°": r["number"],
                    "Nom": r["name"],
                    "Modèle%": f"{r['model_prob']:.1f}",
                    "Marché%": f"{r['market_prob']:.1f}",
                    "Cheval": f"{r['horse_music']:.1f}",
                    "Driver": f"{r['driver_music']:.1f}",
                    "Entraîneur": f"{r['trainer_music']:.1f}",
                    "Kelly%": f"{r['kelly_bet_fraction']*100:.2f}",
                    "ROI%": f"{r['expected_roi']:.1f}",
                    "Value": "🟢" if r["is_value_bet"] else ("🔴" if r["value_ratio"] < 1.0 else "⚪"),
                })
            
            st.dataframe(pd.DataFrame(res_df), use_container_width=True, hide_index=True)
            
            st.markdown("---\n## ⚙️ Poids Utilisés")
            weights_df = []
            for key, val in pred["weights"].items():
                weights_df.append({"Paramètre": key, "Poids": f"{val:.1%}"})
            
            st.dataframe(pd.DataFrame(weights_df), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

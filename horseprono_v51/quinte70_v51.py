"""Générateur intelligent de 70 combinaisons Quinté V5.1.

Le pool reste strictement limité aux 10 premiers du classement officiel. Les
70 tickets ne sont pas un simple élargissement des 50 précédents : ils couvrent
quatre scénarios de course et utilisent les signaux V5.1/V4.6 lorsqu'ils sont
disponibles.

Architecture des 70 tickets :
- 1  NOYAU PUR     : 5 chevaux du Top5 ;
- 25 NOYAU FORT    : 4 Top5 + 1 cheval des rangs 6-10 (toutes les variantes) ;
- 34 EQUILIBRES    : 3 Top5 + 2 chevaux des rangs 6-10 ;
- 10 OUVERTS       : 2 Top5 + 3 chevaux des rangs 6-10.

Les tickets 3+2 et 2+3 sont choisis par un score combinant force individuelle,
consensus marché/fondamental, méta-score V5.1, déficit d'exposition et diversité
des paires/triples déjà utilisés.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


# Exposition cible sur 70 tickets = 350 places.
# Top5 : 227 places ; rangs 6-10 : 123 places.
TARGET_EXPOSURES = [52, 49, 46, 42, 38, 32, 28, 24, 21, 18]
RANK_PRIOR = [1.00, 0.94, 0.88, 0.82, 0.76, 0.65, 0.57, 0.50, 0.43, 0.36]


def _num_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[col].fillna(False).astype(bool)


def _relative(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(0.5, index=s.index, dtype=float)
    lo, hi = float(s.min()), float(s.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=s.index, dtype=float)
    return ((s - lo) / (hi - lo)).fillna(0.5)


def _jaccard(a: set[int], b: set[int]) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def _prepare_pool(ranked: pd.DataFrame) -> pd.DataFrame:
    work = ranked.copy()
    if "quinte_rank" in work.columns:
        work["_rank"] = pd.to_numeric(work["quinte_rank"], errors="coerce")
        work = work.sort_values(["_rank", "horse_number"], na_position="last")
    else:
        work = work.reset_index(drop=True)
        work["_rank"] = range(1, len(work) + 1)

    work["horse_number"] = pd.to_numeric(work["horse_number"], errors="coerce").astype("Int64")
    work = work[work["horse_number"].notna()].drop_duplicates("horse_number").head(10).copy()
    if len(work) < 10:
        return work

    work["_rank"] = range(1, 11)
    work["_rank_prior"] = RANK_PRIOR

    meta_top5 = _relative(_num_series(work, "v51_meta_top5_score", np.nan))
    meta_win = _relative(_num_series(work, "v51_meta_winner_score", np.nan))
    fund = _relative(_num_series(work, "v46_fundamental_score", np.nan))
    dyn_raw = _num_series(work, "v45_dynamics_score", 0.0).clip(-1.0, 1.0)
    dyn = (dyn_raw + 1.0) / 2.0

    consensus = _bool_series(work, "v46_consensus").astype(float)
    shadow = _bool_series(work, "v51_shadow_top7").astype(float)
    fund7 = _bool_series(work, "v46_fundamental_top7").astype(float)

    # Le rang officiel reste la colonne vertébrale. Les autres moteurs peuvent
    # départager et renforcer un cheval, mais ils ne renversent pas le Top10.
    work["_smart_score"] = (
        0.48 * work["_rank_prior"]
        + 0.17 * meta_top5
        + 0.08 * meta_win
        + 0.08 * fund
        + 0.08 * consensus
        + 0.05 * shadow
        + 0.03 * fund7
        + 0.03 * dyn
    ).clip(0.0, 1.2)
    return work


def generate_quinte_70(ranked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retourne (70 tickets, exposition Top10, résumé des scénarios)."""
    if ranked is None or ranked.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    work = _prepare_pool(ranked)
    if len(work) < 10:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    horses = work["horse_number"].astype(int).tolist()
    rank_of = {h: i + 1 for i, h in enumerate(horses)}
    smart = {int(row["horse_number"]): float(row["_smart_score"]) for _, row in work.iterrows()}
    target = {h: TARGET_EXPOSURES[i] for i, h in enumerate(horses)}
    top5 = set(horses[:5])
    lower5 = set(horses[5:])

    chosen: list[tuple[int, ...]] = []
    scenario_of: dict[tuple[int, ...], str] = {}
    score_of: dict[tuple[int, ...], float] = {}
    current = {h: 0 for h in horses}
    pair_counts: dict[tuple[int, int], int] = {}
    triple_counts: dict[tuple[int, int, int], int] = {}

    def register(combo: tuple[int, ...], scenario: str, raw_score: float) -> None:
        combo = tuple(sorted(combo, key=lambda h: rank_of[h]))
        if combo in scenario_of:
            return
        chosen.append(combo)
        scenario_of[combo] = scenario
        score_of[combo] = float(raw_score)
        for h in combo:
            current[h] += 1
        for p in combinations(sorted(combo), 2):
            pair_counts[p] = pair_counts.get(p, 0) + 1
        for t in combinations(sorted(combo), 3):
            triple_counts[t] = triple_counts.get(t, 0) + 1

    # 1) Le noyau pur : ticket de référence.
    pure = tuple(horses[:5])
    register(pure, "Noyau pur 5+0", sum(smart[h] for h in pure) / 5.0)

    # 2) Toutes les variantes 4 Top5 + 1 cheval 6-10.
    # Cela garantit une couverture exhaustive de ce scénario très plausible.
    strong = []
    for favs in combinations(horses[:5], 4):
        for assoc in horses[5:]:
            c = tuple(favs) + (assoc,)
            quality = sum(smart[h] for h in c) / 5.0
            strong.append((quality, c))
    strong.sort(key=lambda x: (-x[0], sum(rank_of[h] for h in x[1])))
    for quality, c in strong:  # exactement 25
        register(c, "Noyau fort 4+1", quality)

    # 3/4) Sélection intelligente des scénarios 3+2 puis 2+3.
    def pick_structure(n_top5: int, quota: int, scenario: str) -> None:
        candidates: list[tuple[int, ...]] = []
        for favs in combinations(horses[:5], n_top5):
            for outs in combinations(horses[5:], 5 - n_top5):
                candidates.append(tuple(favs) + tuple(outs))

        for _ in range(quota):
            best = None
            best_score = float("-inf")
            for combo in candidates:
                ordered = tuple(sorted(combo, key=lambda h: rank_of[h]))
                if ordered in scenario_of:
                    continue
                s = set(ordered)

                quality = sum(smart[h] for h in ordered) / 5.0

                # Déficit par rapport à l'exposition voulue : le générateur
                # couvre les chevaux encore sous-utilisés sans sacrifier la qualité.
                deficit = sum(max(target[h] - current[h], 0) / max(target[h], 1) for h in ordered) / 5.0
                over = sum(max(current[h] + 1 - target[h], 0) / max(target[h], 1) for h in ordered) / 5.0

                # Diversité ticket / paires / triples.
                max_j = max((_jaccard(s, set(prev)) for prev in chosen), default=0.0)
                pair_rep = sum(pair_counts.get(p, 0) for p in combinations(sorted(ordered), 2)) / 10.0
                triple_rep = sum(triple_counts.get(t, 0) for t in combinations(sorted(ordered), 3)) / 10.0

                # Les scénarios ouverts doivent contenir au moins un associé
                # réellement soutenu par le méta/fondamental, pas juste un rang 10.
                lower_strength = max(smart[h] for h in ordered if h in lower5)
                lower_avg = sum(smart[h] for h in ordered if h in lower5) / max(1, len(s & lower5))

                if n_top5 == 3:
                    structure = 0.10 * lower_avg
                    diversity_w = 0.72
                else:  # 2+3 : plus risqué, on exige un outsider qualitatif.
                    structure = 0.18 * lower_strength + 0.08 * lower_avg
                    diversity_w = 0.86

                raw = (
                    1.55 * quality
                    + 1.60 * deficit
                    + structure
                    - diversity_w * max_j
                    - 0.115 * pair_rep
                    - 0.080 * triple_rep
                    - 1.35 * over
                )
                # Tie-break léger en faveur du meilleur rang cumulé.
                raw -= 1e-5 * sum(rank_of[h] for h in ordered)

                if raw > best_score:
                    best_score = raw
                    best = ordered

            if best is None:
                break
            register(best, scenario, best_score)

    pick_structure(3, 34, "Équilibré 3+2")
    pick_structure(2, 10, "Ouvert 2+3")

    # Indice de force lisible, basé uniquement sur les 5 chevaux du ticket.
    # Ce n'est pas une probabilité de gain : il sert à comparer les tickets entre eux.
    rows: list[dict[str, Any]] = []
    for i, combo in enumerate(chosen, start=1):
        ticket_score = float(np.clip(100.0 * sum(smart[h] for h in combo) / 5.0, 0.0, 100.0))
        rows.append({
            "Ticket": i,
            "Scénario": scenario_of[combo],
            "C1": combo[0],
            "C2": combo[1],
            "C3": combo[2],
            "C4": combo[3],
            "C5": combo[4],
            "Combinaison": " - ".join(map(str, combo)),
            "Top5 présents": sum(h in top5 for h in combo),
            "Indice modèle": round(float(ticket_score), 1),
        })

    names = {}
    if "horse_name" in work.columns:
        names = {int(r.horse_number): str(r.horse_name) for r in work.itertuples(index=False)}
    exposure_rows = []
    for i, h in enumerate(horses, start=1):
        row = work.iloc[i - 1]
        exposure_rows.append({
            "Rang": i,
            "N°": h,
            "Cheval": names.get(h, ""),
            "Présence /70": current[h],
            "Cible": target[h],
            "Taux": current[h] / 70.0,
            "Force intelligente": round(100.0 * float(row["_smart_score"]), 1),
            "Consensus": bool(row.get("v46_consensus", False)),
            "V5.1 shadow": bool(row.get("v51_shadow_top7", False)),
        })

    scenario_rows = []
    for label in ["Noyau pur 5+0", "Noyau fort 4+1", "Équilibré 3+2", "Ouvert 2+3"]:
        cnt = sum(1 for c in chosen if scenario_of[c] == label)
        scenario_rows.append({"Scénario": label, "Tickets": cnt, "Part": cnt / max(len(chosen), 1)})

    return pd.DataFrame(rows), pd.DataFrame(exposure_rows), pd.DataFrame(scenario_rows)

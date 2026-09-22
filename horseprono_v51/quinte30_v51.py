"""Générateur de 30 combinaisons Quinté V5.1 à partir du Top10 officiel.

Objectif : construire 30 tickets uniques de 5 chevaux uniquement parmi les
10 premiers du classement. La sélection est déterministe, favorise les rangs
1-5, mais maintient une exposition réelle des rangs 6-10 et pénalise les tickets
trop redondants.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd


# Exposition-cible sur 30 tickets (150 places au total).
# Elle donne beaucoup de poids aux 5 premiers sans éliminer les rangs 6-10.
TARGET_EXPOSURES = [24, 23, 21, 19, 18, 13, 11, 9, 7, 5]
RANK_WEIGHTS = [1.00, 0.95, 0.90, 0.84, 0.78, 0.65, 0.56, 0.47, 0.38, 0.30]


def _jaccard(a: set[int], b: set[int]) -> float:
    u = a | b
    return (len(a & b) / len(u)) if u else 0.0


def generate_quinte_30(ranked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (30 tickets Quinté, tableau d'exposition des 10 chevaux).

    Règles :
    - pool strict = 10 premiers du classement officiel ;
    - 5 chevaux distincts par ticket ;
    - 30 tickets uniques ;
    - au moins 2 chevaux du Top5 et au moins 1 cheval des rangs 6-10 ;
    - sélection gloutonne qui combine qualité de rang, déficit d'exposition et
      diversité vis-à-vis des tickets déjà choisis.
    """
    if ranked is None or ranked.empty:
        return pd.DataFrame(), pd.DataFrame()

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
        return pd.DataFrame(), pd.DataFrame()

    horses = work["horse_number"].astype(int).tolist()
    rank_of = {h: i + 1 for i, h in enumerate(horses)}
    target = {h: TARGET_EXPOSURES[i] for i, h in enumerate(horses)}
    weight = {h: RANK_WEIGHTS[i] for i, h in enumerate(horses)}
    top5 = set(horses[:5])
    lower5 = set(horses[5:])

    candidates: list[tuple[int, ...]] = []
    for combo in combinations(horses, 5):
        s = set(combo)
        n_top5 = len(s & top5)
        n_lower5 = len(s & lower5)
        if n_top5 < 2 or n_lower5 < 1:
            continue
        candidates.append(tuple(combo))

    current = {h: 0 for h in horses}
    chosen: list[tuple[int, ...]] = []
    chosen_sets: list[set[int]] = []
    pair_counts: dict[tuple[int, int], int] = {}

    for _ in range(30):
        best: tuple[int, ...] | None = None
        best_score = float("-inf")
        for combo in candidates:
            if combo in chosen:
                continue
            s = set(combo)

            # Besoin d'exposition : pousse les chevaux encore sous leur cible.
            deficit_score = 0.0
            over_penalty = 0.0
            for h in combo:
                t = max(target[h], 1)
                deficit_score += max(target[h] - current[h], 0) / t
                over_penalty += max(current[h] + 1 - target[h], 0) / t

            # Qualité intrinsèque selon le classement officiel.
            quality = sum(weight[h] for h in combo) / 5.0

            # Diversité : pénalise fortement les tickets presque identiques.
            max_overlap = max((_jaccard(s, prev) for prev in chosen_sets), default=0.0)
            overlap_penalty = max_overlap

            # Diversité des paires : évite de répéter toujours les mêmes duos.
            pair_penalty = 0.0
            for a, b in combinations(sorted(combo), 2):
                pair_penalty += pair_counts.get((a, b), 0)
            pair_penalty /= 10.0

            # Petit bonus structurel : 3 bases Top5 + 2 associés est le coeur,
            # mais 2+3 et 4+1 restent possibles pour diversifier.
            n_top5 = len(s & top5)
            structure_bonus = 0.16 if n_top5 == 3 else (0.08 if n_top5 == 2 else 0.04)

            score = (
                1.85 * deficit_score
                + 1.10 * quality
                + structure_bonus
                - 0.95 * overlap_penalty
                - 0.10 * pair_penalty
                - 1.50 * over_penalty
            )

            # Tie-break déterministe en faveur du meilleur total de rangs.
            rank_sum = sum(rank_of[h] for h in combo)
            score -= 1e-5 * rank_sum
            if score > best_score:
                best_score = score
                best = combo

        if best is None:
            break

        chosen.append(best)
        s = set(best)
        chosen_sets.append(s)
        for h in best:
            current[h] += 1
        for a, b in combinations(sorted(best), 2):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    rows: list[dict[str, Any]] = []
    for i, combo in enumerate(chosen, start=1):
        # Affichage dans l'ordre du classement, pas dans l'ordre numérique.
        ordered = sorted(combo, key=lambda h: rank_of[h])
        rows.append({
            "Ticket": i,
            "C1": ordered[0],
            "C2": ordered[1],
            "C3": ordered[2],
            "C4": ordered[3],
            "C5": ordered[4],
            "Combinaison": " - ".join(map(str, ordered)),
            "Top5 présents": sum(1 for h in ordered if h in top5),
        })

    exposure_rows: list[dict[str, Any]] = []
    names = {}
    if "horse_name" in work.columns:
        names = {int(r.horse_number): str(r.horse_name) for r in work.itertuples(index=False)}
    for i, h in enumerate(horses, start=1):
        exposure_rows.append({
            "Rang": i,
            "N°": h,
            "Cheval": names.get(h, ""),
            "Présence /30": current[h],
            "Cible": target[h],
            "Taux": current[h] / 30.0,
        })

    return pd.DataFrame(rows), pd.DataFrame(exposure_rows)

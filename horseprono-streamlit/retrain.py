"""
Reentraine le modele de production sur un historique a jour, et regenere
tous les fichiers necessaires a l'app Streamlit (dans data/).

Usage :
    python3 retrain.py chemin/vers/historique_complet_a_jour.csv

A relancer periodiquement (par exemple chaque semaine) a mesure que de
nouvelles courses s'accumulent, puis a commiter/pousser sur GitHub pour
mettre a jour l'app deployee.
"""
import sys
import os
import json
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import HorseRaceNet, AdamOptimizer
import features as feat
from build_entity_history import build as build_entity_history

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def grouped_softmax_and_loss(win_score, y_win, race_code, n_races):
    max_per_race = np.full(n_races, -np.inf)
    np.maximum.at(max_per_race, race_code, win_score)
    exp_s = np.exp(win_score - max_per_race[race_code])
    sum_per_race = np.zeros(n_races)
    np.add.at(sum_per_race, race_code, exp_s)
    p = exp_s / sum_per_race[race_code]
    eps = 1e-12
    loss_per_race = np.zeros(n_races)
    np.add.at(loss_per_race, race_code, -y_win * np.log(p + eps))
    return p, loss_per_race.sum() / n_races, (p - y_win) / n_races


def bce_loss_and_grad(prob, y, n):
    eps = 1e-12
    loss = -(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps)).mean()
    return loss, (prob - y) / n


def main(csv_path):
    os.makedirs(DATA_DIR, exist_ok=True)

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    run = df[(df['status'] == 'finished') & (df['is_non_runner'] == False)].copy()
    run = run.dropna(subset=['odds']).copy()
    run['win'] = (run['finish_position'] == 1).astype(int)

    def place_rule(row):
        if pd.isna(row['finish_position']):
            return 0
        fs = row['field_size']
        thresh = 3 if fs >= 8 else (2 if fs >= 4 else 1)
        return int(row['finish_position'] <= thresh)

    run['placed'] = run.apply(place_rule, axis=1)
    run['race_date'] = pd.to_datetime(run['race_date'])
    run['month'] = run['race_date'].dt.to_period('M').astype(str)
    valid_races = run.groupby('race_id')['win'].sum()
    run = run[run['race_id'].isin(valid_races[valid_races == 1].index)].copy()

    months = sorted(run['month'].unique())
    if len(months) < 2:
        raise SystemExit("Il faut au moins 2 mois de donnees (1 pour l'entrainement, 1 pour l'arret anticipe).")
    val_month = months[-1]
    train_months = months[:-1]
    train_mask = run['month'].isin(train_months)
    val_mask = run['month'] == val_month
    print(f"Train: {train_mask.sum()} lignes ({train_months}) | Val (early stopping): {val_mask.sum()} lignes ({val_month})")

    run_feat, priors = feat.build_full_pipeline(run, train_mask)
    train_ids = set(run.loc[train_mask, 'participant_id'])
    val_ids = set(run.loc[val_mask, 'participant_id'])
    run_feat['split'] = np.where(run_feat['participant_id'].isin(train_ids), 'train',
                         np.where(run_feat['participant_id'].isin(val_ids), 'val', 'other'))
    run_feat = run_feat.sort_values(['race_id']).reset_index(drop=True)
    race_codes, _ = pd.factorize(run_feat['race_id'], sort=False)
    run_feat['race_code'] = race_codes

    cat_cardinalities = {c: len(priors['cat_maps'][c]) + 1 for c in feat.CAT_SPECS}
    num_x_all = run_feat[feat.NUM_COLS].values.astype(np.float64)
    cat_idx_all = {c: run_feat[c + '_idx'].values.astype(np.int64) for c in feat.CAT_SPECS}
    win_all = run_feat['win'].values.astype(np.float64)
    placed_all = run_feat['placed'].values.astype(np.float64)
    race_code_all = run_feat['race_code'].values
    split_all = run_feat['split'].values

    with open(os.path.join(DATA_DIR, 'priors_production.pkl'), 'wb') as f:
        pickle.dump(priors, f)
    with open(os.path.join(DATA_DIR, 'cat_cardinalities_production.json'), 'w') as f:
        json.dump(cat_cardinalities, f)

    rng = np.random.default_rng(0)
    net = HorseRaceNet(cat_cardinalities, feat.CAT_SPECS, n_numeric=len(feat.NUM_COLS), hidden=(64, 32), seed=1)
    opt = AdamOptimizer(net.params, lr=2e-3)

    train_idx = np.where(split_all == 'train')[0]
    val_idx = np.where(split_all == 'val')[0]
    train_races = np.unique(race_code_all[train_idx])
    LAMBDA_PLACE, DROPOUT, N_EPOCHS, BATCH_RACES, PATIENCE = 0.5, 0.15, 150, 256, 12
    best_val_loss, best_params, no_improve = np.inf, None, 0

    for epoch in range(N_EPOCHS):
        order = rng.permutation(train_races)
        for start in range(0, len(order), BATCH_RACES):
            batch_races = order[start:start + BATCH_RACES]
            mask = np.isin(race_code_all[train_idx], batch_races)
            idx = train_idx[mask]
            if len(idx) == 0:
                continue
            cat_idx = {c: cat_idx_all[c][idx] for c in feat.CAT_SPECS}
            num_x = num_x_all[idx]
            ws, place_prob, cache = net.forward(cat_idx, num_x, dropout_p=DROPOUT, training=True, rng=rng)
            codes_local, _ = pd.factorize(race_code_all[idx], sort=False)
            n_r = codes_local.max() + 1
            p, loss_win, grad_win = grouped_softmax_and_loss(ws, win_all[idx], codes_local, n_r)
            loss_place, grad_place = bce_loss_and_grad(place_prob, placed_all[idx], len(idx))
            grads = net.backward(cache, grad_win, LAMBDA_PLACE * grad_place)
            opt.step(net.params, grads)

        cat_idx_v = {c: cat_idx_all[c][val_idx] for c in feat.CAT_SPECS}
        ws_v, place_v, _ = net.forward(cat_idx_v, num_x_all[val_idx], training=False)
        codes_v, _ = pd.factorize(race_code_all[val_idx], sort=False)
        _, val_loss_win, _ = grouped_softmax_and_loss(ws_v, win_all[val_idx], codes_v, codes_v.max() + 1)
        val_loss_place, _ = bce_loss_and_grad(place_v, placed_all[val_idx], len(val_idx))
        val_total = val_loss_win + LAMBDA_PLACE * val_loss_place
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d} | val_loss_win {val_loss_win:.4f}")
        if val_total < best_val_loss - 1e-4:
            best_val_loss, best_params, no_improve = val_total, {k: v.copy() for k, v in net.params.items()}, 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping a l'epoch {epoch}")
                break

    net.params = best_params
    net.save(os.path.join(DATA_DIR, 'model_production.npz'))
    print("Modele sauvegarde dans data/model_production.npz")

    build_entity_history(csv_path, out_path=os.path.join(DATA_DIR, 'entity_history.pkl'))
    print("\nTermine. Pensez a commit + push data/ pour mettre a jour l'app deployee.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 retrain.py chemin/vers/historique_complet.csv")
        sys.exit(1)
    main(sys.argv[1])

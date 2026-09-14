"""
Logique de pronostic partagee entre l'app Streamlit et l'usage en ligne de
commande. Utilise le fichier LEGER entity_history.pkl (precalcule par
build_entity_history.py) plutot que de relire le CSV brut a chaque appel.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd

from model import HorseRaceNet
import features as feat

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

RACE_CONTEXT_COLS = ['hippodrome', 'discipline', 'distance_m', 'field_size']
HORSE_COLS = ['horse_number', 'horse_name', 'jockey_name', 'trainer_name',
              'odds', 'weight_kg', 'draw', 'age', 'sex', 'recent_form']

DISCIPLINES = ['PLAT', 'ATTELE_VOLTE', 'ATTELE_AUTOSTART', 'TROT_MONTE',
               'HAIES', 'STEEPLECHASE', 'CROSS_COUNTRY']
SEXES = ['HONGRES', 'FEMELLES', 'MALES']


def load_model(data_dir=DATA_DIR):
    with open(os.path.join(data_dir, 'priors_production.pkl'), 'rb') as f:
        priors = pickle.load(f)
    with open(os.path.join(data_dir, 'cat_cardinalities_production.json')) as f:
        cat_cardinalities = json.load(f)
    net = HorseRaceNet.load(os.path.join(data_dir, 'model_production.npz'), cat_cardinalities)
    return net, priors


def load_entity_history(data_dir=DATA_DIR):
    with open(os.path.join(data_dir, 'entity_history.pkl'), 'rb') as f:
        return pickle.load(f)


def known_hippodromes(priors):
    return sorted(priors['cat_maps']['hippodrome'].keys())


def predict_race(new_race_df, net, priors, entity_hist):
    """new_race_df : une ligne par partant, colonnes RACE_CONTEXT_COLS + HORSE_COLS
    (identique au format historique_complet.csv, sans finish_position)."""
    df = new_race_df.copy()
    for ent in ['jockey_name', 'trainer_name', 'horse_name']:
        df[ent] = df[ent].fillna('__INCONNU__')

    df = feat.add_base_columns(df)

    for ent, prefix in [('jockey_name', 'jockey'), ('trainer_name', 'trainer'), ('horse_name', 'horse')]:
        hist = entity_hist[prefix]
        wr, n = [], []
        for name in df[ent]:
            if name in hist.index:
                s, c = hist.loc[name, 'sum'], hist.loc[name, 'count']
                wr.append(s / c if c > 0 else 0.09)
                n.append(c)
            else:
                wr.append(0.09)
                n.append(0)
        df[f'{prefix}_winrate'] = wr
        df[f'{prefix}_n_log'] = np.log1p(n)
        df[f'{prefix}_has_hist'] = (np.array(n) > 0).astype(float)

    df = feat.apply_normalization(df, priors)

    cat_idx = {c: df[c + '_idx'].values.astype(np.int64) for c in feat.CAT_SPECS}
    num_x = df[feat.NUM_COLS].values.astype(np.float64)
    win_score, place_prob, _ = net.forward(cat_idx, num_x, training=False)

    exp_s = np.exp(win_score - win_score.max())
    p_win_model = exp_s / exp_s.sum()

    p_win_market = 1.0 / df['odds'].values
    p_win_market = p_win_market / p_win_market.sum()

    result = pd.DataFrame({
        'horse_number': df['horse_number'].values,
        'horse_name': df['horse_name'].values,
        'jockey_name': df['jockey_name'].values,
        'odds': df['odds'].values,
        'p_marche': p_win_market,
        'p_modele': p_win_model,
        'edge_ratio': p_win_model / p_win_market,
        'p_place_modele': place_prob,
    }).sort_values('p_modele', ascending=False).reset_index(drop=True)
    return result


def predict_multi_race_csv(df_all, net, priors, entity_hist):
    """Pour un CSV contenant PLUSIEURS courses (colonne race_id ou
    combinaison hippodrome/race_date/race_number) : pronostique course par
    course et renvoie un seul DataFrame concatene avec une colonne race_key."""
    if 'race_id' in df_all.columns:
        key_col = 'race_id'
    else:
        df_all = df_all.copy()
        df_all['_race_key'] = (df_all['hippodrome'].astype(str) + '_' +
                                df_all.get('race_date', '').astype(str) + '_' +
                                df_all.get('race_number', 0).astype(str))
        key_col = '_race_key'

    outs = []
    for key, g in df_all.groupby(key_col):
        res = predict_race(g, net, priors, entity_hist)
        res.insert(0, 'race', str(key))
        outs.append(res)
    return pd.concat(outs, ignore_index=True)

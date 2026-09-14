"""
Feature engineering pour le modele de pronostic hippique.

Principe anti-fuite (leakage-safe) :
  - Les stats jockey / entraineur / cheval (taux de reussite historique) sont
    calculees en "expanding window" sur TOUTE la chronologie du dataset, mais
    pour chaque ligne on n'utilise QUE les courses strictement anterieures.
    Cela reste valide meme si la ligne appartient au val/test : on utilise de
    l'historique reel, exactement ce qu'on aurait en production.
  - Les statistiques de NORMALISATION (moyenne/ecart-type du poids, de la
    distance, etc.) sont elles calculees UNIQUEMENT sur le train, puis
    appliquees telles quelles au val/test (sinon on "regarderait" la
    distribution du futur pendant l'entrainement).
"""
import pandas as pd
import numpy as np
import re

CAT_SPECS = {  # nom -> dim_embedding
    'discipline': 4,
    'sex': 2,
    'hippodrome': 8,
}

NUM_COLS = ['log_odds', 'draw_rel', 'has_draw', 'weight_z', 'has_weight',
            'age_z', 'field_size_z', 'distance_z',
            'form_avg5_z', 'has_form', 'form_dnf_rate5', 'form_n_log',
            'jockey_winrate', 'jockey_n_log', 'jockey_has_hist',
            'trainer_winrate', 'trainer_n_log', 'trainer_has_hist',
            'horse_winrate', 'horse_n_log', 'horse_has_hist']


def parse_form(s):
    if pd.isna(s):
        return []
    s2 = re.sub(r'\(\d+\)', '', s)
    return re.findall(r'[0-9A-Z][a-z]', s2)


def _form_features(s):
    toks = parse_form(s)
    n = len(toks)
    if n == 0:
        return pd.Series({'form_n': 0, 'form_avg5': np.nan, 'form_dnf_rate5': np.nan})

    def val(tok):
        c = tok[0]
        if c.isdigit():
            v = int(c)
            return 10 if v == 0 else v
        return 12
    vals = [val(t) for t in toks]
    avg5 = np.mean(vals[:5])
    dnf5 = sum(1 for t in toks[:5] if not t[0].isdigit()) / min(5, n)
    return pd.Series({'form_n': n, 'form_avg5': avg5, 'form_dnf_rate5': dnf5})


def add_base_columns(df):
    df = df.copy()
    if {'race_date', 'meeting_number', 'race_number'}.issubset(df.columns):
        df['sort_key'] = (df['race_date'].astype(str) + '_' +
                           df['meeting_number'].astype(str).str.zfill(2) + '_' +
                           df['race_number'].astype(str).str.zfill(2))
    ff = df['recent_form'].apply(_form_features)
    df = pd.concat([df, ff], axis=1)
    df['log_odds'] = np.log(df['odds'])
    df['draw_rel'] = (df['draw'] / df['field_size']).fillna(0.5)
    df['has_draw'] = df['draw'].notna().astype(float)
    df['has_weight'] = df['weight_kg'].notna().astype(float)
    df['has_form'] = df['form_avg5'].notna().astype(float)
    df['form_n_log'] = np.log1p(df['form_n'])
    return df


def add_expanding_entity_stats(df):
    """A appliquer UNE FOIS sur le dataframe complet (train+val+test), trie
    chronologiquement, pour que les lignes val/test beneficient de
    l'historique reel construit sur le train (et rien du futur)."""
    df = df.sort_values(['sort_key']).reset_index(drop=True)
    for ent in ['jockey_name', 'trainer_name', 'horse_name']:
        df[ent] = df[ent].fillna('__INCONNU__')
    for ent, prefix in [('jockey_name', 'jockey'), ('trainer_name', 'trainer'), ('horse_name', 'horse')]:
        grp = df.groupby(ent)['win']
        cum_sum = grp.cumsum() - df['win']   # succes STRICTEMENT avant cette ligne
        cum_n = grp.cumcount()               # nb de courses STRICTEMENT avant
        winrate = cum_sum / cum_n.replace(0, np.nan)
        df[f'{prefix}_winrate'] = winrate.fillna(0.09)  # prior neutre = taux moyen global
        df[f'{prefix}_n_log'] = np.log1p(cum_n)
        df[f'{prefix}_has_hist'] = (cum_n > 0).astype(float)
    return df


def fit_train_priors(train_df):
    p = {}
    p['weight_mean'] = train_df.groupby('discipline')['weight_kg'].mean().to_dict()
    p['weight_std'] = train_df.groupby('discipline')['weight_kg'].std().to_dict()
    p['weight_mean_global'] = train_df['weight_kg'].mean()
    p['weight_std_global'] = train_df['weight_kg'].std()
    p['distance_mean'] = train_df['distance_m'].mean()
    p['distance_std'] = train_df['distance_m'].std()
    p['form_avg5_mean'] = train_df['form_avg5'].mean()
    p['form_avg5_std'] = train_df['form_avg5'].std()
    p['age_mean'] = train_df['age'].mean()
    p['age_std'] = train_df['age'].std()
    p['field_size_mean'] = train_df['field_size'].mean()
    p['field_size_std'] = train_df['field_size'].std()
    p['cat_maps'] = {}
    for c in CAT_SPECS:
        cats = sorted(train_df[c].dropna().unique().tolist())
        p['cat_maps'][c] = {v: i + 1 for i, v in enumerate(cats)}  # 0 reserve = inconnu (OOV)
    return p


def apply_normalization(df, priors):
    df = df.copy()

    def z_weight(row):
        m = priors['weight_mean'].get(row['discipline'], priors['weight_mean_global'])
        s = priors['weight_std'].get(row['discipline'], priors['weight_std_global'])
        if pd.isna(row['weight_kg']) or pd.isna(s) or s == 0:
            return 0.0
        return (row['weight_kg'] - m) / s
    df['weight_z'] = df.apply(z_weight, axis=1)

    df['distance_z'] = (df['distance_m'] - priors['distance_mean']) / priors['distance_std']
    df['form_avg5_filled'] = df['form_avg5'].fillna(priors['form_avg5_mean'])
    df['form_avg5_z'] = (df['form_avg5_filled'] - priors['form_avg5_mean']) / priors['form_avg5_std']
    df['form_dnf_rate5'] = df['form_dnf_rate5'].fillna(0.0)
    df['age_filled'] = df['age'].fillna(priors['age_mean'])
    df['age_z'] = (df['age_filled'] - priors['age_mean']) / priors['age_std']
    df['field_size_z'] = (df['field_size'] - priors['field_size_mean']) / priors['field_size_std']

    for c in CAT_SPECS:
        cmap = priors['cat_maps'][c]
        df[c + '_idx'] = df[c].map(cmap).fillna(0).astype(int)

    return df


def build_full_pipeline(run_df, train_mask):
    """
    run_df : toutes les lignes (train+val+test), avec colonne 'win' deja calculee.
    train_mask : booleen (meme index que run_df) indiquant les lignes de train.
    Retourne (df_avec_features, priors).
    """
    df = add_base_columns(run_df)
    df = add_expanding_entity_stats(df)  # sur la chronologie complete -> re-indexe (reset_index)
    # recalcule le masque train sur le nouvel ordre/index via l'identifiant stable participant_id
    train_ids = set(run_df.loc[train_mask, 'participant_id'])
    tmask2 = df['participant_id'].isin(train_ids)
    priors = fit_train_priors(df[tmask2])
    df = apply_normalization(df, priors)
    return df, priors

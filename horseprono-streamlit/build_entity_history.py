"""
Precalcule les stats jockey / entraineur / cheval (nb de courses, nb de
victoires) a partir de l'historique complet, et les sauvegarde dans un
fichier LEGER (quelques centaines de Ko) au lieu d'embarquer le CSV brut
(17 Mo) dans le depot Git / l'app Streamlit deployee.

A relancer quand vous voulez rafraichir l'app avec de nouvelles courses :
    python3 build_entity_history.py chemin/vers/historique_complet.csv
"""
import sys
import pandas as pd
import pickle

def build(csv_path, out_path='data/entity_history.pkl'):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    run = df[(df['status'] == 'finished') & (df['is_non_runner'] == False)].copy()
    run = run.dropna(subset=['odds']).copy()
    run['win'] = (run['finish_position'] == 1).astype(int)
    for ent in ['jockey_name', 'trainer_name', 'horse_name']:
        run[ent] = run[ent].fillna('__INCONNU__')

    out = {}
    for ent, prefix in [('jockey_name', 'jockey'), ('trainer_name', 'trainer'), ('horse_name', 'horse')]:
        g = run.groupby(ent)['win'].agg(['sum', 'count'])
        out[prefix] = g

    out['_meta'] = {
        'source_csv': csv_path,
        'n_rows': len(run),
        'date_min': str(run['race_date'].min()),
        'date_max': str(run['race_date'].max()),
    }

    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print(f"Ecrit {out_path} : {len(out['jockey'])} jockeys, {len(out['trainer'])} entraineurs, "
          f"{len(out['horse'])} chevaux (historique {out['_meta']['date_min']} -> {out['_meta']['date_max']})")


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'historique_complet.csv'
    build(csv_path)

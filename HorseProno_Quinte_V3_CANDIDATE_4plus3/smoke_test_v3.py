from pathlib import Path
import json
import pandas as pd
from quinte_v3 import rank_quinte_v3, load_ranker

B=Path(__file__).resolve().parent
h=pd.read_csv(B/'validated_history.csv')
h['race_date']=pd.to_datetime(h['race_date']).dt.date
# Course historique de test; on efface les résultats de la course courante avant scoring.
g=h[h['race_id'].astype(str).eq('R1C1_2026-09-08')].copy()
if g.empty:
    # fallback dernière course suffisamment fournie
    rid=h.groupby('race_id').size().sort_values(ascending=False).index[0]
    g=h[h.race_id.eq(rid)].copy()
race=g[['race_id','race_date','discipline','hippodrome','distance','terrain','field_size','horse_number','horse_name','jockey','trainer','odds','draw','weight','age','sex','recent_form','is_non_runner']].copy()
history=h[h.race_date < race.race_date.iloc[0]].copy()
artifact=json.loads((B/'quinte_v2_artifact.json').read_text())
ranker=load_ranker(B/'quinte_v2_ranker.txt')
out=rank_quinte_v3(history,race,artifact,ranker)
sel=out[out.selected_top7]
assert len(sel)==7
assert (sel.shortlist_role=='NOYAU_V3').sum()==4
assert sel.shortlist_role.str.startswith('CHALLENGER_').sum()==3
assert out.market_rankpct.between(0,1).all()
assert out.ranker_percentile.between(0,1).all()
print('V3 smoke test OK')
print(sel[['quinte_rank','horse_number','horse_name','shortlist_role','core_score','top5_probability','ranker_rank','market_rank']].to_string(index=False))

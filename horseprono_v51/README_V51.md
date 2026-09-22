# HorseProno Quinté V5.1 — Meta-Consensus

V5.1 ajoute une couche de méta-consensus au-dessus de V4.6. Elle **ne modifie pas** le Top7 officiel tant que la validation forward n'a pas démontré un gain robuste.

## Ce que V5.1 calcule

- score Meta Top5 pour chaque candidat de l'union Marché + Fondamental ;
- score Meta Winner ;
- noyau de consensus et indice de divergence ;
- confiance de consensus 0–100 % ;
- lecture de course : consensus fort, marché plus convaincant, fondamental à surveiller, divergence élevée ;
- Top7 et Winner Top3 alternatifs en **shadow mode**.

## Protocole historique

Le méta-modèle a été réglé sur les 30 premières courses walk-forward puis observé sur 28 courses suivantes. Le modèle libre n'a pas battu la baseline :

- V4.4/V4.5 baseline holdout : 3,32/5 ; 10 Trios ; 8 Quartés ; 5 Quintés ; gagnant Top3 14/28.
- Meta libre holdout : 3,29/5 ; 10 Trios ; 8 Quartés ; 4 Quintés ; gagnant Top3 12/28.

En revanche, l'union Marché + Fondamental sur ces mêmes 28 courses contient 3,75/5 chevaux de l'arrivée en moyenne et 9 Quintés complets. L'information complémentaire existe donc, mais l'arbitrage n'est pas encore assez fiable.

## Validation forward

Chaque lecture Quinté peut être enregistrée dans `public.v51_meta_shadow_predictions` si la clé serveur Supabase est configurée. Le Top7 officiel, le Top7/Top3 shadow, consensus, divergence et scores runners sont figés avant course pour une vraie évaluation forward.

## Déploiement

Main file : `streamlit_app.py`

La production n'a besoin ni de scikit-learn ni de LightGBM : les coefficients du méta-modèle sont exportés en JSON et le ranker reste portable Python.

## e-Trio — 10 combinaisons, 2 bases Top5

L'application génère désormais automatiquement 10 tickets e-Trio :
- les 5 premiers du classement officiel fournissent les bases ;
- les 10 paires possibles parmi ces 5 chevaux sont utilisées exactement une fois ;
- le 3e cheval est toujours hors Top5 ;
- jusqu'à 5 associés sont priorisés par le consensus V5.1/V4.6, le Top7 officiel, le Top7 shadow et la réserve fondamentale ;
- les associés sont répartis de manière équilibrée afin de diversifier les 10 tickets.

Cette couche de construction de tickets ne modifie ni le Top7 officiel ni les scores des modèles.

## Générateur Quinté 70 intelligent

La version actuelle ajoute un portefeuille de 70 combinaisons parmi les 252 combinaisons possibles du Top10. Ce portefeuille n'est pas aléatoire : il mélange quatre structures de risque (5+0, 4+1, 3+2, 2+3), tient compte du consensus V5.1/V4.6 et limite la redondance des paires et triples.

L'`Indice modèle` affiché pour chaque ticket est un indicateur comparatif interne, pas une probabilité de gain.

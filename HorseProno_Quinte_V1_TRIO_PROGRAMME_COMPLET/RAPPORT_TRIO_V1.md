# HorseProno Trio V1 — rétro-analyse dédiée au Top 7 de Quinté V1

## Périmètre

- Source de sélection : **HorseProno Quinté V1**, pas M8.
- 99 supports Quinté officiels identifiés entre le 1er juin et le 8 septembre 2026.
- 98 supports exploitables avec arrivée Top 5 complète et unique.
- Trio = les trois premiers de l'arrivée, sans ordre pour la mesure de couverture.

## Où se trouve le trio dans le classement Quinté V1 ?

Analyse descriptive avec l'artefact final Quinté V1 refitté sur les 98 supports :

| Taille sélection | Trio complet contenu |
|---|---:|
| Top 3 | 1 / 98 (1,0 %) |
| Top 4 | 9 / 98 (9,2 %) |
| Top 5 | 17 / 98 (17,3 %) |
| Top 6 | 25 / 98 (25,5 %) |
| **Top 7** | **36 / 98 (36,7 %)** |

Sur les 294 places de podium (98 × 3), 219 sont dans le Top 7 V1, soit **74,5 %**. Le 7e rang apporte encore 21 chevaux de podium sur l'échantillon.

> Attention : ce bloc 98 est descriptif/in-sample car l'artefact final V1 a ensuite été refitté sur ces 98 courses. Il sert à comprendre la structure du classement, pas à annoncer une performance future.

## Test temporel plus strict

Découpage utilisé pour éviter de choisir le module Trio sur le test final :

- apprentissage initial : 61 Quintés avant le 1er août 2026 ;
- validation : 22 Quintés du 1er au 22 août ;
- test tenu à l'écart : 15 Quintés du 23 août au 8 septembre.

Sur le test final, le Top 7 du Quinté V1 figé contient le trio complet sur **5 / 15 courses (33,3 %)**. Le modèle Trio ne peut évidemment pas retrouver un cheval absent de ce Top 7.

Pour les 5 courses où le trio est entièrement dans le Top 7 :

| Classement des 35 trios possibles | V1 brut | Trio V1 |
|---|---:|---:|
| Bonne combinaison dans Top 10 | 4 / 5 | 4 / 5 |
| Bonne combinaison dans Top 15 | 5 / 5 | 5 / 5 |
| Rang moyen de la bonne combinaison | 9,8 | **9,0** |

Le gain observé est donc **petit** : meilleur rang moyen, mais pas davantage de bonnes combinaisons dans le Top 10 sur ce mini-échantillon. Le module reste expérimental.

## Architecture retenue

1. **Quinté V1 reste inchangé** et produit son Top 7.
2. Trio V1 réutilise les variables pré-course de Quinté V1, son score et son rang relatif.
3. Il apprend une cible `finish_position ∈ {1,2,3}` uniquement sur les supports Quinté officiels.
4. Il re-classe seulement les 7 chevaux de Quinté V1.
5. Les 35 combinaisons possibles de trois chevaux sont classées selon les scores Trio ; l'app affiche 10 combinaisons par défaut (curseur 5 à 20).

Le score affiché est un **indice de classement**, pas une probabilité calibrée ni une estimation de rentabilité.

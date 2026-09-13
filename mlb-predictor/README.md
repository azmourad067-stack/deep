# MLB Predictor — Niveau 1 (Elo + Home Field Advantage + Lanceur titulaire)

Application Streamlit qui calcule des probabilités de victoire pour les matchs
MLB du jour, à partir d'un modèle **statistique simple et interprétable** :

```
Elo(équipe) + ajustement lanceur titulaire (FIP) + avantage du terrain
        ↓
Probabilité de victoire (formule Elo logistique)
```

Aucun machine learning ici volontairement : voir `docs/METHODOLOGIE.md` pour
la justification (c'est le "Niveau 1" d'une roadmap en 3 niveaux — cf.
discussion de conception).

## ⚠️ Disclaimer

Ce projet est un outil **pédagogique et analytique**. Il ne constitue pas un
conseil de pari, ne garantit aucun résultat, et le marché des cotes MLB est
historiquement difficile à battre durablement. Utilise-le pour comprendre le
sabermetrics, pas comme une martingale.

## Architecture du dépôt

```
mlb-predictor/
├── app.py                        # application Streamlit (point d'entrée)
├── requirements.txt
├── src/
│   ├── mlb_api.py                 # accès à l'API MLB Stats (gratuite, sans clé)
│   ├── pitcher_stats.py           # récupération FIP/xFIP via pybaseball + Chadwick
│   ├── elo.py                     # moteur de rating Elo
│   ├── predict.py                 # combinaison Elo + FIP + HFA -> probabilités
│   └── config.py                  # constantes du modèle (K, HFA, scaling...)
├── scripts/
│   └── update_data.py             # job de rafraîchissement (lancé par GitHub Actions)
├── data/cache/                    # snapshots (ratings Elo, historique matchs, stats lanceurs)
├── .github/workflows/
│   └── update_data.yml            # cron quotidien qui rafraîchit data/cache/
├── .streamlit/config.toml         # thème
└── tests/test_elo.py              # tests unitaires sur la logique Elo (pas de réseau)
```

## Installation locale

```bash
git clone <ton-fork>
cd mlb-predictor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) construire/rafraîchir les données locales (Elo historique + stats lanceurs)
python scripts/update_data.py

# 2) lancer l'app
streamlit run app.py
```

## Déploiement sur Streamlit Community Cloud

1. Pousse ce repo sur GitHub (public ou privé).
2. Sur [share.streamlit.io](https://share.streamlit.io), "New app" → sélectionne
   le repo, la branche, et `app.py` comme fichier principal.
3. **Aucun secret requis** : l'API MLB Stats est publique et sans clé. Si tu
   ajoutes plus tard une API de cotes payante, mets la clé dans
   *App settings → Secrets* (jamais dans le code).
4. Le premier build peut être un peu lent (installation de `pybaseball`).
   Une fois déployé, `data/cache/` doit déjà contenir des données (générées
   par le workflow GitHub Actions ou par `scripts/update_data.py` en local
   avant de commit) — l'app ne recalcule pas tout l'historique à chaque
   démarrage, elle lit le cache.

## Rafraîchissement des données

Le workflow `.github/workflows/update_data.yml` tourne une fois par jour
(cron), relance `scripts/update_data.py`, et commit les fichiers mis à jour
dans `data/cache/`. C'est volontairement **périodique et non temps réel** :
les compositions/lanceurs probables ne sont de toute façon fiables que
quelques heures avant le match, et Streamlit Community Cloud n'est pas
dimensionné pour du polling en direct.

## Limites connues (à lire avant de faire confiance aux chiffres)

- **FIP "as of now"** : pendant la saison, on utilise le FIP saison-en-cours
  du lanceur (mis à jour par FanGraphs quotidiennement, donc pas de fuite de
  données du futur). En petit échantillon (début de saison, callup récent),
  on retombe sur le FIP de la saison précédente puis sur la moyenne de ligue.
- **Confrontations directes** : volontairement absentes du modèle (échantillon
  trop petit par saison pour être un signal fiable, cf. discussion de conception).
- **Le facteur d'échelle FIP → points Elo** (`config.py`) est un paramètre à
  calibrer via `scripts/backtest.py`, pas une constante empirique établie.
- **Plafond réaliste** : ne t'attends pas à dépasser ~58-62% de bonne
  prédiction du vainqueur sur la durée — c'est structurel au baseball, pas un
  défaut du modèle.

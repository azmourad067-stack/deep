# HorseProno Quinté V1 — programme PMU complet

## Fonctionnement

1. `streamlit_app.py` récupère automatiquement tout le programme PMU de la date choisie.
2. L'utilisateur sélectionne d'abord la **réunion**, puis la **course** à analyser.
3. L'application récupère automatiquement `/R{n}/C{n}/participants` pour la course choisie.
4. Elle convertit les partants au format attendu par Quinté V1.
5. Elle charge `validated_history.csv` puis, si des secrets Supabase sont présents, ajoute les nouvelles courses historiques validées antérieures à la course cible.
6. Elle produit le classement complet et le **Top 7**.

Aucun CSV de course n'est demandé à l'utilisateur.

Le programme complet peut être affiché dans l'interface. Les courses support du Quinté+ sont signalées avec un badge `⭐ QUINTÉ+` grâce au code officiel PMU `E_QUINTE_PLUS`.

> Important : le modèle Quinté V1 a été calibré sur des supports Quinté officiels. L'application autorise l'analyse des autres courses du programme, mais les performances du modèle hors Quinté n'ont pas été validées par le backtest spécialisé.

## Streamlit Community Cloud

Main file path :

`horse_prono_v2/streamlit_app.py`

Si le dossier du dépôt porte un autre nom, adaptez simplement le chemin.

## Secrets Supabase

Dans Streamlit Cloud > Manage app > Settings > Secrets :

```toml
SUPABASE_URL = "https://VOTRE-PROJET.supabase.co"
SUPABASE_KEY = "VOTRE_CLE_DE_LECTURE"
```

Utiliser de préférence une clé limitée à la lecture des tables nécessaires. Ne jamais committer une vraie clé secrète dans GitHub.

## Module Trio V1

Cette version contient aussi `trio_v1_artifact.json`.
Le Top 7 reste celui de HorseProno Quinté V1. Trio V1 re-classe seulement ces 7 chevaux et classe les 35 trios possibles. L'interface affiche 10 combinaisons par défaut, réglables de 5 à 20.

Le module Trio est expérimental : le test temporel final contient seulement 15 Quintés, dont 5 où les trois premiers étaient tous présents dans le Top 7 V1.

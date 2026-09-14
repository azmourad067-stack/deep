# 🐎 Pronostic hippique — réseau de neurones (Streamlit)

App Streamlit de pronostic hippique, adossée à un réseau de neurones (embeddings +
MLP, softmax par course) entraîné sur 3 mois de courses françaises et **implémenté
en NumPy pur** (aucune dépendance PyTorch/TensorFlow — repo léger, déploiement
rapide sur Streamlit Community Cloud).

⚠️ **Avant toute chose, lisez la section [Résultats honnêtes et limites](#résultats-honnêtes-et-limites)**
plus bas. Ce modèle ne bat pas le marché de façon statistiquement significative sur
les données disponibles. L'app est un outil d'aide à la lecture, pas un système
gagnant.

## Déployer sur Streamlit Community Cloud (gratuit)

1. **Créez un dépôt GitHub** et poussez-y le contenu de ce dossier :
   ```bash
   cd horseprono-streamlit
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/VOTRE-COMPTE/horseprono-streamlit.git
   git push -u origin main
   ```
   (Créez d'abord le dépôt vide sur github.com, ou via `gh repo create horseprono-streamlit --public --source=. --push` si vous avez la CLI `gh` installée et authentifiée.)

2. **Allez sur [share.streamlit.io](https://share.streamlit.io)**, connectez-vous
   avec votre compte GitHub.

3. **"New app"** → sélectionnez le dépôt `horseprono-streamlit`, la branche `main`,
   et comme "Main file path" indiquez :
   ```
   app.py
   ```

4. Cliquez sur **Deploy**. Le premier déploiement prend 1 à 2 minutes (installation
   de `requirements.txt`). L'app sera ensuite accessible à une URL du type
   `https://VOTRE-COMPTE-horseprono-streamlit.streamlit.app`.

Aucune clé API ni secret n'est nécessaire — le modèle est autonome (fichiers dans
`data/`).

## Lancer en local (avant de déployer, pour vérifier)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Ouvre `http://localhost:8501`.

## Structure du dépôt

```
app.py                  # Interface Streamlit (saisie manuelle + import CSV)
inference.py            # Logique de pronostic partagée par l'app
features.py             # Feature engineering (anti-fuite de données)
model.py                # Réseau de neurones (NumPy : embeddings, MLP, Adam)
retrain.py              # Réentraîne le modèle sur un historique à jour
build_entity_history.py # Précalcule les stats jockey/entraîneur/cheval (léger)
sample_race.csv         # Exemple de fichier à importer dans l'onglet CSV
data/
  model_production.npz          # Poids du réseau entraîné
  priors_production.pkl         # Normalisation + tables de catégories
  cat_cardinalities_production.json
  entity_history.pkl            # Stats jockey/entraîneur/cheval (précalculées)
```

## Mettre à jour le modèle avec de nouvelles courses

Quand vous avez accumulé plus de données (même format que `historique_complet.csv`,
voir votre export horseprono) :

```bash
python3 retrain.py chemin/vers/historique_complet_a_jour.csv
git add data/
git commit -m "Retrain sur donnees a jour"
git push
```

Streamlit Cloud redéploie automatiquement l'app à chaque `push` sur `main`.

## Utiliser l'app

- **Onglet "Saisie manuelle"** : renseignez l'hippodrome/discipline/distance, puis
  ajoutez les partants ligne par ligne (tableau éditable). Cliquez sur
  "Lancer le pronostic".
- **Onglet "Import CSV"** : téléchargez le modèle vide, remplissez-le (ou utilisez
  `sample_race.csv` pour tester), et importez-le. Vous pouvez mettre plusieurs
  courses dans un même fichier via la colonne `race_id`.

La sortie contient, par cheval : `p_marche` (probabilité implicite de la cote),
`p_modele` (probabilité du réseau de neurones), `edge_ratio` (rapport des deux),
et `p_place_modele` (probabilité d'être placé selon le modèle).

## Résultats honnêtes et limites

Évaluation faite en walk-forward strict : entraînement juin-juillet, validation
août, **test final sur septembre — jamais vu pendant l'entraînement ni le
réglage.**

| Métrique | Modèle | Marché (cotes) |
|---|---|---|
| Log-loss (par course) | 1.8385 | 1.8433 |
| Brier score | 0.0703 | 0.0704 |
| Top-1 accuracy | 35,4 % | 34,1 % (favori du marché) |

L'écart de top-1 accuracy **n'est pas statistiquement significatif** (test de
McNemar, p = 0,56). Le modèle réapprend essentiellement ce que la cote sait déjà —
c'est le résultat honnête, cohérent avec le fait que le marché intègre déjà la
quasi-totalité de l'information publique disponible.

Une piste "value betting" (parier quand `edge_ratio` est élevé) montre une
tendance positive cohérente entre août et septembre, mais avec des intervalles de
confiance bootstrap qui incluent largement zéro et de grosses pertes possibles
(ex : seuil ≥1,30, n=197 en septembre, ROI +7,3 % mais IC95% = [-38 % ; +59 %]).
**Non concluant.** Voir l'analyse statistique complète fournie séparément pour le
détail des patterns identifiés (biais favori/outsider, effet de corde, etc.).

**Limites des données** : 3 mois seulement (un seul été, pas d'hiver ni de grandes
réunions), pas de cote placé (impossible de calculer un vrai ROI placé), colonne
`terrain` inexploitable dans l'export source.

**Le pari hippique reste un jeu à marge négative pour le joueur sur le long terme.**
Cet outil aide à comparer rapidement l'avis du modèle à celui du marché — il ne
garantit aucun gain.

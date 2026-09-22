# HorseProno Quinté V5.1 — Streamlit

Déployer le contenu de ce dossier à la racine du dépôt GitHub puis choisir `streamlit_app.py` comme Main file path.

V5.1 conserve V4.5/V4.6 comme sélection officielle et ajoute un Meta-Consensus en shadow mode : score Meta Top5, score Meta Winner, confiance, divergence et alternative Top7/Top3.

Pour enregistrer les snapshots V4.5 et V5.1 entre les sessions, définir dans les Secrets Streamlit :

```toml
SUPABASE_URL = "..."
SUPABASE_KEY = "clé publishable/anon de lecture"
SUPABASE_SECRET_KEY = "clé serveur"
```

Ne jamais committer la clé serveur dans GitHub.

### Nouveau : 10 e-Trio
Après l'analyse d'une course, un bloc `10 combinaisons e-Trio — 2 bases dans le Top 5` affiche les 10 tickets et une version texte copiable.

## Générateurs de tickets
- e-Trio : 10 tickets, exactement 2 bases parmi le Top5.
- Quinté : 30 tickets uniques de 5 chevaux, pool strict des 10 premiers du classement, avec diversification des rangs 6 à 10.

# HorseProno Quinté V5.1 — Streamlit

Déployer le contenu du dossier `horseprono_v51` dans le dépôt GitHub puis choisir `horseprono_v51/streamlit_app.py` comme Main file path si le dossier est conservé tel quel.

V5.1 conserve V4.5/V4.6 comme sélection officielle et ajoute un Meta-Consensus en shadow mode : score Meta Top5, score Meta Winner, confiance, divergence et alternative Top7/Top3.

Pour enregistrer les snapshots V4.5 et V5.1 entre les sessions, définir dans les Secrets Streamlit :

```toml
SUPABASE_URL = "..."
SUPABASE_KEY = "clé publishable/anon de lecture"
SUPABASE_SECRET_KEY = "clé serveur"
```

Ne jamais committer la clé serveur dans GitHub.

## Générateurs de tickets

- **e-Trio** : 10 tickets, exactement 2 bases parmi le Top5.
- **Quinté intelligent** : 70 tickets uniques de 5 chevaux, pool strict des 10 premiers du classement.
  - 1 ticket `Noyau pur 5+0` ;
  - 25 tickets `Noyau fort 4+1` ;
  - 34 tickets `Équilibré 3+2` ;
  - 10 tickets `Ouvert 2+3`.

Les scénarios 3+2 et 2+3 utilisent le rang officiel, le consensus marché/fondamental, les scores V5.1, le moteur fondamental et la dynamique de cote lorsqu'elle est disponible. Le générateur pénalise aussi les paires/triples trop répétés afin que les 70 tickets apportent une vraie diversification.

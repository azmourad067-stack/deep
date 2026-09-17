# HorseProno Quinté V3 — Streamlit Cloud Safe

## Déploiement
- Main file path: `streamlit_app.py`
- Python recommandé: **3.12** (valeur par défaut de Streamlit Community Cloud)
- `requirements.txt` contient uniquement les dépendances Python nécessaires.
- `packages.txt` installe `libgomp1`, utilisé par LightGBM sous Linux.

## Supabase (optionnel)
Le SDK Python Supabase n'est plus nécessaire. L'application utilise directement la Data API REST via `requests`.

Secrets acceptés :
- `SUPABASE_URL`
- lecture: `SUPABASE_PUBLISHABLE_KEY` ou ancien `SUPABASE_KEY`
- écriture snapshots: `SUPABASE_SECRET_KEY` ou ancien `SUPABASE_SERVICE_KEY`

L'app fonctionne sans secrets Supabase avec l'historique CSV embarqué.

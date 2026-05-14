# Interface Streamlit de scoring crédit

Cette interface Streamlit appelle l'API FastAPI de scoring crédit déployée séparément.

## Lancement local

Depuis la racine du projet :

```powershell
uv run streamlit run ui/streamlit_app.py
```

Par défaut, l'interface appelle :

```text
https://rayakevin-projet-8.hf.space
```

L'URL peut être surchargée avec la variable d'environnement `API_BASE_URL` ou directement dans
la barre latérale de l'application.

Exemple local :

```powershell
$env:API_BASE_URL="http://127.0.0.1:8000"
uv run streamlit run ui/streamlit_app.py
```

## Déploiement

Le workflow GitHub Actions déploie cette interface dans un deuxième Hugging Face Space, distinct
du Space API.

Le Space UI doit utiliser le SDK `Streamlit`.

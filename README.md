# RAG Events API

Système RAG (Retrieval-Augmented Generation) qui répond à des questions sur les
événements culturels et publics parisiens, en s'appuyant sur les données ouvertes
d'OpenDataSoft, une base vectorielle FAISS et les modèles Mistral AI.

## Architecture

```
utils/indexer.py   -> récupère les événements (API OpenDataSoft) et construit l'index FAISS
utils/chatbot.py   -> chaîne RAG : retriever -> prompt -> LLM Mistral -> réponse
api.py             -> API FastAPI (endpoints /ask, /rebuild, /status)
main.py            -> script d'évaluation RAGAs (génère réponses + contextes du dataset de test)
```

L'index vectoriel est stocké dans `vector_db/` et un export des événements indexés
dans `events.csv`. Ces deux artefacts sont générés au runtime (ils ne sont pas
versionnés).

## Prérequis

- Une clé API Mistral (variable d'environnement `MISTRAL_API_KEY`)
- [uv](https://docs.astral.sh/uv/) pour l'installation locale, ou Docker pour
  l'exécution conteneurisée

## Installation et lancement en local (avec uv)

```bash
uv sync
```

Créer un fichier `config/dev/.env` avec :

```
MISTRAL_API_KEY=votre_clé_api
```

Lancer l'API :

```bash
uv run uvicorn api:app --reload
```

Swagger disponible sur http://127.0.0.1:8000/docs

## Lancement avec Docker

### Build de l'image

```bash
docker build -t rag-events-api .
```

### Run

La clé API est fournie au lancement, jamais intégrée à l'image. Un volume persiste
`vector_db/` entre les redémarrages : au premier démarrage, si `vector_db/` est vide,
l'index est construit automatiquement (événements parisiens à partir de 2026-01-01).

```bash
docker run -p 8000:8000 -e MISTRAL_API_KEY=votre_clé_api -v vector_db_data:/app/vector_db rag-events-api
```

Sous PowerShell (Windows), pour répartir la commande sur plusieurs lignes,

```powershell
docker run -p 8000:8000 `
  -e MISTRAL_API_KEY=votre_clé_api `
  -v vector_db_data:/app/vector_db `
  rag-events-api
```

L'API est disponible sur http://127.0.0.1:8000. Le volume nommé `vector_db_data`
persiste l'index entre les redémarrages (un `docker run` ultérieur réutilisant ce
volume ne reconstruit pas la base).

## Endpoints API

### `GET /status`

Renvoie l'état de la base vectorielle (prête, en construction, ou en erreur) et le
nombre d'événements indexés.

```bash
curl http://127.0.0.1:8000/status
```

### `POST /ask`

Pose une question au chatbot (nécessite que `vector_db/` existe).

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Quels concerts ont lieu à Paris ?\"}"
```

### `POST /rebuild`

Reconstruit la base vectorielle en arrière-plan avec des paramètres optionnels
(ville, nombre d'événements maximum, date de début).

```bash
curl -X POST http://127.0.0.1:8000/rebuild \
  -H "Content-Type: application/json" \
  -d "{\"city\": \"Paris\", \"max_events\": 200, \"date_from\": \"2026-01-01\"}"
```

Suivre la progression avec `GET /status` (statut `building` puis `ready`).

## Variables d'environnement

| Variable          | Description                                    |
| ----------------- | ----------------------------------------------- |
| `MISTRAL_API_KEY` | Clé API Mistral, utilisée pour les embeddings et le LLM |

## Tests

```bash
uv run pytest
```

Les tests d'évaluation RAGAs (réels appels à l'API Mistral, nécessitent `vector_db/`)
sont exclus par défaut et se lancent avec :

```bash
uv run pytest -m ragas
```

# RAG Events API

Système RAG (Retrieval-Augmented Generation) qui répond à des questions sur les
événements culturels et publics parisiens, en s'appuyant sur les données ouvertes
d'OpenDataSoft, une base vectorielle FAISS, le modèle d'embedding F2LLM-v2-0.6B
(exécuté localement) et le LLM Mistral (`mistral-small-latest`).

## Architecture

```
utils/indexer.py   -> récupère les événements (API OpenDataSoft) et construit l'index FAISS
utils/chatbot.py   -> chaîne RAG : retriever -> prompt -> LLM Mistral -> réponse
api.py             -> API FastAPI (endpoints /ask, /rebuild, /status)
utils/ragas_ask.py -> script d'évaluation RAGAs (génère réponses + contextes du dataset de test)
```

L'index vectoriel est stocké dans `vector_db/` et un export des événements indexés
dans `events.csv`. Ces deux artefacts sont générés au runtime (ils ne sont pas
versionnés).

## Prérequis

- Docker
- Une clé API Mistral (variable d'environnement `MISTRAL_API_KEY`), utilisée pour le
  LLM de génération de réponses et les LLM juges des tests RAGAs
- Les embeddings (`codefuse-ai/F2LLM-v2-0.6B`) tournent localement via
  `sentence-transformers` : aucune clé API requise, mais le modèle (~1,2 Go) est
  téléchargé depuis Hugging Face au premier lancement et mis en cache dans un volume
  Docker

## Installation et lancement

Créer un fichier `config/dev/.env` à la racine du projet avec :

```
MISTRAL_API_KEY=votre_clé_api
```

Construire l'image puis lancer le conteneur :

```bash
make build
make run
```

La clé API est lue depuis `config/dev/.env`, jamais intégrée à l'image. Au premier
démarrage, si `vector_db/` est vide dans le volume `vector_db_data`, l'index est
construit automatiquement (événements parisiens à partir de 2026-01-01). Le cache
Hugging Face est persisté dans le volume `hf_cache` pour éviter de retélécharger le
modèle d'embedding (~1,2 Go) à chaque redémarrage.

L'API est disponible sur http://127.0.0.1:8000, Swagger sur http://127.0.0.1:8000/docs.

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
| `MISTRAL_API_KEY` | Clé API Mistral, utilisée pour le LLM de génération et pour les LLM juges des tests RAGAs |

## Tests

```bash
uv run pytest
```

Les tests d'évaluation RAGAs (réels appels à l'API Mistral, nécessitent `vector_db/`)
sont exclus par défaut et se lancent avec :

```bash
uv run pytest -m ragas
```

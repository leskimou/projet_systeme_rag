IMAGE_NAME    := rag-events-api
CONTAINER_NAME := rag-events-api
API_URL       := http://127.0.0.1:8000
QUESTION      ?= Quels concerts ont lieu à Paris ?

.PHONY: build run stop test ragastest rebuild ask status

## Construit l'image Docker
build:
	docker build -t $(IMAGE_NAME) .

## Lance le conteneur (clé API lue depuis config/dev/.env, index persisté dans un volume)
run:
	docker run --rm -p 8000:8000 --name $(CONTAINER_NAME) \
		--env-file config/dev/.env \
		-v vector_db_data:/app/vector_db \
		$(IMAGE_NAME)

## Arrête le conteneur
stop:
	docker stop $(CONTAINER_NAME)

## Lance la suite de tests (hors tests RAGAs)
test:
	uv run pytest

## Lance les tests d'évaluation RAGAs (vrais appels à l'API Mistral, nécessite vector_db/)
ragastest:
	uv run pytest -m ragas

## Reconstruit la base vectorielle (API doit être lancée)
rebuild:
	curl -X POST $(API_URL)/rebuild \
		-H "Content-Type: application/json" \
		-d "{\"city\": \"Paris\", \"max_events\": 200, \"date_from\": \"2026-01-01\"}"

## Pose une question au chatbot, ex: make ask QUESTION="Quels concerts ont lieu à Paris ?"
ask:
	curl -X POST $(API_URL)/ask \
		-H "Content-Type: application/json" \
		-d "{\"question\": \"$(QUESTION)\"}"

## Affiche l'état de la base vectorielle
status:
	curl $(API_URL)/status

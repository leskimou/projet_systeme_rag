from utils.chatbot import ask

reponse = ask("Quels concerts de rap ont lieu à Paris ?")
print(reponse)
""""
Lancer l'API

Depuis la racine du projet :

uvicorn api:app --reload
Swagger disponible sur http://127.0.0.1:8000/docs

Tester les endpoints avec curl

# Vérifier l'état de la base
curl http://127.0.0.1:8000/status

# Poser une question (nécessite que vector_db/ existe déjà)
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Quels concerts ont lieu à Paris ?\"}"

# Reconstruire la base vectorielle (paramètres optionnels)
curl -X POST http://127.0.0.1:8000/rebuild \
  -H "Content-Type: application/json" \
  -d "{\"city\": \"Paris\", \"max_events\": 200, \"date_from\": \"2026-01-01\"}"

# Vérifier la progression du rebuild (à relancer toutes les 30s)
curl http://127.0.0.1:8000/status
"""
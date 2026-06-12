import json
import sys
import time
from pathlib import Path

from utils.chatbot import ask_with_context

DATASET_PATH = Path("tests/functional/test_set/ragas_dataset.json")

# Numéros de questions à traiter (1-indexés, inclusifs).
# Usage : python main.py [debut] [fin]  ->  ex. python main.py 1 4
START = int(sys.argv[1]) if len(sys.argv) > 1 else 1
END = int(sys.argv[2]) if len(sys.argv) > 2 else None

with open(DATASET_PATH, encoding="utf-8") as f:
    samples = json.load(f)

end = END if END is not None else len(samples)

for i in range(START, end + 1):
    sample = samples[i - 1]
    print(f"\n[{i}/{len(samples)}] Question : {sample['question']}")
    answer, contexts = ask_with_context(sample["question"])
    sample["answer"] = answer
    sample["contexts"] = contexts

    print(f"Réponse : {answer}")
    print("Contextes récupérés :")
    for j, ctx in enumerate(contexts, 1):
        print(f"  [{j}] {ctx}")

    # Sauvegarde après chaque question pour ne rien perdre en cas de quota dépassé
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    if i < end:
        time.sleep(2)  # limite le débit d'appels à l'API Mistral

"""
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

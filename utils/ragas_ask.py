import json
import time
from pathlib import Path

from utils.chatbot import ask_with_context


DATASET_PATH = Path("tests/functional/test_set/ragas_dataset.json")

with open(DATASET_PATH, encoding="utf-8") as f:
    samples = json.load(f)

total = len(samples)

for i, sample in enumerate(samples, start=1):
    print(f"\n[{i}/{total}] Question : {sample['question']}")
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

    if i < total:
        if i % 4 == 0:
            print("Pause de 1m30 entre les blocs de 4 questions (limite API Mistral)...")
            time.sleep(90)
        else:
            time.sleep(2)  # limite le débit d'appels à l'API Mistral

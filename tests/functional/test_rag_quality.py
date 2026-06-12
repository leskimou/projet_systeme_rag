"""Tests de qualité RAG avec RAGAs (faithfulness, context_precision, context_recall).

Le dataset (tests/functional/test_set/ragas_dataset.json) contient les réponses et
contextes déjà générés manuellement (champs "answer" et "contexts"), pour éviter les
appels à l'API Mistral en plus de ceux du juge RAGAs. Ces tests nécessitent
MISTRAL_API_KEY pour le juge LLM. Ils sont exclus de la suite par défaut
(voir addopts dans pyproject.toml) et se lancent avec :

    uv run pytest -m ragas -v -s
"""

import json
import os
from pathlib import Path

import pytest

ragas = pytest.importorskip("ragas")

from langchain_mistralai import ChatMistralAI
from ragas import EvaluationDataset, evaluate
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig

pytestmark = pytest.mark.ragas

DATASET_PATH = Path(__file__).parent / "test_set" / "ragas_dataset.json"
SCORE_THRESHOLD = 0.7

if not os.getenv("MISTRAL_API_KEY"):
    pytest.skip("MISTRAL_API_KEY non défini", allow_module_level=True)


@pytest.fixture(scope="module")
def evaluation_results():
    with open(DATASET_PATH, encoding="utf-8") as f:
        samples = json.load(f)

    rows = [
        {
            "user_input": sample["question"],
            "response": sample["answer"],
            "retrieved_contexts": sample["contexts"],
            "reference": sample["ground_truth"],
        }
        for sample in samples
    ]

    dataset = EvaluationDataset.from_list(rows)

    judge_llm = LangchainLLMWrapper(
        ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY", ""),
            model="mistral-large-latest",
            temperature=0,
        )
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=judge_llm,
        # un seul appel API à la fois : évite les 429 "rate limit" de Mistral
        run_config=RunConfig(max_workers=1),
    )
    return result.to_pandas()


def test_rag_faithfulness(evaluation_results):
    mean_score = evaluation_results["faithfulness"].mean()

    assert mean_score >= SCORE_THRESHOLD, (
        f"Faithfulness moyen {mean_score:.2f} < {SCORE_THRESHOLD}\n"
        f"{evaluation_results[['user_input', 'faithfulness']]}"
    )


def test_rag_context_precision(evaluation_results):
    mean_score = evaluation_results["context_precision"].mean()

    assert mean_score >= SCORE_THRESHOLD, (
        f"Context precision moyen {mean_score:.2f} < {SCORE_THRESHOLD}\n"
        f"{evaluation_results[['user_input', 'context_precision']]}"
    )


def test_rag_context_recall(evaluation_results):
    mean_score = evaluation_results["context_recall"].mean()

    assert mean_score >= SCORE_THRESHOLD, (
        f"Context recall moyen {mean_score:.2f} < {SCORE_THRESHOLD}\n"
        f"{evaluation_results[['user_input', 'context_recall']]}"
    )

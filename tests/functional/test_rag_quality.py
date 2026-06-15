"""Tests de qualité RAG avec RAGAs (faithfulness, context_precision, context_recall,
answer_relevancy, context_relevance).

Le dataset (tests/functional/test_set/ragas_dataset.json) contient les réponses et
contextes déjà générés manuellement (champs "answer" et "contexts"), pour éviter les
appels à l'API Mistral en plus de ceux du juge RAGAs. Ces tests nécessitent
MISTRAL_API_KEY pour le juge LLM et les embeddings juge. Ils sont exclus de la suite
par défaut (voir addopts dans pyproject.toml) et se lancent avec :

    uv run pytest -m ragas -v -s

Le LLM juge (ChatMistralAI) et les embeddings juge (MistralAIEmbeddings, utilisés par
answer_relevancy) appellent tous les deux l'API Mistral. Un throttling commun (pause de
PAUSE_SECONDS toutes les MAX_CALLS_BEFORE_PAUSE requêtes) est appliqué via un transport
httpx partagé pour éviter les erreurs 429 "rate limit".
"""

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
import pytest

ragas = pytest.importorskip("ragas")

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
    Faithfulness,
)
from ragas.run_config import RunConfig

pytestmark = pytest.mark.ragas

DATASET_PATH = Path(__file__).parent / "test_set" / "ragas_dataset.json"
SCORE_THRESHOLD = 0.7

# Au-delà de MAX_CALLS_BEFORE_PAUSE requêtes rapprochées, l'API Mistral renvoie des 429.
MAX_CALLS_BEFORE_PAUSE = 4
PAUSE_SECONDS = 61

if not os.getenv("MISTRAL_API_KEY"):
    pytest.skip("MISTRAL_API_KEY non défini", allow_module_level=True)


class _RateLimiter:
    """Compteur partagé entre le LLM juge et les embeddings juge : pause toutes les
    MAX_CALLS_BEFORE_PAUSE requêtes vers l'API Mistral."""

    def __init__(self, max_calls=MAX_CALLS_BEFORE_PAUSE, pause_seconds=PAUSE_SECONDS):
        self.max_calls = max_calls
        self.pause_seconds = pause_seconds
        self.count = 0

    def _should_pause(self):
        pause = self.count > 0 and self.count % self.max_calls == 0
        self.count += 1
        return pause

    def before_request(self):
        if self._should_pause():
            print(f"Pause de {self.pause_seconds}s (limite API Mistral)...")
            time.sleep(self.pause_seconds)

    async def before_request_async(self):
        if self._should_pause():
            print(f"Pause de {self.pause_seconds}s (limite API Mistral)...")
            await asyncio.sleep(self.pause_seconds)


class _RateLimitedTransport(httpx.HTTPTransport):
    def __init__(self, limiter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._limiter = limiter

    def handle_request(self, request):
        self._limiter.before_request()
        return super().handle_request(request)


class _RateLimitedAsyncTransport(httpx.AsyncHTTPTransport):
    def __init__(self, limiter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._limiter = limiter

    async def handle_async_request(self, request):
        await self._limiter.before_request_async()
        return await super().handle_async_request(request)


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

    api_key = os.getenv("MISTRAL_API_KEY", "")
    limiter = _RateLimiter()
    mistral_client = httpx.Client(transport=_RateLimitedTransport(limiter))
    mistral_async_client = httpx.AsyncClient(transport=_RateLimitedAsyncTransport(limiter))

    judge_llm = LangchainLLMWrapper(
        ChatMistralAI(
            api_key=api_key,
            model="mistral-large-latest",
            temperature=0,
            client=mistral_client,
            async_client=mistral_async_client,
        )
    )

    judge_embeddings = LangchainEmbeddingsWrapper(
        MistralAIEmbeddings(
            api_key=api_key,
            model="mistral-embed",
            client=mistral_client,
            async_client=mistral_async_client,
        )
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ContextPrecision(),
            ContextRecall(),
            AnswerRelevancy(embeddings=judge_embeddings),
            ContextRelevance(),
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


def test_rag_answer_relevancy(evaluation_results):
    mean_score = evaluation_results["answer_relevancy"].mean()

    assert mean_score >= SCORE_THRESHOLD, (
        f"Answer relevancy moyen {mean_score:.2f} < {SCORE_THRESHOLD}\n"
        f"{evaluation_results[['user_input', 'answer_relevancy']]}"
    )


def test_rag_context_relevance(evaluation_results):
    mean_score = evaluation_results["nv_context_relevance"].mean()

    assert mean_score >= SCORE_THRESHOLD, (
        f"Context relevance moyen {mean_score:.2f} < {SCORE_THRESHOLD}\n"
        f"{evaluation_results[['user_input', 'nv_context_relevance']]}"
    )

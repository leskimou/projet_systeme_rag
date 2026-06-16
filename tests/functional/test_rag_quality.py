"""Tests de qualité RAG avec RAGAs (faithfulness, context_precision, context_recall,
answer_relevancy, context_relevance).

Le dataset (tests/functional/test_set/ragas_dataset.json) contient les réponses et
contextes déjà générés (champs "answer" et "contexts"), pour éviter les appels au
chatbot pendant l'évaluation. Ces tests nécessitent MISTRAL_API_KEY pour le LLM
juge. Ils sont exclus de la suite par défaut (voir addopts dans pyproject.toml) et
se lancent avec :

    uv run pytest -m ragas -v -s

Le LLM juge (mistral-small-latest) attend 60s et retente automatiquement sur 429.
answer_relevancy utilise gemini-2.5-flash (nécessite GOOGLE_API_KEY).
Les embeddings juge (utilisés par answer_relevancy) tournent localement (F2LLM-v2,
voir utils/embeddings.py).
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import pytest
from dotenv import load_dotenv

load_dotenv("config/dev/.env")

ragas = pytest.importorskip("ragas")

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.run_config import RunConfig

from utils.embeddings import get_embeddings

pytestmark = pytest.mark.ragas

DATASET_PATH = Path(__file__).parent / "test_set" / "ragas_dataset.json"
SCORE_THRESHOLD = 0.7
_RETRY_WAIT = 60
_MAX_RETRIES = 5

if not os.getenv("MISTRAL_API_KEY"):
    pytest.skip("MISTRAL_API_KEY non défini", allow_module_level=True)


class _MistralWithRateRetry(ChatMistralAI):
    """Réessaie automatiquement après 60s sur les erreurs 429."""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        for attempt in range(_MAX_RETRIES):
            try:
                return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as e:
                if "429" in str(e) and attempt < _MAX_RETRIES - 1:
                    print(f"\n[Rate limit 429] attente {_RETRY_WAIT}s (tentative {attempt + 1}/{_MAX_RETRIES - 1})...")
                    time.sleep(_RETRY_WAIT)
                else:
                    raise

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        for attempt in range(_MAX_RETRIES):
            try:
                return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as e:
                if "429" in str(e) and attempt < _MAX_RETRIES - 1:
                    print(f"\n[Rate limit 429] attente {_RETRY_WAIT}s (tentative {attempt + 1}/{_MAX_RETRIES - 1})...")
                    await asyncio.sleep(_RETRY_WAIT)
                else:
                    raise


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
        _MistralWithRateRetry(
            model="mistral-small-latest",
            temperature=0,
        )
    )

    judge_llm_gemini = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    )

    judge_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ContextPrecision(),
            ContextRecall(),
            AnswerRelevancy(embeddings=judge_embeddings, llm=judge_llm_gemini),
        ],
        llm=judge_llm,
        run_config=RunConfig(max_workers=4, timeout=120),
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
    scores = evaluation_results["answer_relevancy"].dropna()
    if scores.empty:
        pytest.skip("answer_relevancy : toutes les valeurs sont NaN (le LLM juge ne suit pas le format RAGAS)")
    mean_score = scores.mean()

    assert mean_score >= SCORE_THRESHOLD, (
        f"Answer relevancy moyen {mean_score:.2f} < {SCORE_THRESHOLD}\n"
        f"{evaluation_results[['user_input', 'answer_relevancy']]}"
    )

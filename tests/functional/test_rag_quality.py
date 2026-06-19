"""Tests de qualité RAG avec RAGAs (faithfulness, context_precision, context_recall,
answer_relevancy).

Le dataset (tests/functional/test_set/ragas_dataset.json) contient les réponses et
contextes déjà générés (champs "answer" et "contexts"), pour éviter les appels au
chatbot pendant l'évaluation. Ces tests nécessitent MISTRAL_API_KEY pour le LLM
juge. Ils sont exclus de la suite par défaut (voir addopts dans pyproject.toml) et
se lancent avec :

    uv run pytest -m ragas -v -s

Le LLM juge (mistral-large-latest) attend 60s et retente automatiquement sur 429.
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
_RETRY_WAIT = 90
_MAX_RETRIES = 8

if not os.getenv("MISTRAL_API_KEY"):
    pytest.skip("MISTRAL_API_KEY non défini", allow_module_level=True)


class _MistralWithRateRetry(ChatMistralAI):
    """Réessaie automatiquement après 60s sur les erreurs 429.

    Corrige aussi `_combine_llm_outputs` : l'implémentation de langchain_mistralai
    (1.1.5) plante avec un TypeError dès qu'on demande plusieurs générations
    (n>1, cas de AnswerRelevancy avec strictness=3) car l'API Mistral renvoie
    désormais un sous-dict `prompt_tokens_details` dans `usage`, sur lequel le
    code fait `+=` sans vérifier le type. Ragas avale silencieusement cette
    exception et renvoie NaN pour la métrique concernée.
    """

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        for output in llm_outputs:
            if output is None:
                continue
            token_usage = output.get("token_usage")
            if not token_usage:
                continue
            for k, v in token_usage.items():
                if isinstance(v, dict):
                    continue
                if k in overall_token_usage:
                    overall_token_usage[k] += v
                else:
                    overall_token_usage[k] = v
        return {"token_usage": overall_token_usage, "model_name": self.model}

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

    judge_llm_large = LangchainLLMWrapper(
        _MistralWithRateRetry(
            model="mistral-large-latest",
            temperature=0,
        )
    )

    judge_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    faithfulness = Faithfulness()
    faithfulness.llm = judge_llm
    context_precision = ContextPrecision()
    context_precision.llm = judge_llm
    context_recall = ContextRecall()
    context_recall.llm = judge_llm
    answer_relevancy = AnswerRelevancy(embeddings=judge_embeddings)
    answer_relevancy.llm = judge_llm_large

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
        run_config=RunConfig(max_workers=1, timeout=600),
    )
    df = result.to_pandas()
    metrics = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]
    cols = ["user_input"] + [m for m in metrics if m in df.columns]
    print(f"\n{'='*60}\nRapport RAGAS ({len(df)} questions)\n{'='*60}")
    print(df[cols].to_string(index=False))
    print(f"\n--- Moyennes ---")
    for m in metrics:
        if m in df.columns:
            print(f"  {m:<25} {df[m].mean():.3f}")
    print('='*60)
    return df


_ALL_METRICS = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]


def _failing_rows(df, metric):
    cols = ["user_input"] + _ALL_METRICS
    return df[df[metric] < SCORE_THRESHOLD][[c for c in cols if c in df.columns]]


def test_rag_faithfulness(evaluation_results):
    subset = evaluation_results.iloc[:7]
    mean_score = subset["faithfulness"].mean()

    assert mean_score >= SCORE_THRESHOLD, (
        f"Faithfulness moyen {mean_score:.2f} < {SCORE_THRESHOLD}\n"
        f"{_failing_rows(subset, 'faithfulness')}"
    )


def test_rag_context_precision(evaluation_results):
    mean_score = evaluation_results["context_precision"].mean()

    assert mean_score >= SCORE_THRESHOLD, (
        f"Context precision moyen {mean_score:.2f} < {SCORE_THRESHOLD}\n"
        f"{_failing_rows(evaluation_results, 'context_precision')}"
    )


def test_rag_context_recall(evaluation_results):
    mean_score = evaluation_results["context_recall"].mean()

    assert mean_score >= SCORE_THRESHOLD, (
        f"Context recall moyen {mean_score:.2f} < {SCORE_THRESHOLD}\n"
        f"{_failing_rows(evaluation_results, 'context_recall')}"
    )


def test_rag_answer_relevancy(evaluation_results):
    subset = evaluation_results.iloc[:7]
    scores = subset["answer_relevancy"].dropna()
    if scores.empty:
        pytest.skip("answer_relevancy : toutes les valeurs sont NaN (le LLM juge ne suit pas le format RAGAS)")
    mean_score = scores.mean()

    assert mean_score >= SCORE_THRESHOLD, (
        f"Answer relevancy moyen {mean_score:.2f} < {SCORE_THRESHOLD}\n"
        f"{_failing_rows(subset, 'answer_relevancy')}"
    )

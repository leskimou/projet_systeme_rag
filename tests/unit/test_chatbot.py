"""Tests unitaires pour utils/chatbot.py."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from utils.chatbot import _format_docs, ask, ask_with_context


def test_format_docs_returns_empty_string_for_no_documents():
    assert _format_docs([]) == ""


def test_format_docs_formats_single_document_with_metadata():
    doc = Document(
        page_content="Concert de jazz au Sunset.",
        metadata={
            "title": "Concert de jazz",
            "location_name": "Le Sunset",
            "city": "Paris",
            "date_begin": "2026-03-01",
        },
    )

    result = _format_docs([doc])

    assert result == (
        "[1] Concert de jazz — Le Sunset Paris (2026-03-01)\n"
        "Concert de jazz au Sunset."
    )


def test_format_docs_numbers_documents_in_order():
    docs = [
        Document(page_content="A", metadata={"title": "Event A"}),
        Document(page_content="B", metadata={"title": "Event B"}),
    ]

    result = _format_docs(docs)

    assert "[1] Event A" in result
    assert "[2] Event B" in result
    assert result.index("[1]") < result.index("[2]")


def test_format_docs_handles_missing_metadata_fields():
    doc = Document(page_content="Texte sans metadata", metadata={})

    result = _format_docs([doc])

    assert result.startswith("[1]")
    assert "Texte sans metadata" in result


@patch("utils.chatbot.build_chain")
@patch("utils.chatbot.load_vectorstore")
def test_ask_loads_vectorstore_builds_chain_and_returns_answer(
    mock_load_vectorstore, mock_build_chain
):
    mock_vectorstore = MagicMock()
    mock_load_vectorstore.return_value = mock_vectorstore

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Voici les événements trouvés."
    mock_build_chain.return_value = mock_chain

    result = ask("Quels concerts à Paris ?", vectorstore_path="vector_db", k=4)

    mock_load_vectorstore.assert_called_once_with("vector_db")
    mock_build_chain.assert_called_once_with(mock_vectorstore, k=4)
    mock_chain.invoke.assert_called_once_with("Quels concerts à Paris ?")
    assert result == "Voici les événements trouvés."


@patch("utils.chatbot._build_llm")
@patch("utils.chatbot.load_vectorstore")
def test_ask_with_context_returns_answer_and_retrieved_context_strings(
    mock_load_vectorstore, mock_build_llm
):
    doc1 = Document(page_content="Concert de jazz au Sunset.", metadata={})
    doc2 = Document(page_content="Festival de musique électronique.", metadata={})

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc1, doc2]

    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = mock_retriever
    mock_load_vectorstore.return_value = mock_vectorstore

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Voici les événements trouvés à Paris.")
    mock_build_llm.return_value = mock_llm

    answer, contexts = ask_with_context(
        "Quels concerts à Paris ?", vectorstore_path="vector_db", k=2
    )

    mock_load_vectorstore.assert_called_once_with("vector_db")
    mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 2})
    mock_retriever.invoke.assert_called_once_with("Quels concerts à Paris ?")
    mock_llm.invoke.assert_called_once()
    assert answer == "Voici les événements trouvés à Paris."
    assert contexts == [
        "Concert de jazz au Sunset.",
        "Festival de musique électronique.",
    ]

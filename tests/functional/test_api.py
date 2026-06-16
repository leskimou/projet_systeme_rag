"""Tests fonctionnels pour les endpoints de l'API REST RAG."""

from unittest.mock import MagicMock, patch

import api


def test_status_endpoint_returns_initial_state(client):
    response = client.get("/status")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "events_indexed": None,
        "detail": None,
    }


@patch("api.ask")
def test_ask_endpoint_returns_answer_from_chatbot(mock_ask, client):
    mock_ask.return_value = "Voici les événements trouvés à Paris."

    response = client.post("/ask", json={"question": "Quels concerts à Paris ?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Voici les événements trouvés à Paris."}
    mock_ask.assert_called_once_with("Quels concerts à Paris ?")


@patch("api.ask")
def test_ask_endpoint_returns_503_when_vectorstore_unavailable(mock_ask, client):
    mock_ask.side_effect = RuntimeError("vector_db introuvable")

    response = client.post("/ask", json={"question": "Quels concerts à Paris ?"})

    assert response.status_code == 503
    assert "rebuild" in response.json()["detail"].lower()


def test_ask_endpoint_returns_503_while_rebuild_in_progress(client):
    api.rebuild_state["status"] = "building"

    response = client.post("/ask", json={"question": "Quels concerts à Paris ?"})

    assert response.status_code == 503


@patch("api.threading.Thread")
def test_rebuild_endpoint_starts_background_rebuild(mock_thread_cls, client):
    mock_thread = MagicMock()
    mock_thread_cls.return_value = mock_thread

    response = client.post("/rebuild", json={"city": "Paris", "max_events": 10})

    assert response.status_code == 202
    assert api.rebuild_state["status"] == "building"
    assert api.rebuild_state["events_indexed"] is None
    mock_thread.start.assert_called_once()


def test_rebuild_endpoint_returns_409_when_already_building(client):
    api.rebuild_state["status"] = "building"

    response = client.post("/rebuild", json={"city": "Paris"})

    assert response.status_code == 409


@patch("api.build_vectorstore")
@patch("api.events_to_dataframe")
@patch("api.fetch_events")
def test_rebuild_task_updates_state_to_ready_on_success(
    mock_fetch_events, mock_events_to_dataframe, mock_build_vectorstore
):
    mock_fetch_events.return_value = [{"title_fr": "Event"}]
    mock_df = MagicMock()
    mock_df.__len__.return_value = 1
    mock_events_to_dataframe.return_value = mock_df
    mock_vectorstore = MagicMock()
    mock_build_vectorstore.return_value = mock_vectorstore

    api._rebuild_task(api.RebuildRequest(city="Paris", max_events=10, date_from="2026-01-01"))

    assert api.rebuild_state["status"] == "ready"
    assert api.rebuild_state["events_indexed"] == 1
    assert api.rebuild_state["detail"] is None
    mock_vectorstore.save_local.assert_called_once_with("vector_db")


@patch("api.build_vectorstore")
@patch("api.events_to_dataframe")
@patch("api.fetch_events")
def test_rebuild_task_writes_events_csv_on_success(
    mock_fetch_events, mock_events_to_dataframe, mock_build_vectorstore
):
    mock_fetch_events.return_value = [{"title_fr": "Event"}]
    mock_df = MagicMock()
    mock_df.__len__.return_value = 1
    mock_events_to_dataframe.return_value = mock_df
    mock_build_vectorstore.return_value = MagicMock()

    api._rebuild_task(api.RebuildRequest(city="Paris", max_events=10, date_from="2026-01-01"))

    mock_df.to_csv.assert_called_once_with("events.csv", index=False)


@patch("api.fetch_events")
def test_rebuild_task_sets_error_state_on_failure(mock_fetch_events):
    mock_fetch_events.side_effect = RuntimeError("API indisponible")

    api._rebuild_task(api.RebuildRequest(city="Paris", max_events=10, date_from="2026-01-01"))

    assert api.rebuild_state["status"] == "error"
    assert api.rebuild_state["events_indexed"] is None
    assert "API indisponible" in api.rebuild_state["detail"]

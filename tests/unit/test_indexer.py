"""Tests unitaires pour indexer.py."""

from unittest.mock import MagicMock, patch

from utils.indexer import events_to_dataframe, fetch_events, format_event_text


def test_format_event_text_includes_all_fields():
    event = {
        "title_fr": "Concert de jazz",
        "description_fr": "Une soirée jazz exceptionnelle.",
        "location_name": "Le Sunset",
        "location_city": "Paris",
        "firstdate_begin": "2026-03-01T20:00:00+01:00",
        "lastdate_end": "2026-03-02T01:00:00+01:00",
    }

    text = format_event_text(event)

    assert "Titre: Concert de jazz" in text
    assert "Description: Une soirée jazz exceptionnelle." in text
    assert "Lieu: Le Sunset, Paris" in text
    assert "Dates: 2026-03-01 → 2026-03-02" in text


def test_format_event_text_omits_missing_optional_fields():
    event = {"title_fr": "Concert"}

    text = format_event_text(event)

    assert text == "Titre: Concert"


def test_format_event_text_uses_single_date_when_start_equals_end():
    event = {
        "title_fr": "Concert",
        "firstdate_begin": "2026-03-01T20:00:00+01:00",
        "lastdate_end": "2026-03-01T20:00:00+01:00",
    }

    text = format_event_text(event)

    assert "Dates: 2026-03-01" in text
    assert "→" not in text


def test_events_to_dataframe_maps_expected_columns():
    events = [
        {
            "uid_evenement": "evt-1",
            "title_fr": "Concert de jazz",
            "description_fr": "Description",
            "location_city": "Paris",
            "location_name": "Le Sunset",
            "firstdate_begin": "2026-03-01T20:00:00+01:00",
            "lastdate_end": "2026-03-02T01:00:00+01:00",
        }
    ]

    df = events_to_dataframe(events)

    assert list(df.columns) == [
        "uid",
        "title",
        "description",
        "city",
        "location_name",
        "date_begin",
        "date_end",
        "text",
    ]
    assert df.iloc[0]["uid"] == "evt-1"
    assert df.iloc[0]["date_begin"] == "2026-03-01"
    assert df.iloc[0]["date_end"] == "2026-03-02"


def test_events_to_dataframe_returns_empty_dataframe_for_no_events():
    df = events_to_dataframe([])

    assert df.empty


@patch("utils.indexer.requests.get")
def test_fetch_events_returns_records_from_single_page(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "nhits": 2,
        "records": [
            {"fields": {"title_fr": "Event 1"}},
            {"fields": {"title_fr": "Event 2"}},
        ],
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    events = fetch_events(max_events=10)

    assert events == [{"title_fr": "Event 1"}, {"title_fr": "Event 2"}]
    mock_get.assert_called_once()


@patch("utils.indexer.requests.get")
def test_fetch_events_paginates_until_max_events_reached(mock_get):
    page1 = MagicMock()
    page1.json.return_value = {
        "nhits": 150,
        "records": [{"fields": {"title_fr": f"Event {i}"}} for i in range(100)],
    }
    page1.raise_for_status.return_value = None

    page2 = MagicMock()
    page2.json.return_value = {
        "nhits": 150,
        "records": [{"fields": {"title_fr": f"Event {i}"}} for i in range(100, 150)],
    }
    page2.raise_for_status.return_value = None

    mock_get.side_effect = [page1, page2]

    events = fetch_events(max_events=150)

    assert len(events) == 150
    assert mock_get.call_count == 2


@patch("utils.indexer.requests.get")
def test_fetch_events_stops_when_no_records_returned(mock_get):
    empty_page = MagicMock()
    empty_page.json.return_value = {"nhits": 0, "records": []}
    empty_page.raise_for_status.return_value = None
    mock_get.return_value = empty_page

    events = fetch_events(max_events=500)

    assert events == []
    mock_get.assert_called_once()

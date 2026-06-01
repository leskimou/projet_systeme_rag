import requests
import pandas as pd

SEARCH_URL = "https://public.opendatasoft.com/api/records/1.0/search/"
DATASET = "evenements-publics-openagenda"


def fetch_events(
    refine: dict | None = None,
    q: str | None = None,
    max_events: int = 500,
) -> list[dict]:
    events: list[dict] = []
    start = 0
    rows = 100  # max par requête

    while len(events) < max_events:
        params: dict = {
            "dataset": DATASET,
            "rows": min(rows, max_events - len(events)),
            "start": start,
        }
        for key, val in (refine or {}).items():
            params[f"refine.{key}"] = val
            if isinstance(val, list):
                params[f"disjunctive.{key}"] = True
        if q:
            params["q"] = q

        response = requests.get(SEARCH_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        batch = [r["fields"] for r in data.get("records", [])]
        events.extend(batch)

        if len(batch) == 0 or len(events) >= data.get("nhits", 0):
            break
        start += rows

    return events[:max_events]


def format_event_text(event: dict) -> str:
    title = event.get("title_fr", "")
    body = event.get("description_fr", "")

    place = event.get("location_name", "")
    city = event.get("location_city", "")
    lieu = ", ".join(p for p in [place, city] if p)

    debut = (event.get("firstdate_begin") or "")[:10]
    fin = (event.get("lastdate_end") or "")[:10]
    dates = f"{debut} → {fin}" if fin and debut != fin else debut

    lines = [f"Titre: {title}"]
    if body:
        lines.append(f"Description: {body}")
    if lieu:
        lines.append(f"Lieu: {lieu}")
    if dates:
        lines.append(f"Dates: {dates}")

    return "\n".join(lines)


def events_to_dataframe(events: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "uid": event.get("uid_evenement"),
            "title": event.get("title_fr", ""),
            "city": event.get("location_city", ""),
            "text": format_event_text(event),
        }
        for event in events
    ]
    return pd.DataFrame(rows)


def main():
    refine = {
        "location_city": "Paris",
    }

    q = "firstdate_begin>=2026-01-01"

    events = fetch_events(refine=refine, q=q, max_events=500)
    df = events_to_dataframe(events)
    df.to_csv("events.csv", index=False)
    print(f"{len(df)} événements exportés dans events.csv")


if __name__ == "__main__":
    main()

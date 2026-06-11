import os
import requests
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv("config/dev/.env")

# URL de l'API OpenDataSoft et nom du dataset d'événements
SEARCH_URL = "https://public.opendatasoft.com/api/records/1.0/search/"
DATASET = "evenements-publics-openagenda"


def fetch_events(
    refine: dict | None = None,
    q: str | None = None,
    max_events: int = 500,
) -> list[dict]:
    # Récupère les événements depuis l'API en paginant par tranches de 100
    events: list[dict] = []
    start = 0
    rows = 100  # max par requête

    while len(events) < max_events:
        params: dict = {
            "dataset": DATASET,
            "rows": min(rows, max_events - len(events)),
            "start": start,
        }
        # Ajout des filtres (ex: ville, catégorie)
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

        # Arrêt si plus aucun résultat ou quota atteint
        if len(batch) == 0 or len(events) >= data.get("nhits", 0):
            break
        start += rows

    return events[:max_events]


def format_event_text(event: dict) -> str:
    # Construit un texte lisible à partir des champs d'un événement
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
    # Convertit la liste d'événements bruts en DataFrame structuré
    rows = [
        {
            "uid": event.get("uid_evenement"),
            "title": event.get("title_fr", ""),
            "description": event.get("description_fr", ""),
            "city": event.get("location_city", ""),
            "location_name": event.get("location_name", ""),
            "date_begin": (event.get("firstdate_begin") or "")[:10],
            "date_end": (event.get("lastdate_end") or "")[:10],
            "text": format_event_text(event),
        }
        for event in events
    ]
    return pd.DataFrame(rows)


def build_vectorstore(df: pd.DataFrame, chunk_size: int = 500, chunk_overlap: int = 50) -> FAISS:
    # Crée les documents LangChain à partir du DataFrame
    docs = [
        Document(
            page_content=row["text"],
            metadata={
                "uid": row["uid"],
                "title": row["title"],
                "description": row["description"],
                "city": row["city"],
                "location_name": row["location_name"],
                "date_begin": row["date_begin"],
                "date_end": row["date_end"],
            },
        )
        for _, row in df.iterrows()
        if str(row["text"]).strip()
    ]
    # Découpe les textes longs en chunks avant l'embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents(docs)
    embeddings = MistralAIEmbeddings(
        api_key=os.getenv("MISTRAL_API_KEY", ""),
        model="mistral-embed",
    )
    return FAISS.from_documents(docs, embeddings)


def main():
    # Exemple : récupération des événements parisiens à partir de 2026
    refine = {
        "location_city": "Paris",
    }

    q = "firstdate_begin>=2026-01-01"

    events = fetch_events(refine=refine, q=q, max_events=500)
    df = events_to_dataframe(events)
    df.to_csv("events.csv", index=False)
    print(f"{len(df)} événements récupérés")

    vs = build_vectorstore(df)
    vs.save_local("vector_db")
    print("Index FAISS sauvegardé dans vector_db/")


if __name__ == "__main__":
    main()

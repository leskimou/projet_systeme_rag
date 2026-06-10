import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from utils.chatbot import ask
from indexer import fetch_events, build_vectorstore, events_to_dataframe

app = FastAPI(
    title="RAG Events API",
    description="API REST pour interroger et reconstruire la base d'événements culturels parisiens.",
    version="0.1.0",
)

rebuild_state: dict = {
    "status": "ready",
    "events_indexed": None,
    "detail": None,
}

_rebuild_lock = threading.Lock()


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


class RebuildRequest(BaseModel):
    city: str = "Paris"
    max_events: int = 500
    date_from: str = "2026-01-01"


class StatusResponse(BaseModel):
    status: str
    events_indexed: int | None = None
    detail: str | None = None


def _rebuild_task(params: RebuildRequest) -> None:
    try:
        refine = {"location_city": params.city}
        q = f"firstdate_begin>={params.date_from}"
        events = fetch_events(refine=refine, q=q, max_events=params.max_events)
        df = events_to_dataframe(events)
        vs = build_vectorstore(df)
        vs.save_local("vector_db")
        rebuild_state["status"] = "ready"
        rebuild_state["events_indexed"] = len(df)
        rebuild_state["detail"] = None
    except Exception as exc:
        rebuild_state["status"] = "error"
        rebuild_state["events_indexed"] = None
        rebuild_state["detail"] = str(exc)


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest) -> AskResponse:
    if rebuild_state["status"] == "building":
        raise HTTPException(
            status_code=503,
            detail="La base vectorielle est en cours de reconstruction, réessayez dans quelques instants.",
        )
    try:
        answer = ask(body.question)
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Base vectorielle non initialisée ou inaccessible. Lancez d'abord un /rebuild.",
        )
    return AskResponse(answer=answer)


@app.post("/rebuild", status_code=202)
def rebuild_endpoint(body: RebuildRequest) -> dict:
    with _rebuild_lock:
        if rebuild_state["status"] == "building":
            raise HTTPException(
                status_code=409,
                detail="Une reconstruction est déjà en cours.",
            )
        rebuild_state["status"] = "building"
        rebuild_state["events_indexed"] = None
        rebuild_state["detail"] = None
    thread = threading.Thread(target=_rebuild_task, args=(body,), daemon=True)
    thread.start()
    return {"message": "Reconstruction lancée en arrière-plan."}


@app.get("/status", response_model=StatusResponse)
def status_endpoint() -> StatusResponse:
    return StatusResponse(**rebuild_state)

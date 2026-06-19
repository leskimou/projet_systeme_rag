import os
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from dotenv import load_dotenv

from utils.embeddings import get_embeddings

load_dotenv("config/dev/.env")

# Prompt système : le modèle répond uniquement à partir des événements retrouvés
SYSTEM_PROMPT = """Tu es un assistant spécialisé dans les événements culturels et publics sur Paris.

Tu dois conseiller les utilisateurs sur les événements à venir en te basant uniquement sur les données que tu retrouves 
dans ta base de connaissances. Donne des réponses et déscriptions détaillées des événements, incluant les dates, lieux, horaires et informations pratiques renseigné.

- Ne mentionne AUCUNE information absente des événements fournis.
- N'invente pas d'horaires, tarifs, adresses ou accès qui ne figurent pas dans la description.
- N'extrapolés pas le contenu supposé d'une exposition ou d'un événement.
- N'ajoute pas de liens ou URLs.
- Si une information n'est pas dans le contexte, dis simplement qu'elle n'est pas disponible.

Si tu ne trouves pas d'événements pertinents, réponds honnêtement que tu n'as pas d'information à ce sujet.

Si on te pose une question qui n'est pas liée aux événements culturels et publics sur Paris, réponds honnêtement que tu ne peux pas répondre à cette question et invite l'utilisateur à poser une question sur les événements culturels et publics sur Paris.

Événements pertinents :
{context}
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def _format_docs(docs: list[Document]) -> str:
    # Formate les documents récupérés en un bloc texte lisible pour le LLM
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        header = f"[{i}] {m.get('title', '')} — {m.get('location_name', '')} {m.get('city', '')} ({m.get('date_begin', '')})"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


def load_vectorstore(vectorstore_path: str = "vector_db") -> FAISS:
    # Charge l'index FAISS depuis le disque avec le même modèle d'embedding
    return FAISS.load_local(vectorstore_path, get_embeddings(), allow_dangerous_deserialization=True)


def _build_llm() -> ChatMistralAI:
    # Construit le client LLM Mistral utilisé pour générer les réponses
    return ChatMistralAI(
        api_key=os.getenv("MISTRAL_API_KEY", ""),
        model="mistral-small-latest",
        temperature=0.2,
        top_p=0.7,
        max_tokens=400,
    )


def build_chain(vectorstore: FAISS, k: int = 4):
    # Construit la chaîne RAG : retriever → prompt → LLM → texte
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | _build_llm()
        | StrOutputParser()
    )
    return chain


def ask(question: str, vectorstore_path: str = "vector_db", k: int = 4) -> str:
    # Point d'entrée principal : prend une question, retourne une réponse
    vs = load_vectorstore(vectorstore_path)
    chain = build_chain(vs, k=k)
    return chain.invoke(question)


def ask_with_context(question: str, vectorstore_path: str = "vector_db", k: int = 4) -> tuple[str, list[str]]:
    # Comme ask(), mais retourne aussi le texte brut des documents récupérés (pour l'évaluation RAGAs).
    # Le retriever n'est appelé qu'une fois (contre deux dans build_chain) : un seul appel
    # d'embedding + un seul appel de complétion, pour limiter les requêtes à l'API Mistral.
    vs = load_vectorstore(vectorstore_path)
    retriever = vs.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    prompt_value = PROMPT.invoke({"context": _format_docs(docs), "question": question})
    response = _build_llm().invoke(prompt_value)
    return response.content, [doc.page_content for doc in docs]

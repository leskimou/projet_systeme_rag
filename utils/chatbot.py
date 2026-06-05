import os
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv("config/dev/.env")

# Prompt système : le modèle répond uniquement à partir des événements retrouvés
SYSTEM_PROMPT = """Tu es un assistant spécialisé dans les événements culturels et publics sur Paris.

Tu dois conseiller les utilisateurs sur les événements à venir en te basant uniquement sur les données que tu retrouves 
dans ta base de connaissances.

Si tu ne trouves pas d'événements pertinents, réponds honnêtement que tu n'as pas d'information à ce sujet.

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
    embeddings = MistralAIEmbeddings(
        api_key=os.getenv("MISTRAL_API_KEY", ""),
        model="mistral-embed",
    )
    return FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)


def build_chain(vectorstore: FAISS, k: int = 4):
    # Construit la chaîne RAG : retriever → prompt → LLM → texte
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatMistralAI(
        api_key=os.getenv("MISTRAL_API_KEY", ""),
        model="mistral-large-latest",
        temperature=0.2,
    )
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def ask(question: str, vectorstore_path: str = "vector_db", k: int = 4) -> str:
    # Point d'entrée principal : prend une question, retourne une réponse
    vs = load_vectorstore(vectorstore_path)
    chain = build_chain(vs, k=k)
    return chain.invoke(question)

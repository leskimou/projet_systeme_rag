from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "codefuse-ai/F2LLM-v2-0.6B"

def get_embeddings() -> HuggingFaceEmbeddings:
    # F2LLM-v2 distingue requêtes et documents : encode_query() préfixe le texte
    # avec une instruction ("Instruct: ...\nQuery: "), encode_document() non.
    # On reproduit cette asymétrie via prompt_name="query" pour embed_query.
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"normalize_embeddings": True, "prompt_name": "query"},
    )

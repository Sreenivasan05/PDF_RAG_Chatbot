from .configs import EMBEDDING_MODEL, VECTOR_DB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(VECTOR_DB, embeddings, allow_dangerous_deserialization=True)


def retrive_docs(query, k=5):
    db = load_vector_store()
    results = db.similarity_search(query, k=k)
    return results

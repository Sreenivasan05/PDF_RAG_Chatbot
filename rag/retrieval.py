from .configs import EMBEDDING_MODEL, VECTOR_DB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS




def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(VECTOR_DB, embeddings, allow_dangerous_deserialization=True)

    return db.as_retriever(search_kwargs={"k": 5})
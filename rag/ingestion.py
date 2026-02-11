from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_classic.schema import Document

from .configs import EMBEDDING_MODEL, VECTOR_DB


def extract_pdf_text(pdfs):
    text = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def convert_to_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", ".", " ", ""], length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def creat_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    docs = [Document(page_content=chunk) for chunk in text_chunks]
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(VECTOR_DB)
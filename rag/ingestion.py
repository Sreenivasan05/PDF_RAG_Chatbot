from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_classic.schema import Document

from .configs import EMBEDDING_MODEL, VECTOR_DB


def extract_pdf_text(pdfs):
    docs = []

    text = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page_num, page in enumerate(reader.pages, start=1):

            text = page.extract_text()
            if not text:
                continue
            
            docs.append(
                Document(
                    page_content=text, 
                    metadata = {"source" : pdf.name,"page" : page_num})
            )
    return docs

def convert_to_text_chunks(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", ".", " ", ""], length_function=len)
    return text_splitter.split_documents(documents=document)

def creat_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.from_documents(text_chunks, embeddings)
    db.save_local(VECTOR_DB)
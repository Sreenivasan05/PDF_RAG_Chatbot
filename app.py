import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.prompts import PromptTemplate

from datetime import datetime

def get_pdf_text(pdfs):
    text = ""
    for pdf in pdf:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text, model_name):
    if model_name == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def convert_to_text_chunks(text, modelname):
    if modelname == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter()
    chunks = text_splitter.split_text(text)
    return chunks
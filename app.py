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



from langchain_huggingface import HuggingFaceEmbeddings


from datetime import datetime

def get_pdf_text(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def convert_to_text_chunks(text, modelname):
    if modelname == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, model_name):
    if model_name == "Google_AI":
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_conversational_chain(modelname, api_key=None):
    if modelname == "Google_AI":

        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        print(prompt_template)
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", teperature=0.3, google_api_key=api_key)
        prompt = PromptTemplate(template=prompt_template, input_types=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    


def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):
    if api_key is None:
        st.warning("Please provide API key before processing")
        return
    if pdf_docs is None:
        st.warning("Please upload PDF files")
    text_chunks = convert_to_text_chunks(get_pdf_text(pdf_docs), model_name)
    vector_store = get_vector_store(text_chunks, model_name)
    user_question_output = ""
    response_output = ""
    if model_name == "Google_AI":
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        print("\n" + "=" * 80)
        print("Retrieved Chunks:\n")
        for i, doc in enumerate(docs, start=1):
            print(f"Chunk {i}:")
            print(doc.page_content)
            print("-" * 80)
        # chain = get_conversational_chain("Google_AI", api_key=api_key)
        
        # response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        # user_question_output = user_question
        # response_output = response['output_text']
        # pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
        # conversation_history.append((user_question_output, response_output, model_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ", ".join(pdf_names)))

def main():
    st.set_page_config(page_title="Chat with multiple PDFs")
    st.header("Chat with PDFs")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    model_name = st.sidebar.radio("Select the model:", ("Google_AI", "me"))

    api_key = None

    if model_name == "Google_AI":
        api_key = st.sidebar.text_input("Enter your Google API key")

        # if not api_key:
        #     st.sidebar.warning("Please enter your Google API key to proceed")
        #     return
        
    with st.sidebar:
        st.title("Menu:")

        col1, col2 = st.columns(2)

        reset_button = col2.button("Reset")
        clear_button = col1.button("Rerun")

        if reset_button:
            st.session_state.conversation_history = []
            st.session_state.user_question = None
        
            api_key = None
            pdf_docs = None

        else:
            if clear_button:
                if "user_question" in st.session_state:
                    st.warning("The previous query will be discarded")
                    st.session_state.user_question = ""
                    if len(st.session_state.conversation_history) > 0:
                        st.session_state.conversation_history.pop()
                else:
                    st.warning("The question in the input will be queried again")

        pdf_docs = st.file_uploader("Upload your PDF file here",accept_multiple_files=True)
        if st.button("Submit"):
            if pdf_docs:
                with st.spinner("Processing"):
                    st.success("Done")
            else:
                st.warning("Please upload PDF file before processing")

        
    user_question = st.text_input("Ask a Question from the PDF files")

    if user_question:
        user_input(user_question,model_name, api_key, pdf_docs, st.session_state.conversation_history)
        st.session_state.user_question = ""


if __name__ == "__main__":
    main()


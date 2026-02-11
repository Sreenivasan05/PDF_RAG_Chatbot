from PyPDF2 import PdfReader
from datetime import datetime
import streamlit as st
import os, shutil, time
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.prompts import PromptTemplate
from langchain_classic.schema import Document
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_huggingface import HuggingFaceEmbeddings


from google.api_core.exceptions import ResourceExhausted


VECTOR_DB = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_SCORE_THRESHOLD = 0.5


def get_pdf_text(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def convert_to_text_chunks(text, modelname):
    if modelname == "Google_AI":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", ".", " ", ""], length_function=len)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ".", " ", ""], length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, model_name):
    if model_name == "Google_AI":
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    docs = [Document(page_content=chunk) for chunk in text_chunks]
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(VECTOR_DB)

def get_pdf_signature(pdf_docs):
    hasher = hashlib.md5()
    for pdf in pdf_docs:
        hasher.update(pdf.getvalue())
    return hasher.hexdigest()

def get_conversational_chain(modelname, api_key):
    if modelname == "Google_AI":

        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n

        Answer the question using clear Markdown formatting.

        Rules:
        - Use bullet points or numbered lists where applicable
        - Add a blank line before lists
        - Use code blocks for code

        Context:
        \n {context}?\n

        Question: 
        \n{question}\n

        Answer:
        """

        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=api_key, max_output_tokens=512)
        prompt = ChatPromptTemplate.from_template(prompt_template)

        document_chain = create_stuff_documents_chain(model, prompt)

        return document_chain

def safe_chain_invoke(chain, inputs, retries=3):
    for i in range(retries):
        try:
            return chain.invoke(inputs)
        except ResourceExhausted:
            time.sleep(2 ** i)
    raise Exception("Rate limit Exceeded")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_faiss():
    embeddings = load_embeddings()
    return FAISS.load_local(VECTOR_DB, embeddings, allow_dangerous_deserialization=True)

def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):
    if not st.session_state.get("vector_store_ready", False) :
        st.warning("Please upload and process PDFs first")
        return

    if api_key is None:
        st.warning("Please provide API key before processing")
        return
    if pdf_docs is None:
        st.warning("Please upload PDF files")

    if model_name == "Google_AI":

        new_db = load_faiss()


        docs_and_scores = new_db.similarity_search_with_score(user_question, k=5)

        docs = [doc for doc, _ in docs_and_scores]

        if not docs:
            st.warning("No relevant context found in PDFs.")
            return
        
        chain = get_conversational_chain("Google_AI", api_key=api_key)

        inputs = {
            "context": docs,
            "question": user_question
        }

        response = safe_chain_invoke(chain, inputs, retries=3)
        
        if isinstance(response, dict):
            response_output = response.get("answer", "")
        else:
            response_output = response

        user_question_output = user_question


        pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
        conversation_history.append((user_question_output, response_output, model_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ", ".join(pdf_names)))

        with st.chat_message("user"):
            st.markdown(user_question_output)

        with st.chat_message("assistant"):
            st.markdown(response_output)

def delete_vector_db():
    if os.path.exists(VECTOR_DB):
        shutil.rmtree(VECTOR_DB)

def main():
    st.set_page_config(page_title="Chat with PDFs", layout="wide")
    st.header("📄 Chat with PDFs")

    # --- Initializing session state ---

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    if "vector_store_ready" not in st.session_state:
        st.session_state.vector_store_ready = False

    if "pdf_signature" not in st.session_state:
        st.session_state.pdf_signature = None

    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = None

    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    # --- Side bars ---

    st.sidebar.title("Menu")

    model_name = st.sidebar.radio("Select the model:", ("Google_AI"))

    api_key = None
    if model_name == "Google_AI":
        api_key = st.sidebar.text_input("Enter your Google API key", type="password")
        if not api_key:
            st.sidebar.warning("Please enter your Google API key")
            st.stop()

    col1, col2 = st.sidebar.columns(2)
    rerun_button = col1.button("Rerun")
    reset_button = col2.button("Reset")

    # --- Reset button ---

    if reset_button:
        st.session_state.clear()
        delete_vector_db()
        st.rerun()

    # -- Rerun button ---
    if rerun_button:
        if st.session_state.conversation_history:
            st.session_state.conversation_history.pop()
            st.warning("Previous question discarded")
        else:
            st.info("Nothing to rerun")

    # --- Submit and Embed PDFs ---
    uploaded_pdfs = st.file_uploader(
        "Upload your PDF files",
        accept_multiple_files=True
    )

    if uploaded_pdfs:
        st.session_state.pdf_docs = uploaded_pdfs

    if st.button("Submit"):
        if not st.session_state.pdf_docs:
            st.warning("Please upload PDF files first")
        else:
            current_signature = get_pdf_signature(st.session_state.pdf_docs)

            if ( not st.session_state.vector_store_ready or st.session_state.pdf_signature != current_signature):
                with st.spinner("Embedding PDFs..."):
                    text = get_pdf_text(st.session_state.pdf_docs)
                    text_chunks = convert_to_text_chunks(text, model_name)
                    get_vector_store(text_chunks, model_name)

                    st.session_state.pdf_signature = current_signature
                    st.session_state.vector_store_ready = True

                st.success("PDFs embedded successfully!")
            else:
                st.info("Using existing embeddings")

    # -- chat history display ---
    for user_msg, bot_msg, model, ts, pdfs in st.session_state.conversation_history:
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            st.markdown(bot_msg)

    # -- User question input ---
    user_question = st.text_input("Ask a question from the PDFs")

    if user_question:
        user_input(
            user_question=user_question,
            model_name=model_name,
            api_key=api_key,
            pdf_docs=st.session_state.pdf_docs,
            conversation_history=st.session_state.conversation_history,
        )
        st.session_state.user_question = ""
        
if __name__ == "__main__":
    main()


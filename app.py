import streamlit as st
from rag.utils import get_pdf_signature
from rag.ingestion import extract_pdf_text, convert_to_text_chunks, creat_vector_store
from rag.retrieval import get_retriever
from rag.llm_chain import build_conversational_chain
import time
from google.api_core.exceptions import ResourceExhausted

def initialization():
    if "pdf_sig" not in st.session_state:
        st.session_state.pdf_sig = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "memory" not in st.session_state:
        st.session_state.memory = None

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

def ingestion(uploaded, sig):

    if sig != st.session_state.pdf_sig:
        with st.spinner("Embedding PDFs..."):
            text = extract_pdf_text(uploaded)
            chunks = convert_to_text_chunks(text)
            creat_vector_store(chunks)

        st.session_state.pdf_sig = sig
        st.success("Vector DB created")

        st.session_state.rag_chain = None
        st.session_state.memory = None

    else:
        st.info("Using existing embeddings")

def build_chain(api_key):
    if (
        st.session_state.rag_chain is None
        and api_key
        and st.session_state.pdf_sig
    ):
        retriever = get_retriever()
        chain, memory = build_conversational_chain(
            api_key=api_key,
            retriever=retriever
        )

        st.session_state.rag_chain = chain
        st.session_state.memory = memory

def safe_invoke(chain, inputs, retries=3, base_delay=2):
    for i in range(retries):
        try:
            return chain.invoke(inputs)   # ← return FULL response
        except ResourceExhausted:
            if i == retries - 1:
                raise RuntimeError("Rate limit exceeded after retries")

            time.sleep(base_delay ** i)

def main():

    st.set_page_config(page_title="Chat with PDFs", layout="wide")

    st.header("📄 Chat with PDFs")

    initialization()


    api_key = st.sidebar.text_input("Gemini API Key", type="password")

    if not api_key:
        st.sidebar.warning("Please enter your Google API key")
        st.stop()


    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.memory = None
        st.rerun()


    uploaded = st.file_uploader("Upload PDFs", accept_multiple_files=True)

    if st.button("Submit") and uploaded:
        sig = get_pdf_signature(uploaded)

        ingestion(uploaded, sig)



    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    build_chain(api_key)



    question = st.chat_input("Ask a question from the PDF...")

    if question and st.session_state.rag_chain:


        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)


        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):


                result = safe_invoke(
                    st.session_state.rag_chain,
                    {
                        "input": question,
                        "chat_history": st.session_state.memory.chat_memory.messages
                    }
                )

                answer = result["answer"]
                st.markdown(answer)


        st.session_state.messages.append({"role": "assistant", "content": answer})


        st.session_state.memory.chat_memory.add_user_message(question)
        st.session_state.memory.chat_memory.add_ai_message(answer)

    elif question and not st.session_state.rag_chain:
        st.warning("Please upload and process a PDF first.")


if __name__ == "__main__":
    main()
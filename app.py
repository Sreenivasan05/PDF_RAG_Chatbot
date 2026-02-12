import streamlit as st
from rag.utils import get_pdf_signature
from rag.ingestion import extract_pdf_text, convert_to_text_chunks, creat_vector_store
from rag.retrieval import retrive_docs
from rag.llm_chain import build_chain, safe_invoke


def main():

    if "pdf_sig" not in st.session_state:
        st.session_state.pdf_sig = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    st.set_page_config(page_title="Chat with PDFs", layout="wide")
    st.header("📄 Chat with PDFs")

    api_key = st.sidebar.text_input("Gemini API Key", type="password")

    if not api_key:
            st.sidebar.warning("Please enter your Google API key")
            st.stop()

    if st.sidebar.button("Clear Chat"):
        st.session_state.chat_history = []


    uploaded = st.file_uploader("Upload PDFs", accept_multiple_files=True)

    # --- Ingestion ---

    if st.button("Submit") and uploaded:
        sig = get_pdf_signature(uploaded)
    
        if sig != st.session_state.pdf_sig:
            with st.spinner("Embedding pdfs..."):
                text = extract_pdf_text(uploaded)
                chunks = convert_to_text_chunks(text)
                creat_vector_store(chunks)

            st.session_state.pdf_sig = sig
            st.success("Vector DB created")

        else:
            st.info("Using existing embeddings")

    # --- render chat history --- 
    for q,a , sources in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("AI"):
            st.markdown(a)
        
        if sources:
            st.markdown("**Sources:**")
            for s in sources:
                st.markdown(f"- {s}")

    # -- chat ---
    query = st.chat_input("Ask a question")

    if query and api_key:
        docs = retrive_docs(query=query)

        if not docs:
            st.warning("No relevant context found in PDF for this question")

        chain = build_chain(api_key)

        with st.chat_message("user"):
            st.markdown(query)

        assistant_placeholder =  st.chat_message("assistant").empty()
        assistant_placeholder.markdown("Thinking....")

        answer = safe_invoke(chain, {
            "context": docs,
            "question": query
        })

        sources = {
        f"{doc.metadata.get('source')} (page {doc.metadata.get('page')})"
        for doc in docs
        }

        st.session_state.chat_history.append(
            (query, answer, sources)
        )

        assistant_placeholder.markdown(answer)
        st.markdown("**Sources:**")
        for s in sorted(sources):
            st.markdown(f"-{s}")

if __name__ == "__main__":
    main()


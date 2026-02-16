from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory
from .configs import LLM





def build_conversational_chain(api_key, retriever):
    llm  = ChatGoogleGenerativeAI(
        model=LLM,
        temperature = 0.3,
        google_api_key=api_key,
        max_output_tokens = 512
    )

    # --- query rewriting

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Given chat history and a follow-up question, "
             "rewrite the question to be standalone."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=contextualize_q_prompt
    )

    # --- qa prompt ---

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
            """Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
            provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n

            {context}

            Answer the question using clear Markdown formatting.

            Rules:
            - Use bullet points or numbered lists where applicable
            - Add a blank line before lists
            - Use code blocks for code"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    combine_docs_chain = create_stuff_documents_chain(
        llm,
        qa_prompt
    )

    rag_chain = create_retrieval_chain(
        history_aware_retriever, combine_docs_chain
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return rag_chain, memory



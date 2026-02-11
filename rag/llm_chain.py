from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from google.api_core.exceptions import ResourceExhausted
from .configs import LLM

import time


def build_chain(api_key):
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

    model = ChatGoogleGenerativeAI(model=LLM, temperature=0.3, google_api_key=api_key, max_output_tokens=512)
    prompt = ChatPromptTemplate.from_template(prompt_template)

    document_chain = create_stuff_documents_chain(model, prompt)

    return document_chain


def safe_invoke(chain, inputs, retries=3):
    for i in range(retries):
        try:
            res = chain.invoke(inputs)
            return res if isinstance(res, str) else res.get("answer", "")
        except ResourceExhausted:
            time.sleep(2 ** i)

    raise RuntimeError("Rate limit exceeded")


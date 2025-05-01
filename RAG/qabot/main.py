import os

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_milvus import Milvus
from langchain import hub

from RAG.config import CHATMODEL, MILVUS_RAG_COLLECTION

load_dotenv()


def answer_question(question: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    llm = ChatOpenAI(model=CHATMODEL)

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": os.environ.get("MILVUS_URI")},
        collection_name=MILVUS_RAG_COLLECTION
    )

    # rag_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    rag_prompt = hub.pull("rlm/rag-prompt")

    combine_docs_chain = create_stuff_documents_chain(llm, rag_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=combine_docs_chain,
    )

    response = retrieval_chain.invoke(input={"question": question, "input": ""})

    print(response)


# answer_question("What the style guide says about values and numbers")


def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)



def answer_question_manual(question: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    llm = ChatOpenAI(model=CHATMODEL)

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": os.environ.get("MILVUS_URI")},
        collection_name=MILVUS_RAG_COLLECTION
    )

    rag_prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Always give a query related quote from computer scientists/well-known programmers, after the answer.
    Question: {question} 
    Context: {context} 
    Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template=rag_prompt)

    rag_chain = (
        {
            "context": vector_store.as_retriever() | format_docs,
            "question": RunnablePassthrough()
        }
        | custom_rag_prompt
        | llm
    )

    response = rag_chain.invoke(question)

    print(response.content)


answer_question_manual("What the style guide says about values and numbers")
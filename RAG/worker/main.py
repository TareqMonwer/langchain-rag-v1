import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus

from RAG.config import MILVUS_RAG_COLLECTION


load_dotenv()

data_path = Path("D:/Personal/Projects/python/genai/langchain-udemy/react-langchain") / "RAG" / "data" / "text"

for file_name in os.listdir(data_path):
    file_path = data_path / file_name

    loader = TextLoader(file_path)
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": os.environ.get("MILVUS_URI")},
    )

    vector_store.from_documents(texts, embeddings, collection_name=MILVUS_RAG_COLLECTION)

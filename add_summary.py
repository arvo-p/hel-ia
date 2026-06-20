import os
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_location = "./data/chroma_langchain_db"

vector_store = Chroma(
    collection_name="helinfos",
    persist_directory=db_location,
    embedding_function=embeddings
)

summary_path = "./data/src/plaintext_hel/global_summary.txt"
with open(summary_path, "r", encoding="utf-8") as f:
    content = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

doc = Document(page_content=content, metadata={"source": "global_summary.txt"})
split_docs = text_splitter.split_documents([doc])

vector_store.add_documents(split_docs)

print(f"Global summary split into {len(split_docs)} chunks and added to vector store successfully.")

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

class HELVectorManager:
    def __init__(self, db_location="./data/chroma_langchain_db", source_dir="./data/src/plaintext_hel"):
        self.db_location = db_location
        self.source_dir = source_dir
        
                               
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        
                                   
        self.vector_store = self._init_vector_store()
        self.retriever = self._init_retriever()

    def _init_vector_store(self):
        is_new_db = not os.path.exists(self.db_location)
        print(f"Initializing vector store. New DB: {is_new_db}")
        
        vector_store = Chroma(
            collection_name="helinfos",
            persist_directory=self.db_location,
            embedding_function=self.embeddings
        )

        if is_new_db:
            print("Loading and indexing documents...")
            self._load_and_index_documents(vector_store)
            
        return vector_store

    def _load_and_index_documents(self, vector_store):
        documents = []
        for root, _, files in os.walk(self.source_dir):
            for filename in files:
                if filename.startswith('.') or filename.endswith('.swp') or filename.endswith('.swo'):
                    continue
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        documents.append(Document(page_content=f.read()))
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
        
        if documents:
            split_docs = self.text_splitter.split_documents(documents)
            print(f"Total chunks to index: {len(split_docs)}")
            
            vector_store.add_documents(documents=split_docs)
            print(f"Indexing complete: {len(split_docs)} chunks added to the vector store.")

    def _init_retriever(self):
        """
        Configures a contextual compression retriever using FlashRank for reranking.
        """
        base_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 8}
        )
        
        compressor = FlashrankRerank()
        
        return ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=base_retriever
        )

_manager = HELVectorManager()
vector_store = _manager.vector_store
retriever = _manager.retriever

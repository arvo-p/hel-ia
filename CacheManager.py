import os
import random
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from CacheDatabase import CacheDatabase

class CacheManager:
    def __init__(self, db_location="./data/chroma_langchain_db"):
        self.sql = CacheDatabase()  
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        self.vector_store = Chroma(
            collection_name="cache2",
            embedding_function=self.embeddings
        )

        documents = self.sql.fetch_sql_as_documents();
        self.vector_store.add_documents(documents)
        
        self.retriever = ContextualCompressionRetriever(
            base_compressor=FlashrankRerank(), 
            base_retriever=self.vector_store.as_retriever(search_kwargs={"k":10})
        )                                  

    def insert(self, question, answer):
        rid = self.sql.insert(question, answer)
        combined_text = f"{question}"
        new_doc = Document(
            page_content=combined_text, 
            metadata={
                "answer": answer, 
                "sql_id": rid,
                "rating": 0,
                "views": 0
            }
        )
        self.vector_store.add_documents([new_doc])
        return rid
    
    def increment_views(self, sql_id):
        self.sql.increment_views(sql_id)

    def increment_rating(self, sql_id):
        self.sql.increment_rating(sql_id)

    def search(self, query):
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=5)
        if not results:
            return None

        valid_docs = []
        weights = []
        for doc, score in results:
            if score >= 0.999:
                print("Possible answer:" + doc.metadata.get("answer"))
                rating = doc.metadata.get("rating", 0)
                views = doc.metadata.get("views", 0)
                weight = (rating + 1) / (views + 1)
                
                valid_docs.append(doc)
                weights.append(weight)

        if not valid_docs:
            return None                                 
            
        selected_doc = random.choices(valid_docs, weights=weights, k=1)[0]
        return selected_doc.metadata.get("answer"), selected_doc.metadata.get("sql_id")

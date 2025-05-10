import getpass
import os
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.documents import Document

class VectorStore():

    def __init__(self):
        self.embeddings = VertexAIEmbeddings(model="text-embedding-004")
        self.vector_store = InMemoryVectorStore(self.embeddings)


    def addToStore(self, text, pdfName):
        vector = self.embeddings.embed_query(text)
        print("vector ========== ", vector)
        documents = [Document(page_content=text, metadata={"source": pdfName})]
        self.vector_store.add_documents(documents=documents)

    def addAllToStore(self, texts, pdfsNames):
        documents = []
        for text, name in zip(texts, pdfsNames):
            documents.append(Document(page_content=text, metadata={"source": name}))
        self.vector_store.add_documents(documents=documents)
        
    def search(self, query, k=10):
        results = self.vector_store.similarity_search(query=query, k=k)
        return results

    def searchWithScore(self, query, k=10):
        return self.vector_store.similarity_search_with_score(query=query, k=k)

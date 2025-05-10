import redis
import os
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.documents import Document

class VectorStoreRedis():
    def __init__(self):
        self.embeddings = VertexAIEmbeddings(model="text-embedding-004")
        self.redis_url = os.getenv("REDIS_URL")
        
        self.config = RedisConfig(
            index_name="cv",
            redis_url=self.redis_url,
            metadata_schema=[
                {"name": "source", "type": "text"},
                {"name": "format", "type": "text"}
            ]
        )
        self.vector_store = RedisVectorStore(embeddings=self.embeddings, config=self.config)

    def addToStore(self, text, pdfName, form):
        """
            text is part that is vector calculated on, pdfName is the cv file name to identify source, form is source of text: [cv, summary, extract]
        """
        documents = [Document(page_content=text, metadata={"source": pdfName, "format": form})]
        self.vector_store.add_documents(documents=documents)

    def addAllToStore(self, texts, pdfsNames, form):
        """
            text is part that is vector calculated on, pdfName is the cv file name to identify source, form is source of text: [cv, summary, extract]
        """
        documents = []
        for text, name in zip(texts, pdfsNames):
            documents.append(Document(page_content=text, metadata={"source": name, "format": form}))
        self.vector_store.add_documents(documents=documents)
        
    def search(self, query, k=10):
        results = self.vector_store.similarity_search(query=query, k=k)
        return results

    def searchWithScore(self, query, k=10):
        return self.vector_store.similarity_search_with_score(query=query, k=k)


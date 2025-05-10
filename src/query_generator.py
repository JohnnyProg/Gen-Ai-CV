from langchain_core.messages import HumanMessage, SystemMessage
import re
import json

class QueryGenerator:
    """
        This class retrives question from the user, generate querry for database, then uses it as a context for the second request 
    """
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = """
You are an AI assistant designed to formulate precise search queries for a vector database. Your task is to understand the user's question about a collection of resumes and translate it into an effective query that will retrieve the most relevant resume vectors. The goal is to retrieve resumes containing information that directly answers the user's question. Consider skills, experience, education, and other relevant fields when crafting the query. Focus on keywords and semantic meaning to maximize retrieval accuracy. Only return the search query string.

VERY IMPORTANT: ANSWER ONLY WITH QUERY, NO ADDITIONAL DESCRIPTIONS OR MESSAGES

JUST QUERY
"""

    async def generateQuery(self, question: str):
        """Extracts entities from resume text."""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=question),
        ]
        response = await self.llm.invoke(messages)
        return response

class QueryAnalizer:
    """
        This class retrives question from the user, generate querry for database, then uses it as a context for the second request 
    """
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = """
You are an AI assistant specialized in analyzing resume data retrieved from a vector database. You will receive a user's original question and the content of several resumes retrieved in response to that question. Your task is to synthesize the information from these resumes to directly answer the user's question. Identify relevant skills, experience levels, job titles, and other details within the resumes to formulate your answer. If the resumes do not contain enough information to definitively answer the question, indicate that the answer cannot be determined with the available data. Present your answer clearly and concisely, focusing on accuracy and relevance. 
Important: each cv have distinct number in metadata source ./directory/*****.extension. Each cv is stored in vector storage in 3 different forms: CV (metadata format: cv), summary (metadata format: summary) and json (metadata format: json). it is possible to one cv to show more than once in retrived data. Remember about it.
Only provide the direct answer to the question.
"""

    async def extract(self, question: str, context: str):
        """Extracts entities from resume text."""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content="question: " + question + " ; context: " + context),
        ]
        response = await self.llm.invoke(messages)
        return response

class VectorStorageQuestions():
    def __init__(self, llm, storage):
        self.generator = QueryGenerator(llm)
        self.analizer = QueryAnalizer(llm)
        self.storage = storage

    async def answer(self, question: str):
        query = await self.generator.generateQuery(question)
        query = query.content
        print("query = ", query)
        results = self.storage.searchWithScore(query, k=20)
        
        print("results len =", len(results))
        print(results)
        
        context = []
        for result in results:
            text = "distinct result: " + json.dumps(result[0].metadata)
            text = text + "database score: 0" + str(results[1])
            text = text + ".   IMPORTANT: text: " + result[0].page_content + "."
            context.append(text)
        print("context = ------------------------------------------------------------------------------ ", " ".join(context))
        return await self.analizer.extract(question, " ".join(context))
        
        
        
import asyncio
MAX_CONCURRENT_REQUESTS = 10  # Tune this based on API limits
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

class CompatibleVertexAI:
    def __init__(self, model_name="gemini-pro"):
        from langchain_google_vertexai import VertexAI
        self.model = VertexAI(model_name=model_name)

    async def invoke(self, prompt):
        async with semaphore:
            
            response = await self.model.ainvoke(prompt)
            return type("Response", (object,), {"content": response})()
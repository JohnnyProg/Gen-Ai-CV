---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.12.3
  nbformat: 4
  nbformat_minor: 5
---

::: {#c820e19f-5e3a-4283-b0a6-61ba708a5158 .cell .markdown}
**Project Overview:**

This project processes a dataset of anonymized resumes in PDF format to
extract information, generate summaries, score candidates against job
descriptions, and create a searchable knowledge base using vector
embeddings. It uses LLMs (Gemini and Vertex AI) for translation, entity
extraction, summarization, and scoring. The project is structured as a
Python project with modular classes.

**Requirements:**

-   Python 3.6+
-   Libraries: `langchain`, `langchain-redis`, `langchain-huggingface`,
    `sentence-transformers`, `scikit-learn`, `numpy`,
    `langchain-google-genai`, `langchain-core`,
    `langchain-google-vertexai`
-   API Keys: Google Gemini API key, Google Cloud Vertex AI credentials,
    Langsmith API key (optional, for tracing), Redis URL.
-   PDF Resume Dataset (resumes_dataset.zip)
:::

::: {#68052c96-e84b-4ea8-a9c0-78437154c676 .cell .markdown}
**Project Structure:**

    .
    ├── README.md  (or README.ipynb)
    ├── keys/
    │   ├── google_cloud_key.json    # Google console api key.
    ├── src/
    │   ├── entity_extractor.py      # Class for extracting entities from resumes.
    │   ├── pdf_reader.py            # Class for reading PDF files.
    │   ├── rank_candidate_llm.py    # Class for ranking candidates using LLM.
    │   ├── rank_candidate.py        # Class for ranking candidates based on skills.
    │   ├── summary_generator.py     # Class for generating resume summaries.
    │   ├── translator.py            # Class for translating resumes to English.
    │   ├── vector_store_inmemory.py # Class for in-memory vector storage.
    │   ├── vector_store_redis.py    # Class for Redis-based vector storage.
    │   ├── resumes_dataset/         # Directory containing the PDF resume files.
    ├── config.py                    # Stores API keys, paths, and configuration details
    └── FinalRating.ipynb            # main file that reads other modules
:::

::: {#903984e5-96ea-4015-b7c8-1a26589b74ec .cell .markdown}
1.  **Clone the repository:**
2.  **Install dependencies:**\
3.  **Configure API Keys:**
4.  **Download the resume dataset**
:::

::: {#c627c31b-3e95-43da-900c-464b5fe7ea1a .cell .code}
``` python
%pip install numpy==2.2.4
%pip install PyPDF2 langchain langchain-google-genai langchain-google-vertexai langchain-openai
%pip install -q -U google-genai
%pip install -qU langchain-redis langchain-huggingface sentence-transformers scikit-learn
```
:::

::: {#4f55dd09-9eaf-4691-9b7a-6423b11acfeb .cell .code}
``` python
from src.entity_extractor import *
from src.pdf_reader import *
from src.rank_candidate_llm import *
from src.rank_candidate import *
from src.summary_generator import *
from src.translator import *
import json
```
:::

::: {#76c6a554-5883-45b9-86c2-e973714db6ca .cell .markdown}
1.  **Loading Gemini model:**
:::

::: {#974914fa-5fe5-4a18-8d7d-20cdc40ced9d .cell .code}
``` python
import os
import config
from langchain_google_genai import ChatGoogleGenerativeAI
from src.compatible_vertex_ai import *
import asyncio

MAX_CONCURRENT_REQUESTS = 10  # Tune this based on API limits
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

os.environ["GOOGLE_API_KEY"] = config.API_TOKEN
os.environ["LANGSMITH_API_KEY"] = config.LANGSMITH_TOKEN
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_ENDPOINT"] = config.LANGSMITH_ENDPOINT
os.environ["LANGSMITH_PROJECT"] = config.LANGSMITH_PROJECT
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.GOOGLE_CRED_PATH
os.environ["REDIS_URL"] = config.REDIS_URL

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

llm = CompatibleVertexAI(modejson.dumps(dictl_name="gemini-2.0-flash")
```
:::

::: {#860a9811-456f-4363-a747-d1862d774aec .cell .markdown}
1.  **Loading CVS:**
:::

::: {#e2ba43aa-374f-4a8d-afd9-10cc83baa79a .cell .code}
``` python
# LOAD ALL CVS
pdfReader = PDFReader()
pdfs = pdfReader.readfiles("./resumes_dataset/resumes_dataset/", "*.pdf")
```
:::

::: {#c4561f60-47a4-4c24-b497-4c32c8cd40de .cell .markdown}
1.  **Helper methods to score single CV:**
:::

::: {#06368475-b8e6-4a81-adf2-8bb3ef5213bf .cell .code}
``` python
async def translate_CV(pdf):
    translator = Translator(llm)
    print("translating")
    res = await translator.translate(pdf)
    print("translated")
    if res.content.strip() == "empty":
        return pdf
    else:
        return res.content

async def extract_skills(pdf):
    extractor = EntityExtractor(llm)

    return await extractor.extract(pdf)
    
async def generate_summary(extract):
    extract_str = json.dumps(extract)
    summaryGenerator = SummaryGenerator(llm)
    return await summaryGenerator.summary(extract_str)

async def rank_user(extract, required_technologies, required_description):
    rankCandidate = RankCandidate(llm)
    maxScore = len(required_technologies)
    
    score, candidate_skills_list = rank_candidate_from_json(extract, required_technologies)
    normalized = score / (maxScore * 1.0)
    extract_str = json.dumps(extract)
    scoreLLM = await rankCandidate.rank(extract_str, required_description + "technologies: " + ' '.join(required_technologies))
    normalizedLLM = int(scoreLLM) / 10.0
    
    return 0.5 * normalized + 0.5 * normalizedLLM, 0.5 * normalized, 0.5 * normalizedLLM, candidate_skills_list

async def generateMetadata(pdf):
    translated = await translate_CV(pdf)
    # print("translated ==== ", translated)
    extract = await extract_skills(translated)
    # print("extract ===== ", extract)
    summary = await generate_summary(extract)
    # print("summary =======", summary)
    return (extract, summary)

async def score_CV(pdf, required_technologies, required_description):
    (extract, summary) = await generateMetadata(pdf)
    finalScore, alg_part, llm_part, skills = await rank_user(extract, required_technologies, required_description)
    print("final score =======", finalScore)
    return extract, summary, finalScore, alg_part, llm_part, skills
```
:::

::: {#34e8bdac-ab84-4d3a-a1e8-0c518ea0a349 .cell .code scrolled="true"}
``` python
extract, summary, score, alg_part, llm_part, skills  = await score_CV(pdfReader.openFile(pdfs[0]), ["Java", "Writing", "Excel"], "needed java developer with 3 years of experience in backend development in spring")
```
:::

::: {#cae4ed0b-16d5-4e0b-bb78-25df415fa1bc .cell .markdown}
1.  **Helper method to limit requests. Gemini 2 Flash in free tier is
    limited to 15 requests per minute:**
:::

::: {#a49f2771-6691-4d09-9a6f-8aeac1aab378 .cell .code}
``` python
import time

def thressholdRequests(pdfs, requirement_technologies, requirement_description):
    """
        this method is required when i use gemini free tier that have limits: 15 requrest per minute
    """
    results = {}
    
    for i in range(0, len(pdfs), 4):
        print("i ->>>>>>>>>>>>>>>>>>>>>>>>>>", i)
        start_time = time.time();
        group = pdfs[i:i+4]  # Slice the list to get a group of 3 (or less at the end)
        for pdf in group:
            text = pdfReader.openFile(pdfs[1])
            result = score_CV(text, requirement_technologies, requirement_description)
            results[pdf] = result
        end_time = time.time()
        diff = end_time - start_time
        if(diff < 70):
            print("sleeping for ", 70 - diff)
            time.sleep(70 - diff)
    return results
            
```
:::

::: {#cba2128e-12e5-4e7e-9dee-a8873c3a3733 .cell .markdown}
Asynchronous methods to process over 2000 CVs. Each cv requires 3 api
calls to generate summary and json extract. To improve performance I
implemented asynch processess.
:::

::: {#4c65614d-f5c7-4d30-932b-30c4e36d7437 .cell .code scrolled="true"}
``` python
async def saveToFile(results, pdf):
    try:
        filename = os.path.splitext(os.path.basename(pdf))[0]
        print("filename", filename)
        
        output_dir = "./resumes_dataset"  #
        
        json_path = os.path.join(output_dir, "jsons", f"{filename}.json")
        print("jsonPath", json_path)
        summary_path = os.path.join(output_dir, "summaries", f"{filename}.txt")

        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)

         # Save JSON data
        with open(json_path, "w") as f:
            json.dump(results[0], f, indent=4)  # Use indent for readability

        # Save summary
        with open(summary_path, "w") as f:
            f.write(results[1])

    except Exception as e:
        print(f"Error saving metadata for {pdf}: {e}")


async def processAsyncPDF(results, pdf):
    async with semaphore:
        print("processing:", pdf)
        result = await generateMetadata(pdfReader.openFile(pdf))
        results[pdf] = result
        await saveToFile(result, pdf)
        
        print("finished:", pdf)

async def main():
    results = {}
    pdfs_part = pdfs[1000:]
    await asyncio.gather(*[processAsyncPDF(results, pdf) for pdf in pdfs_part])
    return results
results = await main()
```
:::

::: {#f4d52013-a23a-45a0-8459-f30f9462b908 .cell .markdown}
1.  **This part load text of all data sources**

This section demonstrates how to create a vector store in Redis, embed
resume text, and add the embeddings to the store. This enables semantic
search and retrieval of resumes based on their content. It leverages the
VectorStoreRedis class, which encapsulates the interaction with the
Redis database. The text embeddings are generated using Vertex AI
embedding model. After creating vector database resumes are added to
vectorstore.

Additionally from previous step I generated summaries and json formats
of all CVs. They are stored in vector storage as separate entries with
specific metadata. This improves research of specific skills or
experience.
:::

::: {#a848c016-dbcc-42db-84c9-7ec22a18f3f7 .cell .code}
``` python
summaries = pdfReader.readfiles("./resumes_dataset/summaries/", "*.txt")
jsons = pdfReader.readfiles("./resumes_dataset/jsons/", "*.json")
```
:::

::: {#cbec6b40-f67f-46e2-a5c1-2c39856b58cd .cell .code}
``` python
from src.vector_store_inmemory import *
from src.vector_store_redis import *

store = VectorStoreRedis()
def addFilesToVectorStorage(store, names, source):
    print("starting to process")
    texts = []
    i = 0
    for name in names:
        texts.append(pdfReader.openJSONFile(name))
        i = i + 1
        if i % 100 == 0:
            print(i)
    print("start vectorizing")
    store.addAllToStore(texts, names, source)
print("finished")

```
:::

::: {#f1db3324-49a8-4f2c-9ce6-0e187ccfbe69 .cell .code}
``` python
addFilesToVectorStorage(store, pdfs, "cv")
addFilesToVectorStorage(store, summaries, "summary")
addFilesToVectorStorage(store, jsons, "json")
```
:::

::: {#d7169818-5ec5-4b4c-a9d1-2b2aea95af57 .cell .markdown}
1.  **Searching vector db with semantic queries**
:::

::: {#e5df0689-c204-4072-bb0a-d7a3e3c286ec .cell .code}
``` python
results = store.searchWithScore("who is the best java developer", k=10)
```
:::

::: {#12818a7d-9313-42f3-b3b9-ef90d20cb733 .cell .code scrolled="true"}
``` python
from src.query_generator import *

analyzer = VectorStorageQuestions(llm, store)
answer = await analyzer.answer("We have offer for middle java developer (at least 2 years of experience) to backend application, how many candidates do we have in database")
```
:::

::: {#1fe8c5ff-13c9-4698-a74c-46451725d828 .cell .code}
``` python
answer.content
```
:::

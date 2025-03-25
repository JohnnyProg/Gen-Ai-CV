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

::: {#c627c31b-3e95-43da-900c-464b5fe7ea1a .cell .code execution_count="18"}
``` python
%pip install numpy==2.2.4
%pip install PyPDF2 langchain langchain-google-genai langchain-google-vertexai langchain-openai
%pip install -q -U google-genai
%pip install -qU langchain-redis langchain-huggingface sentence-transformers scikit-learn
```

::: {.output .stream .stdout}
    Collecting numpy==2.2.4
      Using cached numpy-2.2.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (62 kB)
    Using cached numpy-2.2.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.1 MB)
    Installing collected packages: numpy
      Attempting uninstall: numpy
        Found existing installation: numpy 1.26.4
        Uninstalling numpy-1.26.4:
          Successfully uninstalled numpy-1.26.4
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    langchain-redis 0.2.0 requires numpy<2,>=1, but you have numpy 2.2.4 which is incompatible.
    Successfully installed numpy-2.2.4
    Note: you may need to restart the kernel to use updated packages.
:::
:::

::: {#4f55dd09-9eaf-4691-9b7a-6423b11acfeb .cell .code execution_count="1"}
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

::: {#974914fa-5fe5-4a18-8d7d-20cdc40ced9d .cell .code execution_count="2"}
``` python
import os
import config
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = config.API_TOKEN
os.environ["LANGSMITH_API_KEY"] = config.LANGSMITH_TOKEN
os.environ["LANGSMITH_TRACING"] = "true"
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
```
:::

::: {#860a9811-456f-4363-a747-d1862d774aec .cell .markdown}
1.  **Loading CVS:**
:::

::: {#e2ba43aa-374f-4a8d-afd9-10cc83baa79a .cell .code execution_count="3"}
``` python
# LOAD ALL CVS
pdfReader = PDFReader()
pdfs = pdfReader.readfiles("./resumes_dataset/resumes_dataset/")
```
:::

::: {#c4561f60-47a4-4c24-b497-4c32c8cd40de .cell .markdown}
1.  **Helper methods to score single CV:**
:::

::: {#06368475-b8e6-4a81-adf2-8bb3ef5213bf .cell .code execution_count="4"}
``` python
def translate_CV(pdf):
    translator = Translator(llm)
    res = translator.translate(pdf)
    if res.content == "empty":
        return pdf
    else:
        return res.content

def extract_skills(pdf):
    extractor = EntityExtractor(llm)

    return extractor.extract(pdf)
    
def generate_summary(extract):
    extract_str = json.dumps(extract)
    summaryGenerator = SummaryGenerator(llm)
    return summaryGenerator.summary(extract_str)

def rank_user(extract, required_technologies, required_description):
    rankCandidate = RankCandidate(llm)
    maxScore = len(required_technologies)
    
    score, candidate_skills_list = rank_candidate_from_json(extract, required_technologies)
    normalized = score / (maxScore * 1.0)
    extract_str = json.dumps(extract)
    scoreLLM = rankCandidate.rank(extract_str, required_description + "technologies: " + ' '.join(required_technologies))
    normalizedLLM = int(scoreLLM) / 10.0
    
    return 0.5 * normalized + 0.5 * normalizedLLM, 0.5 * normalized, 0.5 * normalizedLLM, candidate_skills_list
    
def score_CV(pdf, required_technologies, required_description):
    translated = translate_CV(pdf)
    # print("translated ==== ", translated)
    extract = extract_skills(translated)
    # print("extract ===== ", extract)
    summary = generate_summary(extract)
    # print("summary =======", summary)
    finalScore, alg_part, llm_part, skills = rank_user(extract, required_technologies, required_description)
    # print("final score =======", finalScore)
    return extract, summary, finalScore, alg_part, llm_part, skills
```
:::

::: {#34e8bdac-ab84-4d3a-a1e8-0c518ea0a349 .cell .code scrolled="true"}
``` python
extract, summary, score, alg_part, llm_part, skills  = score_CV(pdfReader.openFile(pdfs[0]), ["Java", "Writing", "Excel"], "needed java developer with 3 years of experience in backend development in spring")
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

::: {#c3e63c47-460b-4077-9f05-a3ceeb593bf7 .cell .markdown}
1.  **Example use case:**
:::

::: {#434910bf-18e4-4e06-9320-36b0a8b07992 .cell .code scrolled="true"}
``` python
results = thressholdRequests(pdfs[0:10],["Spring Boot", "RabbitMQ", "AWS Lambda", "Terraform", "Docker", "Kubernetes", "PostgreSQL", "DynamoDB", "GitLab CI/CD", "ArgoCD", "TypeScript", "NestJS", "Redis", "Load Balancers", "VPC"]
, "Seeking a Senior Java Developer with 5+ years of experience in backend and cloud development. Must have expertise in Java, Spring Boot, AWS, TypeScript, and experience with microservices, Kubernetes, CI/CD, and message queues (RabbitMQ).")
```
:::

::: {#f4d52013-a23a-45a0-8459-f30f9462b908 .cell .markdown}
1.  **This part load text of all**

This section demonstrates how to create a vector store in Redis, embed
resume text, and add the embeddings to the store. This enables semantic
search and retrieval of resumes based on their content. It leverages the
VectorStoreRedis class, which encapsulates the interaction with the
Redis database. The text embeddings are generated using Vertex AI
embedding model. After creating vector database resumes are added to
vectorstore.
:::

::: {#5037c379-d05a-497b-aeab-6bc809c97063 .cell .code execution_count="6"}
``` python
from src.vector_store_inmemory import *
from src.vector_store_redis import *

store = VectorStoreRedis()

print("starting to process")

# names = pdfs[1000:]
# texts = []
# i = 0
# for name in names:
#     texts.append(pdfReader.openFile(name))
#     i = i + 1
#     if i % 100 == 0:
#         print(i)
# print("start vectorizing")
# store.addAllToStore(texts, names, "cv")
# print("finished")
```

::: {.output .stream .stdout}
    14:52:41 redisvl.index.index INFO   Index already exists, not overwriting.
    starting to process
:::
:::

::: {#d7169818-5ec5-4b4c-a9d1-2b2aea95af57 .cell .markdown}
1.  **Searching vector db with semantic queries**
:::

::: {#e5df0689-c204-4072-bb0a-d7a3e3c286ec .cell .code execution_count="7"}
``` python
results = store.searchWithScore("who is the best java developer", k=10)

    # print("cv file name = ", result.
```
:::

::: {#a4617eea-e998-4132-9a51-7c78455d49e8 .cell .code execution_count="11" scrolled="true"}
``` python
for result in results:
    print("score = ", result[1])
    print("metadata = ", result[0].metadata)
```

::: {.output .stream .stdout}
    score =  0.409007191658
    metadata =  {'source': './resumes_dataset/resumes_dataset/22351830.pdf', 'format': 'cv'}
    score =  0.426907658577
    metadata =  {'source': './resumes_dataset/resumes_dataset/23464505.pdf', 'format': 'cv'}
    score =  0.44193482399
    metadata =  {'source': './resumes_dataset/resumes_dataset/85101052.pdf', 'format': 'cv'}
    score =  0.449346184731
    metadata =  {'source': './resumes_dataset/resumes_dataset/11813872.pdf', 'format': 'cv'}
    score =  0.449945092201
    metadata =  {'source': './resumes_dataset/resumes_dataset/12763627.pdf', 'format': 'cv'}
    score =  0.450672805309
    metadata =  {'source': './resumes_dataset/resumes_dataset/28762662.pdf', 'format': 'cv'}
    score =  0.452338695526
    metadata =  {'source': './resumes_dataset/resumes_dataset/12144825.pdf', 'format': 'cv'}
    score =  0.460814118385
    metadata =  {'source': './resumes_dataset/resumes_dataset/60489316.pdf', 'format': 'cv'}
    score =  0.462621927261
    metadata =  {'source': './resumes_dataset/resumes_dataset/31169070.pdf', 'format': 'cv'}
    score =  0.46465086937
    metadata =  {'source': './resumes_dataset/resumes_dataset/17823436.pdf', 'format': 'cv'}
:::
:::

::: {#12818a7d-9313-42f3-b3b9-ef90d20cb733 .cell .code}
``` python
```
:::

::: {#1fe8c5ff-13c9-4698-a74c-46451725d828 .cell .code}
``` python
```
:::

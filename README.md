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

1.  **Clone the repository:**
2.  **Install dependencies:**\
3.  **Configure API Keys:**
4.  **Download the resume dataset**

``` python
%pip install numpy==2.2.4
%pip install PyPDF2 langchain langchain-google-genai langchain-google-vertexai langchain-openai
%pip install -q -U google-genai
%pip install -qU langchain-redis langchain-huggingface sentence-transformers scikit-learn
```

``` python
from src.entity_extractor import *
from src.pdf_reader import *
from src.rank_candidate_llm import *
from src.rank_candidate import *
from src.summary_generator import *
from src.translator import *
import json
```
1.  **Loading Gemini model:**


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
1.  **Loading CVS:**
``` python
# LOAD ALL CVS
pdfReader = PDFReader()
pdfs = pdfReader.readfiles("./resumes_dataset/resumes_dataset/")
```

1.  **Helper methods to score single CV:**

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

``` python
extract, summary, score, alg_part, llm_part, skills  = score_CV(pdfReader.openFile(pdfs[0]), ["Java", "Writing", "Excel"], "needed java developer with 3 years of experience in backend development in spring")
```

1.  **Helper method to limit requests. Gemini 2 Flash in free tier is
    limited to 15 requests per minute:**

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

1.  **Example use case:**

``` python
results = thressholdRequests(pdfs[0:10],["Spring Boot", "RabbitMQ", "AWS Lambda", "Terraform", "Docker", "Kubernetes", "PostgreSQL", "DynamoDB", "GitLab CI/CD", "ArgoCD", "TypeScript", "NestJS", "Redis", "Load Balancers", "VPC"]
, "Seeking a Senior Java Developer with 5+ years of experience in backend and cloud development. Must have expertise in Java, Spring Boot, AWS, TypeScript, and experience with microservices, Kubernetes, CI/CD, and message queues (RabbitMQ).")
```

1.  **Load CVs to Vector Storage**

This section demonstrates how to create a vector store in Redis, embed
resume text, and add the embeddings to the store. This enables semantic
search and retrieval of resumes based on their content. It leverages the
VectorStoreRedis class, which encapsulates the interaction with the
Redis database. The text embeddings are generated using Vertex AI
embedding model. After creating vector database resumes are added to
vectorstore.

``` python
from src.vector_store_inmemory import *
from src.vector_store_redis import *

store = VectorStoreRedis()

print("starting to process")

names = pdfs[1000:]
texts = []
i = 0
for name in names:
    texts.append(pdfReader.openFile(name))
    i = i + 1
    if i % 100 == 0:
        print(i)
print("start vectorizing")
store.addAllToStore(texts, names, "cv")
print("finished")
```


1.  **Searching vector db with semantic queries**

``` python
results = store.searchWithScore("who is the best java developer", k=10)
```

``` python
for result in results:
    print("score = ", result[1])
    print("metadata = ", result[0].metadata)
```




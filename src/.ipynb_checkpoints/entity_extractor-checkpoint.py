from langchain_core.messages import HumanMessage, SystemMessage
import re
import json

class EntityExtractor:
    """
        This class extract data about cv to json format
    """
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = """
You are an AI specialized in extracting key information from resumes.
Your task is to extract structured data from the given text and return it in JSON format.

**Extract the following fields if available:**
- "name": The full name of the candidate.
- "job_title": The most relevant job title or the latest position.
- "education": The highest level of education achieved.
- "skills": A list of key technical and soft skills.
- "languages": A list of spoken languages.
- "certifications": Any relevant certifications.

**Output Format (JSON)**:
{
    "name": "John Doe",
    "contact_info": {
        "email": "johndoe@example.com",
        "phone": "+1 234 567 890",
        "linkedin": "https://linkedin.com/in/johndoe"
    },
    "education": [
        {
            "degree": "Bachelor's",
            "field": "Computer Science",
            "institution": "MIT",
        }
    ],
    "experience": [
        {
            "position": "Software Engineer",
            "company": "Google",
            "years_of_experience": 3,
            "technologies_used": ["Python", "Django", "Docker"],
            "responsibilities": "Developed and maintained web applications."
        },
        {
            "position": "Frontend Developer",
            "company": "Meta",
            "years_of_experience": 2,
            "technologies_used": ["React", "TypeScript"],
            "responsibilities": "Built scalable UI components."
        }
    ],
    "skills": ["Software Development", "Web Development"],
    "technologies": ["Python", "React", "AWS"],
    "languages": [
        {
            "language": "English",
            "proficiency": "Fluent"
        },
        {
            "language": "Spanish",
            "proficiency": "Intermediate"
        }
    ],
    "certifications": [
        {
            "name": "AWS Certified Developer",
            "issuer": "AWS",
            "year": "2023"
        }
    ]
}

There are a couple of additional points that you have to remember:
- in education part you add only highest education or the most current
- if experience is set as from 'year' to 'current' calculate it with 'current' being 2025
- responsibilities should be very broad, at most 2 sentences but most of the time 1 will be enough.
- in skills part there should be not more than 10 entries. focus on the most valuable skills for the IT department. prioritie skills that this person have the most experience with.

Only return the JSON object. Do not add any extra explanation.
Remember to return valid json format, NO ENDLINE SYMBOLS
"""

    def extract(self, resume_text: str):
        """Extracts entities from resume text."""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=resume_text),
        ]
        response = self.llm(messages)
        clean_json = re.sub(r"^```json\n|\n```$", "", response.content.strip())
        parsed_data = json.loads(clean_json)
        return parsed_data
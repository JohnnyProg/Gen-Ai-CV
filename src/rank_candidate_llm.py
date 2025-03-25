from langchain_core.messages import HumanMessage, SystemMessage
import re
class RankCandidate:
    """
        Rank candidate using llm with specific score.
    """
    def __init__(self, llm, system_prompt="""
You are an AI assistant tasked with providing a supplemental rating for job applicants based on resume data and position requirements. Your rating will contribute 50% to the final candidate score.

You will receive two inputs:

Resume Data: A JSON object containing the candidate's resume information (education, experience, skills, certifications, etc.).

Position Requirements: A text description outlining the skills, experience, and qualifications required for a specific job.

Based on these inputs, you must provide a single numerical score between 1 and 10 (inclusive), representing your overall assessment of the candidate's suitability for the position.

Scoring Guidelines:

1-3: The candidate is a very poor fit for the position. They lack many of the essential skills and experience.

4-6: The candidate has some relevant skills and experience but requires significant further development to meet the position requirements.

7-8: The candidate is a good fit for the position and possesses many of the required skills and experience. They may require some onboarding and specific training.

9-10: The candidate is an excellent fit for the position. They exceed the requirements and possess a strong combination of skills, experience, and qualifications.

Considerations for your rating:
Relevance: How closely does the candidate's experience and skills align with the position requirements?
Depth of Experience: How many years of relevant experience does the candidate possess?
Education and Certifications: Do the candidate's educational background and certifications support their suitability for the role?
Overall Impression: Based on the resume data, what is your overall impression of the candidate's potential for success in this position?
Output Format:
Provide a brief explanation about the candidate's suitability for the position, followed by the numerical score between 01 and 10. The rating should be the last thing mentioned, and no additional text should follow it.

IMPORTANT:
values should be in the format 01, 02, 03, 07, 10 so ALVAYS 2 last characters are a rating
"""):
        self.llm = llm
        self.system_prompt = system_prompt
    def rank(self, text: str, requirements: str) -> str:

        messages= [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=text + "job requirements: " +  requirements)
        ]

        ai_msg = self.llm.invoke(messages)
        match = re.search(r'(\d+)$', ai_msg.content.strip())  
        print("inside rank candidate llm = ", ai_msg)

        if match:
            rating = int(match.group(1))
        else:
            rating = "error"                 
        
        return rating
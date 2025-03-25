import json
from collections import defaultdict

def rank_candidate_from_json(candidate_extract_json, required_tech_list):
    candidate_skills_list = extract_candidate_technologies(candidate_extract_json)
    print(candidate_skills_list)
    return rank_candidate(required_tech_list, candidate_skills_list)


def extract_candidate_technologies(candidate_extract_json):
    skills = candidate_extract_json["skills"]
    for experience in candidate_extract_json["experience"]:
        skills = skills + experience["technologies_used"]
    return skills

def rank_candidate(required_tech_list, candidate_tech_list, technologies_json_path="./src/technologies.json"):
    with open(technologies_json_path, 'r') as f:
        technologies_data = json.load(f)

    tech_categories = technologies_data["technologies"]

    direct_common = direct_comparison(required_tech_list, candidate_tech_list)
    score = len(direct_common)
    matched_technologies = direct_common
    
    candidate_tech_set = set(candidate_tech_list)
    required_tech_set = set(required_tech_list)

    candidate_tech_set = candidate_tech_set - set(matched_technologies)
    required_tech_set = required_tech_set - set(matched_technologies)
    
    
    for required_tech in required_tech_set:
        found = False
        for category, techs in tech_categories.items():
            if required_tech in techs:
                # if required tech is in this category then check if candidate have skills in the same category
                for candidate_tech in candidate_tech_set:
                    if candidate_tech in techs:
                        found = True
                        score += 0.5
                        matched_technologies.append(category)
    return score, matched_technologies
                

def direct_comparison(required_list, candidate_list):
    return list(set(required_list).intersection(candidate_list))
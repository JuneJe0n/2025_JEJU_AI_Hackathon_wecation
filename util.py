import json
import numpy as np
from collections import defaultdict

# load json data
def load_data(file_path):
    with open(file_path, "r", encoding = "utf-8") as f:
        data = json.load(f)
    return data

# region과 date 기준으로 user를 grouping한 후, filetered_db에 저장
def groupby_date_region(users):
    filtered_db = []
    grouped_users = defaultdict(list)
    for user in users:
        key = (user['added_info']['region'], user['added_info']['date'])
        grouped_users[key].append(user)

    for key,users in grouped_users.items():
        filtered_db.append({"region" : key[0], "date" : key[1], "users" : users})

    return filtered_db

# 프롬프트 전처리
def preprocess_prompt(text):
    return text.strip().replace("\n", " ")


# 적합도 분석
def get_fitness(user_embed, program_embed):
    user_embed = np.array(list(user_embed.values()))
    program_embed = np.array(list(program_embed.values()))

    user_embed = user_embed / np.linalg.norm(user_embed, ord=2, axis=1, keepdims=True)
    program_embed = program_embed / np.linalg.norm(program_embed, ord=2, axis=1, keepdims=True)
    
    cos_similarity = np.dot(user_embed, program_embed.T)
    fitness = np.round((cos_similarity + 1) / 2, 0)
    return fitness
import json
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

# sex, age, job, interest로 user emb 생성
def embed_users(users):
    user_emb = {}
    """
    todo) 도현
    key가 user id, value가 emb
    """
    return user_emb

def embed_programs(program_data):
    program_emb = {}
    """
    todo) 도현
    key가 program id, value가 emb
    """
    return program_emb
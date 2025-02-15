"""
'AI 추천 프로그램' 버튼 눌렀을 때 실행
"""
import json
import argparse
import numpy as np
from util import load_data
from embedding import embed_users, embed_programs

def rank_programs_for_user(user_id, user_data, program_data):
    user = None
    for u in user_data:
        if u["basic_info"]["user_id"] == user_id:
            user = u
            break
    if not user:
        return []
    
    user_emb = embed_users([user])[user_id]
    program_embs = embed_programs(program_data)

    distances = {}
    for program_id, program_emb in program_embs.itmes():
        distance = np.linalg.norm(user_emb-program_emb)
        distances[program_id] = distance
    
    ranked_programs = sorted(distances, key = distances.get)

    return ranked_programs
        

def main():
    parser = argparse.ArgumentParser(description = "rank programs for a given user ID")
    parser.add_argument("user_id", type = str, help = "user ID to rank programs for")
    args = parser.parse_args()

    data = load_data("data.json")
    user_data = data["user_data"]
    program_data = data["program_data"]
    ranked_programs = rank_programs_for_user(args.user_id, user_data, program_data)

    print("Ranked Programs for User", args.user_id, ":", ranked_programs)

    
if __name__ == "__main__":
    main()

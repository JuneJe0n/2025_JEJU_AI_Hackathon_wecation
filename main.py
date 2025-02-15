"""
'다른 사람들과 AI 매칭 시작하기' 버튼 눌렀을 때 실행
"""

import json
from util import load_data, groupby_date_region
from matching import dbscan_clustering, compute_centroid, match_teams
from embedding import embed_users, embed_programs



def main():
    user_data = load_data("user_dummy_data.json")
    program_data = load_data("program_dummy_data.json")
    filtered_db = groupby_date_region(user_data)
    program_embeddings = embed_programs(program_data)

    final_teams = {}
    all_outliers = []

    # 각 (region, date)그룹 돌면서 진행
    for group in filtered_db :
        region, date, users = group["region"], group["date"], group["users"]
        user_embeddings = embed_users(users)
        clusters, outliers = dbscan_clustering(user_embeddings)
        team_db, additional_outliers = match_teams(clusters, user_embeddings, program_embeddings)

        final_teams.update(team_db)
        all_outliers.extend(outliers + additional_outliers)

    print("Final Team Assignments:", final_teams)
    print("Outliers (Not Matched):", all_outliers)

if __name__ == '__main__':
    main()

from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from collections import defaultdict 
import numpy as np
import uuid

# perform DBSCAN clustering on the given embeddings
def dbscan_clustering(embeddings, eps = 0.6, min_samples = 2):
    """
    return)
    clusters : (key=cluster label), (value=cluster에 속하는 user_id)인 dict
    outliers : outlier인 user_id가 들어있는 list
    """
    user_ids = list(embeddings.keys())
    emb_matrix = np.array(list(embeddings.values()))
    clustering = DBSCAN(eps = eps, min_samples = min_samples).fit(emb_matrix)
    labels = clustering.labels_

    clusters = defaultdict(list)
    outliers = []

    for idx, label in enumerate(labels):
        if label == -1: # (outliers list)에 (outlier인 user의 id) append
            outliers.append(user_ids[idx]) 
        else: # (clusters dict)에 (key=cluster label), (value=cluster에 속하는 user_id) append
            clusters[label].append(user_ids[idx]) 

    return clusters, outliers

# compute the centroid for a given cluster
def compute_centroid(user_ids, user_emb):
    user_embs_list = []
    for user_id in user_ids:
        emb_vector = user_emb[user_id]
        user_embs_list.append(emb_vector)
    user_embs = np.array(user_embs_list)
    centroid = np.mean(user_embs, axis = 0)
    return centroid

# find the farthest user from the centroid
def find_farthest_user(user_ids, centroid, user_emb):
    """
    centriod와 가장 먼 사용자의 user id 반환
    """
    distances = {}
    for user_id in user_ids:
        emb_vector = user_emb[user_id] # 사용자 임베딩 가져오기
        distance = np.linalg.norm(emb_vector-centroid) # centroid와의 거리 계산
        distances[user_id] = distance

    farthest_user = max(distances, key=distances.get)
    return farthest_user

# find the 5 closest users to a given user
def find_closest_users(user_ids, target_user, user_emb):
    """
    target_user와 가까운 사용자 5명의 user id를 list로 반환
    """
    distances = {}
    for user_id in user_ids:
        if user_id != target_user:
            emb_vector = np.array(user_emb[user_id])
            target_vector = np.array(user_emb[target_user])
            distance = np.linalg.norm(emb_vector-target_vector)
            distances[user_id] = distance 
    sorted_users = sorted(distances, key=distances.get)
    closest_users = sorted_users[:5] # 가장 가까운 5명 선택
    return closest_users

def find_best_program(team_centroid, program_embs):
    """
    team의 centroid와 가장 가까운 program의 program_id를 반환함
    """
    best_match = None
    min_distance = float('inf')
    for program_id, program_emb in program_embs.items():
        distance = np.linalg.norm(team_centroid-program_emb)
        if distance < min_distance:
            min_distance = distance
            best_match = program_id
    return best_match

def match_teams(clusters, user_emb, program_emb):
    team_db = {}
    outliers = []

    for cluster_id, user_ids in clusters.items(): # cluster 한 개씩 돌면서 진행
        while user_ids: # 해당 cluster에 user_id가 없어질때까지 반복
            centroid = compute_centroid(user_ids, user_emb)
            farthest_user = find_farthest_user(user_ids, centroid, user_emb)
            closest_users = find_closest_users(user_ids, farthest_user, user_emb)

            # 가까운 사용자가 5명 안 찾아지면 outlier로 취급
            if len(closest_users) < 5 :
                outliers.extend(user_ids)
                break

            team = [farthest_user] + closest_users
            team_id = str(uuid.uuid4())
            team_centroid = compute_centroid(team, user_emb)
            recommended_program = find_best_program(team_centroid, program_emb)
            team_db[team_id] = {"users" : team, "recommended_program" : recommended_program}

            for user in team:
                user_ids.remove(user)
    return team_db, outliers

import os
from openai import OpenAI
from util import preprocess_prompt
from dotenv import load_dotenv
load_dotenv()

# upstage api
CLIENT = OpenAI(
    api_key="",
    base_url="https://api.upstage.ai/v1/solar"
)

# matching prompt format
MATCHING_EMBED_PROMPT = """
이 사람은 {sex}이며, {age}세입니다. 직업은 {job}입니다.
관심사는 {interest}이며, 이와 관련된 활동을 즐깁니다.
MBTI 유형은 {mbti}이며, 이에 맞는 성향을 가지고 있습니다.
비슷한 관심사를 가진 사람이나 관련된 프로그램을 찾고 있습니다.
"""

# program prompt format
PROGRAM_EMBED_PROMPT = """
프로그램 이름은 {name}으로, {hashtag}와 관련된 활동을 제공합니다.
"""

# user embedding 생성
def embed_users(users):
    prompts = [
        preprocess_prompt(
            MATCHING_EMBED_PROMPT.format(
                sex="남성" if user["basic_info"]["sex"]=="M" else "여성",
                age=user["basic_info"]["age"],
                job=user["basic_info"]["job"],
                mbti=user["basic_info"]["mbti"],
                interest=", ".join(user["added_info"]["interest"])
            )
        ) for user in users
    ]
    response = CLIENT.embeddings.create(
        model="embedding-query",
        input=prompts
    )
    embeddings = {user["basic_info"]["user_id"]: data.embedding for user, data in zip(users, response.data)}
    return embeddings

# program embeddings 생성
def embed_programs(programs):
    prompts = [
        preprocess_prompt(
            PROGRAM_EMBED_PROMPT.format(
                name=program["name"],
                hashtag=", ".join(list(map(lambda x: x.lstrip("#"), program["hashtag"])))
            )
        ) for program in programs
    ]
    response = CLIENT.embeddings.create(
        model="embedding-query",
        input=prompts
    )
    embeddings = {program["program_id"]: data.embedding for program, data in zip(programs, response.data)}
    return embeddings
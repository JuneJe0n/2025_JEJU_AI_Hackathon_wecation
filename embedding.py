import os
from openai import OpenAI
from util import preprocess_prompt

# openai embedding api
CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# matching prompt format
MATCHING_EMBED_PROMPT = """
이 사람은 {sex}이며, {age}세입니다. 직업은 {job}입니다.
관심사는 {interest}이며, 이와 관련된 활동을 즐깁니다.
비슷한 관심사를 가진 사람이나 관련된 프로그램을 찾고 있습니다.
"""

# program prompt format
PROGRAM_EMBED_PROMPT = """
프로그램 이름은 {name}으로, {hashtag}와 관련된 활동을 제공합니다.
"""

# mathcing embedding 생성
def embed_users(users):
    prompts = [
        preprocess_prompt(
            MATCHING_EMBED_PROMPT.format(
                sex="남성" if user["basic_info"]["sex"]=="M" else "여성",
                age=user["basic_info"]["age"],
                job=user["basic_info"]["job"],
                interest=", ".join(user["added_info"]["interest"])
            )
        ) for user in users
    ]
    response = CLIENT.embeddings.create(
        model="text-embedding-3-small",
        input=prompts
    )
    embeddings = [data.embedding for data in response.data]
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
        model="text-embedding-3-small",
        input=prompts
    )
    embeddings = [data.embedding for data in response.data]
    return embeddings
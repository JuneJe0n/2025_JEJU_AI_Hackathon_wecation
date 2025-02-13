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
def matching_embed(user):
    prompt = preprocess_prompt(
        MATCHING_EMBED_PROMPT.format(
            sex="남성" if user["basic_info"]["sex"]=="M" else "여성",
            age=user["basic_info"]["age"],
            job=user["basic_info"]["job"],
            interest=", ".join(user["added_info"]["interest"])
        )
    )
    response = CLIENT.embeddings.create(
        model="text-embedding-3-small",
        input=prompt
    )
    embedding = response.data[0].embedding
    return embedding

# program embedding 생성
def program_embed(program):
    prompt = preprocess_prompt(
        PROGRAM_EMBED_PROMPT.format(
            name=program["name"],
            hashtag=", ".join(list(map(lambda x: x.lstrip("#"), program["hashtag"])))
        )
    )
    response = CLIENT.embeddings.create(
        model="text-embedding-3-small",
        input=prompt
    )
    embedding = response.data[0].embedding
    return embedding
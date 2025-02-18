import os
from openai import OpenAI
from util import preprocess_prompt

# upstage api
CLIENT = OpenAI(
    api_key="",
    base_url="https://api.upstage.ai/v1/solar"
)


# recommendation system prompt format
RECOMMENDATION_SYSTEM_PROMPT = """
당신은 유저와 프로그램 정보를 기반으로 추천 이유를 분석하는 AI입니다.

답변을 생성할 때 다음 사항을 준수하세요:
1. 유저와 프로그램 정보 간의 유사한 점을 분석하세요.
2. 두 줄 이내로 답변을 생성하세요.
"""

# recommendation user prompt format
RECOMMENDATION_USER_PROMPT = """
아래 정보를 바탕으로 해당 프로그램을 왜 유저에게 추천하는지 설명하세요.

### 유저 정보
- 성별: {sex}
- 나이: {age}
- 직업: {job}
- 관심사: {interest}

### 추천된 프로그램 정보
- 프로그램 이름: {program_name}
- 주요 키워드: {hashtag}
"""

# recommendation reason 생성
def recommend(user, program):
    system_prompt = preprocess_prompt(
        RECOMMENDATION_SYSTEM_PROMPT
    )
    user_prompt = preprocess_prompt(
        RECOMMENDATION_USER_PROMPT.format(
            sex=user["basic_info"]["sex"],
            age=user["basic_info"]["age"],
            job=user["basic_info"]["job"],
            interest=", ".join(user["added_info"]["interest"]),
            program_name=program["name"],
            hashtag=", ".join(list(map(lambda x: x.lstrip("#"), program["hashtag"])))
        )
    )
    response = CLIENT.chat.completions.create(
        model="solar-pro",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    reason = response.choices[0].message.content
    return reason
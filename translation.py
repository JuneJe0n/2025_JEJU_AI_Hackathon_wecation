import os
from openai import OpenAI
from util import preprocess_prompt

# openai embedding api
CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# translation system prompt format
TRANSLATION_SYSTEM_PROMPT = """
You are a highly accurate and fluent translator.
Your task is to translate any input text into natural and grammatically correct English.
Maintain the original meaning, tone, and style of the text.
Do not add explanations or comments—only provide the translated English text.
If the input is already in English, return it unchanged.
"""

# translation user prompt format
TRANSLATION_USER_PROMPT = """
Translate the following text into English:
Text: {text}
"""

# translated text 생성
def translate(data):
    system_prompt = preprocess_prompt(
        TRANSLATION_SYSTEM_PROMPT
    )
    user_prompt = preprocess_prompt(
        TRANSLATION_USER_PROMPT.format(
            text=data["message"]
        )
    )
    response = CLIENT.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    translated = response.choices[0].message.content
    return translated
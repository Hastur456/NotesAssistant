import openai
from openai import OpenAI

from base import (
    BaseLLM,
    LLMConfig,
    LLMResponse
)

import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("PERPLEXITY_API_KEY")


client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.perplexity.ai"
)

response = client.chat.completions.create(
    model="sonar",
    messages=[
        {"role": "system", "content": "Будь точным и кратким."},
        {"role": "user", "content": "Сколько звёзд в нашей галактике?"}
    ]
)

print(response.choices[0].message.content)


class PerplexityAiLLM(BaseLLM):
    def __init__(self, config: LLMConfig, logger):
        super().__init__(config, logger)

        self.api_key = self.config.api_key or os.getenv("PERPLEXITY_API_KEY")

        if self.api_key is None:
            raise ConnectionError("Соединение с API отсутствует")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.perplexity.ai"
        )

    def generate(self, prompt, **kwards):
        response = client.chat.completions.create(
            model="sonar",
            messages=[
                {"role": "system", "content": "Будь точным и кратким."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message

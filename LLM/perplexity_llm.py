import openai
from openai import OpenAI
from datetime import datetime
from logging import Logger
import os
from dotenv import load_dotenv

from base import (
    BaseLLM,
    LLMConfig,
    LLMResponse
)


class PerplexityAiLLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        super().__init__(config)

        self._setup_client()
        self.check_connection()

    def generate(self, prompt, **kwards):
        try:
            self.logger.info("Отправка запроса к API Perplexity")

            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "Будь точным и кратким."},
                    {"role": "user", "content": prompt}
                ]
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens,
                timestamp=datetime.fromtimestamp(response.created),
                metadata={k: v for k, v in response.to_dict().items() if k not in {"choices", "model", "usage", "created"}}
            )
        
        except openai.APIError as e:
            self.logger.error("Ошибка при подключении к API Perplexity: {}".format(e))
            raise e
        
        except Exception as e:
            self.logger.error(f"Ошибка при обработке запроса к API: {e}")
            raise e
        
    def check_connection(self):
        try:
            response = self.client.chat.completions.create(
                model="sonar",
                messages=[
                    {"role": "user", "content": "ping"}
                ],
                max_tokens=1
            )
            self.logger.info("Подключено к Perplexity API")
            self.logger.info("Ответ: {}".format(response.choices[0].message))
            return True
        
        except openai.APIError as e:
            self.logger.error("Ошибка при подключении к API Perplexity: {}".format(e))
            raise e

        except Exception as e:
            self.logger.error("Ошибка при подключении к API Perplexity: {}".format(e))
            raise e
        
    def _setup_client(self):
        load_dotenv()
        self.api_key = self.config.api_key or os.getenv("PERPLEXITY_API_KEY")

        if self.api_key is None:
            raise ConnectionError("Соединение с API отсутствует")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.perplexity.ai"
        )


test_config = LLMConfig()
model = PerplexityAiLLM(test_config)

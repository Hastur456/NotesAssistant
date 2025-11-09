import openai
from datetime import datetime
from logging import Logger
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional, List, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from openai import OpenAI


@dataclass
class LLMConfig:
    model_name: str = "sonar"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 30
    retry_attempts: int = 3
    api_key: Optional[str] = None


class PerplexityAiLLM(BaseChatModel):
    def __init__(self, config: LLMConfig, **kwards):
        super().__init__(**kwards)
        self.config = config
        self._setup_client()
        self.check_connection()
        
    def _setup_client(self):
        load_dotenv()
        self.api_key = self.config.api_key or os.getenv("PERPLEXITY_API_KEY")

        if self.api_key is None:
            raise ValueError("API ключ отсутствует")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.perplexity.ai"
        )

    def _generate(self, message, stop=None, **kwards):
        try:
            prompt = message[-1].content
            self.logger.info("Отправка запроса к API Perplexity")

            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "Будь точным и кратким."},
                    {"role": "user", "content": prompt}
                ]
            )

            message = AIMessage(content=response.choices[0].message.content)
            return ChatResult(generations=[ChatGeneration(message=message)])
        
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
        
    @property
    def _llm_type(self):
        return "perplexity"

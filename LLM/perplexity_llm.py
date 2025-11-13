import openai
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any


from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import BaseModel, Field, ConfigDict
from openai import OpenAI


class LLMConfig(BaseModel):
    model_name: str = "sonar"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 30
    retry_attempts: int = 3
    api_key: Optional[str] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class PerplexityAiLLM(BaseChatModel):
    config: LLMConfig = Field(...)
    client: Optional[OpenAI] = Field(default=None, exclude=True)
    api_key: Optional[str] = Field(default=None)
    logger: Optional[logging.Logger] = Field(default=None, exclude=True)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, config: LLMConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        self._set_default_logger()
        self._setup_client()
        self._check_connection()
        
    def _setup_client(self):
        load_dotenv()
        self.api_key = self.config.api_key or os.getenv("PERPLEXITY_API_KEY")

        if self.api_key is None:
            raise ValueError("API ключ отсутствует")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.perplexity.ai"
        )


    def _generate(self, message, stop=None, **kwargs) -> ChatResult:
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
        
    def _check_connection(self):
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
    def _llm_type(self) -> str:
        return "perplexity"

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "PerplexityAiLLM":
        """
        Привязка инструментов к модели.
        Perplexity API не поддерживает tool_choice/function_calling напрямую,
        поэтому мы просто возвращаем self без изменений.
        """
        return self

    def _set_default_logger(self, log_path: str = 'LLM_debug.log'):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        file_handler = logging.FileHandler(log_path, mode='w', encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

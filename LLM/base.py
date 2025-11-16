import openai
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class LLMConfig(BaseModel):
    model_name: str = "sonar"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 30
    retry_attempts: int = 3
    api_key: Optional[str] = None


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int
    timestamp: datetime
    metadata: Dict[str, Any]


class BaseLLM(ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    llm_config: LLMConfig
    def __init__(self, llm_config: LLMConfig, **kwargs):
        super().__init__(llm_config=llm_config, **kwargs)
        self.llm_config = llm_config
        self.logger: logging.Logger | None = None
        self.client: Any | None = None
        self._set_default_logger()
        self._setup_client()
        self._check_connection()

    @abstractmethod
    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwards) -> LLMResponse:
        pass

    @abstractmethod
    def _check_connection(self) -> bool:
        pass

    @abstractmethod
    def _setup_client(self):
        pass

    def predict(self, text: str, **kwards):
        response = self._generate(text, **kwards)
        return response.content
    
    def generate_with_context(self, query: str, context: str, template: str = None):
        if template is None:
            template = self._get_default_context_template()
        
        prompt = template.format(query=query, context=context)
        return self._generate(prompt)
    
    def _get_default_context_template(self) -> str:
        return """Используя следующий контекст, ответь на вопрос. 
        Контекст: {context}
        Вопрос: {query} 
        Ответ: """
    
    def _set_default_logger(self, logger_path: str = "LLM_debug.log"):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        file_handler = logging.FileHandler(logger_path, mode='w', encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
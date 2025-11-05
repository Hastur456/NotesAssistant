import openai
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class LLMConfig:
    model_name: str
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
    def __init__(self, config: LLMConfig, logger):
        self.config = config
        self.logger = logger

    @abstractmethod
    def generate(self, prompt: str, **kwards):
        pass

    def predict(self, text: str):
        response = self.generate(text)
        return response.content
    
    def generate_with_context(self, query: str, context: str, template: str = None):
        if template is None:
            template = self._get_default_context_template()
        
        prompt = template.format(query=query, context=context)
        return self.generate(prompt)
    
    #Вспомогательные методы
    def _get_default_context_template(self) -> str:
        return """Используя следующий контекст, ответь на вопрос. 
        Контекст: {context}
        Вопрос: {query} 
        Ответ: """
    
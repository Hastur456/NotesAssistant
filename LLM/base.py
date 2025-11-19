# base.py
import openai
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field


class LLMConfig(BaseModel):
    model_name: str = Field(..., description="Name of model")
    temperature: float = Field(..., gt=0, description="Temperature (Greatest then 0)")
    max_tokens: int = Field(..., description="Maximum number of generated tokens")
    timeout: int = Field(..., description="Timeout of request")
    retry_attempts: int = Field(..., description="Retry attemps")
    api_key: str | None = Field(..., description="Your API key")


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int
    timestamp: datetime
    metadata: Dict[str, Any]


class BaseLLM(ABC):
    def __init__(self, llm_config: LLMConfig, **kwargs):
        self.llm_config = llm_config
        self.logger: logging.Logger | None = None
        self.client: Any | None = None
        self.tools_list: List[BaseTool] | None = None
        self._tools_dicts: List[Dict[str, Any]] | None = None
        
        self._set_default_logger()
        self._setup_client()
        self._check_connection()

    @abstractmethod
    def _call_with_tools(
        self, 
        messages: List[BaseMessage], 
        stop: List[str] | None = None,
        tools: List[Dict[str, Any]] | None = None,
        **kwargs
    ) -> BaseMessage:
        pass

    @abstractmethod
    def _check_connection(self) -> bool:
        pass

    @abstractmethod
    def _setup_client(self):
        pass

    def invoke(self, messages: List[BaseMessage], **kwargs) -> BaseMessage:
        return self._call_with_tools(messages, **kwargs)

    def batch(self, messages_list: List[List[BaseMessage]], **kwargs) -> List[BaseMessage]:
        return [self.invoke(messages, **kwargs) for messages in messages_list]

    def predict(self, text: str, **kwargs) -> str:
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content=text)]
        response = self.invoke(messages, **kwargs)
        return response.content

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

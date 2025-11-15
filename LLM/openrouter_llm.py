from openai import OpenAI

from typing import Optional, List, Dict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from pydantic import ConfigDict, Field

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv 
from .base import BaseLLM, LLMConfig
load_dotenv()


class OpenRouterLLM(BaseLLM):
    llm_config: LLMConfig = Field(...)

    def __init__(self, config: LLMConfig):
        self.llm_config = config
        super().__init__()
    
    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwards):
        openai_messages = [
            {
                "role": "system" if isinstance(m, SystemMessage) 
                        else "assistant" if isinstance(m, AIMessage) 
                        else "user",
                "content": m.content
            }
            for m in messages
        ]

        response = self.client.chat.completions.create(
            messages=openai_messages,
            model=self.llm_config.model_name
        )

        return response


    def _setup_client(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API"), 
            base_url="https://openrouter.ai/api/v1"
        )

    def _check_connection(self):
        return super()._check_connection()
        

    @property
    def _llm_type(self) -> str:
        return "deepseek/deepseek-chat"
    

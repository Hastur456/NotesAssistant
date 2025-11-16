# openrouter_llm.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import OpenAI
from typing import Optional, List, Dict, Any
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv 
from .base import BaseLLM, LLMConfig

load_dotenv()


openrouter_config = LLMConfig(
    model_name="deepseek/deepseek-chat",
    temperature=0.4,
    max_tokens=1000,
    timeout=30,
    retry_attempts=3,
    api_key=os.getenv("OPENROUTER_API")
)


class OpenRouterLLM(BaseLLM):

    def __init__(self, llm_config: LLMConfig, **kwargs):
        super().__init__(llm_config=llm_config, **kwargs)
    
    def _call_with_tools(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> BaseMessage:
        
        openai_messages = []
        for m in messages:
            if isinstance(m, SystemMessage):
                openai_messages.append({"role": "system", "content": m.content})
            elif isinstance(m, AIMessage):
                openai_messages.append({"role": "assistant", "content": m.content or ""})
            else:
                openai_messages.append({"role": "user", "content": m.content})
        
        api_params = {
            "messages": openai_messages,
            "model": self.llm_config.model_name,
            "temperature": self.llm_config.temperature,
            "max_tokens": self.llm_config.max_tokens,
        }
        
        if stop:
            api_params["stop"] = stop
        
        tools_to_use = tools or self._tools_dicts
        if tools_to_use:
            api_params["tools"] = tools_to_use
            api_params["tool_choice"] = "auto"
        
        response = self.client.chat.completions.create(**api_params)
        message = response.choices[0].message
        
        return AIMessage(content=message.content or "")

    def _setup_client(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API"), 
            base_url="https://openrouter.ai/api/v1"
        )

    def _check_connection(self) -> bool:
        try:
            return self.client is not None
        except Exception as e:
            if self.logger:
                self.logger.error(f"Connection check failed: {e}")
            return False

    @property
    def _llm_type(self) -> str:
        return "deepseek/deepseek-chat"

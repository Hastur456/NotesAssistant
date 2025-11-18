# openrouter_llm.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import OpenAI
from typing import Optional, List, Dict, Any
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
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
        stop: List[str] | None = None,
        tools: List[Dict[str, Any]] | None = None,
        **kwargs
    ) -> BaseMessage:
        
        converted_messages = self._convert_messages(messages=messages)

        api_params = {
            "messages": converted_messages,
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
            if self.logger:
                self.logger.debug(f"Using {len(tools_to_use)} tools")
        
        try:
            response = self.client.chat.completions.create(**api_params)
            message = response.choices[0].message
            
            if hasattr(message, 'tool_calls') and message.tool_calls:
                if self.logger:
                    self.logger.debug(f"Tool calls detected: {len(message.tool_calls)}")

            return AIMessage(
                content=message.content or "",
                tool_calls=[
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            )
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error calling LLM: {e}")
            raise

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

    def _convert_messages(self, messages: List[BaseMessage]) -> list:
        converted_messages = []
        for m in messages:
            if isinstance(m, SystemMessage):
                converted_messages.append({"role": "system", "content": m.content})
            elif isinstance(m, AIMessage):
                converted_messages.append({"role": "assistant", "content": m.content or ""})
            else:
                converted_messages.append({"role": "user", "content": m.content})
        
        return converted_messages

    @property
    def _llm_type(self) -> str:
        return "deepseek/deepseek-chat"


class OpenRouterAdapter(BaseChatModel):
    llm: Any | None = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.llm = OpenRouterLLM(llm_config=config)

    def _generate(self, messages, **kwargs):
        response = self.llm._call_with_tools(messages, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=response)])
    
    def _llm_type(self):
        return "openrouter"

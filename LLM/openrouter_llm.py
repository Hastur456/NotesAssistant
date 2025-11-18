# openrouter_llm.py - ПОЛНОСТЬЮ ИСПРАВЛЕННЫЙ

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import OpenAI
from typing import Optional, List, Dict, Any
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.messages.tool import tool_call
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from dotenv import load_dotenv
from .base import BaseLLM, LLMConfig
import json

load_dotenv()

openrouter_config = LLMConfig(
    model_name="deepseek/deepseek-chat",
    temperature=0.4,
    max_tokens=2000,
    timeout=30,
    retry_attempts=3,
    api_key=os.getenv("OPENROUTER_API")
)


class OpenRouterLLM(BaseLLM):
    """OpenRouter LLM для использования с create_agent"""
    
    def __init__(self, llm_config: LLMConfig, **kwargs):
        super().__init__(llm_config=llm_config, **kwargs)

    def _call_with_tools(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        tools: List[Dict[str, Any]] | None = None,
        **kwargs
    ) -> BaseMessage:
        """Вызывает LLM с поддержкой tool calling"""
        
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
            
            # ════════════════════════════════════════════════════════════════════
            # ✅ ИСПРАВЛЕННЫЙ КОД: Правильный формат tool_calls
            # ════════════════════════════════════════════════════════════════════
            
            if hasattr(message, 'tool_calls') and message.tool_calls:
                if self.logger:
                    self.logger.debug(f"Tool calls detected: {len(message.tool_calls)}")
                
                # ✅ Используем tool_call() вместо словаря
                tool_calls_list = []
                for tc in message.tool_calls:
                    # Парсим arguments если это строка
                    arguments = tc.function.arguments
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except:
                            arguments = {}
                    
                    # Создаем tool_call объект правильно
                    tool_calls_list.append(
                        tool_call(
                            id=tc.id,
                            name=tc.function.name,
                            args=arguments
                        )
                    )
                
                return AIMessage(
                    content=message.content or "",
                    tool_calls=tool_calls_list  # ✅ Используем правильный формат
                )
            
            return AIMessage(content=message.content or "")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error calling LLM: {e}")
            raise


    def _setup_client(self):
        """Инициализирует OpenAI клиент для OpenRouter"""
        api_key = self.llm_config.api_key or os.getenv("OPENROUTER_API")
        if not api_key:
            raise ValueError("OPENROUTER_API not found in environment or config")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

    def _check_connection(self) -> bool:
        """Проверяет соединение с API"""
        try:
            return self.client is not None
        except Exception as e:
            if self.logger:
                self.logger.error(f"Connection check failed: {e}")
            return False

    def _convert_messages(self, messages: List[BaseMessage]) -> list:
        """Конвертирует LangChain messages в OpenAI format"""
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
    """
    Адаптер для совместимости с create_agent
    
    Реализует все необходимые методы для работы с create_agent
    """
    
    llm: OpenRouterLLM | None = None
    tools_list: List[BaseTool] | None = None
    _tools_dicts: List[Dict[str, Any]] | None = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.llm = OpenRouterLLM(llm_config=config)
        self.tools_list = None
        self._tools_dicts = None

    def _generate(self, messages, **kwargs):
        """Генерирует ответ используя OpenRouter LLM"""
        response = self.llm._call_with_tools(
            messages,
            tools=self._tools_dicts,
            **kwargs
        )
        return ChatResult(generations=[ChatGeneration(message=response)])

    def _llm_type(self) -> str:
        return "openrouter"

    # ════════════════════════════════════════════════════════════════════════
    # ✅ КРИТИЧЕСКИ ВАЖНЫЙ МЕТОД ДЛЯ create_agent
    # ════════════════════════════════════════════════════════════════════════
    
    def bind_tools(
        self,
        tools: List[BaseTool],
        **kwargs
    ) -> "OpenRouterAdapter":
        """
        Привязывает инструменты к модели
        
        Это требуется для create_agent!
        
        Args:
            tools: Список инструментов (BaseTool)
        
        Returns:
            Новый OpenRouterAdapter с привязанными инструментами
        """
        
        # Генерируем JSON Schema для каждого инструмента
        tools_dicts = []
        
        for tool in tools:
            # Получаем параметры инструмента
            if hasattr(tool, 'args_schema') and tool.args_schema:
                try:
                    schema = tool.args_schema.model_json_schema()
                    parameters = {
                        "type": "object",
                        "properties": schema.get("properties", {}),
                        "required": schema.get("required", [])
                    }
                except Exception as e:
                    parameters = {"type": "object", "properties": {}}
            else:
                parameters = {"type": "object", "properties": {}}
            
            # Создаем описание инструмента в формате OpenAI
            tool_dict = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "No description",
                    "parameters": parameters
                }
            }
            tools_dicts.append(tool_dict)
        
        # Создаем новый адаптер с привязанными инструментами
        new_adapter = OpenRouterAdapter(self.llm.llm_config)
        new_adapter.llm = self.llm
        new_adapter.tools_list = tools
        new_adapter._tools_dicts = tools_dicts
        
        return new_adapter

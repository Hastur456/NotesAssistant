import os
import sys
import json
import logging
from typing import Optional, Dict, Any, List
from uuid import uuid4
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool

from AGENT.tools import FileOperationTools
from RAG.notes_rag import RAGAssistant
from LLM.openrouter_llm import OpenRouterAdapter, openrouter_config

class ReActAgent:
    def __init__(
        self,
        notes_dir: str,
        persist_dir: str = "./vectorstorage",
        verbose: bool = True,
        max_iterations: int = 5
    ):
        self.thread_id = str(uuid4())
        
        self.notes_dir = notes_dir
        self.persist_dir = persist_dir
        self.verbose = verbose
        self.max_iterations = max_iterations
        
        self.logger = self._setup_logger()
        self.logger.info("Initializing ReActAgent...")
        
        self._init_llm()
        self._init_rag()
        self._init_tools()
        
        self.agent = self._create_agent()
        
        self.conversation_history: List[Dict[str, str]] = []
        
        self.logger.info("✓ ReActAgent initialized successfully")

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"ReActAgent-{self.thread_id[:8]}")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _init_llm(self):
        self.llm = OpenRouterAdapter(openrouter_config)
        self.logger.debug("LLM initialized")

    def _init_rag(self):
        self.rag_assistant = RAGAssistant(
            notes_dir=self.notes_dir,
            persist_dir=self.persist_dir
        )
        self.logger.debug("RAG system initialized")

    def _init_tools(self):
        self.tools_manager = FileOperationTools(notes_dir=self.notes_dir)
        self.tools_manager.rag_assistant = self.rag_assistant
        self.tools_functions = self.tools_manager.create_tools()
        self.logger.debug(f"Created {len(self.tools_functions)} tools")

    def _create_system_prompt(self) -> str:
        tools_info = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools_functions
        ])

        return f"""Вы - экспертный ассистент для управления текстовыми заметками.

ИНСТРУКЦИИ:
1. Анализируйте запрос пользователя внимательно
2. Используйте необходимые инструменты для выполнения задачи
3. После выполнения инструмента проанализируйте результат
4. Предоставьте четкий ответ пользователю
5. Если нужны уточнения - запросите их

Доступные инструменты:
{tools_info}

РАБОЧИЙ ПРОЦЕСС:
1. Определите, какой инструмент нужен
2. Выполните его с правильными параметрами
3. Проанализируйте результат
4. Дайте финальный ответ

Всегда будьте конкретны и полезны в своих ответах."""

    def _create_agent(self):
        system_prompt = self._create_system_prompt()
        
        agent = create_agent(
            model=self.llm,
            tools=self.tools_functions,
            system_prompt=system_prompt,
        )
        
        self.logger.debug("Agent created with create_agent API")
        return agent

    def answer(self, query: str) -> str:
        self.logger.info(f"Processing query: {query[:80]}...")
        self.conversation_history.append({"role": "user", "content": query})

        try:
            response = self.agent.invoke({
                "messages": [HumanMessage(content=query)]
            })

            answer_text = self._extract_response(response)
            
            self.logger.debug(f"Agent response: {answer_text[:100]}...")
            self.conversation_history.append({
                "role": "assistant",
                "content": answer_text
            })

            return answer_text

        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            error_msg = f"Ошибка: {str(e)}"
            self.conversation_history.append({
                "role": "assistant",
                "content": error_msg
            })
            return error_msg

    def stream(self, query: str):
        self.logger.info(f"Streaming query: {query[:80]}...")

        try:
            for event in self.agent.stream({
                "messages": [HumanMessage(content=query)]
            }):
                yield event

        except Exception as e:
            self.logger.error(f"Error in stream: {e}")
            yield {"error": str(e)}

    def reset_memory(self):
        self.logger.info("Resetting memory")
        self.thread_id = str(uuid4())
        self.conversation_history = []

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.conversation_history.copy()

    def _extract_response(self, response: Any) -> str:
        if isinstance(response, dict):
            if "messages" in response:
                messages = response["messages"]
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, 'content'):
                        return last_msg.content
                    return str(last_msg)
            
            if "output" in response:
                return response["output"]
            
            return str(response)
        
        if hasattr(response, 'content'):
            return response.content
        
        return str(response)

if __name__ == "__main__":
    from pathlib import Path
    
    test_dir = Path("./tests/testnotes")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print("Initializing agent...")
    agent = ReActAgent(notes_dir=str(test_dir), verbose=True)
    
    queries = [
        "Покажи структуру папки с заметками",
        "Создай заметку 'Python Tips' с содержимым 'List comprehension - мощный инструмент'",
        "Найди все заметки про Python",
    ]
    
    print("\n" + "="*80)
    print("RUNNING AGENT EXAMPLES")
    print("="*80)
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}] Query: {query}")
        print("-" * 80)
        
        try:
            response = agent.answer(query)
            print(f"Response: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("\n" + "="*80)
    print("CONVERSATION HISTORY")
    print("="*80)
    for msg in agent.get_conversation_history():
        role = msg['role'].upper()
        content = msg['content'][:60] + "..." if len(msg['content']) > 60 else msg['content']
        print(f"{role}: {content}")
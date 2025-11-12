from langchain.agents import create_agent 
from langchain_core.agents import AgentAction
from langchain.agents.middleware import (
    PIIMiddleware,
    SummarizationMiddleware,
    HumanInTheLoopMiddleware
)
from langchain_core.prompts import PromptTemplate
from langchain.messages import HumanMessage, AIMessage, SystemMessage

from langgraph.checkpoint.memory import InMemorySaver
from uuid import uuid4

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AGENT.tools import FileOperationTools
from LLM.perplexity_llm import PerplexityAiLLM, LLMConfig
from RAG.notes_rag import RAGAssistant


class ReActAgent:
    def __init__(self, notes_dir: str):
        self.notes_dir = notes_dir
        self.llm_config = LLMConfig()
        self.llm = PerplexityAiLLM(config=self.llm_config)
        self.rag_assistant = RAGAssistant(
            notes_dir=notes_dir, 
            persist_dir="./vectorstorage"
        )

        self.tools = FileOperationTools(notes_dir=notes_dir)
        self.tools.rag_assistant = self.rag_assistant 
        
        self.checkpointer = InMemorySaver()

        self.thread_id = str(uuid4())

        self.agent = self._create_agent()

    def _create_agent(self):
        prompt = PromptTemplate(template="""
            Вы - интеллектуальный помощник для управления личными заметками.
            
            У вас есть полный набор инструментов для:
            1. Поиска и чтения существующих заметок (search_notes, read_note, list_notes)
            2. Создания новых заметок (create_note)
            3. Редактирования существующих заметок (update_note)
            4. Удаления заметок (delete_note)
            
            ВАЖНЫЕ ПРАВИЛА:
            - Перед изменением заметки всегда прочитайте её содержимое
            - При создании заметок используйте четкую структуру Markdown
            - Всегда подтверждайте успешное выполнение операций
            - Если пользователь просит отредактировать заметку, сначала найдите нужный файл
            
            Инструменты в вашем распоряжении:
            {tools}
            
            Используйте следующий формат:
            
            Question: входной вопрос
            Thought: о чём вы думаете
            Action: действие (одно из [{tool_names}])
            Action Input: входные данные
            Observation: результат
            ... (повторяйте если нужно)
            Thought: я знаю финальный ответ
            Final Answer: финальный ответ
            
            История разговора:
            {chat_history}
            
            Question: {input}
            {agent_scratchpad}
        """)

        middlewares = [
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            PIIMiddleware(
                "phone_number",
                detector=(
                    r"(?:\+?\d{1,3}[\s.-]?)?"
                    r"(?:\(?\d{2,4}\)?[\s.-]?)?"
                    r"\d{3,4}[\s.-]?\d{4}"
                ),
                strategy="block"
            ),
            SummarizationMiddleware(
                model="claude-sonnet-4-5-20250929",
                max_tokens_before_summary=500
            ),
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_email": {
                        "allowed_decisions": ["approve", "edit", "reject"]
                    }
                }
            ),
        ]

        agent = create_agent(
            model=self.llm,
            tools=self.tools.create_tools(),
            system_prompt=prompt,
            middleware=middlewares,
            checkpointer=self.checkpointer
        )

        return agent

    def answer(self, query: str):
        response = self.agent.invoke(
            HumanMessage(content=query),
            config={
                "configurable": {
                    "thread_id": self.thread_id,
                    "checkpoint_ns": "notes",  # использовать только такие имена!
                    "checkpoint_id": self.thread_id        # можно добавить, если хотите отличать сессии
                }
            }
        )

        return response.content
    
    def reset_memory(self):
        self.thread_id = str(uuid4())
        self.agent = self._create_agent()

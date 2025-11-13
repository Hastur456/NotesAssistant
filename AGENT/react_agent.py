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
from RAG.notes_rag import RAGAssistant
from LLM.amvera_llm import amvera_llm


class ReActAgent:
    def __init__(self, notes_dir: str, persist_dir: str = "./vectorstorage_test"):
        self.notes_dir = notes_dir
        self.llm = amvera_llm
        self.rag_assistant = RAGAssistant(
            notes_dir=notes_dir, 
            persist_dir=persist_dir
        )

        self.tools = FileOperationTools(notes_dir=notes_dir)
        self.tools.rag_assistant = self.rag_assistant 
        self.tools_functions = self.tools.create_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools_functions)
        
        self.checkpointer = InMemorySaver()

        self.thread_id = str(uuid4())

        self.agent = self._create_agent()

    def _create_agent(self):
        prompt = PromptTemplate(template="""
            Вы - ассистент для управления текстовыми заметками.

            ПРАВИЛА (следуйте ТОЧНО):
            1. Анализируйте текущий запрос пользователя.
            2. Определите, какое ОДНО действие требуется.
            3. Выполните ЭТО действие и ТОЛЬКО его.
            4. НИКОГДА не выполняйте другие действия.
            5. ПРЕКРАТИТЕ после первого действия.

            ДЕЙСТВИЯ И КОГДА ИХ ИСПОЛЬЗОВАТЬ:
            - create_note: Когда пользователь просит "создать", "добавить", "написать" новую заметку.
            Параметры: title (название), content (содержимое)
            
            - read_note: Когда пользователь просит "прочитать", "показать", "открыть" заметку по имени файла.
            Параметр: filename (имя файла, например: "Python Tips.md")
            
            - edit_note: Когда пользователь просит "отредактировать", "изменить", "обновить" существующую заметку.
            Параметры: filename, content, title
            
            - delete_note: Когда пользователь просит "удалить", "стереть" заметку.
            Параметр: filename
            
            - get_dir_structure: Когда пользователь просит "список", "все заметки", "структуру".
            Параметр: root_path

            ТЕКУЩИЙ ЗАПРОС: {input}

            На основе этого запроса выберите ОДНО действие:
            - Это запрос на СОЗДАНИЕ? → Используйте create_note
            - Это запрос на ЧТЕНИЕ? → Используйте read_note
            - Это запрос на РЕДАКТИРОВАНИЕ? → Используйте edit_note
            - Это запрос на УДАЛЕНИЕ? → Используйте delete_note
            - Это запрос на СПИСОК? → Используйте get_dir_structure

            Формат ответа (строго):

            Question: {input}
            Thought: Какое действие нужно выполнить и почему?
            Action: [ОДНО действие из списка выше]
            Action Input: [параметры в JSON]
            Observation: [результат выполнения]
            Final Answer: [ответ пользователю на основе результата]

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
            )
        ]

        agent = create_agent(
            model=self.llm_with_tools,
            tools=self.tools_functions,
            system_prompt=prompt.template,
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
                    "checkpoint_ns": "notes",
                    "checkpoint_id": self.thread_id
                }
            }
        )

        return response
    
    def reset_memory(self):
        self.thread_id = str(uuid4())
        self.agent = self._create_agent()

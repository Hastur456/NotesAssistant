from langchain.agents import create_agent 
from langchain_core.agents import AgentAction
from langchain_core.prompts import PromptTemplate

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AGENT.tools import FileOperationTools
from LLM.perplexity_llm import PerplexityAiLLM
from LLM.base import LLMConfig
from RAG.notes_rag import RAGAssistant


class ReActAgent:
    def __init__(self, notes_dir: str):
        self.notes_dir = notes_dir

        self.llm = PerplexityAiLLM(LLMConfig())
        self.tools = FileOperationTools(notes_dir=notes_dir).create_tools()
        self.rag_assistant = RAGAssistant(
            notes_dir=notes_dir, 
            persist_dir="./vectorstorage"
        )

        self._create_agent()

    def _create_agent(self):
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools
        )

    
    
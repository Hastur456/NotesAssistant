from langchain.agents import create_agent 
from langchain_core.agents import AgentAction
from langchain_core.prompts import PromptTemplate

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AGENT.tools import NotesManager


class ReActAgent:
    def __init__(self, notes_dir: str):
        self.notes_dir = notes_dir

    
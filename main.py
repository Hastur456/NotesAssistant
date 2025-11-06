import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
path_to_notes = os.getenv("NOTES_PATH")
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RAG.notes_rag import RAGAssistant
from LLM.perplexity_llm import PerplexityAiLLM, LLMConfig


def main():
    rag_assistant = RAGAssistant(path_to_notes)

    llm = PerplexityAiLLM(LLMConfig())

    

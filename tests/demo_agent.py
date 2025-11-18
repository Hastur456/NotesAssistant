# Простой тест драйв ReActAgent

import asyncio
from pathlib import Path
from langchain.messages import HumanMessage
import traceback

# Импортируем агента
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AGENT.react_agent import ReActAgent


def demo_basic_operations():
    """Демонстрация базовых операций агента"""
    
    # Инициализация
    print("=" * 60)
    print("ИНИЦИАЛИЗАЦИЯ АГЕНТА")
    print("=" * 60)
    
    notes_dir = "./tests/testnotes"
    Path(notes_dir).mkdir(exist_ok=True)
    
    agent = ReActAgent(notes_dir=notes_dir)
    # agent.rag_assistant.initial_indexing()
    print("✓ Агент успешно инициализирован\n")
    
    
    # Демо операции
    demo_queries = [
        "Используя инструмент просмотра файловой директории, который я тебе дал, напиши структуру папки",
        # "Создай заметку с названием 'Python Tips' и содержимым: 'List comprehension - мощный инструмент'",
        # "Найди все заметки содержащие слово 'Python'",
        # "Прочитай заметку 'Python Tips'",
        # "Обнови заметку 'Python Tips' добавив: 'Dictionary comprehension тоже полезен'",
        # "Список всех заметок"
    ]
    
    print("=" * 60)
    print("ВЫПОЛНЕНИЕ ОПЕРАЦИЙ")
    print("=" * 60 + "\n")
    
    for i, query in enumerate(demo_queries, 1):
        print(f"[{i}] Запрос: {query}")
        print("-" * 60)
        
        try:
            response = agent.answer(query)
            print(f"Ответ: {response}\n")
        except Exception:
            print(f"❌ Ошибка: {traceback.format_exc()}\n")
    
    
    # Сброс памяти
    print("=" * 60)
    print("СБРОС ПАМЯТИ")
    print("=" * 60)
    agent.reset_memory()
    print("✓ Память агента сброшена\n")
    
    
    print("=" * 60)
    print("✅ ТЕСТ ДРАЙВ ЗАВЕРШЕН")
    print("=" * 60)


if __name__ == "__main__":
    demo_basic_operations()
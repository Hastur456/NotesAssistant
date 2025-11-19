# demo_agent.py - ОБНОВЛЕН ДЛЯ create_agent

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from AGENT.react_agent import ReActAgent


def demo_basic_operations():
    """Demonstrate basic agent operations"""
    
    print("="*80)
    print("AGENT INITIALIZATION")
    print("="*80)
    
    notes_dir = "./tests/testnotes"
    Path(notes_dir).mkdir(parents=True, exist_ok=True)
    
    agent = ReActAgent(notes_dir=notes_dir, verbose=True)
    print(f"✓ Agent initialized (Thread ID: {agent.thread_id[:8]})\n")
    
    # Demo queries
    demo_queries = [
        "Покажи структуру папки",
        # "Создай заметку 'Test' с содержимым 'Hello World'",
        # "Все заметки",
    ]
    
    print("="*80)
    print("EXECUTING QUERIES")
    print("="*80)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n[{i}] Query: {query}")
        print("-"*80)
        
        try:
            response = agent.answer(query)
            print(f"Response: {response}\n")
        except Exception as e:
            print(f"✗ Error: {e}\n")
    
    # Show history
    print("\n" + "="*80)
    print("CONVERSATION HISTORY")
    print("="*80)
    
    history = agent.get_conversation_history()
    for msg in history:
        role = msg['role'].upper()
        content = msg['content'][:70]
        print(f"{role}: {content}...\n")
    
    # Reset
    print("="*80)
    print("RESETTING MEMORY")
    print("="*80)
    agent.reset_memory()
    print("✓ Memory cleared\n")
    
    print("="*80)
    print("✅ DEMO COMPLETE")
    print("="*80)


if __name__ == "__main__":
    demo_basic_operations()

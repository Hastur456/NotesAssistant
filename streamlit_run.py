import os
import sys
import subprocess
from pathlib import Path

def main():
    """Запустить Streamlit приложение"""
    
    # Получить путь к приложению
    app_path = Path(__file__).parent / "app.py"
    
    if not app_path.exists():
        print("❌ Файл app.py не найден!")
        sys.exit(1)
    
    print("""
    ╔════════════════════════════════════════╗
    ║   Note Assistant - Streamlit Version   ║
    ║            Starting...                 ║
    ╚════════════════════════════════════════╝
    
    📁 Пути:
    """)
    
    print(f"    📝 Notes: {os.getenv('NOTES_PATH', './notes')}")
    print(f"    🗄️  Vector Store: {os.getenv('VECTOR_STORE_PATH', './vectorstorage')}")
    print(f"    🤖 LLM Provider: {os.getenv('LLM_PROVIDER', 'не установлен')}")
    
    print("""
    🌐 Откройте в браузере:
    http://localhost:8501
    
    🛑 Для остановки нажмите Ctrl+C
    """)
    
    # Запустить Streamlit
    try:
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--logger.level=info"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Приложение остановлено")
        sys.exit(0)

if __name__ == "__main__":
    main()

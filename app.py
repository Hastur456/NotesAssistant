import streamlit as st
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import sys

# Загрузить переменные окружения
load_dotenv()

# Добавить пути для импортов
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импорт ваших модулей
try:
    from RAG.notes_rag import RAGAssistant
    from AGENT.react_agent import ReActAgent
    RAG_AVAILABLE = True
except Exception as e:
    st.warning(f"⚠️ RAG модули недоступны: {e}")
    RAG_AVAILABLE = False

# ============================================================================
# КОНФИГУРАЦИЯ STREAMLIT
# ============================================================================

st.set_page_config(
    page_title="Note Assistant",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомный CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        padding: 10px 20px;
    }
    .note-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ SESSION STATE
# ============================================================================

if "notes" not in st.session_state:
    st.session_state.notes = {}

if "rag_assistant" not in st.session_state and RAG_AVAILABLE:
    try:
        notes_path = os.getenv("NOTES_PATH", "./notes")
        vector_store_path = os.getenv("VECTOR_STORE_PATH", "./vectorstorage")
        st.session_state.rag_assistant = RAGAssistant(notes_path, vector_store_path)
    except Exception as e:
        st.session_state.rag_assistant = None

if "llm_assistant" not in st.session_state:
    try:
        api_key = os.getenv("OPENROUTER_API")
        notes_path = os.getenv("NOTES_PATH", "./notes")
        if api_key:
            st.session_state.llm_assistant = ReActAgent(
                notes_dir=notes_path
            )
        else:
            st.session_state.llm_assistant = None
    except Exception as e:
        st.session_state.llm_assistant = None

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def load_notes_from_file():
    """Загрузить заметки из файлов"""
    notes_path = Path(os.getenv("NOTES_PATH", "./notes"))
    notes = {}
    
    if notes_path.exists():
        for file_path in notes_path.glob("*.md"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                notes[file_path.stem] = content
            except Exception as e:
                st.error(f"Ошибка при чтении {file_path}: {e}")
    
    return notes

def save_note(note_id, content):
    """Сохранить заметку в файл"""
    notes_path = Path(os.getenv("NOTES_PATH", "./notes"))
    notes_path.mkdir(exist_ok=True)
    
    file_path = notes_path / f"{note_id}.md"
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        st.error(f"Ошибка при сохранении: {e}")
        return False

def delete_note(note_id):
    """Удалить заметку"""
    notes_path = Path(os.getenv("NOTES_PATH", "./notes"))
    file_path = notes_path / f"{note_id}.md"
    
    try:
        if file_path.exists():
            file_path.unlink()
            return True
    except Exception as e:
        st.error(f"Ошибка при удалении: {e}")
    return False

# ============================================================================
# ГЛАВНОЕ МЕНЮ
# ============================================================================

st.sidebar.title("📝 Note Assistant")
st.sidebar.markdown("---")

# Навигация
page = st.sidebar.radio(
    "Меню",
    ["📄 Мои заметки", "➕ Создать заметку", "🔍 Поиск", "🤖 AI Assistant", "ℹ️ О приложении"]
)

st.sidebar.markdown("---")

# Информация о системе
st.sidebar.subheader("⚙️ Система")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("RAG", "✅" if RAG_AVAILABLE else "❌")
with col2:
    st.metric("LLM", "✅" if st.session_state.llm_assistant else "❌")

# ============================================================================
# СТРАНИЦА 1: МОИ ЗАМЕТКИ
# ============================================================================

if page == "📄 Мои заметки":
    st.title("📝 Мои заметки")
    
    # Загрузить заметки
    notes = load_notes_from_file()
    
    if notes:
        st.success(f"✅ Загружено {len(notes)} заметок")
        
        # Выбрать заметку
        note_id = st.selectbox(
            "Выберите заметку для редактирования:",
            list(notes.keys()),
            key="note_select"
        )
        
        if note_id:
            st.subheader(f"📄 {note_id}")
            
            # Редактор
            updated_content = st.text_area(
                "Содержание заметки:",
                value=notes[note_id],
                height=300,
                key=f"edit_{note_id}"
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💾 Сохранить", key=f"save_{note_id}"):
                    if save_note(note_id, updated_content):
                        st.success("✅ Заметка сохранена!")
                        st.rerun()
            
            with col2:
                if st.button("📋 Копировать", key=f"copy_{note_id}"):
                    st.code(updated_content)
            
            with col3:
                if st.button("🗑️ Удалить", key=f"delete_{note_id}"):
                    if delete_note(note_id):
                        st.success("✅ Заметка удалена!")
                        st.rerun()
            
            # Информация о заметке
            st.divider()
            st.markdown(f"""
            **Информация:**
            - 📏 Размер: {len(updated_content)} символов
            - 📊 Слов: {len(updated_content.split())}
            - 📅 Обновлена: {datetime.now().strftime('%Y-%m-%d %H:%M')}
            """)
    else:
        st.info("📭 Нет заметок. Создайте первую!")

# ============================================================================
# СТРАНИЦА 2: СОЗДАТЬ ЗАМЕТКУ
# ============================================================================

elif page == "➕ Создать заметку":
    st.title("➕ Создать новую заметку")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        title = st.text_input("Название заметки:", placeholder="Моя первая заметка")
    
    with col2:
        tags = st.text_input("Теги (через запятую):", placeholder="tag1, tag2")
    
    content = st.text_area(
        "Содержание:",
        placeholder="Введите содержание заметки...",
        height=300
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Создать заметку", type="primary"):
            if title and content:
                note_id = f"{title.lower().replace(' ', '-')}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                # Добавить метаданные
                full_content = f"""# {title}

**Теги:** {tags}  
**Создана:** {datetime.now().isoformat()}

---

{content}
"""
                
                if save_note(note_id, full_content):
                    st.success("✅ Заметка создана!")
                    st.balloons()
            else:
                st.error("❌ Заполните название и содержание!")
    
    with col2:
        if st.button("🔄 Очистить форму"):
            st.rerun()

# ============================================================================
# СТРАНИЦА 3: ПОИСК
# ============================================================================

elif page == "🔍 Поиск":
    st.title("🔍 Поиск по заметкам")
    
    search_type = st.radio("Тип поиска:", ["Обычный поиск", "Семантический поиск (RAG)"])
    
    query = st.text_input(
        "Введите запрос:",
        placeholder="Что вы ищете?"
    )
    
    if st.button("🔎 Искать", type="primary"):
        if query:
            if search_type == "Обычный поиск":
                # Простой текстовый поиск
                notes = load_notes_from_file()
                results = []
                
                for note_id, content in notes.items():
                    if query.lower() in content.lower():
                        results.append({
                            "id": note_id,
                            "content": content,
                            "relevance": content.lower().count(query.lower())
                        })
                
                # Отсортировать по релевантности
                results.sort(key=lambda x: x['relevance'], reverse=True)
                
                if results:
                    st.success(f"✅ Найдено {len(results)} заметок")
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(f"📄 {i}. {result['id']}"):
                            st.markdown(result['content'][:500] + "...")
                            st.caption(f"Релевантность: {result['relevance']}")
                else:
                    st.info("📭 Ничего не найдено")
            
            elif search_type == "Семантический поиск (RAG)" and RAG_AVAILABLE:
                # RAG поиск
                if st.session_state.rag_assistant:
                    try:
                        with st.spinner("🔍 Ищу похожие заметки..."):
                            results = st.session_state.rag_assistant.query(query, k=5)
                            
                            if results and results.get('documents'):
                                docs = results['documents'][0]
                                st.success(f"✅ Найдено {len(docs)} релевантных документов")
                                
                                for i, doc in enumerate(docs, 1):
                                    with st.expander(f"📄 Результат {i}"):
                                        st.markdown(doc)
                            else:
                                st.info("📭 Ничего не найдено")
                    except Exception as e:
                        st.error(f"❌ Ошибка поиска: {e}")
                else:
                    st.warning("⚠️ RAG система недоступна")
        else:
            st.warning("⚠️ Введите запрос для поиска")

# ============================================================================
# СТРАНИЦА 4: AI ASSISTANT
# ============================================================================

elif page == "🤖 AI Assistant":
    st.title("🤖 AI Assistant")
    
    if st.session_state.llm_assistant:
        assistant_mode = st.radio(
            "Режим работы:",
            ["💬 Обычный вопрос", "📚 С контекстом из заметок", "📝 Анализ заметки"]
        )
        
        if assistant_mode == "💬 Обычный вопрос":
            question = st.text_area(
                "Ваш вопрос:",
                placeholder="Спросите что-нибудь...",
                height=100
            )
            
            if st.button("🚀 Получить ответ", type="primary"):
                if question:
                    with st.spinner("🤔 Думаю..."):
                        try:
                            response = st.session_state.llm_assistant.answer(question)
                            st.success("✅ Ответ готов!")
                            st.markdown(f"""
                            ### Ответ:
                            {response}
                            """)
                        except Exception as e:
                            st.error(f"❌ Ошибка: {e}")
        
        elif assistant_mode == "📚 С контекстом из заметок":
            question = st.text_area(
                "Ваш вопрос:",
                placeholder="Спросите что-нибудь про ваши заметки...",
                height=100
            )
            
            context_size = st.slider("Сколько заметок использовать как контекст:", 1, 10, 3)
            
            if st.button("🚀 Получить ответ", type="primary"):
                if question and RAG_AVAILABLE and st.session_state.rag_assistant:
                    with st.spinner("🔍 Ищу контекст..."):
                        try:
                            # Получить контекст
                            search_results = st.session_state.rag_assistant.query(question, k=context_size)
                            context = ""
                            
                            if search_results and search_results.get('documents'):
                                docs = search_results['documents'][0]
                                context = "\n\n".join([f"[Контекст {i+1}]: {doc}" for i, doc in enumerate(docs)])
                            
                            # Составить промпт с контекстом
                            full_prompt = f"""{context}

Вопрос: {question}

Ответ:"""
                            
                            response = st.session_state.llm_assistant.answer(full_prompt)
                            st.success("✅ Ответ готов!")
                            st.markdown(f"""
                            ### Ответ:
                            {response}
                            
                            ---
                            **Использован контекст из {len(docs)} документов**
                            """)
                        except Exception as e:
                            st.error(f"❌ Ошибка: {e}")
                else:
                    st.warning("⚠️ RAG система недоступна или контекст не найден")
        
        elif assistant_mode == "📝 Анализ заметки":
            notes = load_notes_from_file()
            
            if notes:
                note_id = st.selectbox("Выберите заметку для анализа:", list(notes.keys()))
                
                analysis_type = st.radio(
                    "Тип анализа:",
                    ["📌 Резюме", "🏷️ Ключевые слова", "❓ Вопросы", "📊 Структура"]
                )
                
                if st.button("🔍 Анализировать", type="primary"):
                    content = notes[note_id]
                    
                    prompts = {
                        "📌 Резюме": f"Сделай краткое резюме этого текста:\n\n{content}",
                        "🏷️ Ключевые слова": f"Извлеки ключевые слова из этого текста:\n\n{content}",
                        "❓ Вопросы": f"Сгенерируй 5 важных вопросов по этому тексту:\n\n{content}",
                        "📊 Структура": f"Создай структуру/план этого текста:\n\n{content}"
                    }
                    
                    with st.spinner("🤔 Анализирую..."):
                        try:
                            response = st.session_state.llm_assistant.answer(prompts[analysis_type])
                            st.success("✅ Анализ готов!")
                            st.markdown(response)
                        except Exception as e:
                            st.error(f"❌ Ошибка: {e}")
            else:
                st.info("📭 Нет заметок для анализа")
    else:
        st.error("❌ AI Assistant недоступен. Проверьте API ключи в .env")

# ============================================================================
# СТРАНИЦА 5: О ПРИЛОЖЕНИИ
# ============================================================================

elif page == "ℹ️ О приложении":
    st.title("ℹ️ О приложении")
    
    st.markdown("""
    # Note Assistant
    
    ## 📝 Описание
    Note Assistant - это мощный веб-интерфейс для управления заметками с поддержкой:
    - ✅ CRUD операций над заметками
    - ✅ Семантического поиска (RAG)
    - ✅ AI Assistant с LLM
    - ✅ Анализа заметок
    
    ## 🚀 Возможности
    
    ### 📄 Управление заметками
    - Создание новых заметок
    - Редактирование существующих
    - Удаление заметок
    - Просмотр информации о заметке
    
    ### 🔍 Поиск
    - Обычный текстовый поиск
    - Семантический поиск через RAG (если доступен)
    
    ### 🤖 AI Assistant
    - Ответы на вопросы
    - Поиск ответов с контекстом из ваших заметок
    - Анализ заметок (резюме, ключевые слова, вопросы, структура)
    
    ## 🛠️ Технологии
    - **Frontend**: Streamlit
    - **Backend**: FastAPI
    - **RAG**: Chroma + sentence-transformers
    - **LLM**: Perplexity AI / OpenAI
    
    ## 📊 Статистика
    """)
    
    notes = load_notes_from_file()
    total_chars = sum(len(content) for content in notes.values())
    total_words = sum(len(content.split()) for content in notes.values())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 Заметок", len(notes))
    with col2:
        st.metric("📝 Символов", total_chars)
    with col3:
        st.metric("📝 Слов", total_words)
    
    st.markdown("""
    ## 🔧 Конфигурация
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**LLM Provider:**", os.getenv("LLM_PROVIDER", "не указан"))
        st.write("**Model:**", os.getenv("MODEL_NAME", "не указан"))
        st.write("**Temperature:**", os.getenv("TEMPERATURE", "0.7"))
    
    with col2:
        st.write("**Notes Path:**", os.getenv("NOTES_PATH", "./notes"))
        st.write("**Vector Store:**", os.getenv("VECTOR_STORE_PATH", "./vectorstorage"))
        st.write("**Max Tokens:**", os.getenv("MAX_TOKENS", "2000"))
    
    st.markdown("""
    ## 📚 Документация
    - [GitHub](https://github.com/yourusername/note-assistant)
    - [API Docs](http://localhost:8000/docs)
    
    ## 📧 Контакты
    - Email: support@noteassistant.dev
    - Issues: https://github.com/yourusername/note-assistant/issues
    
    ---
    **Version:** 1.0.0  
    **Made with ❤️ for note management**
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    <p>📝 Note Assistant v1.0.0 | Powered by Streamlit</p>
    <p>Made with ❤️ for better note management</p>
</div>
""", unsafe_allow_html=True)

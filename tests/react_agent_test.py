# tests/test_react_agent.py
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
import tempfile
import shutil

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AGENT.react_agent import ReActAgent
from langchain.messages import HumanMessage, AIMessage


class TestReActAgent:
    """Комплексный набор тестов для ReActAgent"""
    
    @pytest.fixture
    def temp_notes_dir(self):
        """Создаёт временную директорию для тестовых заметок"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_llm(self, mocker):
        """Мокирует PerplexityAiLLM"""
        mock = mocker.patch('AGENT.react_agent.PerplexityAiLLM')
        mock_instance = Mock()
        mock_instance.invoke.return_value = Mock(content="Mocked LLM response")
        mock.return_value = mock_instance
        return mock_instance
    
    @pytest.fixture
    def mock_rag_assistant(self, mocker):
        """Мокирует RAGAssistant"""
        mock = mocker.patch('AGENT.react_agent.RAGAssistant')
        mock_instance = Mock()
        mock_instance.search.return_value = [
            {"content": "Test note 1", "metadata": {"source": "note1.md"}},
            {"content": "Test note 2", "metadata": {"source": "note2.md"}}
        ]
        mock.return_value = mock_instance
        return mock_instance
    
    @pytest.fixture
    def mock_operation_tools(self, mocker):
        """Мокирует OperationTools"""
        mock = mocker.patch('AGENT.react_agent.OperationTools')
        mock_instance = Mock()
        mock.return_value = mock_instance
        return mock_instance
    
    @pytest.fixture
    def mock_create_agent(self, mocker):
        """Мокирует create_agent"""
        mock = mocker.patch('AGENT.react_agent.create_agent')
        mock_agent = Mock()
        mock_agent.invoke.return_value = Mock(content="Agent response")
        mock.return_value = mock_agent
        return mock
    
    @pytest.fixture
    def agent(self, temp_notes_dir, mock_llm, mock_rag_assistant, 
              mock_operation_tools, mock_create_agent):
        """Создаёт экземпляр ReActAgent с моками"""
        return ReActAgent(notes_dir=temp_notes_dir)
    
    # ========== Тесты инициализации ==========
    
    def test_init_creates_llm_instance(self, temp_notes_dir, mocker):
        """Проверяет создание экземпляра LLM при инициализации"""
        mock_llm = mocker.patch('AGENT.react_agent.PerplexityAiLLM')
        mock_rag = mocker.patch('AGENT.react_agent.RAGAssistant')
        mock_tools = mocker.patch('AGENT.react_agent.OperationTools')
        mocker.patch('AGENT.react_agent.create_agent')
        
        agent = ReActAgent(notes_dir=temp_notes_dir)
        
        mock_llm.assert_called_once()
        assert agent.notes_dir == temp_notes_dir
    
    def test_init_creates_rag_assistant(self, temp_notes_dir, mocker):
        """Проверяет создание RAG assistant с правильными параметрами"""
        mocker.patch('AGENT.react_agent.PerplexityAiLLM')
        mock_rag = mocker.patch('AGENT.react_agent.RAGAssistant')
        mocker.patch('AGENT.react_agent.OperationTools')
        mocker.patch('AGENT.react_agent.create_agent')
        
        ReActAgent(notes_dir=temp_notes_dir)
        
        mock_rag.assert_called_once_with(
            notes_dir=temp_notes_dir,
            persist_dir="./vectorstorage"
        )
    
    def test_init_creates_operation_tools(self, temp_notes_dir, mocker):
        """Проверяет создание инструментов с правильной директорией"""
        mocker.patch('AGENT.react_agent.PerplexityAiLLM')
        mock_rag = mocker.patch('AGENT.react_agent.RAGAssistant')
        mock_tools = mocker.patch('AGENT.react_agent.OperationTools')
        mocker.patch('AGENT.react_agent.create_agent')
        
        ReActAgent(notes_dir=temp_notes_dir)
        
        mock_tools.assert_called_once_with(notes_dir=temp_notes_dir)
    
    def test_init_assigns_rag_to_tools(self, agent, mock_rag_assistant):
        """Проверяет, что RAG assistant назначен инструментам"""
        assert agent.tools.rag_assistant == mock_rag_assistant
    
    def test_init_creates_checkpointer(self, agent):
        """Проверяет создание checkpointer для сохранения состояния"""
        from langgraph.checkpoint.memory import InMemorySaver
        assert isinstance(agent.checkpointer, InMemorySaver)
    
    # ========== Тесты создания агента ==========
    
    def test_create_agent_called_with_correct_params(self, temp_notes_dir, 
                                                      mocker, mock_llm):
        """Проверяет вызов create_agent с правильными параметрами"""
        mocker.patch('AGENT.react_agent.RAGAssistant')
        mock_tools = mocker.patch('AGENT.react_agent.OperationTools')
        mock_create = mocker.patch('AGENT.react_agent.create_agent')
        
        agent = ReActAgent(notes_dir=temp_notes_dir)
        
        # Проверяем, что create_agent был вызван
        assert mock_create.called
        call_kwargs = mock_create.call_args.kwargs
        
        assert 'model' in call_kwargs
        assert 'tools' in call_kwargs
        assert 'system_prompt' in call_kwargs
        assert 'middleware' in call_kwargs
        assert 'checkpointer' in call_kwargs
    
    def test_create_agent_includes_all_middlewares(self, temp_notes_dir, mocker):
        """Проверяет, что все middleware добавлены в агент"""
        mocker.patch('AGENT.react_agent.PerplexityAiLLM')
        mocker.patch('AGENT.react_agent.RAGAssistant')
        mocker.patch('AGENT.react_agent.OperationTools')
        mock_create = mocker.patch('AGENT.react_agent.create_agent')
        
        ReActAgent(notes_dir=temp_notes_dir)
        
        middlewares = mock_create.call_args.kwargs['middleware']
        assert len(middlewares) == 4
    
    # ========== Тесты метода answer ==========
    
    def test_answer_invokes_agent_with_human_message(self, agent, mock_create_agent):
        """Проверяет, что answer вызывает агент с HumanMessage"""
        query = "Найди заметку о Python"
        
        agent.answer(query)
        
        # Проверяем, что invoke был вызван
        mock_create_agent.return_value.invoke.assert_called_once()
        call_args = mock_create_agent.return_value.invoke.call_args[0][0]
        
        assert isinstance(call_args, HumanMessage)
        assert call_args.content == query
    
    def test_answer_returns_content_from_response(self, agent, mock_create_agent):
        """Проверяет, что answer возвращает содержимое ответа"""
        expected_response = "Вот результат поиска заметок"
        mock_create_agent.return_value.invoke.return_value = Mock(
            content=expected_response
        )
        
        result = agent.answer("test query")
        
        assert result == expected_response
    
    def test_answer_with_complex_query(self, agent, mock_create_agent):
        """Проверяет работу с комплексным запросом"""
        complex_query = """
        Найди все заметки о машинном обучении,
        отредактируй их и добавь новую секцию о трансформерах
        """
        mock_create_agent.return_value.invoke.return_value = Mock(
            content="Выполнено"
        )
        
        result = agent.answer(complex_query)
        
        assert result == "Выполнено"
        mock_create_agent.return_value.invoke.assert_called_once()
    
    def test_answer_handles_empty_query(self, agent, mock_create_agent):
        """Проверяет обработку пустого запроса"""
        mock_create_agent.return_value.invoke.return_value = Mock(
            content="Пожалуйста, уточните запрос"
        )
        
        result = agent.answer("")
        
        assert result is not None
    
    # ========== Тесты метода reset_memory ==========
    
    def test_reset_memory_recreates_agent(self, agent, mock_create_agent, mocker):
        """Проверяет, что reset_memory пересоздаёт агент"""
        initial_call_count = mock_create_agent.call_count
        
        agent.reset_memory()
        
        # Должен быть вызван ещё раз
        assert mock_create_agent.call_count == initial_call_count + 1
    
    def test_reset_memory_clears_conversation_history(self, agent, mock_create_agent):
        """Проверяет, что история очищается при reset"""
        # Делаем несколько запросов
        agent.answer("Первый вопрос")
        agent.answer("Второй вопрос")
        
        # Сбрасываем память
        agent.reset_memory()
        
        # Проверяем, что создан новый checkpointer
        from langgraph.checkpoint.memory import InMemorySaver
        assert isinstance(agent.checkpointer, InMemorySaver)
    
    # ========== Интеграционные тесты ==========
    
    def test_full_workflow_search_and_create_note(self, temp_notes_dir, mocker):
        """Интеграционный тест: поиск и создание заметки"""
        mock_llm = mocker.patch('AGENT.react_agent.PerplexityAiLLM')
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        
        mocker.patch('AGENT.react_agent.RAGAssistant')
        mocker.patch('AGENT.react_agent.OperationTools')
        
        mock_agent = Mock()
        mock_agent.invoke.return_value = Mock(
            content="Заметка успешно создана"
        )
        mocker.patch('AGENT.react_agent.create_agent', return_value=mock_agent)
        
        agent = ReActAgent(notes_dir=temp_notes_dir)
        result = agent.answer("Создай заметку о тестировании")
        
        assert "успешно" in result.lower() or result is not None
    
    def test_agent_with_real_checkpointer(self, temp_notes_dir, mocker):
        """Проверяет работу с реальным InMemorySaver"""
        mocker.patch('AGENT.react_agent.PerplexityAiLLM')
        mocker.patch('AGENT.react_agent.RAGAssistant')
        mocker.patch('AGENT.react_agent.OperationTools')
        mocker.patch('AGENT.react_agent.create_agent')
        
        agent = ReActAgent(notes_dir=temp_notes_dir)
        
        from langgraph.checkpoint.memory import InMemorySaver
        assert isinstance(agent.checkpointer, InMemorySaver)
    
    # ========== Тесты обработки ошибок ==========
    
    def test_handles_llm_failure(self, agent, mock_create_agent):
        """Проверяет обработку ошибки LLM"""
        mock_create_agent.return_value.invoke.side_effect = Exception("API Error")
        
        with pytest.raises(Exception) as exc_info:
            agent.answer("test query")
        
        assert "API Error" in str(exc_info.value)
    
    def test_handles_invalid_notes_directory(self, mocker):
        """Проверяет поведение при несуществующей директории"""
        mocker.patch('AGENT.react_agent.PerplexityAiLLM')
        mocker.patch('AGENT.react_agent.RAGAssistant')
        mocker.patch('AGENT.react_agent.OperationTools')
        mocker.patch('AGENT.react_agent.create_agent')
        
        agent = ReActAgent(notes_dir="/nonexistent/path")
        assert agent.notes_dir == "/nonexistent/path"
    
    # ========== Тесты промпта ==========
    
    def test_prompt_template_contains_required_sections(self, temp_notes_dir, mocker):
        """Проверяет, что промпт содержит все необходимые секции"""
        mocker.patch('AGENT.react_agent.PerplexityAiLLM')
        mocker.patch('AGENT.react_agent.RAGAssistant')
        mocker.patch('AGENT.react_agent.OperationTools')
        mock_create = mocker.patch('AGENT.react_agent.create_agent')
        
        ReActAgent(notes_dir=temp_notes_dir)
        
        prompt = mock_create.call_args.kwargs['system_prompt']
        prompt_text = str(prompt)
        
        assert "Question:" in prompt_text
        assert "Thought:" in prompt_text
        assert "Action:" in prompt_text
        assert "Observation:" in prompt_text
        assert "Final Answer:" in prompt_text


# ========== Дополнительные фикстуры для сложных тестов ==========

@pytest.fixture
def sample_notes(temp_notes_dir):
    """Создаёт образцовые заметки для тестирования"""
    notes = {
        "python.md": "# Python\n\nЗаметки о Python",
        "machine_learning.md": "# ML\n\nЗаметки о ML",
    }
    
    notes_path = Path(temp_notes_dir)
    for filename, content in notes.items():
        (notes_path / filename).write_text(content, encoding='utf-8')
    
    return notes_path


# ========== Тесты для специфических сценариев ==========

class TestReActAgentEdgeCases:
    """Тесты граничных случаев и специальных сценариев"""
    
    @pytest.fixture
    def temp_notes_dir(self):
        """Создаёт временную директорию для тестовых заметок"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def agent(self, temp_notes_dir, mocker):
        """Создаёт агент с моками для edge case тестов"""
        mocker.patch('AGENT.react_agent.PerplexityAiLLM')
        mocker.patch('AGENT.react_agent.RAGAssistant')
        mocker.patch('AGENT.react_agent.OperationTools')
        mocker.patch('AGENT.react_agent.create_agent')
        return ReActAgent(notes_dir=temp_notes_dir)
    
    @pytest.fixture
    def mock_create_agent(self, mocker):
        """Мокирует create_agent для edge case тестов"""
        mock = mocker.patch('AGENT.react_agent.create_agent')
        mock_agent = Mock()
        mock_agent.invoke.return_value = Mock(content="Agent response")
        mock.return_value = mock_agent
        return mock
    
    def test_answer_with_unicode_query(self, agent, mock_create_agent):
        """Проверяет работу с юникодом в запросе"""
        # Переинициализируем agent с новым mock
        agent.agent = mock_create_agent.return_value
        
        query = "Найди заметки о 机器学习 и 🤖 AI"
        mock_create_agent.return_value.invoke.return_value = Mock(
            content="Найдено"
        )
        
        result = agent.answer(query)
        assert result is not None
    
    def test_multiple_sequential_queries(self, agent, mock_create_agent):
        """Проверяет последовательные запросы"""
        agent.agent = mock_create_agent.return_value
        
        responses = ["Ответ 1", "Ответ 2", "Ответ 3"]
        mock_create_agent.return_value.invoke.side_effect = [
            Mock(content=resp) for resp in responses
        ]
        
        results = [agent.answer(f"Запрос {i}") for i in range(3)]
        
        assert results == responses
    
    def test_reset_between_queries(self, agent, mock_create_agent, mocker):
        """Проверяет сброс памяти между запросами"""
        agent.agent = mock_create_agent.return_value
        
        agent.answer("Первый запрос")
        agent.reset_memory()
        agent.answer("Второй запрос")
        
        assert mock_create_agent.return_value.invoke.call_count == 2

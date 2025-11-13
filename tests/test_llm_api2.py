from langchain_amvera import AmveraLLM
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import json
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AGENT.file_manager.notes_manager import NotesManager

load_dotenv()

# Определяем схемы аргументов
class ReadNoteArgs(BaseModel):
    filename: str = Field(description="Имя файла для чтения")

class CreateNoteArgs(BaseModel):
    title: str = Field(description="Заголовок заметки")
    content: str = Field(description="Содержимое заметки")

class EditNoteArgs(BaseModel):
    filename: str = Field(description="Имя файла для редактирования")
    content: str = Field(description="Новое содержимое")
    title: str = Field(description="Новый заголовок")

class DeleteNoteArgs(BaseModel):
    filename: str = Field(description="Имя файла для удаления")

class DirStructureArgs(BaseModel):
    root_path: str = Field(description="Корневой путь для получения структуры")

# Создаём инструменты
notes_manager = NotesManager(notes_directory="./tests/testnotes")

@tool("read_note", args_schema=ReadNoteArgs)
def read_note(filename: str):
    """Read the content of a note file by filename."""
    result = notes_manager.read_note(filename)
    return json.dumps(result, ensure_ascii=False)

@tool("create_note", args_schema=CreateNoteArgs)
def create_note(title: str, content: str):
    """Create a new note with the given title and content."""
    result = notes_manager.create_note(title, content)
    return json.dumps(result, ensure_ascii=False)

@tool("edit_note", args_schema=EditNoteArgs)
def edit_note(filename: str, content: str, title: str):
    """Edit an existing note by updating its content and title."""
    result = notes_manager.edit_note(filename, content, title)
    return json.dumps(result, ensure_ascii=False)

@tool("delete_note", args_schema=DeleteNoteArgs)
def delete_note(filename: str):
    """Delete a note file by filename."""
    result = notes_manager.delete_note(filename)
    return json.dumps(result, ensure_ascii=False)

@tool("get_dir_structure", args_schema=DirStructureArgs)
def get_dir_structure(root_path: str):
    """Get the directory structure starting from the root path."""
    result = notes_manager.get_dir_structure(root_path)
    return json.dumps(result, ensure_ascii=False)

# Создаём список инструментов
tools_list = [
    read_note,
    create_note,
    edit_note,
    delete_note,
    get_dir_structure,
]

# Инициализируем модель с инструментами
amvera_llm = AmveraLLM(
    model="llama70b", 
    api_token=os.getenv("AMVERA_API_TOKEN"), 
    temperature=0.4,
    max_tokens=1000,
    tools=tools_list  # ← Передаём инструменты при инициализации
)

# Используем модель
response = amvera_llm.invoke([HumanMessage(content="Выведи структуру директории.")])
print(response)

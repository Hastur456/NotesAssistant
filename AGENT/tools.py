
import json 
import os
import sys
from langchain.tools import tool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AGENT.file_manager.notes_manager import NotesManager


class FileOperationTools:
    def __init__(self, notes_dir: str):
        self.notes_dir = notes_dir
        self.notes_manager = NotesManager(notes_directory=notes_dir)

    def create_tools(self):
        @tool
        def read_note(filename):
            result = self.notes_manager.read_note(filename)
            return json.dumps(result, ensure_ascii=False)

        @tool
        def create_note(title: str, content: str):
            result = self.notes_manager.create_note(title, content)
            return json.dumps(result, ensure_ascii=False)
        
        @tool
        def edit_note(filename: str, content: str, title: str):
            result = self.notes_manager.edit_note(filename, content, title)
            return json.dumps(result, ensure_ascii=False)
        
        @tool
        def delete_note(filename: str):
            result = self.notes_manager.delete_note(filename)
            return json.dumps(result, ensure_ascii=False)
        
        @tool
        def get_dir_structure(root_path):
            result = self.notes_manager.get_dir_structure(root_path)
            return json.dumps(result, ensure_ascii=False)

        tools = [
            read_note,
            create_note,
            edit_note,
            delete_note,
            get_dir_structure, 
        ] 

        return tools

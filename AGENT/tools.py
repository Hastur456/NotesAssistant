
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
        self.rag_assistant = None

    def create_tools(self):
        @tool
        def read_note(filename: str):
            """Read the content of a note file by filename."""
            result = self.notes_manager.read_note(filename)
            return json.dumps(result, ensure_ascii=False)

        @tool
        def create_note(title: str, content: str):
            """Create a new note with the given title and content."""
            result = self.notes_manager.create_note(title, content)
            return json.dumps(result, ensure_ascii=False)

        @tool
        def edit_note(filename: str, content: str, title: str):
            """Edit an existing note by updating its content and title."""
            result = self.notes_manager.edit_note(filename, content, title)
            return json.dumps(result, ensure_ascii=False)

        @tool
        def delete_note(filename: str):
            """Delete a note file by filename."""
            result = self.notes_manager.delete_note(filename)
            return json.dumps(result, ensure_ascii=False)

        @tool
        def get_dir_structure(root_path: str):
            """Get the directory structure starting from the root path."""
            result = self.notes_manager.get_dir_structure(root_path)
            return json.dumps(result, ensure_ascii=False)

        @tool
        def search_notes(query: str, k: int = 5):
            """Search for notes using RAG system with the given query and return top k results."""
            if self.rag_assistant is None:
                return json.dumps({"error": "RAG система не инициализирована"}, ensure_ascii=False)

            try:
                results = self.rag_assistant.query(query, k=k)
                documents = results.get('documents', [[]])[0]
                formatted_results = []
                for i, doc in enumerate(documents):
                    formatted_results.append({
                        "rank": i + 1,
                        "content": doc
                    })

                return json.dumps({
                    "status": "success",
                    "query": query,
                    "results_count": len(formatted_results),
                    "results": formatted_results
                }, ensure_ascii=False)

            except Exception as e:
                return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

        tools = [
            read_note,
            create_note,
            edit_note,
            delete_note,
            get_dir_structure,
            search_notes
        ]

        return tools

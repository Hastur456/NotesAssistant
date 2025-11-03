from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
<<<<<<< HEAD:components/documents_processor.py
=======
from pathlib import Path
from RAG.logging_config import logger
>>>>>>> d15280e (Creating new project structure):RAG/components/documents_processor.py


class DocumentsProcessor():
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_documents(self, notes_path):
        loader = DirectoryLoader(
            notes_path,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader
        )

        documents = loader.load()
        
        return documents 
    
    def process_documents(self, notes_path):
        documents = self.load_documents(notes_path)
        chunks = self.text_splitter.split_documents(documents)

        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx
            chunk.metadata["file_path"] = chunk.metadata.get("source", "")
            chunk.metadata["paragraph_number"] = idx

        return chunks

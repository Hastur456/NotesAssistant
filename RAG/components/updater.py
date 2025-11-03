import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from components.documents_processor import DocumentsProcessor
from components.vectorstore import ChromaVectorStore
from components.embedding_model import EmbeddingModel
<<<<<<< HEAD:components/updater.py
=======
from RAG.logging_config import logger
>>>>>>> d15280e (Creating new project structure):RAG/components/updater.py


class IncrementalHandler():
    def __init__(self, vectorstore: ChromaVectorStore, embedding_model: EmbeddingModel, processor: DocumentsProcessor):
        self.vecotorstore = vectorstore
        self.embedding_model = embedding_model
        self.processor = processor

    def update_handler(self, filepath, event_type):
        if event_type == "delete":
            self.vecotorstore.delete_by_source(filepath)

        elif event_type in ["create", "modifed"]:
            self.vecotorstore.delete_by_source(filepath)

            chunks = self.processor.process_documents(filepath)

            texts = [chunk.page_content for chunk in chunks]
            embeddings = self.embedding_model.embed_documents(texts)

            self.vecotorstore.add_documents(chunks, embeddings)

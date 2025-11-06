import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RAG.components.documents_processor import DocumentsProcessor
from RAG.components.vectorstorage import ChromaVectorStorage
from RAG.components.embedding_model import EmbeddingModel
from RAG.logging_config import logger


class IncrementalHandler():
    def __init__(self, vectorstorage: ChromaVectorStorage, embedding_model: EmbeddingModel, processor: DocumentsProcessor):
        self.vectorstorage = vectorstorage
        self.embedding_model = embedding_model
        self.processor = processor

    def update_handler(self, filepath, event_type):
        if event_type == "delete":
            self.vectorstorage.delete_by_source(filepath)

        elif event_type in ["create", "modifed"]:
            self.vectorstorage.delete_by_source(filepath)

            chunks = self.processor.process_documents(filepath)

            texts = [chunk.page_content for chunk in chunks]
            embeddings = self.embedding_model.embed_documents(texts)

            self.vectorstorage.add_documents(chunks, embeddings)

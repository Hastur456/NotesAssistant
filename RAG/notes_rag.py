import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
path_to_notes = os.getenv("NOTES_PATH")
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from components.documents_processor import DocumentsProcessor
from components.vectorstorage import ChromaVectorStorage
from components.embedding_model import EmbeddingModel
from components.notes_handler import start_monitoring
from components.updater import IncrementalHandler
from RAG.logging_config import logger


class RAGAssistant():
    def __init__(self, notes_dir, persist_dir="./vectorstorage"):
        self.notes_dir = notes_dir

        self.documents_processor = DocumentsProcessor()
        self.vectorstorage = ChromaVectorStorage(persist_directory=persist_dir)
        self.embedding_model = EmbeddingModel()

        self.updater = IncrementalHandler(
            vectorstorage=self.vectorstorage,
            embedding_model=self.embedding_model,
            processor=self.documents_processor
        )

    def initial_indexing(self):
        logger.info(f"Начало индексации директории...")

        try:
            chunks = self.documents_processor.process_documents(self.notes_dir)
            texts = [chunk.page_content for chunk in chunks]

            embeddings = self.embedding_model.embed_documents(texts)

            self.vectorstorage.add_documents(chunks=chunks, embeddings=embeddings)

            logger.info(f"Индексация выполнена успешно...")

        except Exception as e:
            logger.error(f"Ошибка при индексации директории {self.notes_dir}: {str(e)}")
            raise
    
    def query(self, query, k=5):
        embedding = self.embedding_model.embed_query(query)

        results = self.vectorstorage.search(embedding, k=k)

        return results
    
    def start_monitoring(self):
        logger.info(f"Начало мониторинга директории с заметками: {self.notes_dir}")
        
        try:
            def update_callback(filepath, event): 
                self.updater.update_handler(filepath, event)

            start_monitoring(str(self.notes_dir), update_callback)

        except KeyboardInterrupt:
            logger.debug(f"Завершение мониторинга директории: {self.notes_dir}")
            raise

        except Exception as e:
            logger.error(f"Ошибка при мониторинге директории: {e}")
            raise


if __name__ == "__main__":
    persist_directory = "./vectorstorage"

    assistant = RAGAssistant(
        path_to_notes,
        persist_dir=persist_directory
    )

    # if not os.path.isdir(persist_directory):
    #     logger.debug(f"Директория {persist_directory}")
    # assistant.initial_indexing()
    
    results = assistant.query("Решение задач с leetcode?", k=3)
    print("\nРезультаты поиска:")
    for i, result in enumerate(results['documents'][0], 1):
        print(f"\n{i}. {result[:200]}...")
    
    assistant.start_monitoring()

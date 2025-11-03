import chromadb
from chromadb.config import Settings
from pathlib import Path

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RAG.logging_config import logger


class ChromaVectorStorage:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = Path(persist_directory)
        
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Инициализация Chroma в директории: {self.persist_directory}")
        
        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
            logger.info("✓ Chroma client успешно инициализирован")
        except Exception as e:
            logger.error(f"✗ Ошибка при инициализации Chroma: {e}")
            raise
        
        try:
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"✓ Коллекция получена/создана. Документов в базе: {self.collection.count()}")
        except Exception as e:
            logger.error(f"✗ Ошибка при создании коллекции: {e}")
            raise
    
    def add_documents(self, chunks, embeddings):
        logger.info(f"Начало добавления {len(chunks)} документов")
        
        try:
            ids = [f"doc_{i}" for i in range(len(chunks))]
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"✓ Успешно добавлено {len(chunks)} документов")
            logger.info(f"✓ Всего документов в базе: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"✗ Ошибка при добавлении документов: {e}")
            raise
    
    def search(self, query_embedding, k=5):
        logger.debug(f"Поиск {k} релевантных документов")
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            logger.debug(f"✓ Найдено {len(results['documents'][0])} результатов")
            return results
        except Exception as e:
            logger.error(f"✗ Ошибка при поиске: {e}")
            raise
    
    def delete_by_source(self, filepath):
        logger.info(f"Удаление документов из файла: {filepath}")
        
        try:
            results = self.collection.get(
                where={"file_path": filepath}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"✓ Удалено {len(results['ids'])} документов")
            else:
                logger.info(f"⚠ Документы из {filepath} не найдены")
                
        except Exception as e:
            logger.error(f"✗ Ошибка при удалении: {e}")
            raise

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from RAG.logging_config import logger


class DocumentsProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        logger.debug(f"Инициализация DocumentsProcessor с chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.LOADERS = {
            '.md': (UnstructuredMarkdownLoader, {'mode': 'single'}),
            '.txt': (TextLoader, {'encoding': 'utf-8'})
        }
        
        logger.info("DocumentsProcessor успешно инициализирован")
    
    def get_loader(self, filepath):
        ext = Path(filepath).suffix.lower()
        logger.debug(f"Определение загрузчика для расширения: {ext}")
        
        if ext not in self.LOADERS:
            logger.error(f"Данный формат файла не поддерживается: {filepath}")
            raise ValueError(f"Unsupported file format: {ext}")
        
        loader_class, kwargs = self.LOADERS[ext]
        logger.debug(f"Загрузчик найден: {loader_class.__name__}")
        
        return loader_class(filepath, **kwargs)
    
    def load_documents(self, notes_path):
        logger.info(f"Начало загрузки документов из директории: {notes_path}")
        
        try:
            loader = DirectoryLoader(
                notes_path,
                glob="**/*.md",
                loader_cls=UnstructuredMarkdownLoader,
                show_progress=True
            )
            
            documents = loader.load()
            logger.info(f"✓ Успешно загружено {len(documents)} документов из {notes_path}")
            logger.debug(f"Загруженные документы: {[doc.metadata.get('source', 'unknown') for doc in documents]}")
            
            return documents
        
        except FileNotFoundError:
            logger.error(f"Директория не найдена: {notes_path}")
            raise
        except Exception as e:
            logger.error(f"Ошибка при загрузке документов из {notes_path}: {str(e)}")
            raise
    
    def load_document(self, filepath):
        logger.info(f"Начало загрузки документа: {filepath}")
        
        try:
            loader = self.get_loader(filepath)
            documents = loader.load()
            
            logger.info(f"✓ Документ успешно загружен: {filepath}")
            logger.debug(f"Количество документов: {len(documents)}")
            
            return documents
        
        except FileNotFoundError:
            logger.error(f"Файл не найден: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Ошибка при загрузке документа {filepath}: {str(e)}")
            raise
    
    def processing_chunks_metadata(self, chunks):
        logger.debug(f"Начало обогащения метаданных для {len(chunks)} чанков")
        
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx
            chunk.metadata["file_path"] = chunk.metadata.get("source", "")
            chunk.metadata["paragraph_number"] = idx
        
        logger.debug(f"✓ Метаданные добавлены ко всем {len(chunks)} чанкам")
        
        return chunks
    
    def documents_processor(self, notes_path):
        logger.info(f"Начало обработки документов из директории: {notes_path}")
        
        try:
            documents = self.load_documents(notes_path)
            logger.debug(f"Количество загруженных документов: {len(documents)}")
            
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"✓ Документы разбиты на {len(chunks)} чанков")
            
            processed_chunks = self.processing_chunks_metadata(chunks)
            logger.info(f"✓ Обработка директории {notes_path} завершена успешно")
            
            return processed_chunks
        
        except Exception as e:
            logger.error(f"Ошибка при обработке директории {notes_path}: {str(e)}")
            raise
    
    def document_processor(self, filepath):
        logger.info(f"Начало обработки документа: {filepath}")
        
        try:
            documents = self.load_document(filepath)
            logger.debug(f"Количество загруженных документов: {len(documents)}")
            
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"✓ Документ разбит на {len(chunks)} чанков")
            
            processed_chunks = self.processing_chunks_metadata(chunks)
            logger.info(f"✓ Обработка документа {filepath} завершена успешно")
            
            return processed_chunks
        
        except Exception as e:
            logger.error(f"Ошибка при обработке документа {filepath}: {str(e)}")
            raise

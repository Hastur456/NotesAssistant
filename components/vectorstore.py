from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.basicConfig(filename='debug.log', level=logging.DEBUG, filemode='w',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ChromaVectorStore():
    def __init__(self, persist_dir="./chroma_db"):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        ))

        self.collection = self.client.get_or_create_collection(
            name="notes_collection",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, chunks, embeddings):
        ids = [f"{chunk.metadata["source"]}_{i}" for i, chunk in enumerate(chunks)]
        documents = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        self.collection.upsert(
            ids=ids, 
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        logging.info("Добавлены данные ids: {}".format(ids))
    
    def delete_by_source(self, filepath):
        results = self.collection.get(
            where={"source": filepath}
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])

        logging.info("Удалены данные ids: {}".format(results["ids"]))
        

    def search(self, query, k=5):
        results = self.collection.query(
            query_embeddings=[query.tolist()],
            n_results=k
        )

        return results
    
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

import os
from dotenv import load_dotenv


load_dotenv()
path_to_notes = os.getenv("NOTES_PATH")

loader = DirectoryLoader(
    path_to_notes,
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(documents)

for idx, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = idx
    chunk.metadata["file_path"] = chunk.metadata.get("source", "")
    chunk.metadata["paragraph_number"] = idx

# Локальная модель эмбеддингов через Ollama
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Создание индекса
vectorstore = FAISS.from_documents(chunks, embeddings)

# Сохранение индекса
vectorstore.save_local("faiss_index")

# Загрузка индекса
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

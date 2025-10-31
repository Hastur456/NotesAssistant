from langchain_community.embeddings import OllamaEmbeddings


class EmbeddingModel():
    def __init__(self, model="evilfreelancer/enbeddrus"):
        self.model = model
        self.embedding_model = OllamaEmbeddings(
            model=self.model,
            show_progress=True
        )

    def embed_documents(self, documents):   
        return self.embedding_model.embed_documents(documents)
    
    def embed_query(self, query):
        return self.embedding_model.embed_query(query)
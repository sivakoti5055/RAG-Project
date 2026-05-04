from typing import List,Any
from sentence_transformers import SentenceTransformer
import os
import numpy as np
import faiss
import pickle
from src.embeddings import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self,presist_dir: str = "faiss_store",embedding_model = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.presist_dir = presist_dir
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.metadata = []
        os.makedirs(presist_dir,exist_ok=True)
        print(f"Loaded Embedding Model:{embedding_model}")

    def build_from_documents(self,documents:List[Any]):
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunk(chunks)
        metadatas= [{'text':chunk.page_content} for chunk in chunks]
        self.add_embedding(np.array(embeddings).astype('float32'),metadatas)
        self.save()

    def add_embedding(self,embeddings:np.ndarray,metadatas:List[Any]):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self):
        faiss_path = os.path.join(self.presist_dir,'faiss.index')
        meta_path = os.path.join(self.presist_dir,'metadata.pkl')
        faiss.write_index(self.index,faiss_path)
        with open(meta_path,'wb') as f:
            pickle.dump(self.metadata,f)
        print(f"[INFO] Saved Faiss index and metadata to {self.presist_dir}")

    def load(self):
        faiss_path = os.path.join(self.presist_dir,'faiss.index')
        meta_path = os.path.join(self.presist_dir,'metadata.pkl')
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index and metadata from {self.presist_dir}")

    def search(self,embedding_query:np.ndarray,top_k = 3):
        D,I = self.index.search(embedding_query,top_k)
        results = []
        for idx,dist in zip(I[0],D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index":idx,"distance":dist,"metadata":meta})
            print(f"Results Appended")
        return results
    
    def query(self,query_text,top_k=3):
        query_emb = self.model.encode([query_text]).astype('float32')
        print(f"print query as :{query_text}")
        return self.search(query_emb,top_k)




    
                                           


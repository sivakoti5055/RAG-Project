from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np


from typing import List, Any

class EmbeddingPipeline:
    def __init__(self,model_name='all-MiniLM-L6-v2',chunk_size=1000,chunk_overlap=200):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self,documents:list[Any])->List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(documents)
        print(f'Split {len(documents)} documents in to {len(chunks)} chunks')

        return chunks
    
    def embed_chunk(self,chunks:List[Any])-> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.model.encode(texts,show_progress_bar=True)
        return embeddings




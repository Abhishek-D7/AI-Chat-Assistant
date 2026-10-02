"""
app/ml/embeddings.py
Embedding generation using Hugging Face Inference API and SentenceTransformers
"""

import logging
from typing import List, Union, Optional
import numpy as np
from huggingface_hub import InferenceClient
from app.config import Config

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingClient:
    """
    Generates text embeddings using Hugging Face Inference API.
    Defaults to BAAI/bge-large-en-v1.5 (1024-dimensional embeddings) to match Pinecone.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        token: Optional[str] = None
    ):
        self.model_name = model_name
        self.token = token or Config.HF_TOKEN
        self.dimension = 1024
        
        if not self.token:
            logger.warning("⚠️ HF_TOKEN not configured. Hugging Face Inference API calls may fail.")
            
        self.client = InferenceClient(token=self.token)

    def embed_query(self, text: str) -> List[float]:
        """
        Encode a single query string to a 1024-dimensional embedding vector.
        
        Args:
            text: Query text string
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            res = self.client.feature_extraction(text, model=self.model_name)
            if hasattr(res, "tolist"):
                vec = res.tolist()
            else:
                vec = list(res)
            
            # If batch output shape (1, 1024)
            if vec and isinstance(vec[0], list):
                vec = vec[0]
                
            return [float(x) for x in vec]
        except Exception as e:
            logger.error(f"❌ Failed to generate embedding for query: {e}", exc_info=True)
            raise e

    def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """
        Encode a list of text chunks in batches.
        
        Args:
            texts: List of document text chunks
            batch_size: Number of chunks per batch
            
        Returns:
            List of embedding vectors (each 1024-dim)
        """
        all_embeddings: List[List[float]] = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                res = self.client.feature_extraction(batch, model=self.model_name)
                if hasattr(res, "tolist"):
                    batch_vecs = res.tolist()
                else:
                    batch_vecs = [list(v) for v in res]
                
                for vec in batch_vecs:
                    all_embeddings.append([float(x) for x in vec])
            except Exception as e:
                logger.warning(f"⚠️ Batch feature extraction failed, falling back to individual items: {e}")
                for single_text in batch:
                    all_embeddings.append(self.embed_query(single_text))
                    
        return all_embeddings


class EmbeddingGenerator:
    """Generates embeddings for text using local SentenceTransformers (optional fallback)"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = 384
    
    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=normalize, show_progress_bar=False)
    
    def encode_query(self, query: str) -> np.ndarray:
        return self.encode(query, normalize=True)[0]
    
    def batch_encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

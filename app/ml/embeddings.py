"""
app/ml/embeddings.py
Embedding generation using SentenceTransformers
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union


class EmbeddingGenerator:
    """Generates embeddings for text using SentenceTransformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of the SentenceTransformer model
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # all-MiniLM-L6-v2 outputs 384-dim embeddings
    
    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Encode text(s) to embeddings
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to normalize embeddings
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query to embedding
        
        Args:
            query: Query text
            
        Returns:
            1D numpy array of embedding
        """
        return self.encode(query, normalize=True)[0]
    
    def batch_encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts in batches
        
        Args:
            texts: List of texts
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )

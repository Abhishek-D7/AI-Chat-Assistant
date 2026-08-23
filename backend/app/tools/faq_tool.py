"""
app/tools/faq_tool.py  
FAQ Agent Tool - Retrieves answers from FAISS vector database
"""

from langchain.tools import tool
import faiss
import pickle
import random
from sentence_transformers import SentenceTransformer
from typing import List, Dict


class FAQRetriever:
    """FAISS-based FAQ retrieval system"""
    
    def __init__(self):
        self.faiss_index_path = "vector.faiss"
        self.metadata_path = "vector.pkl"
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.metadata = []
        self._load()
    
    def _load(self):
        """Load FAISS index and metadata"""
        try:
            self.index = faiss.read_index(self.faiss_index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"✅ Loaded FAQ database: {len(self.metadata)} entries")
        except FileNotFoundError:
            print("⚠️ FAQ database not found. Using empty database.")
            self.metadata = []
    
    def search(self, query: str, top_k: int = 1) -> List[Dict]:
        """Search FAQ using semantic similarity"""
        if not self.index or not self.metadata:
            return []
        
        # Generate embedding
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Search FAISS
        similarities, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k
        )
        
        # Format results
        results = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if idx < len(self.metadata) and similarity > 0.3:
                results.append({
                    "question": self.metadata[idx]["question"],
                    "answer": self.metadata[idx]["answer"],
                    "similarity": float(similarity)
                })
        
        return results
    
    def get_random_faqs(self, k: int = 5) -> List[Dict]:
        """Get random FAQ questions"""
        if not self.metadata:
            return []
        
        sample_size = min(len(self.metadata), k)
        return random.sample(self.metadata, sample_size)


# Global FAQ retriever instance
faq_retriever = FAQRetriever()


@tool
def faq_agent_tool(user_message: str) -> str:
    """
    FAQ Agent - Answers informational questions using semantic search.
    
    Use this tool when the user asks:
    - Questions about services, pricing, features
    - How things work
    - Company/product information
    - "What is...", "How do...", "Tell me about..."
    
    Args:
        user_message: The user's question
        
    Returns:
        Answer from FAQ database or list of common questions
    """
    
    # Check if user wants to see FAQ list
    if user_message.lower() in ["faq", "show faq", "help", "what can you do"]:
        # Show random FAQs
        random_faqs = faq_retriever.get_random_faqs(k=5)
        
        if random_faqs:
            response = "📚 **Common Questions:**\n\n"
            for i, faq in enumerate(random_faqs, 1):
                response += f"{i}. {faq['question']}\n"
            
            response += "\n💬 Feel free to ask any of these questions or ask your own!"
        else:
            response = "❓ How can I help you today? Feel free to ask any questions!"
        
        return response
    
    else:
        # Search for specific answer
        results = faq_retriever.search(user_message, top_k=1)
        
        if results and results[0]['similarity'] > 0.5:
            result = results[0]
            response = f"""✅ **Answer Found:**

**Q:** {result['question']}

**A:** {result['answer']}

---
Need more information or have another question? Just ask! 💬"""
        
        else:
            response = """❓ I couldn't find a specific answer to that question.

Would you like to:
1. **Rephrase** - Try asking in a different way
2. **See common questions** - Type "FAQ"
3. **Talk to a specialist** - Type "book a meeting"

How can I help?"""
        
        return response

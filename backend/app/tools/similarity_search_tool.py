"""
app/tools/similarity_search_tool.py
Vector database similarity search tool powered by Hugging Face Inference API and Pinecone.
"""

from langchain.tools import tool
import logging
from typing import Optional

logger = logging.getLogger(__name__)


@tool
def similarity_search_tool(query: str) -> str:
    """
    Knowledge Base & Document Similarity Search Tool.
    
    Use this tool whenever the user:
    - Asks any question about company products, services, policies, or documentation
    - Queries information from uploaded PDFs or ingested documents
    - Asks informational questions ("What is...", "How does...", "Can you tell me about...", "Do you provide...")
    - Needs factual answers grounded in the knowledge base
    
    Args:
        query: The search question or keywords to search for in the vector database.
        
    Returns:
        Relevant knowledge excerpts with sources and similarity relevance scores.
    """
    from app.main import persistence
    
    logger.info(f"🔍 [similarity_search_tool] Searching for: '{query}'")
    
    try:
        # 1. Search in Pinecone via Hugging Face Inference API embeddings (top_k=4)
        matches = persistence.similarity_search(query=query, top_k=4, min_score=0.20)
        
        if matches:
            context_blocks = []
            for i, match in enumerate(matches, 1):
                source = match.get("source", "Uploaded Document")
                text = match.get("text", "").strip()
                score = match.get("score", 0.0)
                context_blocks.append(f"--- Document Excerpt {i} (Source: {source} | Similarity: {score * 100:.1f}%) ---\n{text}")
                
            formatted = "\n\n".join(context_blocks)
            logger.info(f"✅ [similarity_search_tool] Retrieved {len(matches)} matching chunks for query '{query[:30]}'")
            return f"Found relevant information in vector database:\n\n{formatted}"
        
        return f"No matching documents or records found in the knowledge base for query: '{query}'."
        
    except Exception as e:
        logger.error(f"❌ Error during similarity search: {e}", exc_info=True)
        return f"Unable to complete vector similarity search: {str(e)}"


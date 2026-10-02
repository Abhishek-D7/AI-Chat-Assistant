"""
app/ml/llm_client.py
Resilient LLM client wrapper for OpenRouter free tier.
Automatically falls back across top free instruct models and intercepts/rejects
content-safety classifier outputs (e.g. 'User Safety: safe').
"""

import logging
import re
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage
from app.config import Config

logger = logging.getLogger(__name__)

# Curated list of active, completely FREE chat/instruct models on OpenRouter
# (Excludes content-safety classifiers like nvidia/nemotron-3.5-content-safety)
FREE_CHAT_MODELS: List[str] = [
    "liquid/lfm-2.5-2.6b:free",
    "qwen/qwen3.8-27b:free",
    "inclusionai/ling-3.0-flash-sante:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free"
]

def is_safety_or_invalid_output(text: str) -> bool:
    """Check if the text is merely a content safety classification rather than a real answer."""
    if not text or not text.strip():
        return True
    cleaned = text.strip().lower()
    # Reject NVIDIA Nemotron Content Safety and similar guardrail model outputs
    if "user safety:" in cleaned:
        return True
    if cleaned in ["safe", "unsafe", "user safety: safe", "user safety: unsafe"]:
        return True
    return False

def clean_llm_text(text: str) -> str:
    """Strip internal reasoning tags (<think>...</think>) from open reasoning models."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text.strip()

def get_resilient_llm(streaming: bool = True):
    """
    Returns a ChatOpenAI instance using the top free instruct model
    with built-in LangChain fallbacks to subsequent free models.
    """
    primary_model = FREE_CHAT_MODELS[0]
    fallback_models = FREE_CHAT_MODELS[1:]
    
    primary = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=Config.OPENROUTER_API_KEY,
        model=primary_model,
        temperature=0.2,
        streaming=streaming,
        timeout=25
    )
    
    fallbacks = [
        ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=Config.OPENROUTER_API_KEY,
            model=m,
            temperature=0.2,
            streaming=streaming,
            timeout=25
        )
        for m in fallback_models
    ]
    
    return primary.with_fallbacks(fallbacks)

def invoke_resilient_chat(messages: List[BaseMessage], streaming: bool = False) -> AIMessage:
    """
    Invokes OpenRouter across the curated free instruct models.
    If an upstream model is rate-limited (429) or outputs a safety classification
    (e.g., 'User Safety: safe'), it automatically switches to the next free model.
    """
    last_error = None
    
    for model_name in FREE_CHAT_MODELS:
        try:
            logger.info(f"🤖 Invoking free model: {model_name}")
            client = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=Config.OPENROUTER_API_KEY,
                model=model_name,
                temperature=0.2,
                streaming=streaming,
                timeout=25
            )
            response = client.invoke(messages)
            raw_content = str(getattr(response, "content", "")).strip()
            
            # Check if this model produced a safety classification tag
            if is_safety_or_invalid_output(raw_content):
                logger.warning(
                    f"⚠️ Model '{model_name}' returned a safety classifier response: '{raw_content}'. "
                    f"Discarding and falling back to next instruct model."
                )
                continue
                
            # Clean think tags if any
            clean_content = clean_llm_text(raw_content)
            response.content = clean_content
            return response
            
        except Exception as e:
            logger.warning(f"⚠️ Model '{model_name}' failed ({e}). Falling back to next free model...")
            last_error = e
            continue
            
    # Final emergency fallback: openrouter/free
    logger.warning("⚠️ All curated models had issues, trying openrouter/free fallback...")
    emergency_client = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=Config.OPENROUTER_API_KEY,
        model="openrouter/free",
        temperature=0.2,
        timeout=30
    )
    resp = emergency_client.invoke(messages)
    content = str(getattr(resp, "content", "")).strip()
    if is_safety_or_invalid_output(content):
        resp.content = (
            "I encountered a temporary service limitation with the free model pool. "
            "Please try submitting your question again in a moment."
        )
    else:
        resp.content = clean_llm_text(content)
    return resp

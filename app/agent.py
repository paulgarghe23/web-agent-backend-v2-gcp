# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import time
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.utils.rag import search

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

_client = None

def _get_client() -> ChatVertexAI:
    global _client
    if _client is None:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "europe-west1")
        if not project_id:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set")
        logger.info("VERTEX_AI_CLIENT_INITIALIZED", extra={
            "event": "vertex_ai_client_created",
            "project_id": project_id,
            "location": location,
        })
        _client = ChatVertexAI(
            model_name="gemini-2.5-flash",
            project=project_id,
            location=location,
            temperature=0.2,
            max_tokens=300,
        )
    return _client


def answer_question(question: str) -> str:
    """Direct LLM call like the working project."""
    total_start = time.time()
    
    logger.info("ANSWER_QUESTION_START", extra={
        "event": "answer_question_started",
        "question": question,
        "question_length": len(question),
    })
    
    # Get context from RAG
    rag_start = time.time()
    context = search(question)
    rag_time = time.time() - rag_start
    
    logger.info("RAG_SEARCH_COMPLETED", extra={
        "event": "rag_search_completed",
        "question": question,
        "context_found": bool(context),
        "context_length": len(context) if context else 0,
        "context_preview": context[:300] + "..." if context and len(context) > 300 else (context if context else ""),
        "rag_time_seconds": round(rag_time, 3),
    })
    
    if not context:
        logger.warning("NO_CONTEXT_FOUND", extra={
            "event": "no_context_found",
            "question": question,
        })
        return "I don't have that information in my knowledge base yet and don't know it."
    
    # Direct LLM call with same format as working project
    system_prompt = (
        "You are Paul's personal AI agent. "
        "Answer using the provided context about Paul. "
        "Synthesize information into your own words - never copy text verbatim. "
        "Keep answers brief (2-4 sentences). "
        "If the answer is not in the context, say you don't know based on the information you have been provided until now. "
        "Always reply in the same language the user asks."
    )
    
    user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
    
    # LangChain message format for Vertex AI
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    
    # LOG: Llamada al LLM
    prompt_tokens_estimate = len(system_prompt + context + question) // 4  # Rough estimate
    logger.info("LLM_CALL_STARTED", extra={
        "event": "llm_call_started",
        "question": question,
        "model": "gemini-2.5-flash",
        "temperature": 0.2,
        "max_tokens": 300,
        "system_prompt_length": len(system_prompt),
        "user_message_length": len(user_content),
        "estimated_prompt_tokens": prompt_tokens_estimate,
    })
    
    llm_start = time.time()
    llm = _get_client()
    resp = llm.invoke(messages)
    llm_time = time.time() - llm_start
    
    answer = resp.content.strip() if hasattr(resp, 'content') else str(resp).strip()
    total_time = time.time() - total_start
    
    # Extract token usage (Vertex AI response structure)
    tokens_prompt = 0
    tokens_completion = 0
    tokens_total = 0
    if hasattr(resp, 'response_metadata'):
        usage = resp.response_metadata.get('token_count', {}) if resp.response_metadata else {}
        tokens_prompt = usage.get("prompt_token_count", 0)
        tokens_completion = usage.get("candidates_token_count", 0)
        tokens_total = tokens_prompt + tokens_completion
    
    # LOG: Respuesta del LLM
    logger.info("LLM_CALL_COMPLETED", extra={
        "event": "llm_call_completed",
        "question": question,
        "answer": answer,
        "answer_length": len(answer),
        "llm_time_seconds": round(llm_time, 3),
        "total_time_seconds": round(total_time, 3),
        "tokens_prompt": tokens_prompt,
        "tokens_completion": tokens_completion,
        "tokens_total": tokens_total,
        "tokens_per_second": round(tokens_total / llm_time, 2) if llm_time > 0 else 0,
        "model": "gemini-2.5-flash",
        "finish_reason": getattr(resp, 'response_metadata', {}).get('finish_reason', 'unknown') if hasattr(resp, 'response_metadata') else "unknown",
    })
    
    return answer

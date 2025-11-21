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
from openai import OpenAI

from app.utils.rag import search

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

_client = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        logger.info("OPENAI_CLIENT_INITIALIZED", extra={
            "event": "openai_client_created",
            "api_key_prefix": api_key[:7] + "..." if api_key else "None",
        })
        _client = OpenAI(api_key=api_key)
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
    
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
        },
    ]
    
    # LOG: Llamada al LLM
    prompt_tokens_estimate = len(system_prompt + context + question) // 4  # Rough estimate
    logger.info("LLM_CALL_STARTED", extra={
        "event": "llm_call_started",
        "question": question,
        "model": "gpt-4o-mini",
        "temperature": 0.2,
        "max_tokens": 300,
        "system_prompt_length": len(system_prompt),
        "user_message_length": len(messages[1]["content"]),
        "estimated_prompt_tokens": prompt_tokens_estimate,
    })
    
    llm_start = time.time()
    resp = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=300,
    )
    llm_time = time.time() - llm_start
    
    answer = resp.choices[0].message.content.strip()
    total_time = time.time() - total_start
    
    # Extract token usage
    usage = resp.usage.model_dump() if hasattr(resp, 'usage') and resp.usage else {}
    tokens_prompt = usage.get("prompt_tokens", 0)
    tokens_completion = usage.get("completion_tokens", 0)
    tokens_total = usage.get("total_tokens", 0)
    
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
        "model": "gpt-4o-mini",
        "finish_reason": resp.choices[0].finish_reason if hasattr(resp.choices[0], 'finish_reason') else "unknown",
    })
    
    return answer

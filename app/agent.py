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
from dotenv import load_dotenv
from openai import OpenAI

from app.utils.rag import search

# Load environment variables from .env file
load_dotenv()

_client = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        _client = OpenAI(api_key=api_key)
    return _client


def answer_question(question: str) -> str:
    """Direct LLM call like the working project."""
    # Get context from RAG
    context = search(question)
    
    if not context:
        return "I don't have that information in my knowledge base yet and don't know it."
    
    # Direct LLM call with same format as working project
    messages = [
        {
            "role": "system",
            "content": (
                "You are Paul's personal AI agent. "
                "Answer using the provided context about Paul. "
                "Synthesize information into your own words - never copy text verbatim. "
                "Keep answers brief (2-4 sentences). "
                "If the answer is not in the context, say you don't know based on the information you have been provided until now. "
                "Always reply in the same language the user asks."
            ),
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
        },
    ]
    
    resp = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()

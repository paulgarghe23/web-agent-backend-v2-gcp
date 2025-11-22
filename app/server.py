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

import logging
import os
from collections.abc import Generator
import time

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import logging as google_cloud_logging
from langchain_core.runnables import RunnableConfig
from traceloop.sdk import Instruments, Traceloop

from app.agent import get_agent
from app.utils.tracing import CloudTraceLoggingSpanExporter
from app.utils.typing import Feedback, InputChat, Request, dumps, ensure_valid_config

# Load environment variables from .env file
load_dotenv()


# Initialize FastAPI app and logging
app = FastAPI(
    title="web-agent",
    description="API for interacting with the Agent web-agent",
)

# Use standard Python logging for local dev, Google Cloud Logging for production
# Check if running locally (not in Cloud Run/GKE) by checking environment
is_local = os.getenv("K_SERVICE") is None and os.getenv("GAE_SERVICE") is None

if is_local:
    # LOCAL: Always use console logging when running locally with detailed format
    class DetailedFormatter(logging.Formatter):
        def format(self, record):
            # Base format
            result = super().format(record)
            
            # Add file location
            if hasattr(record, 'pathname'):
                filename = record.pathname.split('/')[-1] if '/' in record.pathname else record.pathname
                func_name = record.funcName if hasattr(record, 'funcName') else 'unknown'
                result += f"\n  📁 {filename}:{record.lineno} in {func_name}()"
            
            # Add extra fields if they exist
            if hasattr(record, 'event'):
                result += f"\n  📍 Event: {record.event}"
            if hasattr(record, 'question'):
                result += f"\n  ❓ Question: {record.question[:200]}{'...' if len(getattr(record, 'question', '')) > 200 else ''}"
            if hasattr(record, 'answer'):
                result += f"\n  💬 Answer: {record.answer[:200]}{'...' if len(getattr(record, 'answer', '')) > 200 else ''}"
            if hasattr(record, 'context_preview'):
                preview = getattr(record, 'context_preview', '')
                context_len = getattr(record, 'context_length', 0)
                result += f"\n  📄 Context ({context_len} chars): {preview[:600]}{'...' if len(preview) > 600 else ''}"
            if hasattr(record, 'context_full') and len(getattr(record, 'context_full', '')) < 2000:
                result += f"\n  📄 Full Context: {getattr(record, 'context_full', '')}"
            if hasattr(record, 'scores'):
                result += f"\n  🎯 Scores: {record.scores}"
            if hasattr(record, 'tokens_total'):
                result += f"\n  🪙 Tokens: {record.tokens_total} (prompt: {getattr(record, 'tokens_prompt', 0)}, completion: {getattr(record, 'tokens_completion', 0)})"
            if hasattr(record, 'total_time_seconds'):
                result += f"\n  ⏱️  Time: {record.total_time_seconds}s"
            if hasattr(record, 'rag_time_seconds'):
                result += f"\n  🔍 RAG: {record.rag_time_seconds}s"
            if hasattr(record, 'llm_time_seconds'):
                result += f"\n  🤖 LLM: {record.llm_time_seconds}s"
            if hasattr(record, 'chunks_used'):
                result += f"\n  📚 Chunks: {record.chunks_used}"
            if hasattr(record, 'vector_count'):
                result += f"\n  🗂️  Vectors: {record.vector_count}"
            if hasattr(record, 'synthesized_answer'):
                result += f"\n  🔧 Tool Answer: {record.synthesized_answer}"
            if hasattr(record, 'answer_being_returned'):
                result += f"\n  🔧 Tool Returning: {record.answer_being_returned}"
            if hasattr(record, 'content_full'):
                result += f"\n  📝 Content Full: {record.content_full}"
            if hasattr(record, 'document_content_full'):
                result += f"\n  📄 Document Content Full: {record.document_content_full}"
            if hasattr(record, 'is_from_cv'):
                result += f"\n  📋 Is From CV: {record.is_from_cv}"
            if hasattr(record, 'document_source'):
                result += f"\n  📁 Document Source: {record.document_source}"
            if hasattr(record, 'message_index'):
                result += f"\n  📊 Message #{record.message_index}"
            if hasattr(record, 'is_duplicate'):
                result += f"\n  ⚠️  Is Duplicate: {record.is_duplicate}"
            if hasattr(record, 'messages_detail'):
                result += f"\n  📋 Messages Detail:"
                for msg in record.messages_detail:
                    result += f"\n    - #{msg.get('index', '?')} [{msg.get('type', '?')}] ({msg.get('length', 0)} chars): {msg.get('content_full', '')[:100]}"
            if hasattr(record, 'has_content'):
                result += f"\n  ✅ Has Content: {record.has_content}"
            if hasattr(record, 'content_type'):
                result += f"\n  📦 Content Type: {record.content_type}"
            if hasattr(record, 'raw_content_length'):
                result += f"\n  📏 Raw Content Length: {record.raw_content_length}"
            if hasattr(record, 'raw_content_full'):
                result += f"\n  🔍 RAW CONTENT FULL: {record.raw_content_full}"
            
            return result
    
    handler = logging.StreamHandler()
    handler.setFormatter(DetailedFormatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
        force=True,
    )
elif os.getenv("GOOGLE_CLOUD_PROJECT"):
    # Production: Use Google Cloud Logging with structured logging
    try:
        logging_client = google_cloud_logging.Client()
        # Setup logging to capture extra fields in jsonPayload
        logging_client.setup_logging(log_level=logging.INFO)
        # Create a custom handler that properly formats structured logs
        cloud_handler = logging_client.get_default_handler()
        if cloud_handler:
            cloud_handler.setLevel(logging.INFO)
    except Exception as e:
        # Fallback to console if Cloud Logging fails
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )
        logging.warning(f"Failed to setup Cloud Logging, using console: {e}")

logger = logging.getLogger(__name__)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://paulgarghe.com",
        "https://www.paulgarghe.com",
        "https://agent.paulgarghe.com",
        "https://identity-forge-page-4ac4aatga-paul-iulian-garghes-projects.vercel.app",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Telemetry
try:
    Traceloop.init(
        app_name=app.title,
        disable_batch=False,
        exporter=CloudTraceLoggingSpanExporter(),
        instruments={Instruments.LANGCHAIN, Instruments.CREW},
    )
except Exception as e:
    logging.error("Failed to initialize Telemetry: %s", str(e))


def set_tracing_properties(config: RunnableConfig) -> None:
    """Sets tracing association properties for the current request.

    Args:
        config: Optional RunnableConfig containing request metadata
    """
    Traceloop.set_association_properties(
        {
            "log_type": "tracing",
            "run_id": str(config.get("run_id", "None")),
            "user_id": config["metadata"].pop("user_id", "None"),
            "session_id": config["metadata"].pop("session_id", "None"),
            "commit_sha": os.environ.get("COMMIT_SHA", "None"),
        }
    )


def stream_messages(
    input: InputChat,
    config: RunnableConfig | None = None,
) -> Generator[str, None, None]:
    """Stream events in response to an input chat.

    Args:
        input: The input chat messages
        config: Optional configuration for the runnable

    Yields:
        JSON serialized event data
    """
    request_start = time.time()
    
    try:
        config = ensure_valid_config(config=config)
        set_tracing_properties(config)
        
        # LOG: Request recibido
        logger.info("API_REQUEST_RECEIVED", extra={
            "event": "request_received",
            "messages_count": len(input.messages),
            "run_id": str(config.get("run_id", "None")),
            "user_id": config.get("metadata", {}).get("user_id", "None"),
            "session_id": config.get("metadata", {}).get("session_id", "None"),
        })
        
        # Prepare input for LangGraph agent
        messages = input.messages
        if not messages:
            logger.warning("INVALID_REQUEST_NO_MESSAGES", extra={
                "event": "invalid_request",
                "messages_count": 0,
            })
            error_message = {"type": "error", "content": "No messages found"}
            yield dumps(error_message) + "\n"
            return
        
        # Log conversation context
        last_message = messages[-1] if messages else None
        question = None
        if last_message and hasattr(last_message, 'content'):
            question = last_message.content
            if isinstance(question, list):
                text_parts = [part.get('text', '') for part in question if isinstance(part, dict) and part.get('type') == 'text']
                question = ' '.join(text_parts) if text_parts else str(question)
            elif not isinstance(question, str):
                question = str(question)
        
        logger.info("CONVERSATION_CONTEXT", extra={
            "event": "conversation_context",
            "messages_count": len(messages),
            "last_message_type": last_message.type if last_message else "None",
            "question": question if question else "None",
            "has_conversation_history": len(messages) > 1,
        })
        
        # Stream from LangGraph agent
        agent = get_agent()
        input_dict = {"messages": messages}
        chunks_sent = 0
        final_answer = ""
        
        logger.info("LANGGRAPH_STREAM_START", extra={
            "event": "langgraph_stream_start",
            "messages_count": len(messages),
        })
        
        # With stream_mode="values", we get the full state after each step
        # We'll extract the last AI message from the final state
        last_state = None
        tool_call_logged = False  # Only log tool call once
        
        for state in agent.stream(input_dict, config=config, stream_mode="values"):
            # state is a dict with the full graph state
            last_state = state
            
            # Log tool calls only once (first time we see them)
            if not tool_call_logged:
                messages = state.get("messages", [])
                for msg in messages:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        tool_names = [tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown') for tc in msg.tool_calls]
                        logger.info("TOOL_CALL_DETECTED", extra={
                            "event": "tool_call_detected",
                            "tool_calls_count": len(msg.tool_calls),
                            "tool_names": tool_names,
                        })
                        tool_call_logged = True
                        break
        
        # After stream completes, extract the final AI message from the last state
        if last_state:
            messages = last_state.get("messages", [])
            # Find the last AI message (without tool_calls)
            final_ai_message = None
            for msg in reversed(messages):
                if hasattr(msg, 'type') and msg.type == "ai":
                    if not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                        final_ai_message = msg
                        break
            
            if final_ai_message:
                # Extract content
                content = ""
                if hasattr(final_ai_message, 'content'):
                    content = final_ai_message.content
                    if isinstance(content, list):
                        text_parts = [part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text']
                        content = ' '.join(text_parts) if text_parts else str(content)
                    elif not isinstance(content, str):
                        content = str(content)
                
                if content:
                    logger.info("FINAL_AI_MESSAGE_EXTRACTED", extra={
                        "event": "final_ai_message_extracted",
                        "content_full": content,
                        "content_length": len(content),
                    })
                    
                    final_answer = content
                    
                    # Send the complete response
                    yield dumps({"type": "AIMessage", "content": content}) + "\n"
                    chunks_sent = 1
        
        request_time = time.time() - request_start
        
        # LOG: Respuesta generada
        logger.info("ANSWER_GENERATED", extra={
            "event": "answer_generated",
            "question": question if question else "conversation",
            "answer": final_answer,
            "answer_length": len(final_answer),
            "total_request_time_seconds": round(request_time, 3),
            "run_id": str(config.get("run_id", "None")),
            "chunks_sent": chunks_sent,
        })
        
        logger.info("LANGGRAPH_STREAM_COMPLETED", extra={
            "event": "langgraph_stream_completed",
            "chunks_sent": chunks_sent,
            "total_chars": len(final_answer),
        })
            
    except Exception as e:
        request_time = time.time() - request_start
        logger.error("ERROR_IN_STREAM_MESSAGES", extra={
            "event": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "request_time_seconds": round(request_time, 3),
        }, exc_info=True)
        error_message = {"type": "error", "content": f"Error: {str(e)}"}
        yield dumps(error_message) + "\n"


# Routes
@app.get("/", response_class=RedirectResponse)
def redirect_root_to_docs() -> RedirectResponse:
    """Redirect the root URL to the API documentation."""
    return RedirectResponse(url="/docs")


@app.post("/stream_messages")
def stream_chat_events(request: Request) -> StreamingResponse:
    """Stream chat events in response to an input request.

    Args:
        request: The chat request containing input and config

    Returns:
        Streaming response of chat events
    """
    return StreamingResponse(
        stream_messages(input=request.input, config=request.config),
        media_type="text/event-stream",
    )


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
    logger.info(f"Feedback received: {feedback.model_dump()}")
    return {"status": "success"}


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

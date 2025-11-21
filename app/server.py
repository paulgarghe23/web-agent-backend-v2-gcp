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

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import logging as google_cloud_logging
from langchain_core.runnables import RunnableConfig
from traceloop.sdk import Instruments, Traceloop

from app.agent import answer_question
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
if os.getenv("GOOGLE_CLOUD_PROJECT"):
    logging_client = google_cloud_logging.Client()
    logging_client.setup_logging()  # This hooks into standard logging
else:
    # LOCAL: Configurar logging para ver logs en consola
    logging.basicConfig(
        level=logging.INFO,  # Ver INFO, WARNING, ERROR
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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
    try:
        config = ensure_valid_config(config=config)
        set_tracing_properties(config)
        
        # Get last user message
        messages = input.messages
        last_message = messages[-1] if messages else None
        if not last_message or last_message.type != "human":
            error_message = {"type": "error", "content": "No valid question found"}
            yield dumps(error_message) + "\n"
            return
        
        question = last_message.content
        # Extract text from content if it's a list of content parts
        if isinstance(question, list):
            text_parts = [part.get('text', '') for part in question if isinstance(part, dict) and part.get('type') == 'text']
            question = ' '.join(text_parts) if text_parts else str(question)
        elif not isinstance(question, str):
            question = str(question)
        
        # Direct call - get answer
        answer = answer_question(question)
        
        # Stream answer as chunks (simulate streaming for compatibility)
        chunk_size = 10
        for i in range(0, len(answer), chunk_size):
            chunk = answer[i:i + chunk_size]
            yield dumps({"type": "AIMessageChunk", "content": chunk}) + "\n"
            
    except Exception as e:
        logger.error(f"Error in stream_messages: {str(e)}", exc_info=True)
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
